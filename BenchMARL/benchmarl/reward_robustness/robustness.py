"""Glue between the experiment loop and the reward-robustness components.

Per collected batch, :meth:`RewardRobustness.process`

1. keeps a copy of the true rewards (only used by the oracle and the diagnostics),
2. perturbs every group's rewards with the kernel K (if enabled),
3. groups the agents of the task's agent group and replaces their observed rewards
   by the consensus rewards (Algorithm 1, lines 9-14),
4. optionally computes grouping / reward-error diagnostics.
"""

from typing import Dict, Optional, Tuple

import torch
from tensordict import TensorDictBase

from benchmarl.reward_robustness.consensus import apply_consensus, Assignment
from benchmarl.reward_robustness.groupers import (
    GroupData,
    Grouper,
    GroupingConfig,
    make_grouper,
    oracle_labels,
)
from benchmarl.reward_robustness.metrics import (
    adjusted_rand_index,
    pairwise_grouping_scores,
    reward_error_metrics,
)
from benchmarl.reward_robustness.perturbation import perturb_rewards
from benchmarl.reward_robustness.task_spec import get_task_spec


def _resolve_auto(value: str, perturbation_type: str, if_normal: str, otherwise: str) -> str:
    if value != "auto":
        return value
    return if_normal if perturbation_type == "normal" else otherwise


class RewardRobustness:
    """Reward perturbation + grouped-consensus correction for one experiment.

    Args:
        config: the :class:`~benchmarl.experiment.ExperimentConfig`.
        task_name: lower-case task name (e.g. ``"pursuit"``).
        group_map: the experiment's group map (all groups get perturbed rewards).
        action_spec: the experiment's action spec (to one-hot encode discrete actions).
        device: device on which grouping networks are trained.
    """

    def __init__(self, config, task_name: str, group_map, action_spec, device="cpu"):
        self.config = config
        self.group_map = group_map
        self.perturbation = config.reward_perturbation
        self.perturbation_type = config.reward_perturbation_type
        self.consensus = _resolve_auto(
            config.consensus, self.perturbation_type, "mean", "median"
        )
        self.method = config.grouping_method
        self.active = self.method != "none" or config.evaluate_grouping

        self.spec = get_task_spec(task_name) if self.active else None
        self.agent_group = self.spec.agent_group if self.spec is not None else None
        if self.agent_group is not None and self.agent_group not in group_map:
            raise ValueError(
                f"Agent group '{self.agent_group}' not in group map {list(group_map)}"
            )

        self.grouping_cfg = GroupingConfig(
            method=self.method,
            num_groups=config.num_groups,
            epochs=config.grouping_epochs,
            lr=config.grouping_lr,
            batch_size=config.grouping_batch_size,
            use_next_obs=config.grouping_use_next_obs,
            out_of_sample=config.grouping_out_of_sample,
            gmm_sort=config.grouping_gmm_sort,
            reward_model_loss=_resolve_auto(
                config.reward_model_loss, self.perturbation_type, "mse", "l1"
            ),
            reward_model_assign=config.reward_model_assign,
            reward_model_residual=config.reward_model_residual,
            epsilon=config.grouping_epsilon,
        )
        self.grouper: Optional[Grouper] = (
            make_grouper(self.grouping_cfg, self.spec, device) if self.method != "none" else None
        )
        self.n_discrete_actions = (
            self._discrete_action_size(action_spec) if self.active else None
        )

    # ------------------------------------------------------------------ helpers
    def _discrete_action_size(self, action_spec) -> Optional[int]:
        spec = action_spec[self.agent_group, "action"]
        space = getattr(spec, "space", None)
        n = getattr(space, "n", None)
        return int(n) if n is not None else None

    def _group_data(self, batch: TensorDictBase, true_reward: Optional[torch.Tensor]) -> GroupData:
        g = self.agent_group
        action = batch.get((g, "action"))
        if not torch.is_floating_point(action):
            action = torch.nn.functional.one_hot(action.long(), self.n_discrete_actions).float()
        if self.config.grouping_use_agent_id:
            # Append a one-hot agent index so models can represent role-dependent rewards
            n_agents = action.shape[-2]
            agent_id = torch.eye(n_agents, dtype=action.dtype, device=action.device)
            action = torch.cat([action, agent_id.expand(*action.shape[:-1], n_agents)], -1)
        reward = batch.get(("next", g, "reward")).squeeze(-1)
        mask = batch.get((g, "mask"), None)
        mask = torch.ones_like(reward, dtype=torch.bool) if mask is None else mask.bool()
        return GroupData(
            obs=batch.get((g, "observation")),
            next_obs=batch.get(("next", g, "observation")),
            action=action,
            reward=reward,
            mask=mask,
            true_reward=true_reward,
        )

    # --------------------------------------------------------------- interface
    @property
    def needs_warmup(self) -> bool:
        return self.grouper is not None and self.grouper.needs_warmup

    def state_dict(self) -> Dict:
        return {"grouper": self.grouper.state_dict()} if self.grouper is not None else {}

    def load_state_dict(self, state: Dict) -> None:
        if self.grouper is not None and "grouper" in state:
            self.grouper.load_state_dict(state["grouper"])

    def warmup_update(self, batch: TensorDictBase) -> Dict[str, float]:
        """Train the grouper on a batch with clean (unperturbed) rewards."""
        reward = batch.get(("next", self.agent_group, "reward")).squeeze(-1)
        return self.grouper.warmup_update(self._group_data(batch, true_reward=reward))

    def process(self, batch: TensorDictBase) -> Tuple[TensorDictBase, Dict[str, float]]:
        """Perturb rewards and apply the grouped consensus correction in place."""
        true_reward = (
            batch.get(("next", self.agent_group, "reward")).squeeze(-1).clone()
            if self.active
            else None
        )
        if self.perturbation:
            for group in self.group_map.keys():
                key = ("next", group, "reward")
                batch.set(
                    key,
                    perturb_rewards(
                        batch.get(key),
                        self.perturbation_type,
                        stdev=self.config.reward_perturbation_stdev,
                        flip_prob=self.config.reward_perturbation_flip_prob,
                        outlier_value=self.config.reward_perturbation_outlier_value,
                    ),
                )
        if not self.active:
            return batch, {}

        data = self._group_data(batch, true_reward=true_reward)
        if self.grouper is not None:
            assignment, logs = self.grouper(data)
        else:
            assignment, logs = Assignment(), {}
        corrected = apply_consensus(
            data.reward,
            assignment,
            mask=data.mask,
            method=self.consensus,
            trim=self.config.consensus_trim,
        )
        batch.set(("next", self.agent_group, "reward"), corrected.unsqueeze(-1))

        if self.config.evaluate_grouping:
            logs.update(self.diagnostics(data, assignment, corrected))
        return batch, {f"reward_robustness/{k}": v for k, v in logs.items()}

    def diagnostics(
        self, data: GroupData, assignment: Assignment, corrected: torch.Tensor
    ) -> Dict[str, float]:
        truth = oracle_labels(data.true_reward, self.spec, self.config.num_groups, data.mask)
        logs = reward_error_metrics(
            data.true_reward,
            data.reward,
            corrected,
            mask=data.mask,
            levels=truth,
            event_level=self.config.num_groups - 1,
        )
        if assignment.labels is not None:
            logs.update(pairwise_grouping_scores(truth, assignment.labels, data.mask))
            logs["ari"] = adjusted_rand_index(truth, assignment.labels, data.mask)
        return logs

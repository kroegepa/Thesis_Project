"""Grouping functions F (Definition 3 / Algorithm 1 of the paper).

A grouper receives the transitions of one iteration (:class:`GroupData`) and returns
an :class:`~benchmarl.reward_robustness.consensus.Assignment` describing how each
agent's corrected reward is formed.

Methods (``grouping_method`` in the experiment config):

========================  ===============================================================
``oracle``                levels of the *true* reward (upper bound, paper "Oracle")
``global``                all agents in one group (paper ablation, Fig. 5)
``warmup``                classifier trained on oracle labels during a clean warm-up
                          phase, frozen afterwards (paper "Warmup")
``gmm``                   classifier trained online on GMM clusters of the *noisy*
                          rewards (paper "No-Warmup", see ``gmm_sort``/``out_of_sample``)
``reward_model``          **new, warm-up free**: regress the noisy rewards on
                          ``(o, a[, o'])``, then group agents whose *predicted* rewards
                          are similar (k-means / ε-gap / kernel weights), optionally with
                          residual consensus
``reward_model_only``     baseline: replace rewards by the learned reward model
                          (reward-estimation baseline in the spirit of Romoff et al. 2018)
========================  ===============================================================

Why the reward-model grouper works without clean data: under Assumption 2 the noise
is zero-mean and independent of the transition, so the MSE minimiser of
``E[(f(o,a,o') - r~)^2]`` is ``E[r~ | o,a,o'] = E[r | o,a,o']``: the regression target is
identifiable from corrupted rewards alone (for non-zero-mean kernels such as the
Bernoulli one, an L1 loss recovers the conditional median, which still orders the
reward levels correctly).

Why ``out_of_sample`` matters: if the groups of a batch are computed from a model
that was fitted on that same batch (or directly from its noisy rewards, as a GMM
does), the grouping depends on the noise realisation and agents with similar *noise*
get grouped together, which biases the consensus. With ``out_of_sample=True`` the
grouping of iteration ``k`` uses only parameters fitted on iterations ``< k``, so it
is independent of the current noise and the unbiasedness/variance results
(Prop. 1, Lemma 1), which condition on a fixed grouping, apply as stated.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from benchmarl.reward_robustness.clustering import (
    epsilon_labels,
    gmm_labels,
    kmeans_1d_labels,
    threshold_labels,
)
from benchmarl.reward_robustness.consensus import Assignment
from benchmarl.reward_robustness.models import TransitionNet
from benchmarl.reward_robustness.task_spec import TaskSpec


@dataclass
class GroupingConfig:
    """Hyper-parameters of the groupers (mirrors the ``grouping_*`` experiment keys)."""

    method: str = "none"
    num_groups: int = 3
    epochs: int = 35
    lr: float = 1e-3
    batch_size: int = 512
    use_next_obs: bool = False
    out_of_sample: bool = True
    gmm_sort: bool = True
    reward_model_loss: str = "mse"
    reward_model_assign: str = "kmeans"
    reward_model_residual: bool = False
    epsilon: float = 0.5


@dataclass
class GroupData:
    """Transitions of one agent group for one iteration.

    Shapes: ``obs``/``next_obs`` ``[B, T, A, *obs_shape]``, ``action`` ``[B, T, A, act_dim]``
    (encoded, float), ``reward``/``mask`` ``[B, T, A]``. ``true_reward`` is only
    available to the oracle, the warm-up phase and the diagnostics.
    """

    obs: torch.Tensor
    next_obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    mask: torch.Tensor
    true_reward: Optional[torch.Tensor] = None

    @property
    def obs_shape(self):
        return tuple(self.obs.shape[3:])

    @property
    def act_dim(self) -> int:
        return self.action.shape[-1]


def oracle_labels(
    true_reward: torch.Tensor, spec: TaskSpec, num_groups: int, mask: torch.Tensor
) -> torch.Tensor:
    """Reward levels of the true rewards (thresholds if the task defines them, else GMM)."""
    if spec.oracle_thresholds is not None:
        thresholds = spec.oracle_thresholds[: max(num_groups - 1, 0)]
        labels = threshold_labels(true_reward, thresholds, right=spec.oracle_right)
        return torch.where(mask, labels, torch.full_like(labels, -1))
    return gmm_labels(true_reward, num_groups, mask=mask, sort_components=True)


class Grouper:
    """Base class. Subclasses implement :meth:`assign` and optionally :meth:`update`."""

    needs_warmup: bool = False
    uses_true_reward: bool = False

    def __init__(self, cfg: GroupingConfig, spec: TaskSpec, device="cpu"):
        self.cfg = cfg
        self.spec = spec
        self.device = device

    def warmup_update(self, data: GroupData) -> Dict[str, float]:
        """Update from clean-reward data (warm-up phase)."""
        return {}

    def update(self, data: GroupData) -> Dict[str, float]:
        """Online update from the current (perturbed) batch."""
        return {}

    def assign(self, data: GroupData) -> Assignment:
        raise NotImplementedError

    def state_dict(self) -> Dict:
        return {}

    def load_state_dict(self, state: Dict) -> None:
        pass

    def __call__(self, data: GroupData) -> Tuple[Assignment, Dict[str, float]]:
        if self.cfg.out_of_sample:
            assignment = self.assign(data)
            logs = self.update(data)
        else:
            logs = self.update(data)
            assignment = self.assign(data)
        return assignment, logs


class OracleGrouper(Grouper):
    uses_true_reward = True

    def assign(self, data: GroupData) -> Assignment:
        return Assignment(
            labels=oracle_labels(data.true_reward, self.spec, self.cfg.num_groups, data.mask)
        )


class GlobalGrouper(Grouper):
    def assign(self, data: GroupData) -> Assignment:
        labels = torch.zeros_like(data.reward, dtype=torch.long)
        return Assignment(labels=torch.where(data.mask, labels, torch.full_like(labels, -1)))


class _NetworkGrouper(Grouper):
    """Shared machinery for groupers backed by a :class:`TransitionNet`."""

    out_dim: int = 1

    def __init__(self, cfg: GroupingConfig, spec: TaskSpec, device="cpu"):
        super().__init__(cfg, spec, device)
        self.net: Optional[TransitionNet] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.n_updates = 0

    def state_dict(self) -> Dict:
        if self.net is None:
            return {}
        return {
            "obs_shape": self.net.obs_shape,
            "act_dim": self.act_dim,
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "n_updates": self.n_updates,
        }

    def load_state_dict(self, state: Dict) -> None:
        if state:
            self._build(state["obs_shape"], state["act_dim"])
            self.net.load_state_dict(state["net"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.n_updates = state["n_updates"]

    def _ensure_net(self, data: GroupData):
        if self.net is None:
            self._build(data.obs_shape, data.act_dim)

    def _build(self, obs_shape, act_dim: int):
        if self.net is None:
            self.act_dim = act_dim
            self.net = TransitionNet(
                obs_shape=obs_shape,
                act_dim=act_dim,
                out_dim=self.out_dim,
                use_next_obs=self.cfg.use_next_obs,
                encoder=self.spec.encoder,
                cnn_channels=self.spec.cnn_channels,
                cnn_paddings=self.spec.cnn_paddings,
                hidden=self.spec.hidden,
            ).to(self.device)
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)

    def _flat_inputs(self, data: GroupData):
        obs = data.obs.reshape(-1, *data.obs_shape).to(self.device, torch.float32)
        act = data.action.reshape(-1, data.act_dim).to(self.device, torch.float32)
        nxt = (
            data.next_obs.reshape(-1, *data.obs_shape).to(self.device, torch.float32)
            if self.cfg.use_next_obs
            else None
        )
        return obs, act, nxt

    def _fit(self, data: GroupData, targets: torch.Tensor, loss_fn, select: torch.Tensor):
        """Train on the selected flattened samples for ``cfg.epochs`` epochs."""
        self._ensure_net(data)
        obs, act, nxt = self._flat_inputs(data)
        idx_all = select.reshape(-1).nonzero(as_tuple=True)[0].to(self.device)
        targets = targets.reshape(-1).to(self.device)
        self.net.train()
        last_loss = float("nan")
        for _ in range(self.cfg.epochs):
            perm = idx_all[torch.randperm(idx_all.numel(), device=self.device)]
            total, count = 0.0, 0
            for start in range(0, perm.numel(), self.cfg.batch_size):
                idx = perm[start : start + self.cfg.batch_size]
                out = self.net(obs[idx], act[idx], None if nxt is None else nxt[idx])
                loss = loss_fn(out, targets[idx])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total += loss.item() * idx.numel()
                count += idx.numel()
            last_loss = total / max(count, 1)
        self.n_updates += 1
        return last_loss

    @torch.no_grad()
    def _predict(self, data: GroupData, chunk: int = 65536) -> torch.Tensor:
        """Network outputs reshaped to ``[B, T, A, out_dim]``."""
        self._ensure_net(data)
        self.net.eval()
        obs, act, nxt = self._flat_inputs(data)
        outs = [
            self.net(
                obs[i : i + chunk], act[i : i + chunk], None if nxt is None else nxt[i : i + chunk]
            )
            for i in range(0, obs.shape[0], chunk)
        ]
        return torch.cat(outs).reshape(*data.reward.shape, self.out_dim).to(data.reward.device)


class _ClassifierGrouper(_NetworkGrouper):
    """Predicts a group id per agent; groups are agents sharing the predicted id."""

    def __init__(self, cfg: GroupingConfig, spec: TaskSpec, device="cpu"):
        super().__init__(cfg, spec, device)
        self.out_dim = cfg.num_groups

    def _fit_labels(self, data: GroupData, labels: torch.Tensor) -> float:
        ce = nn.CrossEntropyLoss()
        return self._fit(data, labels, lambda out, y: ce(out, y.long()), labels >= 0)

    def assign(self, data: GroupData) -> Assignment:
        if self.n_updates == 0 and not self.needs_warmup:
            return Assignment()  # nothing learned yet: leave rewards unchanged
        labels = self._predict(data).argmax(-1)
        return Assignment(labels=torch.where(data.mask, labels, torch.full_like(labels, -1)))


class WarmupClassifierGrouper(_ClassifierGrouper):
    """Paper "Warmup": supervised on oracle labels from a clean warm-up phase."""

    needs_warmup = True

    def warmup_update(self, data: GroupData) -> Dict[str, float]:
        labels = oracle_labels(data.true_reward, self.spec, self.cfg.num_groups, data.mask)
        return {"grouper/warmup_loss": self._fit_labels(data, labels)}


class GMMClassifierGrouper(_ClassifierGrouper):
    """Paper "No-Warmup": classifier trained on GMM clusters of the noisy rewards.

    ``gmm_sort=False, out_of_sample=False`` reproduces the original implementation.
    """

    def update(self, data: GroupData) -> Dict[str, float]:
        labels = gmm_labels(
            data.reward, self.cfg.num_groups, mask=data.mask, sort_components=self.cfg.gmm_sort
        )
        return {"grouper/loss": self._fit_labels(data, labels)}


_LOSSES = {"mse": nn.MSELoss, "huber": nn.HuberLoss, "l1": nn.L1Loss}


class RewardModelGrouper(_NetworkGrouper):
    """Warm-up-free grouping from a reward model fitted on the noisy rewards.

    ``reward_model_assign``:
        * ``kmeans``: 1-D k-means with ``num_groups`` clusters over all predictions of
          the batch (same number of groups as the paper's classifiers);
        * ``epsilon``: per-timestep ε-gap clustering of the predictions (ε = ``epsilon``,
          the tolerance of Assumption 1, in reward units);
        * ``kernel``: soft consensus weights ``exp(-(f_i - f_j)^2 / (2 ε^2))``;
        * ``global``: one group per timestep (useful together with residual consensus).

    ``reward_model_residual``: consensus is applied to residuals ``r~ - f`` and the
    prediction is added back, so only the *model errors* need to be ε-similar.
    """

    out_dim = 1
    replace_with_prediction = False

    def __init__(self, cfg: GroupingConfig, spec: TaskSpec, device="cpu"):
        super().__init__(cfg, spec, device)
        if cfg.reward_model_loss not in _LOSSES:
            raise ValueError(
                f"Unknown reward_model_loss '{cfg.reward_model_loss}', expected {list(_LOSSES)}"
            )
        self.loss = _LOSSES[cfg.reward_model_loss]()

    def predict_rewards(self, data: GroupData) -> torch.Tensor:
        return self._predict(data).squeeze(-1)

    def update(self, data: GroupData) -> Dict[str, float]:
        logs = {}
        if self.n_updates > 0:
            with torch.no_grad():
                resid = (data.reward - self.predict_rewards(data))[data.mask]
            # Out-of-sample residual variance: sigma^2 + model error (upper bound on sigma^2)
            logs["grouper/residual_var"] = resid.var().item() if resid.numel() > 1 else 0.0
        logs["grouper/loss"] = self._fit(
            data, data.reward, lambda out, y: self.loss(out.squeeze(-1), y), data.mask
        )
        return logs

    def assign(self, data: GroupData) -> Assignment:
        if self.n_updates == 0:
            return Assignment()  # nothing learned yet: leave rewards unchanged
        return assignment_from_predictions(
            self.predict_rewards(data),
            data.mask,
            mode="replace" if self.replace_with_prediction else self.cfg.reward_model_assign,
            num_groups=self.cfg.num_groups,
            epsilon=self.cfg.epsilon,
            residual=self.cfg.reward_model_residual,
        )


def assignment_from_predictions(
    pred: torch.Tensor,
    mask: torch.Tensor,
    mode: str = "kmeans",
    num_groups: int = 3,
    epsilon: float = 0.5,
    residual: bool = False,
) -> Assignment:
    """Turn predicted rewards ``[..., A]`` into an :class:`Assignment`.

    ``mode``: ``kmeans`` | ``epsilon`` | ``kernel`` | ``global`` (see
    :class:`RewardModelGrouper`) or ``replace`` (use the predictions as rewards).
    """
    if mode == "replace":
        return Assignment(replace=pred)
    offset = pred if residual else None
    none = torch.full_like(pred, -1, dtype=torch.long)
    if mode == "kmeans":
        return Assignment(labels=kmeans_1d_labels(pred, num_groups, mask=mask), offset=offset)
    if mode == "epsilon":
        return Assignment(labels=epsilon_labels(pred, epsilon, mask=mask), offset=offset)
    if mode == "global":
        return Assignment(labels=torch.where(mask, torch.zeros_like(none), none), offset=offset)
    if mode == "kernel":
        diff = pred.unsqueeze(-1) - pred.unsqueeze(-2)
        weights = torch.exp(-0.5 * (diff / max(epsilon, 1e-8)) ** 2)
        return Assignment(weights=weights, offset=offset)
    raise ValueError(f"Unknown reward_model_assign '{mode}'")


class RewardModelOnlyGrouper(RewardModelGrouper):
    """Baseline: use the learned reward model ``f(o, a[, o'])`` as the reward."""

    replace_with_prediction = True


GROUPERS = {
    "oracle": OracleGrouper,
    "global": GlobalGrouper,
    "warmup": WarmupClassifierGrouper,
    "gmm": GMMClassifierGrouper,
    "reward_model": RewardModelGrouper,
    "reward_model_only": RewardModelOnlyGrouper,
}


def make_grouper(cfg: GroupingConfig, spec: TaskSpec, device="cpu") -> Grouper:
    try:
        cls = GROUPERS[cfg.method]
    except KeyError:
        raise ValueError(
            f"Unknown grouping_method '{cfg.method}', expected 'none' or one of {sorted(GROUPERS)}"
        ) from None
    return cls(cfg, spec, device)

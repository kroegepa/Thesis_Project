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
``certified``             **certified consensus**: reward-model proposals (optionally restricted to
                          persistent identity clusters) used only once out-of-sample statistics
                          certify that merged agents' squared reward difference is <= ε²
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
    reward_model_heads: int = 1
    reward_model_min_skill_z: float = 3.0
    epsilon: float = 0.5
    certify_z: float = 1.0
    certify_decay: float = 0.5
    certify_noise_std: Optional[float] = None
    certify_noise_estimator: str = "replicates"
    certify_replicate_key: str = "transition"
    certify_replicate_consensus: bool = False
    certify_residual: bool = False
    certify_skill_gate: bool = True
    certify_temporal_window: int = 0
    certify_ivw: bool = False
    certify_warmup: bool = False
    consensus: str = "mean"


@dataclass
class GroupData:
    """Transitions of one agent group for one iteration.

    Shapes: ``obs``/``next_obs`` ``[B, T, A, *obs_shape]``, ``action`` ``[B, T, A, act_dim]``
    (encoded, float), ``reward``/``mask`` ``[B, T, A]``, ``done`` ``[B, T]`` (episode ended
    after step t; consecutive steps run along T). ``true_reward`` is only available to
    the oracle, the warm-up phase and the diagnostics.
    """

    obs: torch.Tensor
    next_obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    mask: torch.Tensor
    true_reward: Optional[torch.Tensor] = None
    done: Optional[torch.Tensor] = None

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

    ``reward_model_heads > 1``: the model is a bootstrapped ensemble (shared encoder,
    one output head per member, each trained with Poisson(1) bootstrap weights). Agents
    whose predictive std across heads exceeds ``epsilon`` are left ungrouped and keep
    their own (unbiased) reward.

    ``reward_model_min_skill_z > 0``: rewards are only corrected once the model beats a
    constant predictor out of sample, measured by a paired z-test on the squared errors
    of the previous batch (computed before training on it). Before that, a model that
    has not yet learned the rare reward events predicts nearly the same value for
    every agent, so similarity grouping would degenerate into global grouping and erase
    the credit of exactly those events; with the gate it falls back to no correction.
    """

    replace_with_prediction = False

    def __init__(self, cfg: GroupingConfig, spec: TaskSpec, device="cpu"):
        super().__init__(cfg, spec, device)
        if cfg.reward_model_loss not in _LOSSES:
            raise ValueError(
                f"Unknown reward_model_loss '{cfg.reward_model_loss}', expected {list(_LOSSES)}"
            )
        self.loss = _LOSSES[cfg.reward_model_loss](reduction="none")
        self.out_dim = max(cfg.reward_model_heads, 1)
        self.skill_z: Optional[float] = None

    def state_dict(self) -> Dict:
        state = super().state_dict()
        if state:
            state["skill_z"] = self.skill_z
        return state

    def load_state_dict(self, state: Dict) -> None:
        super().load_state_dict(state)
        if state:
            self.skill_z = state.get("skill_z")

    @property
    def skilled(self) -> bool:
        """Whether the model passed the out-of-sample skill test (always if disabled)."""
        thr = self.cfg.reward_model_min_skill_z
        return thr <= 0 or (self.skill_z is not None and self.skill_z >= thr)

    @staticmethod
    def skill_zscore(reward: torch.Tensor, pred: torch.Tensor) -> float:
        """Paired z-statistic of ``(r~ - mean)^2 - (r~ - f)^2`` (> 0: f beats a constant)."""
        d = (reward - reward.mean()) ** 2 - (reward - pred) ** 2
        if d.numel() < 2 or d.std() == 0:
            return 0.0
        return (d.mean() / (d.std() / d.numel() ** 0.5)).item()

    def predict_mean_std(self, data: GroupData) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Mean prediction ``[..., A]`` and, for ensembles, the std across heads."""
        out = self._predict(data)
        if self.out_dim == 1:
            return out.squeeze(-1), None
        return out.mean(-1), out.std(-1)

    def predict_rewards(self, data: GroupData) -> torch.Tensor:
        return self.predict_mean_std(data)[0]

    def _loss(self, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        per_head = self.loss(out, y.unsqueeze(-1).expand_as(out))  # [N, heads]
        if self.out_dim > 1:  # online bootstrap: each head sees a resampled batch
            weights = torch.poisson(torch.ones_like(per_head))
            return (per_head * weights).sum() / weights.sum().clamp(min=1)
        return per_head.mean()

    def update(self, data: GroupData, pred_out: Optional[torch.Tensor] = None) -> Dict[str, float]:
        logs = {}
        if self.n_updates > 0:
            with torch.no_grad():
                if pred_out is None or self.out_dim > 1:
                    pred, std = self.predict_mean_std(data)
                else:
                    pred, std = pred_out, None
                resid = (data.reward - pred)[data.mask]
            # Out-of-sample residual variance: sigma^2 + model error (upper bound on sigma^2)
            logs["grouper/residual_var"] = resid.var().item() if resid.numel() > 1 else 0.0
            self.skill_z = self.skill_zscore(data.reward[data.mask], pred[data.mask])
            logs["grouper/skill_z"] = self.skill_z
            logs["grouper/skilled"] = float(self.skilled)
            if std is not None:
                logs["grouper/confident_frac"] = (
                    (std[data.mask] <= self.cfg.epsilon).float().mean().item()
                )
        logs["grouper/loss"] = self._fit(data, data.reward, self._loss, data.mask)
        return logs

    def assign(self, data: GroupData) -> Assignment:
        if self.n_updates == 0 or not self.skilled:
            return Assignment()  # nothing (reliable) learned yet: leave rewards unchanged
        pred, std = self.predict_mean_std(data)
        return assignment_from_predictions(
            pred,
            data.mask,
            mode="replace" if self.replace_with_prediction else self.cfg.reward_model_assign,
            num_groups=self.cfg.num_groups,
            epsilon=self.cfg.epsilon,
            residual=self.cfg.reward_model_residual,
            confident=None if std is None else std <= self.cfg.epsilon,
        )


def assignment_from_predictions(
    pred: torch.Tensor,
    mask: torch.Tensor,
    mode: str = "kmeans",
    num_groups: int = 3,
    epsilon: float = 0.5,
    residual: bool = False,
    confident: Optional[torch.Tensor] = None,
) -> Assignment:
    """Turn predicted rewards ``[..., A]`` into an :class:`Assignment`.

    ``mode``: ``kmeans`` | ``epsilon`` | ``kernel`` | ``global`` (see
    :class:`RewardModelGrouper`) or ``replace`` (use the predictions as rewards).
    ``confident``: optional bool ``[..., A]``; other agents keep their observed reward.
    """
    if confident is not None:
        mask = mask & confident
    if mode == "replace":
        return Assignment(replace=pred, active=confident)
    offset = pred if residual else None
    none = torch.full_like(pred, -1, dtype=torch.long)
    if mode == "kmeans":
        labels = kmeans_1d_labels(pred, num_groups, mask=mask)
    elif mode == "epsilon":
        labels = epsilon_labels(pred, epsilon, mask=mask)
    elif mode == "global":
        labels = torch.where(mask, torch.zeros_like(none), none)
    elif mode == "kernel":
        diff = pred.unsqueeze(-1) - pred.unsqueeze(-2)
        weights = torch.exp(-0.5 * (diff / max(epsilon, 1e-8)) ** 2)
        return Assignment(weights=weights, offset=offset, active=confident)
    else:
        raise ValueError(f"Unknown reward_model_assign '{mode}'")
    return Assignment(labels=labels, offset=offset, active=confident)


class CertifiedGrouper(RewardModelGrouper):
    """Certified consensus: groupings are only used once the data certify them.

    Each iteration (all out of sample):

    1. *Proposals* from the reward model fitted on earlier noisy rewards: merge agents of
       a step whose predicted rewards are epsilon-close (``pred``), and the same restricted
       to persistent identity clusters (``pred_in_id``, if any exist).
    2. *Certification*: every proposal rule is scored on later batches by the mean of
       ``(r~_i - r~_j)^2 - s_i^2 - s_j^2`` over the same-step pairs it merges (an unbiased
       estimate of their squared reward difference). Only rules whose upper confidence
       bound (``certify_z`` standard errors) is <= epsilon^2 may be used; among those, the
       rule that merges most agents is applied. Otherwise rewards are left unchanged.
    3. Identity clusters can only *restrict* merges: they are blind to rare, short-lived
       reward events, which at high noise are invisible in reward statistics.

    Noise variances ``s_i^2``:

    * ``certify_noise_std`` given: known noise.
    * ``certify_noise_estimator="replicates"`` (default when unknown): one common variance
      (homoscedastic noise), estimated from replicated transitions, i.e. identical (o, a, o')
      (pure error) or nearest neighbours extrapolated to distance 0. This assumes the true
      reward is a deterministic function of the local transition. The certifiers accumulate
      raw squared differences; the noise floor 2 sigma^2 is subtracted when certifying, with
      the latest pooled estimate, and its standard error is added to the margin.
    * ``certify_noise_estimator="temporal"``: per agent from temporal differences of the model
      residuals, with lag-1 identity statistics. Unexplained single-step reward events
      inflate this estimate, so the noise-free skill test (``reward_model_min_skill_z``,
      ``certify_skill_gate``) must also pass.

    Options:

    * ``certify_residual``: also propose *residual* rules (consensus on ``r~ - f`` with the model
      prediction f added back): merged agents then need equal model errors rather than equal
      rewards, so once the model is good whole steps (``res_all``) or identity clusters
      (``res_id``) can be merged, including agents with rare events, and temporal merging works
      through predicted spikes (unvetted neighbours, certified on residual differences). Residual
      and plain rules are certified side by side; the one that merges most is used.
    * ``certify_replicate_consensus``: first average the noisy rewards of all samples in the batch
      with identical replicate keys (``certify_replicate_key``: the transition (o, a, o'), or o'
      only). Under local determinism this has zero bias, so it is applied from the first
      iteration, before anything is certified. Spatial and temporal merges then act on these
      averages.
    * ``certify_ivw``: inverse-variance weighted group means (weights 1/s_j^2), the best
      linear unbiased group estimate when noise levels differ between agents. Weights use the
      per-agent variance estimates from earlier batches (also with a known common noise std).
    * ``certify_temporal_window`` (odd w >= 3; known noise or replicate estimate): additionally average each
      agent's corrected reward over its own +-w//2 neighbouring steps in the same episode
      whose predicted rewards are epsilon-close. The temporal rule is certified the same
      way, from ``(r~_{i,t+k} - r~_{i,t})^2 - 2 s_i^2`` over merged pairs.
    * ``certify_warmup``: fit the proposal model on clean rewards during the experiment's
      warm-up phase (``warmup_iters``, a few iterations suffice).
    """

    RULES = ("pred", "pred_in_id")
    RESIDUAL_RULES = ("res_pred", "res_pred_in_id", "res_id", "res_all")
    WARMUP_EPOCHS = 10

    def __init__(self, cfg: GroupingConfig, spec: TaskSpec, device="cpu"):
        super().__init__(cfg, spec, device)
        from benchmarl.reward_robustness.certification import (
            IdentityStructure,
            NoiseVarianceEstimator,
            ReplicateNoiseEstimator,
            RuleCertifier,
            TemporalCertifier,
        )

        self.rule_names = self.RULES + (self.RESIDUAL_RULES if cfg.certify_residual else ())
        self.rules = {
            r: RuleCertifier(cfg.epsilon, z=cfg.certify_z, decay=cfg.certify_decay) for r in self.rule_names
        }
        if cfg.certify_noise_estimator not in ("replicates", "temporal"):
            raise ValueError(f"Unknown certify_noise_estimator '{cfg.certify_noise_estimator}'")
        self.identity = IdentityStructure(cfg.epsilon, z=cfg.certify_z)
        self.noise = NoiseVarianceEstimator()  # per agent, from residual temporal differences
        # common sigma^2 from replicated transitions
        self.replicates = ReplicateNoiseEstimator(z=cfg.certify_z, key=cfg.certify_replicate_key)
        self.temporal = TemporalCertifier(cfg.epsilon, z=cfg.certify_z, decay=cfg.certify_decay)
        # residual temporal merging: unvetted neighbours, certified on r~ - f
        self.temporal_res = TemporalCertifier(cfg.epsilon, z=cfg.certify_z, decay=cfg.certify_decay)
        self.active_rule = "-"
        self.residual_active = False
        self.temporal_active = False
        if cfg.certify_temporal_window >= 3 and self._noise_source == "temporal":
            import warnings

            warnings.warn(
                "certify_temporal_window needs a known noise std or certify_noise_estimator="
                "'replicates': a noise variance estimated from temporal differences would make the "
                "temporal test vacuous. Temporal merging is disabled."
            )

    @property
    def needs_warmup(self) -> bool:
        return self.cfg.certify_warmup

    def warmup_update(self, data: GroupData) -> Dict[str, float]:
        """Fit the proposal model on clean rewards (warm-up phase)."""
        loss = float("nan")
        for _ in range(self.WARMUP_EPOCHS):
            loss = self._fit(data, data.reward, self._loss, data.mask)
        return {"grouper/warmup_loss": loss}

    @property
    def half_window(self) -> int:
        w = self.cfg.certify_temporal_window
        return w // 2 if (w >= 3 and self._noise_source != "temporal") else 0

    @property
    def known_noise(self) -> bool:
        return self.cfg.certify_noise_std is not None

    @property
    def _noise_source(self) -> str:
        return "known" if self.known_noise else self.cfg.certify_noise_estimator

    def noise_variance(self, n_agents: int) -> Optional[torch.Tensor]:
        """Per-agent noise variances (known, or the current estimate; None before any)."""
        source = self._noise_source
        if source == "known":
            return torch.full((n_agents,), float(self.cfg.certify_noise_std) ** 2)
        if source == "replicates":
            v = self.replicates.variance
            return None if v is None else torch.full((n_agents,), float(v))
        return self.noise.variance

    def _update_noise_var(self, n_agents: int) -> Optional[torch.Tensor]:
        """Noise variance subtracted inside the certifier updates. With the replicate estimate the
        certifiers keep raw squared differences and subtract the noise at query time instead."""
        if self._noise_source == "replicates":
            return torch.zeros(n_agents)
        return self.noise_variance(n_agents)

    def _sync_noise_offsets(self) -> None:
        if self._noise_source != "replicates":
            return
        est, se = self.replicates.estimate()
        offset, offset_se = (0.0, 0.0) if est is None else (2.0 * est, 2.0 * se)
        for cert in [*self.rules.values(), self.temporal, self.temporal_res]:
            cert.set_noise_offset(offset, offset_se)

    def proposals(self, pred: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        props = {"pred": epsilon_labels(pred, self.cfg.epsilon, mask=mask)}
        a = pred.shape[-1]
        if self.identity.labels is not None and self.identity.n_clusters < a:
            ident = torch.as_tensor(self.identity.labels, device=pred.device).expand_as(pred)
            within = torch.full_like(ident, -1)
            for g in ident.unique().tolist():
                m = (ident == g) & mask
                within = torch.where(m, epsilon_labels(pred, self.cfg.epsilon, mask=m) + g * a, within)
            props["pred_in_id"] = within
        if self.cfg.certify_residual:
            # residual rules: consensus on r~ - f, so merged agents need equal model *errors*, not equal
            # rewards (certified on residuals); f is added back afterwards
            props["res_pred"] = props["pred"]
            if "pred_in_id" in props:
                props["res_pred_in_id"] = props["pred_in_id"]
                ident = torch.as_tensor(self.identity.labels, device=pred.device).expand_as(pred)
                props["res_id"] = torch.where(mask, ident, torch.full_like(ident, -1))
            props["res_all"] = torch.where(mask, torch.zeros_like(props["pred"]), torch.full_like(props["pred"], -1))
        return props

    @staticmethod
    def is_residual(rule: str) -> bool:
        return rule.startswith("res_")

    @staticmethod
    def _mean_group_size(labels: torch.Tensor) -> float:
        lab = labels.reshape(-1, labels.shape[-1])
        same = (lab.unsqueeze(-1) == lab.unsqueeze(-2)) & (lab >= 0).unsqueeze(-1)
        return same.float().sum(-1).mean().item()

    def _select(self, props: Dict[str, torch.Tensor]) -> Tuple[str, Assignment]:
        # the skill gate guards the temporal-difference noise estimate, which unexplained single-step
        # reward events inflate; known noise and replicate-based estimates do not need it
        allowed = self._noise_source != "temporal" or not self.cfg.certify_skill_gate or self.skilled
        cands = [(r, lab) for r, lab in props.items() if allowed and self.rules[r].certified]
        if not cands:
            return "-", Assignment()

        def key(c):  # merge most; on ties (same groups, plain vs residual) the smaller certified bound
            delta, margin = self.rules[c[0]].stats()
            return self._mean_group_size(c[1]), -(delta + margin)

        rule, lab = max(cands, key=key)
        return rule, Assignment(labels=lab)

    def assign(self, data: GroupData) -> Assignment:
        if self.n_updates == 0:
            return Assignment()
        self._sync_noise_offsets()
        pred = self.predict_rewards(data)
        rule, assignment = self._select(self.proposals(pred, data.mask))
        if self.is_residual(rule):
            assignment.offset = pred
        return assignment

    def _corrected(
        self,
        data: GroupData,
        labels: Optional[torch.Tensor],
        var: Optional[torch.Tensor],
        neighbours,
        base: Optional[torch.Tensor] = None,
        offset: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Corrected rewards when replicate consensus, IVW, temporal merging or a residual rule is
        active (else None: plain labels). ``base``: replicate-consensus rewards, used in place of the
        noisy rewards. ``offset``: model predictions f; merges then act on ``base - f``, and f is
        added back (residual consensus)."""
        from benchmarl.reward_robustness.certification import temporal_average
        from benchmarl.reward_robustness.consensus import apply_consensus

        # IVW weights only affect efficiency, not bias, as long as they do not depend on the current
        # noise: use the per-agent estimates from earlier batches whenever available
        if self.cfg.certify_ivw and self.noise.variance is not None:
            var = self.noise.variance
        use_ivw = self.cfg.certify_ivw and var is not None and labels is not None
        use_temporal = neighbours is not None
        if not (use_ivw or use_temporal or base is not None or offset is not None):
            return None
        x = data.reward if base is None else base
        if offset is not None:
            x = x - offset
        if labels is None:
            values = x.clone()
        elif use_ivw:
            same = (labels.unsqueeze(-1) == labels.unsqueeze(-2)) & (labels >= 0).unsqueeze(-1)
            weights = same.to(x.dtype) * (1.0 / var.to(x.device)).clamp(max=1e6)
            values = apply_consensus(x, Assignment(weights=weights), data.mask, method="mean")
        else:
            values = apply_consensus(x, Assignment(labels=labels), data.mask, method=self.cfg.consensus)
        if use_temporal:
            values = temporal_average(values, neighbours)
        return values if offset is None else values + offset

    def __call__(self, data: GroupData) -> Tuple[Assignment, Dict[str, float]]:
        from benchmarl.reward_robustness.certification import replicate_consensus, temporal_neighbours

        logs: Dict[str, float] = {}
        assignment, pred, base = Assignment(), None, None
        n_agents = data.reward.shape[-1]
        if self.cfg.certify_replicate_consensus:  # zero bias under local determinism: no certification
            base, frac = replicate_consensus(
                data.obs, data.next_obs, data.action, data.reward, data.mask, self.cfg.certify_replicate_key
            )
            assignment = Assignment(replace=base)
            logs["grouper/replicated_frac"] = frac
        if self.n_updates > 0:
            self._sync_noise_offsets()  # noise estimate from earlier batches only
            pred = self.predict_rewards(data)  # model fitted on earlier batches only
            props = self.proposals(pred, data.mask)
            self.active_rule, assignment = self._select(props)
            var = self.noise_variance(n_agents)
            residual = self.cfg.certify_residual
            neighbours = neighbours_res = None
            if self.half_window:
                neighbours = temporal_neighbours(
                    pred, data.mask, data.done, self.half_window, self.cfg.epsilon
                )
                if residual:  # residual temporal merges need no vetting: certification decides
                    neighbours_res = temporal_neighbours(pred, data.mask, data.done, self.half_window, None)
            # residual pipeline: a residual spatial rule, or no spatial rule but residual temporal merging
            self.residual_active = residual and (
                self.is_residual(self.active_rule)
                or (self.active_rule == "-" and self.half_window > 0 and self.temporal_res.certified)
            )
            if self.residual_active:
                self.temporal_active = self.half_window > 0 and self.temporal_res.certified
                nb, offset = neighbours_res, pred
            else:
                self.temporal_active = self.half_window > 0 and self.temporal.certified
                nb, offset = neighbours, None
            values = self._corrected(
                data, assignment.labels, var, nb if self.temporal_active else None, base, offset
            )
            if values is not None:
                assignment = Assignment(labels=assignment.labels, replace=values)
            cert_var = self._update_noise_var(n_agents)
            if cert_var is not None:  # certify the rules on this batch, for later batches
                resid = data.reward - pred
                for r, lab in props.items():
                    x = resid if self.is_residual(r) else data.reward
                    self.rules[r].update(lab, x, cert_var, data.mask)
                if neighbours is not None:
                    self.temporal.update(data.reward, cert_var, neighbours)
                if neighbours_res is not None:
                    self.temporal_res.update(resid, cert_var, neighbours_res)
            self.noise.update(data.reward - pred, data.mask, data.done)
        source = self._noise_source
        if source == "temporal":  # no noise variance: lag-1 statistics (blind to single-step events)
            self.identity.update(data.reward, data.mask, data.done, noise_var=None)
        elif self.noise_variance(n_agents) is not None:  # uncentred second moments
            self.identity.update(data.reward, data.mask, data.done, noise_var=self.noise_variance(n_agents))
        if source == "replicates":
            self.replicates.update(data.obs, data.next_obs, data.action, data.reward, data.mask)
        logs.update(super().update(data, pred_out=pred))
        logs["grouper/certified_rule"] = float(["-", *self.rule_names].index(self.active_rule))
        logs["grouper/residual_active"] = float(self.residual_active)
        logs["grouper/identity_clusters"] = float(self.identity.n_clusters)
        for r, cert in self.rules.items():
            delta, margin = cert.stats()
            logs[f"grouper/{r}_delta"] = delta
            logs[f"grouper/{r}_margin"] = margin
        if self.noise.variance is not None:
            logs["grouper/noise_var_mean"] = self.noise.variance.mean().item()
        if self._noise_source == "replicates" and self.replicates.variance is not None:
            est, se = self.replicates.estimate()
            logs["grouper/noise_var_replicates"] = est
            logs["grouper/noise_var_replicates_se"] = se
            logs["grouper/noise_replicate_dof"] = self.replicates.dof
        if self.half_window:
            delta, margin = self.temporal.stats()
            logs["grouper/temporal_active"] = float(self.temporal_active)
            logs["grouper/temporal_delta"] = delta
            logs["grouper/temporal_margin"] = margin
            if self.cfg.certify_residual:
                delta, margin = self.temporal_res.stats()
                logs["grouper/temporal_res_delta"] = delta
                logs["grouper/temporal_res_margin"] = margin
        return assignment, logs

    def state_dict(self) -> Dict:
        state = super().state_dict()
        if state:
            state.update(
                rules={r: c.state_dict() for r, c in self.rules.items()},
                identity=self.identity.state_dict(),
                noise=self.noise.state_dict(),
                temporal=self.temporal.state_dict(),
                temporal_res=self.temporal_res.state_dict(),
                replicates=self.replicates.state_dict(),
            )
        return state

    def load_state_dict(self, state: Dict) -> None:
        super().load_state_dict(state)
        if state:
            for r, c in self.rules.items():
                if r in state["rules"]:
                    c.load_state_dict(state["rules"][r])
            self.identity.load_state_dict(state["identity"])
            self.noise.load_state_dict(state["noise"])
            if "temporal" in state:
                self.temporal.load_state_dict(state["temporal"])
            if "temporal_res" in state:
                self.temporal_res.load_state_dict(state["temporal_res"])
            if "replicates" in state:
                self.replicates.load_state_dict(state["replicates"])
            self._sync_noise_offsets()


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
    "certified": CertifiedGrouper,
}


def make_grouper(cfg: GroupingConfig, spec: TaskSpec, device="cpu") -> Grouper:
    try:
        cls = GROUPERS[cfg.method]
    except KeyError:
        raise ValueError(
            f"Unknown grouping_method '{cfg.method}', expected 'none' or one of {sorted(GROUPERS)}"
        ) from None
    return cls(cfg, spec, device)

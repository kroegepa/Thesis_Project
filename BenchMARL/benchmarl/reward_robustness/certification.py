"""Statistics behind *certified consensus* (``grouping_method=certified``).

The consensus is only unbiased if merged agents have (nearly) equal true rewards
(Assumption 1). With additive noise that is zero-mean and independent across agents and
steps, the noisy rewards give an unbiased handle on exactly that quantity:

    E[(r~_i - r~_j)^2] = E[(r_i - r_j)^2] + s_i^2 + s_j^2           (same step, i != j)

where s_i^2 is agent i's noise variance. No Gaussian or symmetric noise is needed.

* :class:`RuleCertifier` averages this over all same-step pairs that a grouping *rule*
  merges and accepts the rule only if an upper confidence bound on the mean squared
  reward difference of merged pairs is <= epsilon^2. Then, with high probability, the RMS
  bias of the consensus is <= epsilon (Jensen).
* :class:`IdentityStructure` finds persistent (identity/role) clusters. In the
  noise-variance-free mode it uses lag-1 products of pairwise difference series,
  E[D_t D_{t-1}] with D_t = r~_i,t - r~_j,t, which contain no noise term at all.
* :class:`NoiseVarianceEstimator` estimates each agent's noise variance from temporal
  differences of reward-model residuals (noise is fresh every step, rewards persist).

Limitation (identifiability): reward differences that last a single step and are not
explained by the reward model are, in second-order statistics, indistinguishable from
noise. Without a known noise variance they inflate the variance estimate and make the
certification optimistic until the reward model has learned them.
"""

from typing import Dict, List, Optional

import numpy as np
import torch


def _valid_lag_pairs(mask: torch.Tensor, done: Optional[torch.Tensor]) -> torch.Tensor:
    """``[B, T-1, A]``: agent alive at t-1 and t, and no episode boundary in between."""
    valid = mask[:, 1:] & mask[:, :-1]
    if done is not None:
        valid = valid & ~done[:, :-1].unsqueeze(-1)
    return valid


class RuleCertifier:
    """Certify a grouping rule from the same-step pairs it merges.

    For each (step, group) unit the mean over its pairs of
    ``(r~_i - r~_j)^2 - s_i^2 - s_j^2`` is computed; the certified quantity is the
    pair-weighted average over units, with an empirical standard error over units.
    Statistics are exponentially decayed (``decay`` per update) so they track the
    evolving proposal model; they must come from data that did not shape the proposal
    (out of sample).
    """

    def __init__(self, epsilon: float, z: float = 1.0, decay: float = 0.5, min_units: float = 10.0):
        self.epsilon, self.z, self.decay, self.min_units = epsilon, z, decay, min_units
        self.w = self.s1 = self.s2 = self.n = 0.0

    def update(
        self,
        labels: torch.Tensor,
        reward: torch.Tensor,
        noise_var: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        a = labels.shape[-1]
        lab = labels.reshape(-1, a)
        x = reward.reshape(-1, a).double()
        if mask is not None:
            lab = torch.where(mask.reshape(-1, a), lab, torch.full_like(lab, -1))
        var = torch.as_tensor(noise_var, dtype=torch.float64, device=x.device).expand_as(x)
        self.w *= self.decay
        self.s1 *= self.decay
        self.s2 *= self.decay
        self.n *= self.decay
        for k in lab.unique().tolist():
            if k < 0:
                continue
            m = (lab == k).double()
            cnt = m.sum(1)
            rows = cnt >= 2
            if not bool(rows.any()):
                continue
            sx, sxx = (x * m).sum(1), (x * x * m).sum(1)
            pairs = (cnt * (cnt - 1) / 2)[rows]
            # sum_{i<j}(x_i - x_j)^2 = n sum x^2 - (sum x)^2 ; noise part = (n-1) sum_i s_i^2
            stat = ((cnt * sxx - sx ** 2) - (cnt - 1) * (var * m).sum(1))[rows] / pairs
            self.w += pairs.sum().item()
            self.s1 += (pairs * stat).sum().item()
            self.s2 += (pairs * stat ** 2).sum().item()
            self.n += float(rows.sum())

    def stats(self):
        """(estimated mean squared reward difference of merged pairs, z * SE)."""
        if self.w <= 0:
            return float("nan"), float("nan")
        mean = self.s1 / self.w
        var = max(self.s2 / self.w - mean ** 2, 0.0)
        return mean, self.z * float(np.sqrt(var / max(self.n, 1.0)))

    @property
    def certified(self) -> bool:
        if self.n < self.min_units:
            return False
        delta, margin = self.stats()
        return delta + margin <= self.epsilon ** 2

    def state_dict(self) -> Dict:
        return {"w": self.w, "s1": self.s1, "s2": self.s2, "n": self.n}

    def load_state_dict(self, state: Dict) -> None:
        self.w, self.s1, self.s2, self.n = state["w"], state["s1"], state["s2"], state["n"]


class IdentityStructure:
    """Persistent clusters of agents (same agent index across steps) with equal rewards.

    Per agent pair, an unbiased estimate of the persistent squared reward difference:

    * ``noise_var`` given (known noise): E[(r~_i - r~_j)^2] - s_i^2 - s_j^2 (uncentred, so
      constant offsets between roles count);
    * ``noise_var=None`` (variance-free): E[D_t D_{t-1}] with D_t = r~_i,t - r~_j,t, which
      has no noise term (noise independent over time) but is blind to differences lasting
      a single step.

    Clusters are certified top-down along an average-linkage dendrogram: a cluster is
    accepted if every member's mean estimate to the other members has UCB <= epsilon^2.
    Because it cannot see single-step differences, the result should only *restrict*
    merges, never justify them on its own.
    """

    def __init__(self, epsilon: float, z: float = 1.0, decay: float = 0.9, min_steps: float = 100.0):
        self.epsilon, self.z, self.decay, self.min_steps = epsilon, z, decay, min_steps
        self.s = self.ss = self.cnt = None
        self.labels: Optional[np.ndarray] = None

    def update(
        self,
        reward: torch.Tensor,
        mask: torch.Tensor,
        done: Optional[torch.Tensor] = None,
        noise_var: Optional[torch.Tensor] = None,
    ) -> None:
        """``reward``/``mask``: ``[B, T, A]`` (consecutive steps along T), ``done``: ``[B, T]``."""
        a = reward.shape[-1]
        if self.s is None:
            self.s, self.ss, self.cnt = np.zeros((a, a)), np.zeros((a, a)), np.zeros((a, a))
            self.labels = np.arange(a)
        x = reward.double()
        if noise_var is None:
            if x.shape[1] < 2:
                return
            d = x.unsqueeze(-1) - x.unsqueeze(-2)  # [B, T, A, A]
            valid_t = _valid_lag_pairs(mask, done)  # [B, T-1, A]
            valid = valid_t.unsqueeze(-1) & valid_t.unsqueeze(-2)
            stat = d[:, 1:] * d[:, :-1]
        else:
            d = x.unsqueeze(-1) - x.unsqueeze(-2)
            var = torch.as_tensor(noise_var, dtype=torch.float64, device=x.device)
            stat = d ** 2 - var.unsqueeze(-1) - var.unsqueeze(-2)
            valid = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        stat = torch.where(valid, stat, torch.zeros_like(stat)).reshape(-1, a, a).numpy()
        valid = valid.reshape(-1, a, a).double().numpy()
        self.s = self.s * self.decay + stat.sum(0)
        self.ss = self.ss * self.decay + (stat ** 2).sum(0)
        self.cnt = self.cnt * self.decay + valid.sum(0)
        self._recluster()

    def _recluster(self) -> None:
        from scipy.cluster.hierarchy import linkage, to_tree
        from scipy.spatial.distance import squareform

        a = self.s.shape[0]
        n = np.maximum(self.cnt, 1.0)
        delta = self.s / n
        se = np.sqrt(np.clip(self.ss / n - delta ** 2, 0, None) / n)
        enough = self.cnt >= self.min_steps
        if a < 2 or not enough[~np.eye(a, dtype=bool)].any():
            self.labels = np.arange(a)
            return
        dist = np.clip((delta + delta.T) / 2, 0, None)
        np.fill_diagonal(dist, 0.0)
        tree = to_tree(linkage(squareform(dist, checks=False), "average"))
        eps2 = self.epsilon ** 2

        def certify(node) -> List[List[int]]:
            members = node.pre_order()
            k = len(members)
            if k == 1:
                return [members]
            sub = np.ix_(members, members)
            off = ~np.eye(k, dtype=bool)
            ok_data = enough[sub][off].all()
            loo = np.where(off, delta[sub], 0).sum(1) / (k - 1)
            loo_se = np.sqrt(np.where(off, se[sub] ** 2, 0).sum(1)) / (k - 1)
            if ok_data and np.all(loo + self.z * loo_se <= eps2):
                return [members]
            return certify(node.get_left()) + certify(node.get_right())

        labels = np.empty(a, dtype=np.int64)
        for k, cluster in enumerate(certify(tree)):
            labels[cluster] = k
        self.labels = labels

    @property
    def n_clusters(self) -> int:
        return 0 if self.labels is None else len(set(self.labels.tolist()))

    def state_dict(self) -> Dict:
        return {"s": self.s, "ss": self.ss, "cnt": self.cnt, "labels": self.labels}

    def load_state_dict(self, state: Dict) -> None:
        self.s, self.ss, self.cnt, self.labels = state["s"], state["ss"], state["cnt"], state["labels"]


class NoiseVarianceEstimator:
    """Per-agent noise variance from temporal differences of residuals ``e = r~ - f``.

    ``E[(e_t - e_{t-1})^2] / 2 = s_i^2 + E[(delta(r - f))^2] / 2``: shape-free and per agent,
    biased upwards only by reward changes that the model ``f`` does not explain.
    """

    def __init__(self, decay: float = 0.7):
        self.decay = decay
        self.ss = self.cnt = None

    def update(self, resid: torch.Tensor, mask: torch.Tensor, done: Optional[torch.Tensor] = None) -> None:
        if resid.shape[1] < 2:
            return
        a = resid.shape[-1]
        valid = _valid_lag_pairs(mask, done)
        d = (resid[:, 1:] - resid[:, :-1]).double()
        d2 = torch.where(valid, d ** 2, torch.zeros_like(d)).reshape(-1, a).sum(0).numpy()
        c = valid.reshape(-1, a).double().sum(0).numpy()
        if self.ss is None:
            self.ss, self.cnt = np.zeros(a), np.zeros(a)
        self.ss = self.ss * self.decay + d2 / 2
        self.cnt = self.cnt * self.decay + c

    @property
    def variance(self) -> Optional[torch.Tensor]:
        if self.ss is None or (self.cnt < 2).any():
            return None
        return torch.as_tensor(self.ss / self.cnt, dtype=torch.float32)

    def state_dict(self) -> Dict:
        return {"ss": self.ss, "cnt": self.cnt}

    def load_state_dict(self, state: Dict) -> None:
        self.ss, self.cnt = state["ss"], state["cnt"]


def _no_boundary(done: Optional[torch.Tensor], t_len: int, k: int, shape) -> torch.Tensor:
    """``[B, T]`` bool: steps t and t+k lie in the same episode (no done in [t, t+k-1])."""
    if done is None:
        return torch.ones(shape[:2], dtype=torch.bool)
    csum = torch.cat([torch.zeros_like(done[:, :1], dtype=torch.long), done.long().cumsum(1)], 1)  # [B, T+1]
    ok = torch.zeros(shape[:2], dtype=torch.bool)
    if k < t_len:
        ok[:, : t_len - k] = (csum[:, k: t_len] - csum[:, : t_len - k]) == 0
    return ok


def temporal_neighbours(
    pred: torch.Tensor,
    mask: torch.Tensor,
    done: Optional[torch.Tensor],
    half_window: int,
    epsilon: Optional[float],
) -> Dict[int, torch.Tensor]:
    """For each offset k in [-h, h] \\ {0}: ``[B, T, A]`` bool, (i, t+k) may be merged with (i, t).

    Same agent, same episode, both alive and, if ``epsilon`` is given, the reward model's
    predictions at t and t+k are epsilon-close (vetting).
    """
    b, t_len, a = pred.shape
    out = {}
    for k in range(1, half_window + 1):
        if k >= t_len:
            break
        fwd = torch.zeros_like(mask)
        same_ep = _no_boundary(done, t_len, k, pred.shape).unsqueeze(-1)
        fwd[:, : t_len - k] = (mask[:, : t_len - k] & mask[:, k:] & same_ep[:, : t_len - k])
        if epsilon is not None:
            close = torch.zeros_like(mask)
            close[:, : t_len - k] = (pred[:, k:] - pred[:, : t_len - k]).abs() <= epsilon
            fwd &= close
        bwd = torch.zeros_like(mask)
        bwd[:, k:] = fwd[:, : t_len - k]  # (i, t) pairs with (i, t-k)
        out[k], out[-k] = fwd, bwd
    return out


def temporal_average(values: torch.Tensor, neighbours: Dict[int, torch.Tensor]) -> torch.Tensor:
    """Average each (i, t) with its allowed temporal neighbours (i, t+k)."""
    num, den = values.clone(), torch.ones_like(values)
    t_len = values.shape[1]
    for k, ok in neighbours.items():
        shifted = torch.zeros_like(values)
        if k > 0:
            shifted[:, : t_len - k] = values[:, k:]
        else:
            shifted[:, -k:] = values[:, : t_len + k]
        num = num + torch.where(ok, shifted, torch.zeros_like(values))
        den = den + ok.to(values.dtype)
    return num / den


class TemporalCertifier:
    """Certify merging an agent's reward with its own neighbouring steps.

    Per merged temporal pair: ``(r~_{i,t+k} - r~_{i,t})^2 - 2 s_i^2`` is unbiased for the squared
    change of the true reward. Requires a *known* noise variance: a variance estimated from
    temporal differences already absorbs temporal reward changes, which would make this
    test vacuous.
    """

    def __init__(self, epsilon: float, z: float = 1.0, decay: float = 0.5, min_pairs: float = 100.0):
        self.epsilon, self.z, self.decay, self.min_pairs = epsilon, z, decay, min_pairs
        self.n = self.s = self.ss = 0.0

    def update(self, reward: torch.Tensor, noise_var: torch.Tensor, neighbours: Dict[int, torch.Tensor]) -> None:
        var = torch.as_tensor(noise_var, dtype=torch.float64).expand_as(reward)
        x = reward.double()
        t_len = x.shape[1]
        self.n *= self.decay
        self.s *= self.decay
        self.ss *= self.decay
        for k, ok in neighbours.items():
            if k <= 0:
                continue
            diff = torch.zeros_like(x)
            diff[:, : t_len - k] = x[:, k:] - x[:, : t_len - k]
            stat = (diff ** 2 - 2 * var)[ok]
            self.n += stat.numel()
            self.s += stat.sum().item()
            self.ss += (stat ** 2).sum().item()

    def stats(self):
        if self.n <= 0:
            return float("nan"), float("nan")
        mean = self.s / self.n
        return mean, self.z * float(np.sqrt(max(self.ss / self.n - mean ** 2, 0.0) / self.n))

    @property
    def certified(self) -> bool:
        if self.n < self.min_pairs:
            return False
        delta, margin = self.stats()
        return delta + margin <= self.epsilon ** 2

    def state_dict(self) -> Dict:
        return {"n": self.n, "s": self.s, "ss": self.ss}

    def load_state_dict(self, state: Dict) -> None:
        self.n, self.s, self.ss = state["n"], state["s"], state["ss"]

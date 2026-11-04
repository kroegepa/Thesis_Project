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
* :class:`ReplicateNoiseEstimator` estimates one common noise variance (homoscedastic
  noise) from replicated transitions, assuming the true reward is a deterministic function
  of the local transition (o, a, o'). This avoids the limitation below.

Limitation (identifiability): reward differences that last a single step and are not
explained by the reward model are, in second-order statistics, indistinguishable from
noise. Without a known noise variance (or an extra assumption such as the replicate one)
they inflate the variance estimate and make the certification optimistic until the reward
model has learned them.
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
        self.offset = self.offset_se = 0.0

    def set_noise_offset(self, offset: float, se: float = 0.0) -> None:
        """Noise term subtracted at query time (with standard error), for statistics accumulated
        with ``noise_var=0``. With a common noise variance sigma^2 the pair offset is 2 sigma^2,
        so an estimated sigma^2 can be refined after the fact and its uncertainty is added in
        quadrature to the margin."""
        self.offset, self.offset_se = float(offset), float(se)

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
        raw = self.s1 / self.w
        var = max(self.s2 / self.w - raw ** 2, 0.0)
        return raw - self.offset, self.z * float(np.sqrt(var / max(self.n, 1.0) + self.offset_se ** 2))

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
    change of the true reward. Requires a noise variance that does not come from temporal
    differences (known, or :class:`ReplicateNoiseEstimator`): a variance estimated from temporal
    differences already absorbs temporal reward changes, which would make this test vacuous.
    """

    def __init__(self, epsilon: float, z: float = 1.0, decay: float = 0.5, min_pairs: float = 100.0):
        self.epsilon, self.z, self.decay, self.min_pairs = epsilon, z, decay, min_pairs
        self.n = self.s = self.ss = 0.0
        self.offset = self.offset_se = 0.0

    set_noise_offset = RuleCertifier.set_noise_offset

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
        raw = self.s / self.n
        var = max(self.ss / self.n - raw ** 2, 0.0) / self.n
        return raw - self.offset, self.z * float(np.sqrt(var + self.offset_se ** 2))

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


class ReplicateNoiseEstimator:
    """Common noise variance from (near-)replicated transitions, without knowing sigma.

    Assumptions: the noise variance is the same for all agents (homoscedastic) and the true
    reward is a deterministic function of the agent's local transition (o, a, o'). Then two
    samples with the same transition have the same true reward and their noisy rewards differ
    only by noise:

    * **exact replicates** (pure error, as with replicated design points in regression): the
      within-class variance of noisy rewards over identical transitions is unbiased for
      sigma^2. Typical for discrete observations (grid worlds).
    * **near replicates** (continuous observations, Tong & Wang 2005 style): for sampled query
      transitions, take their k nearest neighbours in standardised (o, a, o') space and regress
      (r~_q - r~_nn)^2 / 2 on the squared distance; the intercept estimates sigma^2 for rewards
      that are smooth in the transition.

    Unlike temporal differences, this compares only (near-)identical transitions, so rare
    single-step reward events do not inflate the estimate. The noise kernel is stationary, so
    statistics are pooled over all batches (``decay=1``). :meth:`estimate` returns the estimate
    and its standard error. Precision matters: certifying a mean squared difference of eps^2
    against a noise floor of 2 sigma^2 needs sigma^2 to a relative precision of about
    eps^2 / (2 sigma^2) (1% at sigma = 3.5, eps = 0.5).
    """

    CHUNK = 2 ** 24  # max distance-matrix entries per block

    def __init__(self, decay: float = 1.0, min_dof: float = 2000.0, k: int = 5, n_query: int = 16000, z: float = 1.0):
        self.decay, self.min_dof, self.k, self.n_query, self.z = decay, min_dof, k, n_query, z
        self.dof = self.ss = self.ss_var = 0.0  # exact replicates: pooled dof, SS, var of SS
        self.knn_w = self.knn_s = 0.0  # near replicates: inverse-variance pooled intercepts

    @staticmethod
    def _features(obs, next_obs, action, mask):
        n = mask.numel()
        f = torch.cat([obs.reshape(n, -1), next_obs.reshape(n, -1), action.reshape(n, -1)], 1).float()
        return f[mask.reshape(-1)]

    def update(self, obs, next_obs, action, reward, mask) -> None:
        f = self._features(obs, next_obs, action, mask)
        x = reward.reshape(-1)[mask.reshape(-1)].double()
        if x.numel() < 3:
            return
        self.dof *= self.decay
        self.ss *= self.decay
        self.ss_var *= self.decay ** 2
        self.knn_w *= self.decay
        self.knn_s *= self.decay
        # ---- exact replicates: within-class variance
        _, inv, cnt = torch.unique(f, dim=0, return_inverse=True, return_counts=True)
        k = cnt.numel()
        s1 = torch.zeros(k, dtype=torch.float64, device=x.device).index_add_(0, inv, x)
        s2 = torch.zeros(k, dtype=torch.float64, device=x.device).index_add_(0, inv, x * x)
        c = cnt.double()
        rep = c > 1
        if bool(rep.any()):
            ss_c = (s2 - s1 ** 2 / c)[rep]
            dof_c = (c - 1)[rep]
            self.dof += dof_c.sum().item()
            self.ss += ss_c.sum().item()
            est = ss_c.sum() / dof_c.sum()
            self.ss_var += ((ss_c - dof_c * est) ** 2).sum().item()  # sandwich variance of SS
        # ---- near replicates (only needed when exact ones are scarce)
        if self.dof < self.min_dof:
            g = (f - f.mean(0)) / f.std(0).clamp(min=1e-6)
            q = torch.randperm(g.shape[0], device=g.device)[: self.n_query]
            kk = min(self.k, g.shape[0] - 1)
            dist, idx = [], []
            for qc in q.split(max(1, self.CHUNK // g.shape[0])):  # bounded memory
                d2 = torch.cdist(g[qc], g) ** 2
                d2[torch.arange(qc.numel(), device=g.device), qc] = float("inf")  # exclude self
                dc, ic = d2.topk(kk, largest=False)
                dist.append(dc)
                idx.append(ic)
            dist, idx = torch.cat(dist), torch.cat(idx)
            x = x.to(g.device)
            h = ((x[q].unsqueeze(1) - x[idx]) ** 2 / 2).reshape(-1)
            dd = dist.reshape(-1).double()
            a = torch.stack([torch.ones_like(dd), dd], 1)
            coef = torch.linalg.lstsq(a, h.unsqueeze(1)).solution.squeeze(1)
            resid = h - a @ coef
            cov = torch.linalg.pinv(a.T @ a) * resid.var() * kk  # neighbours share queries: inflate
            var0 = max(cov[0, 0].item(), 1e-12)
            self.knn_w += 1.0 / var0
            self.knn_s += coef[0].item() / var0

    @property
    def source(self) -> str:
        if self.dof >= self.min_dof:
            return "replicates"
        return "near_replicates" if self.knn_w > 0 else "none"

    def estimate(self):
        """(sigma^2 estimate, standard error), or (None, None) before any data."""
        if self.dof >= self.min_dof:
            return self.ss / self.dof, np.sqrt(self.ss_var) / self.dof
        if self.knn_w > 0:
            return self.knn_s / self.knn_w, 1.0 / np.sqrt(self.knn_w)
        return None, None

    @property
    def variance(self) -> Optional[float]:
        return self.estimate()[0]

    @property
    def variance_lcb(self) -> Optional[float]:
        est, se = self.estimate()
        return None if est is None else max(est - self.z * se, 0.0)

    def state_dict(self) -> Dict:
        return {k: getattr(self, k) for k in ["dof", "ss", "ss_var", "knn_w", "knn_s"]}

    def load_state_dict(self, state: Dict) -> None:
        for k, v in state.items():
            setattr(self, k, v)

"""Consensus-based reward correction (Definition 4 of the paper).

All functions operate on the trailing agent dimension: ``values`` has shape
``[..., A]`` and every leading index (environment, timestep) is an independent
consensus problem. A group is the set of agents of one timestep sharing a label.

Conventions:
    * ``labels``: long tensor ``[..., A]``. ``-1`` means "not grouped" (the agent keeps
      its own reward).
    * ``mask``: bool tensor ``[..., A]``. Masked-out agents (e.g. dead agents) neither
      contribute to nor receive a consensus value.
"""

from dataclasses import dataclass
from typing import Optional

import torch

CONSENSUS_METHODS = ("mean", "median", "trimmed_mean")


@dataclass
class Assignment:
    """Output of a grouper: how to compute the corrected reward of each agent.

    Exactly one of ``labels`` / ``weights`` / ``replace`` is normally set. If none is
    set the rewards are left unchanged.

    Attributes:
        labels: hard groups ``[..., A]`` (``-1`` = ungrouped).
        weights: soft consensus weights ``[..., A, A]``; row ``i`` holds the weight
            of every agent ``j`` in agent ``i``'s consensus (mean consensus only).
        offset: optional per-agent baseline ``[..., A]``. If given, consensus is run on
            the residuals ``r~ - offset`` and the offset is added back
            ("residual consensus"). This relaxes ε-similarity of rewards to
            ε-similarity of residuals.
        replace: per-agent rewards ``[..., A]`` that replace the observed rewards
            (used by the reward-model-only baseline).
    """

    labels: Optional[torch.Tensor] = None
    weights: Optional[torch.Tensor] = None
    offset: Optional[torch.Tensor] = None
    replace: Optional[torch.Tensor] = None


def _num_labels(labels: torch.Tensor) -> int:
    if labels.numel() == 0:
        return 0
    return int(labels.max().item()) + 1


def group_mean(
    values: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Equal-weight group estimator (Lemma 1)."""
    k = _num_labels(labels)
    if k == 0:
        return values.clone()
    idx = labels.clamp(min=0)
    member = torch.nn.functional.one_hot(idx, k).to(values.dtype)
    member = member * ((labels >= 0) & mask).unsqueeze(-1).to(values.dtype)
    sums = (member * values.unsqueeze(-1)).sum(-2)  # [..., K]
    counts = member.sum(-2)  # [..., K]
    means = sums / counts.clamp(min=1)
    own_mean = torch.gather(means, -1, idx)
    own_count = torch.gather(counts, -1, idx)
    use = (labels >= 0) & mask & (own_count > 0)
    return torch.where(use, own_mean, values)


def group_median(
    values: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Group median. For even group sizes the lower median is used (``torch.nanmedian``)."""
    out = values.clone()
    nan = torch.tensor(float("nan"), dtype=values.dtype, device=values.device)
    for g in range(_num_labels(labels)):
        members = (labels == g) & mask
        med = torch.nanmedian(torch.where(members, values, nan), dim=-1, keepdim=True)
        med = med.values.expand_as(values)
        out = torch.where(members & ~torch.isnan(med), med, out)
    return out


def group_trimmed_mean(
    values: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, trim: float = 0.2
) -> torch.Tensor:
    """Group mean after discarding the ``floor(trim * n)`` lowest and highest values."""
    if not 0.0 <= trim < 0.5:
        raise ValueError(f"trim must be in [0, 0.5), got {trim}")
    out = values.clone()
    a = values.shape[-1]
    pos = torch.arange(a, device=values.device)
    inf = torch.tensor(float("inf"), dtype=values.dtype, device=values.device)
    for g in range(_num_labels(labels)):
        members = (labels == g) & mask
        n = members.sum(-1, keepdim=True)
        k = torch.floor(n * trim).long()
        sorted_vals, _ = torch.sort(torch.where(members, values, inf), dim=-1)
        keep = (pos >= k) & (pos < n - k)
        total = torch.where(keep, sorted_vals, torch.zeros_like(sorted_vals)).sum(-1)
        tmean = (total / keep.sum(-1).clamp(min=1)).unsqueeze(-1).expand_as(values)
        out = torch.where(members & (n > 0), tmean, out)
    return out


def weighted_mean(
    values: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Soft consensus: ``r^_i = sum_j w_ij r~_j / sum_j w_ij`` over contributing agents."""
    w = weights * mask.unsqueeze(-2).to(weights.dtype)
    den = w.sum(-1)
    num = (w * values.unsqueeze(-2)).sum(-1)
    out = num / den.clamp(min=1e-12)
    return torch.where(mask & (den > 0), out, values)


def apply_consensus(
    reward: torch.Tensor,
    assignment: Assignment,
    mask: Optional[torch.Tensor] = None,
    method: str = "mean",
    trim: float = 0.2,
) -> torch.Tensor:
    """Compute the corrected rewards ``r^`` (eq. 3) from observed rewards ``[..., A]``."""
    if mask is None:
        mask = torch.ones_like(reward, dtype=torch.bool)
    if assignment.replace is not None:
        return torch.where(mask, assignment.replace.to(reward.dtype), reward)

    offset = assignment.offset
    base = reward - offset if offset is not None else reward

    if assignment.weights is not None:
        if method != "mean":
            raise ValueError("Soft (weighted) consensus only supports method='mean'")
        corrected = weighted_mean(base, assignment.weights, mask)
    elif assignment.labels is not None:
        if method == "mean":
            corrected = group_mean(base, assignment.labels, mask)
        elif method == "median":
            corrected = group_median(base, assignment.labels, mask)
        elif method == "trimmed_mean":
            corrected = group_trimmed_mean(base, assignment.labels, mask, trim)
        else:
            raise ValueError(
                f"Unknown consensus method '{method}', expected one of {CONSENSUS_METHODS}"
            )
    else:
        corrected = base

    return corrected + offset if offset is not None else corrected

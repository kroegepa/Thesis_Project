"""Diagnostics for groupings and corrected rewards.

All grouping metrics are invariant to label permutations and are computed per
timestep: two agents count as "grouped together" only if they share a label at the
same environment step, which is what the consensus actually uses. (The F1 score of
predicted vs. oracle *label ids* is not meaningful for clustering-based groupers,
whose ids are arbitrary.)
"""

from typing import Dict, Optional

import torch


def _same_group(labels: torch.Tensor) -> torch.Tensor:
    """``[..., A] -> [..., A, A]`` bool, ungrouped agents (-1) are only alone."""
    same = labels.unsqueeze(-1) == labels.unsqueeze(-2)
    grouped = (labels >= 0).unsqueeze(-1) & (labels >= 0).unsqueeze(-2)
    return same & grouped


def pairwise_grouping_scores(
    true_labels: torch.Tensor,
    pred_labels: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Pairwise precision/recall of "same group" decisions, plus group-size stats.

    ``precision`` = fraction of predicted same-group pairs that are truly similar
    (``1 - precision`` is the pairwise analogue of the misgrouping rate η of
    Proposition 3); ``recall`` = fraction of truly similar pairs that were grouped.
    """
    a = true_labels.shape[-1]
    if mask is None:
        mask = torch.ones_like(true_labels, dtype=torch.bool)
    eye = torch.eye(a, dtype=torch.bool, device=true_labels.device)
    valid = mask.unsqueeze(-1) & mask.unsqueeze(-2) & ~eye
    same_t = _same_group(true_labels) & valid
    same_p = _same_group(pred_labels) & valid
    tp = (same_t & same_p).sum().item()
    n_pred = same_p.sum().item()
    n_true = same_t.sum().item()
    precision = tp / n_pred if n_pred else float("nan")
    recall = tp / n_true if n_true else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if n_pred and n_true and (precision + recall) > 0
        else float("nan")
    )
    n_alive = mask.sum().clamp(min=1).item()
    return {
        "pair_precision": precision,
        "pair_recall": recall,
        "pair_f1": f1,
        "mean_group_size": 1.0 + same_p.sum().item() / n_alive,
        "mean_oracle_group_size": 1.0 + same_t.sum().item() / n_alive,
    }


def _per_timestep_ids(labels: torch.Tensor) -> torch.Tensor:
    """Make labels unique across timesteps; ungrouped agents become singletons."""
    a = labels.shape[-1]
    flat = labels.reshape(-1, a)
    rows = torch.arange(flat.shape[0], device=labels.device).unsqueeze(-1)
    cols = torch.arange(a, device=labels.device).unsqueeze(0)
    k = max(int(flat.max().item()) + 1, 0) if flat.numel() else 0
    local = torch.where(flat >= 0, flat, k + cols)
    return (rows * (k + a) + local).reshape(labels.shape)


def adjusted_rand_index(
    true_labels: torch.Tensor,
    pred_labels: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """ARI of the per-timestep partitions."""
    from sklearn.metrics import adjusted_rand_score

    if mask is None:
        mask = torch.ones_like(true_labels, dtype=torch.bool)
    t = _per_timestep_ids(true_labels)[mask].cpu().numpy()
    p = _per_timestep_ids(pred_labels)[mask].cpu().numpy()
    return float(adjusted_rand_score(t, p))


def event_contrast_sums(
    true_reward: torch.Tensor,
    corrected_reward: torch.Tensor,
    event: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
):
    """Within-timestep credit contrast between event agents and the other agents.

    For every timestep that has both event agents (e.g. capturers) and non-event
    agents, compares mean reward of event agents minus mean reward of the others, for
    the corrected and the true rewards. Returns the two sums and the number of such
    timesteps; ``contrast = sum_corrected / sum_true`` is 1 for an unbiased correction
    and 0 if the event reward is shared equally by all agents of the timestep (credit
    assignment destroyed, as with global grouping).
    """
    if mask is None:
        mask = torch.ones_like(true_reward, dtype=torch.bool)
    ev = (event & mask).float()
    other = (~event & mask).float()
    n_ev, n_other = ev.sum(-1), other.sum(-1)
    rows = (n_ev > 0) & (n_other > 0)

    def diff(x):
        return (x * ev).sum(-1) / n_ev.clamp(min=1) - (x * other).sum(-1) / n_other.clamp(min=1)

    return diff(corrected_reward)[rows].sum().item(), diff(true_reward)[rows].sum().item(), int(rows.sum())


def reward_error_metrics(
    true_reward: torch.Tensor,
    observed_reward: torch.Tensor,
    corrected_reward: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    levels: Optional[torch.Tensor] = None,
    event_level: Optional[int] = None,
) -> Dict[str, float]:
    """How close the corrected rewards are to the true rewards.

    Reports MSE before/after correction and, if oracle ``levels`` are given, the mean
    error (systematic bias) and MSE per reward level. Bias matters more than MSE for
    learning: variance averages out over samples, a level-dependent bias does not.
    ``event_level`` (default: highest level present) defines the "event" agents for
    the within-timestep ``event_contrast`` (see :func:`event_contrast_sums`).
    """
    if mask is None:
        mask = torch.ones_like(true_reward, dtype=torch.bool)
    t, o, c = true_reward[mask], observed_reward[mask], corrected_reward[mask]
    out = {
        "mse_observed": ((o - t) ** 2).mean().item(),
        "mse_corrected": ((c - t) ** 2).mean().item(),
        "bias_corrected": (c - t).mean().item(),
    }
    if t.numel() > 1 and t.std() > 0 and c.std() > 0:
        out["corr_corrected"] = torch.corrcoef(torch.stack([t, c]))[0, 1].item()
    if levels is not None:
        if event_level is None:
            event_level = int(levels[mask].max())
        num, den, n_rows = event_contrast_sums(
            true_reward, corrected_reward, levels == event_level, mask
        )
        if n_rows > 0 and den != 0:
            out["event_contrast"] = num / den
        lv = levels[mask]
        for level in torch.unique(lv).tolist():
            sel = lv == level
            err = c[sel] - t[sel]
            out[f"level{level}_frac"] = sel.float().mean().item()
            out[f"level{level}_bias"] = err.mean().item()
            out[f"level{level}_mse"] = (err**2).mean().item()
    return out

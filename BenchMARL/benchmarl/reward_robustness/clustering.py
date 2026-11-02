"""Turning scalar per-agent values (true, noisy or predicted rewards) into group labels.

Two families of label are produced:

* *global* labels (thresholds, GMM, k-means): the same cluster ids are used for every
  timestep; a group is the set of agents of one timestep sharing a cluster id.
  Clusters are always ordered by their centre so ids are stable across batches.
* *per-timestep* labels (ε-gap clustering): agents of one timestep are split
  wherever consecutive sorted values differ by more than ε, which is a direct
  implementation of the ε-similarity in Assumption 1 (single-linkage in 1-D).
"""

from typing import Optional, Sequence

import numpy as np
import torch


def threshold_labels(
    values: torch.Tensor, thresholds: Sequence[float], right: bool = False
) -> torch.Tensor:
    """Bin ``values`` with the given edges (see ``torch.bucketize`` for ``right``)."""
    if len(thresholds) == 0:
        return torch.zeros_like(values, dtype=torch.long)
    edges = torch.as_tensor(thresholds, dtype=values.dtype, device=values.device)
    return torch.bucketize(values, edges, right=right)


def gmm_labels(
    values: torch.Tensor,
    n_components: int,
    mask: Optional[torch.Tensor] = None,
    sort_components: bool = True,
    random_state: int = 0,
) -> torch.Tensor:
    """Fit a 1-D Gaussian mixture on all (unmasked) values and return hard labels.

    Args:
        sort_components: relabel components in increasing order of their mean. Without
            this, sklearn's component order is arbitrary and changes between fits,
            so labels from different batches are not comparable (the original
            No-Warmup implementation trained a classifier on such labels).
    """
    from sklearn.mixture import GaussianMixture

    if mask is None:
        mask = torch.ones_like(values, dtype=torch.bool)
    x = values[mask].detach().double().cpu().numpy().reshape(-1, 1)
    labels = torch.full_like(values, -1, dtype=torch.long)
    if x.shape[0] < n_components:
        labels[mask] = 0
        return labels
    gmm = GaussianMixture(
        n_components=n_components, covariance_type="full", random_state=random_state
    ).fit(x)
    raw = gmm.predict(x)
    if sort_components:
        order = np.argsort(gmm.means_.ravel())
        rank = np.empty_like(order)
        rank[order] = np.arange(n_components)
        raw = rank[raw]
    labels[mask] = torch.as_tensor(raw, dtype=torch.long, device=values.device)
    return labels


def kmeans_1d_labels(
    values: torch.Tensor,
    n_clusters: int,
    mask: Optional[torch.Tensor] = None,
    n_iters: int = 30,
) -> torch.Tensor:
    """Lloyd's algorithm in 1-D, initialised on an even grid between min and max.

    In 1-D Lloyd's algorithm preserves the order of the centres, so label ``0`` is
    always the lowest cluster. The grid initialisation (instead of quantiles) keeps
    rare but large values (e.g. capture rewards) in their own cluster.
    """
    if mask is None:
        mask = torch.ones_like(values, dtype=torch.bool)
    labels = torch.full_like(values, -1, dtype=torch.long)
    x = values[mask].detach()
    if x.numel() == 0:
        return labels
    lo, hi = x.min(), x.max()
    if n_clusters <= 1 or hi - lo < 1e-12:
        labels[mask] = 0
        return labels
    centres = torch.linspace(lo.item(), hi.item(), n_clusters, device=x.device, dtype=x.dtype)
    for _ in range(n_iters):
        assign = (x.unsqueeze(-1) - centres).abs().argmin(-1)
        sums = torch.zeros_like(centres).scatter_add_(0, assign, x)
        counts = torch.zeros_like(centres).scatter_add_(0, assign, torch.ones_like(x))
        new_centres = torch.where(counts > 0, sums / counts.clamp(min=1), centres)
        if torch.allclose(new_centres, centres):
            break
        centres = new_centres
    labels[mask] = (x.unsqueeze(-1) - centres).abs().argmin(-1)
    return labels


def epsilon_labels(
    values: torch.Tensor, epsilon: float, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Per-timestep 1-D single-linkage clustering with tolerance ``epsilon``.

    Agents are sorted by value; a new group starts wherever the gap to the previous
    agent exceeds ``epsilon``. With ``epsilon = 0`` only identical values are grouped.
    """
    if mask is None:
        mask = torch.ones_like(values, dtype=torch.bool)
    inf = torch.tensor(float("inf"), dtype=values.dtype, device=values.device)
    sorted_vals, order = torch.sort(torch.where(mask, values, inf), dim=-1)
    gaps = sorted_vals[..., 1:] - sorted_vals[..., :-1]
    new_group = torch.cat(
        [torch.zeros_like(gaps[..., :1], dtype=torch.long), (~(gaps <= epsilon)).long()],
        dim=-1,
    )
    sorted_labels = new_group.cumsum(-1)
    labels = torch.empty_like(sorted_labels).scatter_(-1, order, sorted_labels)
    return torch.where(mask, labels, torch.full_like(labels, -1))

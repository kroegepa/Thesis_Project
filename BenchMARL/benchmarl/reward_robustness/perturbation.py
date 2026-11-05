"""Reward perturbation kernels K (Definition 2 of the paper).

Each agent's reward is perturbed independently at every timestep.
"""

import torch

PERTURBATION_TYPES = ("normal", "bernoulli")


def perturb_rewards(
    reward: torch.Tensor,
    kind: str,
    stdev: float = 0.0,
    flip_prob: float = 0.0,
    outlier_value: float = 10.0,
) -> torch.Tensor:
    """Return a perturbed copy of ``reward``.

    Args:
        reward: true rewards of any shape.
        kind: ``"normal"`` adds i.i.d. ``N(0, stdev^2)`` noise (zero-mean, Assumption 2).
            ``"bernoulli"`` replaces each reward with probability ``flip_prob`` by
            ``+outlier_value`` or ``-outlier_value`` (50/50). This kernel is *not*
            mean-preserving: ``E[r~] = (1 - flip_prob) * r``.
        stdev: standard deviation of the Gaussian kernel.
        flip_prob: corruption probability of the Bernoulli kernel.
        outlier_value: magnitude of Bernoulli outliers.
    """
    if kind == "normal":
        return reward + torch.randn_like(reward) * stdev
    if kind == "bernoulli":
        attacked = torch.rand_like(reward) < flip_prob
        sign = torch.where(torch.rand_like(reward) < 0.5, 1.0, -1.0)
        return torch.where(attacked, sign * outlier_value, reward)
    raise ValueError(
        f"Unknown reward perturbation type '{kind}', expected one of {PERTURBATION_TYPES}"
    )

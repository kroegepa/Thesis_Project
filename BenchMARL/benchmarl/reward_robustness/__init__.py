"""Grouped-Reward Robust MARL (GR²-MARL): reward perturbation and consensus correction.

See :mod:`benchmarl.reward_robustness.groupers` for the available grouping methods.
"""

from benchmarl.reward_robustness.consensus import apply_consensus, Assignment
from benchmarl.reward_robustness.groupers import (
    GroupData,
    Grouper,
    GROUPERS,
    GroupingConfig,
    make_grouper,
    oracle_labels,
)
from benchmarl.reward_robustness.perturbation import perturb_rewards
from benchmarl.reward_robustness.robustness import RewardRobustness
from benchmarl.reward_robustness.task_spec import get_task_spec, TASK_SPECS, TaskSpec

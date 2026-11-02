"""Per-task information needed by the reward-robustness components.

Everything that used to be hard-coded in ``if task_name == ...`` chains (agent group
name, oracle thresholds, grouping-network architecture) lives here, so adding a task
means adding one entry to :data:`TASK_SPECS`.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class TaskSpec:
    """Static description of how GR²-MARL handles a task.

    Args:
        agent_group: key of the agent group whose rewards are grouped (e.g. ``"pursuer"``).
        oracle_thresholds: bin edges used by the oracle to turn true rewards into reward
            levels. ``None`` means the oracle clusters true rewards with a GMM instead.
        oracle_right: ``torch.bucketize`` ``right`` flag. ``False`` means a reward equal to a
            threshold falls in the lower bin (``r > t`` semantics), ``True`` means it falls
            in the upper bin (``r >= t`` semantics).
        encoder: ``"cnn"`` for grid observations ``[H, W, C]``, ``"mlp"`` for flat ones.
        cnn_channels / cnn_paddings: conv stack of the grouping network (3x3 kernels).
        hidden: width of the hidden layer(s) of the grouping network head.
    """

    agent_group: str
    oracle_thresholds: Optional[Tuple[float, ...]]
    oracle_right: bool = False
    encoder: str = "mlp"
    cnn_channels: Tuple[int, ...] = (16, 32)
    cnn_paddings: Tuple[int, ...] = (1, 1)
    hidden: int = 128


# The architectures below reproduce the grouping networks used in the paper.
TASK_SPECS = {
    # Pursuit: r = -0.1 (urgency) + tag shaping (< 0.05) + 5 per capture.
    # Levels: {urgency only, tagging, capture}.
    "pursuit": TaskSpec(
        agent_group="pursuer",
        oracle_thresholds=(-0.1, 2.0),
        oracle_right=False,
        encoder="cnn",
        cnn_channels=(16, 32),
        cnn_paddings=(1, 1),
        hidden=128,
    ),
    # SimpleSpread: shared distance term + local collision penalty. No natural levels,
    # the oracle clusters the true rewards with a GMM.
    "simple_spread": TaskSpec(
        agent_group="agent",
        oracle_thresholds=None,
        encoder="mlp",
        hidden=128,
    ),
    "adversarial_pursuit": TaskSpec(
        agent_group="predator",
        oracle_thresholds=(0.0, 0.8),
        oracle_right=True,
        encoder="cnn",
        cnn_channels=(32, 64, 64),
        cnn_paddings=(1, 0, 0),
        hidden=256,
    ),
    "battle": TaskSpec(
        agent_group="red",
        oracle_thresholds=(0.0, 1.0),
        oracle_right=True,
        encoder="cnn",
        cnn_channels=(32, 64, 64),
        cnn_paddings=(1, 0, 0),
        hidden=256,
    ),
}


def get_task_spec(task_name: str) -> TaskSpec:
    try:
        return TASK_SPECS[task_name.lower()]
    except KeyError:
        raise NotImplementedError(
            f"Task '{task_name}' has no reward-robustness TaskSpec. "
            f"Supported tasks: {sorted(TASK_SPECS)}"
        ) from None

"""Selection utilities for teacher correction seeds from student rollouts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class RecoveryStepSelection:
    step: int
    reason: str
    gripper_switch_count: int
    burst_start: int | None


def select_recovery_step(
    actions: npt.ArrayLike,
    *,
    minimum_step: int = 20,
    pre_roll_steps: int = 5,
    burst_window: int = 20,
    minimum_burst_switches: int = 4,
) -> RecoveryStepSelection:
    """Choose a conservative teacher takeover point before gripper oscillation."""

    array = np.asarray(actions)
    if (
        not np.issubdtype(array.dtype, np.floating)
        or array.ndim != 2
        or array.shape[1] != 7
        or len(array) == 0
        or not np.isfinite(array).all()
    ):
        raise ValueError("correction-source actions must be finite floating [T,7]")
    if (
        minimum_step < 0
        or pre_roll_steps < 0
        or burst_window <= 0
        or minimum_burst_switches <= 0
    ):
        raise ValueError("recovery selection parameters are invalid")

    closed = array[:, 6] >= 0.0
    switches = np.flatnonzero(closed[1:] != closed[:-1]) + 1
    burst_start: int | None = None
    for switch in switches:
        count = int(np.count_nonzero((switches >= switch) & (switches < switch + burst_window)))
        if count >= minimum_burst_switches:
            burst_start = int(switch)
            break

    last_step = len(array) - 1
    bounded_minimum = min(minimum_step, last_step)
    if burst_start is not None:
        step = max(bounded_minimum, burst_start - pre_roll_steps)
        reason = "before_gripper_switch_burst"
    else:
        step = max(bounded_minimum, len(array) // 3)
        step = min(step, last_step)
        reason = "trajectory_one_third_fallback"
    return RecoveryStepSelection(
        step=step,
        reason=reason,
        gripper_switch_count=len(switches),
        burst_start=burst_start,
    )

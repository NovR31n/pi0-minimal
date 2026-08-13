import numpy as np
import pytest

from pi0_minimal.correction_data import select_recovery_step


def test_recovery_step_precedes_first_dense_gripper_switch_burst() -> None:
    actions = np.zeros((100, 7), dtype=np.float32)
    actions[:, 6] = -1.0
    for start, value in [(40, 1.0), (43, -1.0), (46, 1.0), (49, -1.0)]:
        actions[start:, 6] = value

    selected = select_recovery_step(actions)

    assert selected.step == 35
    assert selected.reason == "before_gripper_switch_burst"
    assert selected.burst_start == 40
    assert selected.gripper_switch_count == 4


def test_recovery_step_falls_back_to_one_third_for_stable_gripper() -> None:
    actions = np.zeros((90, 7), dtype=np.float32)
    actions[:, 6] = -1.0

    selected = select_recovery_step(actions)

    assert selected.step == 30
    assert selected.reason == "trajectory_one_third_fallback"
    assert selected.burst_start is None


def test_recovery_step_is_bounded_for_short_trace() -> None:
    selected = select_recovery_step(np.zeros((5, 7), dtype=np.float32))

    assert selected.step == 4


def test_recovery_step_rejects_invalid_actions() -> None:
    with pytest.raises(ValueError, match=r"\[T,7\]"):
        select_recovery_step(np.zeros((2, 6), dtype=np.float32))

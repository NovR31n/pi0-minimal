from __future__ import annotations

import numpy as np

from pi0_minimal.failure_analysis import FailureProxyThresholds, diagnose_trace


def _positions(steps: int, distance: float = 0.5) -> np.ndarray:
    positions = np.zeros((steps, 3), dtype=np.float32)
    positions[:, 0] = np.linspace(0.0, distance, steps)
    return positions


def test_diagnose_trace_marks_missing_close_command() -> None:
    actions = np.zeros((12, 7), dtype=np.float32)
    actions[:, 6] = 1.0
    diagnosis = diagnose_trace(actions, _positions(12))
    assert diagnosis["primary_proxy"] == "no_close_command"
    assert "no_close_command" in diagnosis["proxy_flags"]
    assert diagnosis["gripper"]["first_close_step"] is None


def test_diagnose_trace_marks_gripper_oscillation() -> None:
    actions = np.zeros((14, 7), dtype=np.float32)
    actions[:, 6] = np.asarray([1.0, -1.0] * 7)
    diagnosis = diagnose_trace(
        actions,
        _positions(14),
        thresholds=FailureProxyThresholds(
            excessive_gripper_switches=6,
            severe_gripper_switches=12,
        ),
    )
    assert diagnosis["primary_proxy"] == "severe_gripper_switching"
    assert "excessive_gripper_switching" in diagnosis["proxy_flags"]
    assert "severe_gripper_switching" in diagnosis["proxy_flags"]


def test_diagnose_trace_marks_no_reopen_after_close() -> None:
    actions = np.zeros((10, 7), dtype=np.float32)
    actions[:3, 6] = 1.0
    actions[3:, 6] = -1.0
    diagnosis = diagnose_trace(actions, _positions(10))
    assert diagnosis["primary_proxy"] == "no_reopen_after_close"
    assert diagnosis["gripper"]["first_close_step"] == 3
    assert diagnosis["gripper"]["reopened_after_close"] is False


def test_diagnose_trace_does_not_claim_semantic_failure_stage() -> None:
    actions = np.zeros((12, 7), dtype=np.float32)
    actions[:4, 6] = 1.0
    actions[4:8, 6] = -1.0
    actions[8:, 6] = 1.0
    diagnosis = diagnose_trace(actions, _positions(12))
    assert diagnosis["semantic_failure_stage"] == (
        "unknown_requires_video_or_object_state"
    )
    assert diagnosis["requires_video_review"] is True

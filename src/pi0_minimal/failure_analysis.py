"""Conservative proxy diagnostics for saved closed-loop rollout traces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from pi0_minimal.metrics import trajectory_metrics


@dataclass(frozen=True)
class FailureProxyThresholds:
    """Predeclared thresholds for trace-level failure proxies.

    These thresholds identify motion and gripper symptoms. They do not infer
    object contact, grasp success, drops, or placement success because the
    current trace schema does not contain object poses or contact signals.
    """

    gripper_deadband: float = 0.1
    excessive_gripper_switches: int = 10
    severe_gripper_switches: int = 24
    low_endpoint_displacement: float = 0.31
    low_path_efficiency: float = 0.28
    high_stagnation_ratio: float = 0.02

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def diagnose_trace(
    normalized_actions: npt.ArrayLike,
    end_effector_positions: npt.ArrayLike,
    *,
    thresholds: FailureProxyThresholds | None = None,
) -> dict[str, Any]:
    """Return non-semantic proxy flags for one saved rollout trace."""

    limits = thresholds or FailureProxyThresholds()
    actions = np.asarray(normalized_actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 7 or not len(actions):
        raise ValueError("normalized_actions must have shape [T,7] with T > 0")
    if not np.isfinite(actions).all():
        raise ValueError("normalized_actions must contain only finite values")

    metrics = trajectory_metrics(end_effector_positions, actions)
    gripper = actions[:, 6]
    discrete = np.where(
        gripper > limits.gripper_deadband,
        1,
        np.where(gripper < -limits.gripper_deadband, -1, 0),
    )
    nonzero_indices = np.flatnonzero(discrete)
    close_indices = np.flatnonzero(discrete < 0)
    first_close = int(close_indices[0]) if len(close_indices) else None
    reopened_after_close = bool(
        first_close is not None and np.any(discrete[first_close + 1 :] > 0)
    )
    active_fraction = float(np.mean(discrete != 0))

    flags: list[str] = []
    if first_close is None:
        flags.append("no_close_command")
    if metrics.gripper_switch_count >= limits.excessive_gripper_switches:
        flags.append("excessive_gripper_switching")
    if metrics.gripper_switch_count >= limits.severe_gripper_switches:
        flags.append("severe_gripper_switching")
    if first_close is not None and not reopened_after_close:
        flags.append("no_reopen_after_close")
    if metrics.endpoint_displacement <= limits.low_endpoint_displacement:
        flags.append("low_endpoint_displacement")
    if metrics.path_efficiency <= limits.low_path_efficiency:
        flags.append("low_path_efficiency")
    if metrics.stagnation_ratio >= limits.high_stagnation_ratio:
        flags.append("high_stagnation")
    if not flags:
        flags.append("no_proxy_detected")

    priority = (
        "no_close_command",
        "severe_gripper_switching",
        "excessive_gripper_switching",
        "no_reopen_after_close",
        "low_endpoint_displacement",
        "low_path_efficiency",
        "high_stagnation",
        "no_proxy_detected",
    )
    primary = next(item for item in priority if item in flags)
    return {
        "primary_proxy": primary,
        "proxy_flags": flags,
        "semantic_failure_stage": "unknown_requires_video_or_object_state",
        "requires_video_review": True,
        "gripper": {
            "first_close_step": first_close,
            "reopened_after_close": reopened_after_close,
            "active_command_fraction": active_fraction,
            "nonzero_command_steps": len(nonzero_indices),
        },
        "trajectory_metrics": metrics.to_dict(),
    }

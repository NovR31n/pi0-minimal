"""Safety-critical conversion between LIBERO observations and policy tensors."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from pi0_minimal.data.normalization import QuantileStats


@dataclass(frozen=True, slots=True)
class SafeActionChunk:
    values: npt.NDArray[np.float32]
    normalized_clip_count: int
    environment_clip_count: int

    def __post_init__(self) -> None:
        if self.values.dtype != np.float32 or self.values.ndim != 2:
            raise TypeError("safe LIBERO actions must be float32 [H,7]")
        if self.values.shape[1] != 7:
            raise ValueError("safe LIBERO actions must have width 7")
        if not np.isfinite(self.values).all() or np.any(np.abs(self.values) > 1.0):
            raise ValueError("safe LIBERO actions must be finite and lie in [-1,1]")
        if self.normalized_clip_count < 0 or self.environment_clip_count < 0:
            raise ValueError("clip counts must be non-negative")


@dataclass(slots=True)
class DebouncedBipolarGripper:
    """Emit stable bipolar commands after repeated evidence for a state change."""

    threshold: float = 0.5
    debounce_steps: int = 5
    initial_state: float = -1.0
    _state: float = field(init=False)
    _candidate: float = field(init=False)
    _candidate_steps: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError("gripper threshold must lie in (0,1]")
        if self.debounce_steps <= 0:
            raise ValueError("gripper debounce steps must be positive")
        if self.initial_state not in (-1.0, 1.0):
            raise ValueError("initial gripper state must be -1 or 1")
        self._state = self.initial_state
        self._candidate = self.initial_state

    def apply(self, action: npt.ArrayLike) -> npt.NDArray[np.float32]:
        array = np.asarray(action)
        if (
            not np.issubdtype(array.dtype, np.floating)
            or array.shape != (7,)
            or not np.isfinite(array).all()
            or np.any(np.abs(array) > 1.0 + 1e-6)
        ):
            raise ValueError("LIBERO action must be a finite floating point [7] in [-1,1]")

        command = float(array[6])
        desired = self._state
        if command <= -self.threshold:
            desired = -1.0
        elif command >= self.threshold:
            desired = 1.0

        if desired == self._state:
            self._candidate = self._state
            self._candidate_steps = 0
        elif desired != self._candidate:
            self._candidate = desired
            self._candidate_steps = 1
        else:
            self._candidate_steps += 1

        if (
            self._candidate != self._state
            and self._candidate_steps >= self.debounce_steps
        ):
            self._state = self._candidate
            self._candidate_steps = 0

        result = array.astype(np.float32, copy=True)
        result[6] = self._state
        return result


def libero_simulator_state(environment: object) -> npt.NDArray[np.float64]:
    """Copy the flattened MuJoCo state needed to restore an exact rollout point."""

    getter = getattr(environment, "get_sim_state", None)
    if not callable(getter):
        raise TypeError("LIBERO environment must provide callable get_sim_state()")
    state = np.asarray(getter())
    if not np.issubdtype(state.dtype, np.floating) or state.ndim != 1:
        raise ValueError("LIBERO simulator state must be a floating-point vector")
    if not np.isfinite(state).all():
        raise ValueError("LIBERO simulator state must contain only finite values")
    return state.astype(np.float64, copy=True)


def libero_controller_state(environment: object) -> npt.NDArray[np.float64]:
    """Copy the OSC goal that is not represented in the flattened MuJoCo state."""

    robots = getattr(environment, "robots", None)
    if robots is None or len(robots) != 1:
        raise TypeError("LIBERO trace capture requires exactly one robot")
    controller = getattr(robots[0], "controller", None)
    if controller is None:
        raise TypeError("LIBERO robot must provide a controller")
    if (
        getattr(controller, "interpolator_pos", None) is not None
        or getattr(controller, "interpolator_ori", None) is not None
    ):
        raise ValueError("interpolated LIBERO controllers are not trace-supported")
    goal_position = np.asarray(getattr(controller, "goal_pos", None))
    goal_orientation = np.asarray(getattr(controller, "goal_ori", None))
    if (
        goal_position.shape != (3,)
        or goal_orientation.shape != (3, 3)
        or not np.issubdtype(goal_position.dtype, np.floating)
        or not np.issubdtype(goal_orientation.dtype, np.floating)
    ):
        raise ValueError("LIBERO OSC controller goals have an unexpected schema")
    state = np.concatenate((goal_position, goal_orientation.reshape(-1)))
    if not np.isfinite(state).all():
        raise ValueError("LIBERO controller state must contain only finite values")
    return state.astype(np.float64, copy=True)


def restore_libero_controller_state(
    environment: object,
    state: npt.ArrayLike,
) -> None:
    """Restore a controller goal captured by :func:`libero_controller_state`."""

    array = np.asarray(state)
    if (
        not np.issubdtype(array.dtype, np.floating)
        or array.shape != (12,)
        or not np.isfinite(array).all()
    ):
        raise ValueError("LIBERO controller state must be a finite length-12 vector")
    robots = getattr(environment, "robots", None)
    if robots is None or len(robots) != 1:
        raise TypeError("LIBERO controller restore requires exactly one robot")
    controller = getattr(robots[0], "controller", None)
    if controller is None:
        raise TypeError("LIBERO robot must provide a controller")
    if (
        getattr(controller, "interpolator_pos", None) is not None
        or getattr(controller, "interpolator_ori", None) is not None
    ):
        raise ValueError("interpolated LIBERO controllers are not trace-supported")
    controller.goal_pos = array[:3].astype(np.float64, copy=True)
    controller.goal_ori = array[3:].reshape(3, 3).astype(np.float64, copy=True)


def libero_numeric_observation(
    observation: Mapping[str, object],
) -> dict[str, npt.NDArray[np.float32]]:
    """Copy finite numeric observation fields while excluding camera images."""

    result: dict[str, npt.NDArray[np.float32]] = {}
    for key in sorted(observation):
        if key.endswith("_image"):
            continue
        value = np.asarray(observation[key])
        if not np.issubdtype(value.dtype, np.number):
            continue
        if not np.isfinite(value).all():
            raise ValueError(f"LIBERO observation field {key!r} is not finite")
        result[key] = value.astype(np.float32, copy=True)
    return result


def stack_libero_observation_trace(
    observations: Sequence[Mapping[str, npt.NDArray[np.float32]]],
    *,
    prefix: str,
) -> dict[str, npt.NDArray[np.float32]]:
    """Stack fixed-schema numeric observations into named NPZ payload fields."""

    if not observations:
        return {}
    expected_keys = tuple(observations[0])
    expected_shapes = {
        key: np.asarray(observations[0][key]).shape for key in expected_keys
    }
    payload: dict[str, npt.NDArray[np.float32]] = {}
    for index, observation in enumerate(observations):
        if tuple(observation) != expected_keys:
            raise ValueError(
                f"LIBERO observation schema changed at trace index {index}"
            )
        for key in expected_keys:
            if np.asarray(observation[key]).shape != expected_shapes[key]:
                raise ValueError(
                    f"LIBERO observation field {key!r} changed shape at "
                    f"trace index {index}"
                )
    for key in expected_keys:
        payload[f"{prefix}{key}"] = np.stack(
            [observation[key] for observation in observations]
        ).astype(np.float32, copy=False)
    return payload


def libero_state_from_observation(
    observation: Mapping[str, object],
) -> npt.NDArray[np.float32]:
    """Return xyz + axis-angle + finger opening width from one simulator observation."""

    required = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
    missing = [key for key in required if key not in observation]
    if missing:
        raise KeyError(f"LIBERO observation is missing fields: {missing}")
    position = _vector(observation["robot0_eef_pos"], "end-effector position", 3)
    quaternion = _vector(observation["robot0_eef_quat"], "end-effector quaternion", 4)
    gripper = _vector(observation["robot0_gripper_qpos"], "gripper qpos", 2)
    opening_width = gripper[0] - gripper[1]
    return np.concatenate(
        (position, quaternion_to_axis_angle(quaternion), [opening_width])
    ).astype(np.float32)


def quaternion_to_axis_angle(
    quaternion: npt.ArrayLike,
) -> npt.NDArray[np.float32]:
    """Convert LIBERO/robosuite xyzw quaternion to a three-vector."""

    quat = _vector(quaternion, "quaternion", 4).astype(np.float64)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        raise ValueError("quaternion norm must be positive")
    quat /= norm
    scalar = float(np.clip(quat[3], -1.0, 1.0))
    denominator = math.sqrt(max(0.0, 1.0 - scalar * scalar))
    if math.isclose(denominator, 0.0, abs_tol=1e-8):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * (2.0 * math.acos(scalar) / denominator)).astype(np.float32)


def safe_libero_action_chunk(
    normalized_actions: npt.ArrayLike,
    action_stats: QuantileStats,
) -> SafeActionChunk:
    """Restore actions, convert the binary gripper command, and enforce LIBERO's box."""

    normalized = np.asarray(normalized_actions)
    if (
        not np.issubdtype(normalized.dtype, np.floating)
        or normalized.ndim != 2
        or normalized.shape[1] != 7
    ):
        raise ValueError("normalized actions must be floating point [H,7]")
    if not np.isfinite(normalized).all():
        raise ValueError("normalized actions must be finite")
    normalized_clip_count = int(np.count_nonzero(np.abs(normalized) > 1.0))
    clipped_normalized = np.clip(normalized, -1.0, 1.0).astype(np.float32)
    restored = action_stats.inverse(clipped_normalized)
    _require_binary_gripper_stats(action_stats)
    restored = training_actions_to_libero_space(restored)
    environment_clip_count = int(np.count_nonzero(np.abs(restored) > 1.0))
    safe = np.clip(restored, -1.0, 1.0).astype(np.float32)
    return SafeActionChunk(safe, normalized_clip_count, environment_clip_count)


def libero_actions_to_training_space(
    actions: npt.ArrayLike,
) -> npt.NDArray[np.float32]:
    """Map LIBERO commands to LeRobot labels (1=open, 0=closed)."""

    array = np.asarray(actions)
    if (
        not np.issubdtype(array.dtype, np.floating)
        or array.ndim != 2
        or array.shape[1] != 7
    ):
        raise ValueError("executed LIBERO actions must be floating point [T,7]")
    if not np.isfinite(array).all() or np.any(np.abs(array) > 1.0 + 1e-6):
        raise ValueError("executed LIBERO actions must be finite and lie in [-1,1]")
    training_actions = array.astype(np.float32, copy=True)
    training_actions[..., 6] = (1.0 - training_actions[..., 6]) / 2.0
    return training_actions


def training_actions_to_libero_space(
    actions: npt.ArrayLike,
) -> npt.NDArray[np.float32]:
    """Map LeRobot labels (1=open, 0=closed) to LIBERO commands."""

    array = np.asarray(actions)
    if (
        not np.issubdtype(array.dtype, np.floating)
        or array.ndim != 2
        or array.shape[1] != 7
    ):
        raise ValueError("training actions must be floating point [T,7]")
    if not np.isfinite(array).all():
        raise ValueError("training actions must be finite")
    if np.any(array[:, 6] < -1e-6) or np.any(array[:, 6] > 1.0 + 1e-6):
        raise ValueError("training gripper actions must lie in [0,1]")
    libero_actions = array.astype(np.float32, copy=True)
    libero_actions[..., 6] = 1.0 - libero_actions[..., 6] * 2.0
    return libero_actions


def _require_binary_gripper_stats(action_stats: QuantileStats) -> None:
    bounds = np.asarray(
        [action_stats.lower[6], action_stats.upper[6]],
        dtype=np.float64,
    )
    if not np.allclose(bounds, [0.0, 1.0], rtol=0.0, atol=1e-6):
        raise ValueError(
            "LIBERO action statistics must encode binary gripper labels with bounds [0,1]"
        )


def _vector(value: object, name: str, width: int) -> npt.NDArray[np.float32]:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.floating) or array.shape != (width,):
        raise ValueError(f"{name} must be a floating-point vector of length {width}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array.astype(np.float32)

import numpy as np
import pytest

from pi0_minimal.data import QuantileStats
from pi0_minimal.libero_adapter import (
    DebouncedBipolarGripper,
    libero_actions_to_training_space,
    libero_controller_state,
    libero_numeric_observation,
    libero_simulator_state,
    libero_state_from_observation,
    quaternion_to_axis_angle,
    restore_libero_controller_state,
    safe_libero_action_chunk,
    stack_libero_observation_trace,
    training_actions_to_libero_space,
)


class _FakeLiberoEnvironment:
    class _Controller:
        goal_pos = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        goal_ori = np.eye(3, dtype=np.float64)
        interpolator_pos = None
        interpolator_ori = None

    class _Robot:
        controller = None

    def __init__(self) -> None:
        robot = self._Robot()
        robot.controller = self._Controller()
        self.robots = [robot]

    def get_sim_state(self) -> np.ndarray:
        return np.array([1.0, 2.0, 3.0], dtype=np.float64)


def test_libero_observation_converts_quaternion_and_finger_width() -> None:
    state = libero_state_from_observation(
        {
            "robot0_eef_pos": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "robot0_gripper_qpos": np.array([0.04, -0.03], dtype=np.float32),
        }
    )

    np.testing.assert_allclose(
        state,
        np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.07], dtype=np.float32),
    )


def test_libero_simulator_state_is_copied_as_float64() -> None:
    state = libero_simulator_state(_FakeLiberoEnvironment())

    np.testing.assert_array_equal(state, [1.0, 2.0, 3.0])
    assert state.dtype == np.float64


def test_libero_controller_state_round_trip() -> None:
    environment = _FakeLiberoEnvironment()
    state = libero_controller_state(environment)
    environment.robots[0].controller.goal_pos[:] = 0.0
    environment.robots[0].controller.goal_ori[:] = 0.0

    restore_libero_controller_state(environment, state)

    np.testing.assert_array_equal(
        environment.robots[0].controller.goal_pos,
        [0.1, 0.2, 0.3],
    )
    np.testing.assert_array_equal(
        environment.robots[0].controller.goal_ori,
        np.eye(3),
    )


def test_numeric_observation_trace_excludes_images_and_stacks_fields() -> None:
    first = libero_numeric_observation(
        {
            "agentview_image": np.zeros((4, 4, 3), dtype=np.uint8),
            "object-state": np.array([1.0, 2.0], dtype=np.float64),
            "robot0_gripper_qpos": np.array([0.04, -0.04]),
            "language": "ignored",
        }
    )
    second = libero_numeric_observation(
        {
            "agentview_image": np.ones((4, 4, 3), dtype=np.uint8),
            "object-state": np.array([3.0, 4.0], dtype=np.float64),
            "robot0_gripper_qpos": np.array([0.03, -0.03]),
            "language": "ignored",
        }
    )

    payload = stack_libero_observation_trace([first, second], prefix="before__")

    assert set(payload) == {
        "before__object-state",
        "before__robot0_gripper_qpos",
    }
    np.testing.assert_array_equal(
        payload["before__object-state"],
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )


def test_observation_trace_rejects_schema_drift() -> None:
    with pytest.raises(ValueError, match="schema changed"):
        stack_libero_observation_trace(
            [
                {"object-state": np.zeros(2, dtype=np.float32)},
                {"other": np.zeros(2, dtype=np.float32)},
            ],
            prefix="before__",
        )


def test_quaternion_axis_angle_is_invariant_to_scale() -> None:
    expected = np.array([0.0, 0.0, np.pi / 2], dtype=np.float32)

    np.testing.assert_allclose(
        quaternion_to_axis_angle([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]),
        expected,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        quaternion_to_axis_angle([0.0, 0.0, 2 * np.sqrt(0.5), 2 * np.sqrt(0.5)]),
        expected,
        rtol=1e-6,
    )


def test_safe_action_chunk_clips_before_and_after_inverse() -> None:
    stats = QuantileStats(
        lower=np.array([-2.0] * 6 + [0.0]),
        upper=np.array([2.0] * 6 + [1.0]),
        count=np.full(7, 10, dtype=np.int64),
    )
    generated = np.array([[2.0, -2.0, 0.0, 0.5, -0.5, 0.25, -0.25]], dtype=np.float32)

    chunk = safe_libero_action_chunk(generated, stats)

    assert chunk.normalized_clip_count == 2
    assert chunk.environment_clip_count == 2
    assert np.max(np.abs(chunk.values)) <= 1.0
    np.testing.assert_allclose(
        chunk.values,
        np.array([[1.0, -1.0, 0.0, 1.0, -1.0, 0.5, 0.25]], dtype=np.float32),
    )


def test_safe_action_chunk_maps_binary_gripper_to_libero_bipolar_commands() -> None:
    stats = QuantileStats(
        lower=np.array([-1.0] * 6 + [0.0]),
        upper=np.ones(7),
        count=np.full(7, 10, dtype=np.int64),
    )
    generated = np.zeros((3, 7), dtype=np.float32)
    generated[:, 6] = [-1.0, 0.0, 1.0]

    chunk = safe_libero_action_chunk(generated, stats)

    np.testing.assert_allclose(chunk.values[:, 6], [1.0, 0.0, -1.0])


def test_safe_action_chunk_rejects_non_binary_gripper_statistics() -> None:
    stats = QuantileStats(
        lower=np.full(7, -1.0),
        upper=np.ones(7),
        count=np.full(7, 10, dtype=np.int64),
    )

    with pytest.raises(ValueError, match="binary gripper labels"):
        safe_libero_action_chunk(np.zeros((1, 7), dtype=np.float32), stats)


def test_safe_action_chunk_rejects_non_finite_values() -> None:
    stats = QuantileStats(
        lower=np.array([-1.0] * 6 + [0.0]),
        upper=np.full(7, 1.0),
        count=np.full(7, 10, dtype=np.int64),
    )

    with pytest.raises(ValueError, match="finite"):
        safe_libero_action_chunk(np.full((10, 7), np.nan), stats)


def test_executed_libero_actions_map_back_to_binary_gripper_space() -> None:
    actions = np.zeros((3, 7), dtype=np.float32)
    actions[:, 6] = [-1.0, 0.0, 1.0]

    converted = libero_actions_to_training_space(actions)

    np.testing.assert_allclose(converted[:, 6], [1.0, 0.5, 0.0])
    np.testing.assert_allclose(actions[:, 6], [-1.0, 0.0, 1.0])


def test_libero_training_action_conversion_round_trip_is_exact() -> None:
    actions = np.array(
        [
            [0.25, -0.5, 0.0, 0.1, -0.2, 0.3, -1.0],
            [-0.25, 0.5, 0.0, -0.1, 0.2, -0.3, 1.0],
        ],
        dtype=np.float32,
    )

    restored = training_actions_to_libero_space(
        libero_actions_to_training_space(actions)
    )

    np.testing.assert_array_equal(restored, actions)


def test_training_action_conversion_rejects_non_binary_gripper_range() -> None:
    actions = np.zeros((2, 7), dtype=np.float32)
    actions[:, 6] = [-0.01, 1.01]

    with pytest.raises(ValueError, match="gripper"):
        training_actions_to_libero_space(actions)


def test_debounced_gripper_requires_repeated_evidence_and_does_not_mutate() -> None:
    gripper = DebouncedBipolarGripper(threshold=0.5, debounce_steps=5)
    action = np.zeros(7, dtype=np.float32)
    emitted = []
    for command in [0.6] * 5 + [0.0] * 2 + [-0.6] * 5:
        action[6] = command
        emitted.append(float(gripper.apply(action)[6]))

    assert emitted == [-1.0] * 4 + [1.0] * 7 + [-1.0]
    assert action[6] == pytest.approx(-0.6)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"threshold": 0.0}, "threshold"),
        ({"debounce_steps": 0}, "debounce"),
        ({"initial_state": 0.0}, "initial"),
    ],
)
def test_debounced_gripper_rejects_invalid_configuration(
    kwargs: dict[str, float | int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DebouncedBipolarGripper(**kwargs)

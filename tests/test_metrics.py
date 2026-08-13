import math

import numpy as np
import pytest

from pi0_minimal.metrics import (
    action_prediction_metrics,
    action_smoothness_metrics,
    bootstrap_mean_interval,
    paired_difference_interval,
    runtime_metrics,
    trajectory_metrics,
    wilson_interval,
)


def test_action_prediction_metrics_clip_mask_and_score_gripper() -> None:
    target = np.zeros((1, 2, 7), dtype=np.float32)
    target[..., 6] = [-1.0, 1.0]
    predicted = target.copy()
    predicted[0, 0, 0] = 2.0
    predicted[0, 1, 0] = 999.0
    predicted[0, 1, 6] = -0.25
    valid = np.ones_like(target, dtype=np.bool_)
    valid[0, 1, 0] = False

    metrics = action_prediction_metrics(predicted, target, valid)

    assert metrics.normalized_action_mae == pytest.approx(2.25 / 13)
    assert metrics.normalized_arm_mae == pytest.approx(1.0 / 11)
    assert metrics.zero_action_baseline_mae == pytest.approx(2.0 / 13)
    assert metrics.zero_arm_baseline_mae == 0.0
    assert metrics.gripper_accuracy == 0.5
    assert metrics.prediction_clip_ratio == pytest.approx(1 / 13)
    assert metrics.per_dimension_mae[0] == 1.0
    assert metrics.per_horizon_mae == pytest.approx((1 / 7, 1.25 / 6))
    assert metrics.per_horizon_valid_scalars == (7, 6)
    assert metrics.valid_scalars == 13
    assert metrics.valid_gripper_steps == 2


def test_action_prediction_metrics_reject_bad_shapes_and_empty_masks() -> None:
    actions = np.zeros((1, 2, 7), dtype=np.float32)

    with pytest.raises(ValueError, match="floating point"):
        action_prediction_metrics(
            actions[..., :6],
            actions[..., :6],
            np.ones((1, 2, 6), dtype=np.bool_),
        )
    with pytest.raises(ValueError, match="valid action"):
        action_prediction_metrics(
            actions,
            actions,
            np.zeros_like(actions, dtype=np.bool_),
        )


def test_constant_actions_have_zero_temporal_and_frequency_metrics() -> None:
    actions = np.ones((16, 3), dtype=np.float32)

    metrics = action_smoothness_metrics(actions)

    assert metrics.first_difference_rms == 0.0
    assert metrics.second_difference_rms == 0.0
    assert metrics.high_frequency_energy_ratio == 0.0


def test_linear_actions_have_constant_first_and_zero_second_difference() -> None:
    actions = np.arange(8, dtype=np.float32)[:, None]

    metrics = action_smoothness_metrics(actions)

    assert metrics.first_difference_rms == 1.0
    assert metrics.second_difference_rms == 0.0


def test_oscillation_has_more_high_frequency_energy_than_step() -> None:
    step = np.concatenate((np.zeros(16), np.ones(16)))[:, None]
    oscillation = np.tile([0.0, 1.0], 16)[:, None]

    step_metrics = action_smoothness_metrics(step)
    oscillation_metrics = action_smoothness_metrics(oscillation)

    assert oscillation_metrics.first_difference_rms > step_metrics.first_difference_rms
    assert (
        oscillation_metrics.second_difference_rms
        > step_metrics.second_difference_rms
    )
    assert (
        oscillation_metrics.high_frequency_energy_ratio
        > step_metrics.high_frequency_energy_ratio
    )


def test_padding_is_excluded_and_must_be_a_prefix() -> None:
    actions = np.asarray([[0.0], [1.0], [2.0], [999.0]])
    valid = np.asarray([[True], [True], [True], [False]])

    metrics = action_smoothness_metrics(actions, valid)

    assert metrics.first_difference_rms == 1.0
    assert metrics.second_difference_rms == 0.0
    with pytest.raises(ValueError, match="contiguous prefix"):
        action_smoothness_metrics(actions, [[True], [False], [True], [False]])


def test_trajectory_metrics_match_straight_line_and_action_events() -> None:
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    )
    actions = np.zeros((3, 7))
    actions[0, 0] = 1.0
    actions[:, 6] = [-1.0, 1.0, -1.0]

    metrics = trajectory_metrics(positions, actions)

    assert metrics.path_length == 2.0
    assert metrics.endpoint_displacement == 2.0
    assert metrics.path_efficiency == 1.0
    assert metrics.stagnation_ratio == 0.0
    assert metrics.action_saturation_ratio == pytest.approx(4 / 21)
    assert metrics.gripper_switch_count == 2


def test_stationary_trajectory_has_zero_efficiency_and_full_stagnation() -> None:
    positions = np.zeros((4, 3))
    actions = np.zeros((4, 7))

    metrics = trajectory_metrics(positions, actions)

    assert metrics.path_length == 0.0
    assert metrics.path_efficiency == 0.0
    assert metrics.stagnation_ratio == 1.0


def test_runtime_separates_first_call_from_stable_calls() -> None:
    metrics = runtime_metrics([0.5, 0.1, 0.2])

    assert metrics.calls == 3
    assert metrics.first_seconds == 0.5
    assert metrics.stable_mean_seconds == pytest.approx(0.15)
    assert metrics.mean_seconds == pytest.approx(0.8 / 3)
    assert metrics.total_seconds == pytest.approx(0.8)


def test_wilson_interval_has_known_center_behavior_and_validates_counts() -> None:
    lower, upper = wilson_interval(5, 10)

    assert lower == pytest.approx(0.236593, abs=1e-6)
    assert upper == pytest.approx(0.763407, abs=1e-6)
    assert math.isclose(sum(wilson_interval(0, 10)), 0.277533, abs_tol=1e-6)
    with pytest.raises(ValueError):
        wilson_interval(11, 10)


def test_bootstrap_intervals_are_deterministic_and_preserve_pairing() -> None:
    first = np.asarray([3.0, 4.0, 5.0])
    second = np.asarray([1.0, 2.0, 3.0])

    mean_interval = bootstrap_mean_interval(
        [2.0, 2.0, 2.0],
        resamples=100,
    )
    paired = paired_difference_interval(
        first,
        second,
        resamples=100,
    )

    assert mean_interval.estimate == 2.0
    assert mean_interval.lower == 2.0
    assert mean_interval.upper == 2.0
    assert paired.estimate == 2.0
    assert paired.lower == 2.0
    assert paired.upper == 2.0

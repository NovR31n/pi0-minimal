"""Deterministic metrics for policy action, trajectory, and runtime traces."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class ActionSmoothnessMetrics:
    first_difference_rms: float
    second_difference_rms: float
    high_frequency_energy_ratio: float
    per_dimension_first_difference_rms: tuple[float, ...]
    per_dimension_second_difference_rms: tuple[float, ...]
    per_dimension_high_frequency_energy_ratio: tuple[float, ...]

    def to_dict(self) -> dict[str, float | list[float]]:
        result = asdict(self)
        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in result.items()
        }


@dataclass(frozen=True, slots=True)
class TrajectoryMetrics:
    path_length: float
    endpoint_displacement: float
    path_efficiency: float
    stagnation_ratio: float
    action_saturation_ratio: float
    gripper_switch_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RuntimeMetrics:
    calls: int
    first_seconds: float
    stable_mean_seconds: float
    mean_seconds: float
    median_seconds: float
    p95_seconds: float
    total_seconds: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ActionPredictionMetrics:
    normalized_action_mae: float
    normalized_action_rmse: float
    normalized_arm_mae: float
    zero_action_baseline_mae: float
    zero_arm_baseline_mae: float
    gripper_accuracy: float
    prediction_clip_ratio: float
    per_dimension_mae: tuple[float, ...]
    per_horizon_mae: tuple[float | None, ...]
    per_horizon_valid_scalars: tuple[int, ...]
    valid_scalars: int
    valid_gripper_steps: int

    def to_dict(
        self,
    ) -> dict[str, float | int | list[float | int | None]]:
        result = asdict(self)
        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in result.items()
        }


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    resamples: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def action_smoothness_metrics(
    actions: npt.ArrayLike,
    valid: npt.ArrayLike | None = None,
    *,
    high_frequency_cutoff: float = 0.25,
) -> ActionSmoothnessMetrics:
    """Measure temporal differences and detrended high-frequency energy.

    ``high_frequency_cutoff`` is in cycles per sample and must lie below the
    Nyquist frequency. Valid samples for each dimension must form one
    contiguous prefix, matching padded action trajectories.
    """

    values = _finite_matrix(actions, "actions")
    mask = _valid_matrix(valid, values.shape)
    if not 0.0 < high_frequency_cutoff < 0.5:
        raise ValueError("high_frequency_cutoff must lie in (0,0.5)")
    _require_prefix_mask(mask)

    first_by_dim = []
    second_by_dim = []
    frequency_by_dim = []
    all_first: list[np.ndarray] = []
    all_second: list[np.ndarray] = []
    for dimension in range(values.shape[1]):
        sequence = values[mask[:, dimension], dimension].astype(np.float64)
        first = np.diff(sequence)
        second = np.diff(sequence, n=2)
        all_first.append(first)
        all_second.append(second)
        first_by_dim.append(_rms(first))
        second_by_dim.append(_rms(second))
        frequency_by_dim.append(
            _high_frequency_energy_ratio(sequence, high_frequency_cutoff)
        )
    return ActionSmoothnessMetrics(
        first_difference_rms=_rms(_concatenate_nonempty(all_first)),
        second_difference_rms=_rms(_concatenate_nonempty(all_second)),
        high_frequency_energy_ratio=float(np.mean(frequency_by_dim)),
        per_dimension_first_difference_rms=tuple(first_by_dim),
        per_dimension_second_difference_rms=tuple(second_by_dim),
        per_dimension_high_frequency_energy_ratio=tuple(frequency_by_dim),
    )


def action_prediction_metrics(
    predicted: npt.ArrayLike,
    target: npt.ArrayLike,
    valid: npt.ArrayLike,
) -> ActionPredictionMetrics:
    """Compare generated normalized actions with targets after rollout clipping."""

    predictions = np.asarray(predicted)
    targets = np.asarray(target)
    mask = np.asarray(valid)
    if (
        not np.issubdtype(predictions.dtype, np.floating)
        or not np.issubdtype(targets.dtype, np.floating)
        or predictions.ndim != 3
        or predictions.shape != targets.shape
        or predictions.shape[-1] != 7
    ):
        raise ValueError("predicted and target must be floating point [N,H,7]")
    if mask.dtype != np.bool_ or mask.shape != predictions.shape:
        raise ValueError("valid must be boolean and match the action tensors")
    if not np.isfinite(predictions).all() or not np.isfinite(targets).all():
        raise ValueError("action tensors must be finite")
    if not mask.any() or not mask[..., 6].any():
        raise ValueError("metrics require valid action and gripper elements")

    clipped = np.clip(predictions, -1.0, 1.0)
    absolute_error = np.abs(clipped - targets)
    squared_error = np.square(clipped - targets)
    arm_valid = mask[..., :6]
    gripper_valid = mask[..., 6]
    per_horizon_valid = tuple(
        int(mask[:, step].sum()) for step in range(mask.shape[1])
    )
    per_dimension = tuple(
        float(absolute_error[..., dimension][mask[..., dimension]].mean())
        for dimension in range(7)
    )
    per_horizon = tuple(
        (
            float(absolute_error[:, step][mask[:, step]].mean())
            if per_horizon_valid[step]
            else None
        )
        for step in range(mask.shape[1])
    )
    return ActionPredictionMetrics(
        normalized_action_mae=float(absolute_error[mask].mean()),
        normalized_action_rmse=float(np.sqrt(squared_error[mask].mean())),
        normalized_arm_mae=float(absolute_error[..., :6][arm_valid].mean()),
        zero_action_baseline_mae=float(np.abs(targets)[mask].mean()),
        zero_arm_baseline_mae=float(np.abs(targets[..., :6])[arm_valid].mean()),
        gripper_accuracy=float(
            np.mean(
                (clipped[..., 6][gripper_valid] >= 0.0)
                == (targets[..., 6][gripper_valid] >= 0.0)
            )
        ),
        prediction_clip_ratio=float(
            np.mean(np.abs(predictions[mask]) > 1.0)
        ),
        per_dimension_mae=per_dimension,
        per_horizon_mae=per_horizon,
        per_horizon_valid_scalars=per_horizon_valid,
        valid_scalars=int(mask.sum()),
        valid_gripper_steps=int(gripper_valid.sum()),
    )


def trajectory_metrics(
    end_effector_positions: npt.ArrayLike,
    actions: npt.ArrayLike,
    *,
    stagnation_threshold: float = 1e-4,
    saturation_threshold: float = 0.999,
    gripper_deadband: float = 0.1,
) -> TrajectoryMetrics:
    """Measure geometric progress, inactivity, saturation, and gripper switches."""

    positions = _finite_matrix(end_effector_positions, "positions")
    action_values = _finite_matrix(actions, "actions")
    if positions.shape[1] != 3:
        raise ValueError("positions must have shape [T,3]")
    if action_values.shape[1] != 7:
        raise ValueError("actions must have shape [T,7]")
    if len(positions) != len(action_values):
        raise ValueError("positions and actions must have the same length")
    if len(positions) < 1:
        raise ValueError("trajectory must contain at least one step")
    if stagnation_threshold < 0.0:
        raise ValueError("stagnation_threshold must be non-negative")
    if not 0.0 < saturation_threshold <= 1.0:
        raise ValueError("saturation_threshold must lie in (0,1]")
    if not 0.0 <= gripper_deadband < 1.0:
        raise ValueError("gripper_deadband must lie in [0,1)")

    displacement_steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    path_length = float(displacement_steps.sum())
    endpoint_displacement = float(np.linalg.norm(positions[-1] - positions[0]))
    path_efficiency = (
        endpoint_displacement / path_length if path_length > 0.0 else 0.0
    )
    stagnation_ratio = (
        float(np.mean(displacement_steps <= stagnation_threshold))
        if len(displacement_steps)
        else 0.0
    )
    saturation_ratio = float(
        np.mean(np.abs(action_values) >= saturation_threshold)
    )
    gripper = action_values[:, 6]
    discrete = np.where(
        gripper > gripper_deadband,
        1,
        np.where(gripper < -gripper_deadband, -1, 0),
    )
    nonzero = discrete[discrete != 0]
    switches = int(np.count_nonzero(np.diff(nonzero))) if len(nonzero) > 1 else 0
    return TrajectoryMetrics(
        path_length=path_length,
        endpoint_displacement=endpoint_displacement,
        path_efficiency=path_efficiency,
        stagnation_ratio=stagnation_ratio,
        action_saturation_ratio=saturation_ratio,
        gripper_switch_count=switches,
    )


def runtime_metrics(inference_times: npt.ArrayLike) -> RuntimeMetrics:
    """Summarize first-call and steady-state policy generation latency."""

    times = np.asarray(inference_times, dtype=np.float64)
    if times.ndim != 1 or not len(times):
        raise ValueError("inference_times must be a non-empty vector")
    if not np.isfinite(times).all() or np.any(times < 0.0):
        raise ValueError("inference_times must be finite and non-negative")
    stable = times[1:] if len(times) > 1 else times
    return RuntimeMetrics(
        calls=len(times),
        first_seconds=float(times[0]),
        stable_mean_seconds=float(stable.mean()),
        mean_seconds=float(times.mean()),
        median_seconds=float(np.median(times)),
        p95_seconds=float(np.percentile(times, 95)),
        total_seconds=float(times.sum()),
    )


def wilson_interval(
    successes: int,
    trials: int,
    *,
    z: float = 1.95996398454,
) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""

    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("require 0 <= successes <= trials and trials > 0")
    if z <= 0.0 or not math.isfinite(z):
        raise ValueError("z must be positive and finite")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def bootstrap_mean_interval(
    values: npt.ArrayLike,
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 7,
) -> BootstrapInterval:
    """Return a deterministic percentile-bootstrap interval for a mean."""

    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError("values must be a non-empty finite vector")
    if not 0.0 < confidence < 1.0 or resamples <= 0:
        raise ValueError("confidence must lie in (0,1) and resamples be positive")
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0,
        len(samples),
        size=(resamples, len(samples)),
    )
    bootstrap_means = samples[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(bootstrap_means, [tail, 1.0 - tail])
    return BootstrapInterval(
        estimate=float(samples.mean()),
        lower=float(lower),
        upper=float(upper),
        confidence=confidence,
        resamples=resamples,
    )


def paired_difference_interval(
    first: npt.ArrayLike,
    second: npt.ArrayLike,
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 7,
) -> BootstrapInterval:
    """Bootstrap the mean paired difference ``first - second``."""

    first_values = np.asarray(first, dtype=np.float64)
    second_values = np.asarray(second, dtype=np.float64)
    if first_values.shape != second_values.shape:
        raise ValueError("paired samples must have matching shapes")
    return bootstrap_mean_interval(
        first_values - second_values,
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )


def _finite_matrix(values: npt.ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.number) or array.ndim != 2:
        raise ValueError(f"{name} must be a numeric rank-2 array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _valid_matrix(
    valid: npt.ArrayLike | None,
    shape: tuple[int, int],
) -> np.ndarray:
    if valid is None:
        return np.ones(shape, dtype=np.bool_)
    mask = np.asarray(valid)
    if mask.dtype != np.bool_ or mask.shape != shape:
        raise ValueError("valid must be boolean and match actions")
    if not mask.any(axis=0).all():
        raise ValueError("every action dimension must contain a valid sample")
    return mask


def _require_prefix_mask(mask: np.ndarray) -> None:
    for dimension in range(mask.shape[1]):
        invalid_seen = np.maximum.accumulate(~mask[:, dimension])
        if np.any(mask[:, dimension] & invalid_seen):
            raise ValueError("valid samples must form a contiguous prefix")


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else 0.0


def _concatenate_nonempty(values: list[np.ndarray]) -> np.ndarray:
    nonempty = [value for value in values if len(value)]
    return np.concatenate(nonempty) if nonempty else np.empty(0, dtype=np.float64)


def _high_frequency_energy_ratio(
    sequence: np.ndarray,
    cutoff: float,
) -> float:
    if len(sequence) < 2:
        return 0.0
    detrended = sequence - sequence.mean()
    power = np.abs(np.fft.rfft(detrended)) ** 2
    frequencies = np.fft.rfftfreq(len(detrended))
    non_dc = frequencies > 0.0
    total = float(power[non_dc].sum())
    if total == 0.0:
        return 0.0
    return float(power[frequencies >= cutoff].sum() / total)

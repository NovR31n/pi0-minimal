"""Source-safe dataset merging and sampling for offline policy distillation."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

_SAMPLE_FIELDS = (
    "images",
    "state",
    "actions",
    "action_valid",
    "prompts",
    "episode_index",
)


def merge_demo_and_teacher_data(
    demonstrations: Mapping[str, np.ndarray],
    teacher: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Append teacher samples while requiring disjoint episode identities."""

    teacher_indices = teacher_indices_after_demonstrations(demonstrations, teacher)
    merged = {
        field: np.concatenate((demonstrations[field], teacher[field]), axis=0)
        for field in _SAMPLE_FIELDS
    }
    merged["image_keys"] = np.asarray(demonstrations["image_keys"]).copy()
    return merged, teacher_indices


def teacher_indices_after_demonstrations(
    demonstrations: Mapping[str, np.ndarray],
    teacher: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Validate source identity and return teacher indices in a virtual merge."""

    if not np.array_equal(demonstrations["image_keys"], teacher["image_keys"]):
        raise ValueError("demonstration and teacher image keys must match")
    demo_episodes = set(np.asarray(demonstrations["episode_index"]).tolist())
    teacher_episodes = set(np.asarray(teacher["episode_index"]).tolist())
    if demo_episodes & teacher_episodes:
        raise ValueError("demonstration and teacher episode identities must be disjoint")
    demo_count = len(demonstrations["state"])
    teacher_count = len(teacher["state"])
    if not demo_count or not teacher_count:
        raise ValueError("both distillation sources must contain samples")
    return np.arange(demo_count, demo_count + teacher_count, dtype=np.int64)


def source_balanced_sampling_probabilities(
    demonstration_probabilities: np.ndarray | None,
    *,
    demonstration_count: int,
    teacher_count: int,
    teacher_fraction: float,
    teacher_task_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Assign exact source mass and optionally balance teacher mass by task."""

    if demonstration_count <= 0 or teacher_count <= 0:
        raise ValueError("both sources must contribute at least one training sample")
    if not 0.0 < teacher_fraction < 1.0:
        raise ValueError("teacher sampling fraction must lie strictly between 0 and 1")
    demo = _normalized_probabilities(
        demonstration_probabilities,
        count=demonstration_count,
        source_name="demonstration",
    )
    teacher = _task_balanced_probabilities(
        count=teacher_count,
        task_indices=teacher_task_indices,
        source_name="teacher",
    )
    return np.concatenate(
        (demo * (1.0 - teacher_fraction), teacher * teacher_fraction)
    )


def three_source_balanced_sampling_probabilities(
    demonstration_probabilities: np.ndarray | None,
    *,
    demonstration_count: int,
    teacher_count: int,
    correction_count: int,
    teacher_fraction: float,
    correction_fraction: float,
    teacher_task_indices: np.ndarray | None = None,
    correction_task_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Assign exact mass to demonstrations, teacher data, and corrections."""

    if demonstration_count <= 0 or teacher_count <= 0 or correction_count <= 0:
        raise ValueError("all three sources must contribute at least one sample")
    if teacher_fraction <= 0.0 or correction_fraction <= 0.0:
        raise ValueError("teacher and correction fractions must be positive")
    demonstration_fraction = 1.0 - teacher_fraction - correction_fraction
    if demonstration_fraction <= 0.0:
        raise ValueError("teacher and correction fractions must sum to less than one")
    demo = _normalized_probabilities(
        demonstration_probabilities,
        count=demonstration_count,
        source_name="demonstration",
    )
    teacher = _task_balanced_probabilities(
        count=teacher_count,
        task_indices=teacher_task_indices,
        source_name="teacher",
    )
    correction = _task_balanced_probabilities(
        count=correction_count,
        task_indices=correction_task_indices,
        source_name="correction",
    )
    return np.concatenate(
        (
            demo * demonstration_fraction,
            teacher * teacher_fraction,
            correction * correction_fraction,
        )
    )


def _normalized_probabilities(
    probabilities: np.ndarray | None,
    *,
    count: int,
    source_name: str,
) -> np.ndarray:
    if probabilities is None:
        return np.full(count, 1.0 / count)
    normalized = np.asarray(probabilities, dtype=np.float64)
    if normalized.shape != (count,):
        raise ValueError(f"{source_name} probabilities have the wrong shape")
    if (
        not np.isfinite(normalized).all()
        or np.any(normalized < 0.0)
        or normalized.sum() <= 0.0
    ):
        raise ValueError(
            f"{source_name} probabilities must be finite and non-negative"
        )
    return normalized / normalized.sum()


def _task_balanced_probabilities(
    *,
    count: int,
    task_indices: np.ndarray | None,
    source_name: str,
) -> np.ndarray:
    if task_indices is None:
        return np.full(count, 1.0 / count)
    indices = np.asarray(task_indices)
    if indices.shape != (count,) or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(f"{source_name} task indices must be an integer vector")
    tasks, counts = np.unique(indices, return_counts=True)
    if not len(tasks):
        raise ValueError(f"{source_name} task indices must not be empty")
    probabilities = np.zeros(count, dtype=np.float64)
    for task, task_count in zip(tasks, counts, strict=True):
        probabilities[indices == task] = 1.0 / (len(tasks) * int(task_count))
    return probabilities

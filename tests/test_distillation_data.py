import numpy as np
import pytest

from pi0_minimal.data import (
    merge_demo_and_teacher_data,
    source_balanced_sampling_probabilities,
    teacher_indices_after_demonstrations,
    three_source_balanced_sampling_probabilities,
)


def _source(count: int, episode: int) -> dict[str, np.ndarray]:
    return {
        "images": np.zeros((count, 2, 3, 4, 4), dtype=np.uint8),
        "image_keys": np.asarray(["base", "wrist"]),
        "state": np.zeros((count, 7), dtype=np.float32),
        "actions": np.zeros((count, 10, 7), dtype=np.float32),
        "action_valid": np.ones((count, 10, 7), dtype=np.bool_),
        "prompts": np.asarray(["task"] * count),
        "episode_index": np.full(count, episode, dtype=np.int64),
    }


def test_distillation_merge_keeps_demo_first_and_teacher_disjoint() -> None:
    merged, teacher_indices = merge_demo_and_teacher_data(
        _source(3, 1),
        _source(2, 1_000_000),
    )
    assert merged["state"].shape == (5, 7)
    np.testing.assert_array_equal(
        merged["episode_index"],
        [1, 1, 1, 1_000_000, 1_000_000],
    )
    np.testing.assert_array_equal(teacher_indices, [3, 4])


def test_distillation_merge_rejects_episode_identity_leakage() -> None:
    with pytest.raises(ValueError, match="episode identities"):
        merge_demo_and_teacher_data(_source(2, 1), _source(2, 1))


def test_virtual_teacher_indices_do_not_require_array_merging() -> None:
    indices = teacher_indices_after_demonstrations(
        _source(3, 1),
        _source(2, 1_000_000),
    )
    np.testing.assert_array_equal(indices, [3, 4])


def test_source_balancing_assigns_exact_probability_mass() -> None:
    probabilities = source_balanced_sampling_probabilities(
        np.asarray([0.75, 0.25]),
        demonstration_count=2,
        teacher_count=4,
        teacher_fraction=0.4,
    )
    assert probabilities.shape == (6,)
    assert probabilities[:2].sum() == pytest.approx(0.6)
    assert probabilities[2:].sum() == pytest.approx(0.4)
    assert probabilities.sum() == pytest.approx(1.0)


def test_teacher_probability_mass_is_balanced_across_tasks() -> None:
    probabilities = source_balanced_sampling_probabilities(
        None,
        demonstration_count=2,
        teacher_count=4,
        teacher_fraction=0.4,
        teacher_task_indices=np.asarray([0, 1, 1, 1], dtype=np.int64),
    )
    teacher = probabilities[2:]
    assert teacher[0] == pytest.approx(0.2)
    assert teacher[1:].sum() == pytest.approx(0.2)
    assert probabilities[:2].sum() == pytest.approx(0.6)


def test_teacher_task_balancing_rejects_invalid_indices() -> None:
    with pytest.raises(ValueError, match="integer vector"):
        source_balanced_sampling_probabilities(
            None,
            demonstration_count=2,
            teacher_count=2,
            teacher_fraction=0.5,
            teacher_task_indices=np.asarray([0.0, 1.0]),
        )


def test_three_source_balancing_assigns_exact_mass_per_source_and_task() -> None:
    probabilities = three_source_balanced_sampling_probabilities(
        np.asarray([0.75, 0.25]),
        demonstration_count=2,
        teacher_count=4,
        correction_count=3,
        teacher_fraction=0.25,
        correction_fraction=0.25,
        teacher_task_indices=np.asarray([0, 1, 1, 1]),
        correction_task_indices=np.asarray([2, 6, 6]),
    )
    assert probabilities[:2].sum() == pytest.approx(0.5)
    assert probabilities[2:6].sum() == pytest.approx(0.25)
    assert probabilities[6:].sum() == pytest.approx(0.25)
    assert probabilities[2] == pytest.approx(0.125)
    assert probabilities[3:6].sum() == pytest.approx(0.125)
    assert probabilities[6] == pytest.approx(0.125)
    assert probabilities[7:].sum() == pytest.approx(0.125)
    assert probabilities.sum() == pytest.approx(1.0)


def test_three_source_balancing_rejects_exhausted_demo_mass() -> None:
    with pytest.raises(ValueError, match="sum to less than one"):
        three_source_balanced_sampling_probabilities(
            None,
            demonstration_count=2,
            teacher_count=2,
            correction_count=2,
            teacher_fraction=0.5,
            correction_fraction=0.5,
        )

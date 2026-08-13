import json

import numpy as np
import pytest

from pi0_minimal.teacher_data import (
    COMPLETE_MARKER,
    build_teacher_query,
    canonicalize_teacher_action_chunk,
    is_teacher_episode_complete,
    load_teacher_episode_training_samples,
    save_teacher_episode,
    select_teacher_episode_dirs,
    teacher_state_from_observation,
)


def _observation() -> dict[str, np.ndarray]:
    return {
        "robot0_eef_pos": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.asarray([0.04, -0.03], dtype=np.float32),
    }


def test_teacher_and_student_state_contracts_are_explicit() -> None:
    observation = _observation()
    teacher_state = teacher_state_from_observation(observation)
    query = build_teacher_query(
        observation,
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.ones((8, 8, 3), dtype=np.uint8),
        np.zeros((10, 7), dtype=np.float32),
        student_horizon=10,
        replan_steps=5,
    )
    np.testing.assert_allclose(
        teacher_state,
        np.asarray([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.04, -0.03]),
    )
    assert query["student_state"].shape == (7,)
    assert query["teacher_state"].shape == (8,)


def test_teacher_chunk_is_converted_and_padded_for_student() -> None:
    actions = np.zeros((7, 7), dtype=np.float32)
    actions[:3, 6] = -1.0
    actions[3:, 6] = 1.0
    chunk = canonicalize_teacher_action_chunk(
        actions,
        student_horizon=10,
        replan_steps=5,
    )
    assert chunk.raw_libero_actions.shape == (7, 7)
    np.testing.assert_array_equal(chunk.target_training_actions[:3, 6], 1.0)
    np.testing.assert_array_equal(chunk.target_training_actions[3:7, 6], 0.0)
    np.testing.assert_array_equal(chunk.target_training_actions[7:], 0.0)
    assert chunk.target_valid[:7].all()
    assert not chunk.target_valid[7:].any()


@pytest.mark.parametrize(
    "actions",
    [
        np.zeros((4, 7), dtype=np.float32),
        np.zeros((10, 8), dtype=np.float32),
        np.full((10, 7), np.nan, dtype=np.float32),
        np.full((10, 7), np.inf, dtype=np.float32),
    ],
)
def test_teacher_chunk_rejects_invalid_actions(actions: np.ndarray) -> None:
    with pytest.raises(ValueError):
        canonicalize_teacher_action_chunk(
            actions,
            student_horizon=10,
            replan_steps=5,
        )


def test_teacher_chunk_preserves_raw_values_and_clips_student_target() -> None:
    actions = np.full((10, 7), 1.1, dtype=np.float32)
    chunk = canonicalize_teacher_action_chunk(
        actions,
        student_horizon=10,
        replan_steps=5,
    )
    np.testing.assert_allclose(chunk.raw_libero_actions, 1.1)
    np.testing.assert_allclose(chunk.target_training_actions[:, :6], 1.0)
    np.testing.assert_allclose(chunk.target_training_actions[:, 6], 0.0)


def test_teacher_episode_save_is_atomic_and_resumable(tmp_path) -> None:
    episode_dir = tmp_path / "task00" / "init00"
    save_teacher_episode(
        episode_dir,
        metadata={"task_id": 0, "success": True},
        arrays={"actions": np.zeros((2, 10, 7), dtype=np.float32)},
    )
    assert is_teacher_episode_complete(episode_dir)
    assert (episode_dir / COMPLETE_MARKER).read_text().strip() == "complete"
    assert json.loads((episode_dir / "result.json").read_text())["schema_version"] == 1
    with np.load(episode_dir / "teacher_episode.npz") as episode:
        assert episode["actions"].shape == (2, 10, 7)
    with pytest.raises(FileExistsError):
        save_teacher_episode(
            episode_dir,
            metadata={},
            arrays={"actions": np.zeros((1, 10, 7), dtype=np.float32)},
        )


def test_complete_teacher_episode_converts_to_training_contract(tmp_path) -> None:
    episode_dir = tmp_path / "task03" / "init07"
    query_count = 2
    save_teacher_episode(
        episode_dir,
        metadata={
            "task_id": 3,
            "init_state_index": 7,
            "task_description": "place the object",
            "success": True,
            "completed_without_exception": True,
            "infrastructure_error": False,
        },
        arrays={
            "base_images": np.zeros((query_count, 8, 8, 3), dtype=np.uint8),
            "wrist_images": np.ones((query_count, 8, 8, 3), dtype=np.uint8),
            "student_states": np.zeros((query_count, 7), dtype=np.float32),
            "target_actions": np.zeros((query_count, 10, 7), dtype=np.float32),
            "target_valid": np.ones((query_count, 10, 7), dtype=np.bool_),
            "query_steps": np.asarray([0, 5], dtype=np.int32),
        },
    )
    samples = load_teacher_episode_training_samples(episode_dir)
    assert samples["images"].shape == (2, 2, 3, 8, 8)
    assert samples["images"].dtype == np.uint8
    assert samples["image_keys"].tolist() == ["base_0_rgb", "wrist_0_rgb"]
    assert samples["state"].shape == (2, 7)
    assert samples["actions"].shape == (2, 10, 7)
    np.testing.assert_array_equal(samples["episode_index"], 1_003_007)
    np.testing.assert_array_equal(samples["task_index"], 3)


def test_correction_episode_uses_disjoint_identity_namespace(tmp_path) -> None:
    episode_dir = tmp_path / "task03" / "init07"
    query_count = 1
    save_teacher_episode(
        episode_dir,
        metadata={
            "model_type": "official_pi05_libero_teacher_correction",
            "task_id": 3,
            "init_state_index": 7,
            "task_description": "place the object",
            "success": True,
            "completed_without_exception": True,
            "infrastructure_error": False,
        },
        arrays={
            "base_images": np.zeros((query_count, 8, 8, 3), dtype=np.uint8),
            "wrist_images": np.ones((query_count, 8, 8, 3), dtype=np.uint8),
            "student_states": np.zeros((query_count, 7), dtype=np.float32),
            "target_actions": np.zeros((query_count, 10, 7), dtype=np.float32),
            "target_valid": np.ones((query_count, 10, 7), dtype=np.bool_),
            "query_steps": np.asarray([0], dtype=np.int32),
        },
    )
    samples = load_teacher_episode_training_samples(episode_dir)
    np.testing.assert_array_equal(samples["episode_index"], 2_003_007)


def _save_selection_episode(
    episode_dir, *, task_id: int, init_state_index: int, success: bool
) -> None:
    save_teacher_episode(
        episode_dir,
        metadata={
            "task_id": task_id,
            "init_state_index": init_state_index,
            "task_description": "test task",
            "success": success,
            "completed_without_exception": True,
            "infrastructure_error": False,
        },
        arrays={"placeholder": np.zeros(1, dtype=np.float32)},
    )


def test_teacher_episode_selection_skips_failures_and_balances_tasks(tmp_path) -> None:
    _save_selection_episode(
        tmp_path / "task00" / "init00", task_id=0, init_state_index=0, success=True
    )
    _save_selection_episode(
        tmp_path / "task00" / "init01", task_id=0, init_state_index=1, success=False
    )
    _save_selection_episode(
        tmp_path / "task00" / "init02", task_id=0, init_state_index=2, success=True
    )
    _save_selection_episode(
        tmp_path / "task01" / "init03", task_id=1, init_state_index=3, success=True
    )
    selected = select_teacher_episode_dirs(
        [tmp_path], successes_per_task=1, expected_task_ids=[0, 1]
    )
    assert [(path.parent.name, path.name) for path in selected] == [
        ("task00", "init00"),
        ("task01", "init03"),
    ]


def test_teacher_episode_selection_requires_requested_success_count(tmp_path) -> None:
    _save_selection_episode(
        tmp_path / "task00" / "init00", task_id=0, init_state_index=0, success=True
    )
    with pytest.raises(ValueError, match="task 0 has 1 eligible successes; 2 required"):
        select_teacher_episode_dirs(
            [tmp_path], successes_per_task=2, expected_task_ids=[0]
        )

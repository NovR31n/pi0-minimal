"""Validated, resumable artifacts for offline LIBERO policy distillation."""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

TEACHER_DATA_SCHEMA_VERSION = 1
COMPLETE_MARKER = "_COMPLETE"


@dataclass(frozen=True)
class CanonicalTeacherChunk:
    """One raw teacher action chunk and its fixed-horizon student target."""

    raw_libero_actions: npt.NDArray[np.float32]
    target_training_actions: npt.NDArray[np.float32]
    target_valid: npt.NDArray[np.bool_]

    def __post_init__(self) -> None:
        if (
            self.raw_libero_actions.dtype != np.float32
            or self.raw_libero_actions.ndim != 2
            or self.raw_libero_actions.shape[1] != 7
        ):
            raise TypeError("raw teacher actions must be float32 [T,7]")
        if (
            self.target_training_actions.dtype != np.float32
            or self.target_training_actions.ndim != 2
            or self.target_training_actions.shape[1] != 7
        ):
            raise TypeError("student targets must be float32 [H,7]")
        if self.target_valid.dtype != np.bool_ or self.target_valid.shape != (
            *self.target_training_actions.shape,
        ):
            raise TypeError("student target validity must be bool [H,7]")


def teacher_state_from_observation(
    observation: Mapping[str, object],
) -> npt.NDArray[np.float32]:
    """Return the official OpenPI LIBERO xyz + axis-angle + two-finger state."""

    required = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
    missing = [key for key in required if key not in observation]
    if missing:
        raise KeyError(f"LIBERO observation is missing fields: {missing}")
    position = _floating_vector(observation["robot0_eef_pos"], "position", 3)
    quaternion = _floating_vector(observation["robot0_eef_quat"], "quaternion", 4)
    fingers = _floating_vector(observation["robot0_gripper_qpos"], "gripper", 2)
    return np.concatenate(
        (position, _quaternion_to_axis_angle(quaternion), fingers)
    ).astype(np.float32)


def canonicalize_teacher_action_chunk(
    actions: npt.ArrayLike,
    *,
    student_horizon: int,
    replan_steps: int,
) -> CanonicalTeacherChunk:
    """Validate official actions and construct a padded student-space target."""

    if student_horizon <= 0 or replan_steps <= 0:
        raise ValueError("student horizon and replanning interval must be positive")
    array = np.asarray(actions)
    if (
        not np.issubdtype(array.dtype, np.floating)
        or array.ndim != 2
        or array.shape[1] != 7
    ):
        raise ValueError("teacher actions must be a floating-point [T,7] array")
    if len(array) < replan_steps:
        raise ValueError("teacher action chunk is shorter than the replanning interval")
    if not np.isfinite(array).all():
        raise ValueError("teacher actions must be finite")

    raw = array.astype(np.float32)
    environment_actions = np.clip(raw, -1.0, 1.0)
    available = min(len(raw), student_horizon)
    target = np.zeros((student_horizon, 7), dtype=np.float32)
    valid = np.zeros((student_horizon, 7), dtype=np.bool_)
    target[:available] = _libero_actions_to_training_space(
        environment_actions[:available]
    )
    valid[:available] = True
    return CanonicalTeacherChunk(raw, target, valid)


def build_teacher_query(
    observation: Mapping[str, object],
    base_image: npt.ArrayLike,
    wrist_image: npt.ArrayLike,
    teacher_actions: npt.ArrayLike,
    *,
    student_horizon: int,
    replan_steps: int,
) -> dict[str, npt.NDArray[Any]]:
    """Build one fully validated query record for an episode artifact."""

    base = _uint8_image(base_image, "base image")
    wrist = _uint8_image(wrist_image, "wrist image")
    if base.shape != wrist.shape:
        raise ValueError("base and wrist images must have the same shape")
    chunk = canonicalize_teacher_action_chunk(
        teacher_actions,
        student_horizon=student_horizon,
        replan_steps=replan_steps,
    )
    return {
        "base_image": base,
        "wrist_image": wrist,
        "student_state": _student_state_from_observation(observation),
        "teacher_state": teacher_state_from_observation(observation),
        "raw_teacher_actions": chunk.raw_libero_actions,
        "target_actions": chunk.target_training_actions,
        "target_valid": chunk.target_valid,
    }


def save_teacher_episode(
    output_dir: Path,
    *,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, npt.ArrayLike],
) -> None:
    """Atomically save one complete episode and its resumability marker."""

    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / COMPLETE_MARKER
    if marker.exists():
        raise FileExistsError(f"teacher episode is already complete: {output_dir}")
    payload = dict(metadata)
    payload["schema_version"] = TEACHER_DATA_SCHEMA_VERSION
    array_payload = {name: np.asarray(value) for name, value in arrays.items()}
    for name, value in array_payload.items():
        if value.dtype == np.dtype("O"):
            raise TypeError(f"teacher array {name!r} must not use object dtype")

    data_path = output_dir / "teacher_episode.npz"
    data_temp = output_dir / ".teacher_episode.npz.tmp"
    metadata_path = output_dir / "result.json"
    metadata_temp = output_dir / ".result.json.tmp"
    marker_temp = output_dir / f".{COMPLETE_MARKER}.tmp"
    with data_temp.open("wb") as file:
        np.savez_compressed(file, **array_payload)
        file.flush()
        os.fsync(file.fileno())
    with metadata_temp.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    marker_temp.write_text("complete\n", encoding="utf-8")
    os.replace(data_temp, data_path)
    os.replace(metadata_temp, metadata_path)
    os.replace(marker_temp, marker)


def is_teacher_episode_complete(output_dir: Path) -> bool:
    """Return whether all committed files for one episode are present."""

    return all(
        (output_dir / name).is_file()
        for name in (COMPLETE_MARKER, "teacher_episode.npz", "result.json")
    )


def load_teacher_episode_training_samples(
    episode_dir: Path,
    *,
    require_success: bool = True,
) -> dict[str, npt.NDArray[Any]]:
    """Load one complete teacher episode in the compact training NPZ contract."""

    if not is_teacher_episode_complete(episode_dir):
        raise ValueError(f"teacher episode is incomplete: {episode_dir}")
    metadata = json.loads((episode_dir / "result.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != TEACHER_DATA_SCHEMA_VERSION:
        raise ValueError("teacher episode schema is unsupported")
    if metadata.get("infrastructure_error") or not metadata.get(
        "completed_without_exception"
    ):
        raise ValueError("infrastructure-error episodes cannot enter training")
    if require_success and not metadata.get("success"):
        raise ValueError("unsuccessful teacher episode is excluded from Student-v1")
    task_id = int(metadata["task_id"])
    init_state_index = int(metadata["init_state_index"])
    model_type = str(metadata.get("model_type", ""))
    episode_namespace = (
        2_000_000
        if model_type == "official_pi05_libero_teacher_correction"
        else 1_000_000
    )
    episode_index = episode_namespace + task_id * 1_000 + init_state_index
    prompt = str(metadata["task_description"])
    with np.load(episode_dir / "teacher_episode.npz", allow_pickle=False) as episode:
        base = episode["base_images"]
        wrist = episode["wrist_images"]
        state = episode["student_states"]
        actions = episode["target_actions"]
        action_valid = episode["target_valid"]
        frame_index = episode["query_steps"]
    sample_count = len(state)
    if (
        base.dtype != np.uint8
        or wrist.dtype != np.uint8
        or base.ndim != 4
        or base.shape != wrist.shape
        or base.shape[-1] != 3
    ):
        raise ValueError("teacher images must be matching uint8 [N,H,W,3] arrays")
    if state.dtype != np.float32 or state.shape != (sample_count, 7):
        raise ValueError("teacher student_states must be float32 [N,7]")
    if actions.dtype != np.float32 or actions.shape != (sample_count, 10, 7):
        raise ValueError("teacher target_actions must be float32 [N,10,7]")
    if action_valid.dtype != np.bool_ or action_valid.shape != actions.shape:
        raise ValueError("teacher target_valid must be bool with the action shape")
    if frame_index.shape != (sample_count,):
        raise ValueError("teacher query_steps must have one entry per query")
    if not np.isfinite(state).all() or not np.isfinite(actions).all():
        raise ValueError("teacher training values must be finite")
    images = np.stack((base, wrist), axis=1).transpose(0, 1, 4, 2, 3)
    return {
        "images": np.ascontiguousarray(images),
        "image_keys": np.asarray(["base_0_rgb", "wrist_0_rgb"]),
        "state": state,
        "actions": actions,
        "action_valid": action_valid,
        "prompts": np.asarray([prompt] * sample_count),
        "episode_index": np.full(sample_count, episode_index, dtype=np.int64),
        "frame_index": frame_index.astype(np.int64),
        "task_index": np.full(sample_count, task_id, dtype=np.int64),
    }


def select_teacher_episode_dirs(
    input_roots: Iterable[Path],
    *,
    include_failures: bool = False,
    successes_per_task: int | None = None,
    expected_task_ids: Sequence[int] | None = None,
) -> list[Path]:
    """Select complete teacher episodes with optional exact task balancing."""

    if successes_per_task is not None and successes_per_task <= 0:
        raise ValueError("successes_per_task must be positive")
    if include_failures and successes_per_task is not None:
        raise ValueError("failure inclusion is incompatible with success balancing")
    episode_dirs = sorted(
        {
            marker.parent
            for root in input_roots
            for marker in root.rglob(COMPLETE_MARKER)
        }
    )
    if not episode_dirs:
        raise ValueError("no complete teacher episodes were found")

    selected: list[tuple[int, int, Path]] = []
    for episode_dir in episode_dirs:
        if not is_teacher_episode_complete(episode_dir):
            raise ValueError(f"teacher episode is incomplete: {episode_dir}")
        metadata = json.loads(
            (episode_dir / "result.json").read_text(encoding="utf-8")
        )
        if metadata.get("infrastructure_error") or not metadata.get(
            "completed_without_exception"
        ):
            raise ValueError("infrastructure-error episodes cannot enter training")
        if not include_failures and not metadata.get("success"):
            continue
        selected.append(
            (
                int(metadata["task_id"]),
                int(metadata["init_state_index"]),
                episode_dir,
            )
        )
    if not selected:
        raise ValueError("no eligible teacher episodes were found")
    selected.sort(key=lambda item: (item[0], item[1], str(item[2])))

    if successes_per_task is None:
        if expected_task_ids is not None:
            present = {task_id for task_id, _, _ in selected}
            missing = sorted(set(expected_task_ids) - present)
            if missing:
                raise ValueError(f"teacher data is missing task IDs: {missing}")
        return [episode_dir for _, _, episode_dir in selected]

    grouped: dict[int, list[tuple[int, Path]]] = defaultdict(list)
    for task_id, init_state_index, episode_dir in selected:
        grouped[task_id].append((init_state_index, episode_dir))
    required_tasks = (
        sorted(set(expected_task_ids))
        if expected_task_ids is not None
        else sorted(grouped)
    )
    balanced: list[Path] = []
    for task_id in required_tasks:
        candidates = grouped.get(task_id, [])
        if len(candidates) < successes_per_task:
            raise ValueError(
                f"task {task_id} has {len(candidates)} eligible successes; "
                f"{successes_per_task} required"
            )
        balanced.extend(path for _, path in candidates[:successes_per_task])
    return balanced


def _floating_vector(value: object, name: str, width: int) -> npt.NDArray[np.float32]:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.floating) or array.shape != (width,):
        raise ValueError(f"{name} must be a floating-point vector of length {width}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array.astype(np.float32)


def _student_state_from_observation(
    observation: Mapping[str, object],
) -> npt.NDArray[np.float32]:
    position = _floating_vector(observation["robot0_eef_pos"], "position", 3)
    quaternion = _floating_vector(observation["robot0_eef_quat"], "quaternion", 4)
    fingers = _floating_vector(observation["robot0_gripper_qpos"], "gripper", 2)
    opening_width = fingers[0] - fingers[1]
    return np.concatenate(
        (position, _quaternion_to_axis_angle(quaternion), [opening_width])
    ).astype(np.float32)


def _quaternion_to_axis_angle(
    quaternion: npt.ArrayLike,
) -> npt.NDArray[np.float32]:
    quat = _floating_vector(quaternion, "quaternion", 4).astype(np.float64)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        raise ValueError("quaternion norm must be positive")
    quat /= norm
    scalar = float(np.clip(quat[3], -1.0, 1.0))
    denominator = math.sqrt(max(0.0, 1.0 - scalar * scalar))
    if math.isclose(denominator, 0.0, abs_tol=1e-8):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * (2.0 * math.acos(scalar) / denominator)).astype(np.float32)


def _libero_actions_to_training_space(
    actions: npt.ArrayLike,
) -> npt.NDArray[np.float32]:
    array = np.asarray(actions, dtype=np.float32)
    result = array.copy()
    result[..., 6] = (1.0 - result[..., 6]) / 2.0
    return result


def _uint8_image(value: npt.ArrayLike, name: str) -> npt.NDArray[np.uint8]:
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{name} must be uint8 [H,W,3]")
    return np.ascontiguousarray(array)

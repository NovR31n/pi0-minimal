"""Create a tiny, real LIBERO batch from local LeRobot v2.1 files.

This extraction utility intentionally keeps PyArrow, PyAV, and OpenCV outside
the core runtime environment. Run it with an environment that already provides
those packages and expose this project's ``src`` directory through PYTHONPATH.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np
import pyarrow.parquet as pq

from pi0_minimal.data import build_action_chunk, compact_libero_state

_CAMERAS = ("observation.images.image", "observation.images.wrist_image")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--samples-per-episode", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--action-horizon", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples_per_episode < 2:
        raise ValueError("samples-per-episode must be at least 2")

    tasks = _read_jsonl(args.dataset_root / "meta" / "tasks.jsonl")
    all_images: list[np.ndarray] = []
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_action_valid: list[np.ndarray] = []
    prompts: list[str] = []
    episode_indices: list[int] = []
    frame_indices: list[int] = []

    for episode_index in args.episodes:
        parquet_path = (
            args.dataset_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        )
        table = pq.read_table(parquet_path)
        states = compact_libero_state(np.asarray(table["observation.state"].to_pylist()))
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        task_index = int(table["task_index"][0].as_py())
        selected = np.unique(
            np.linspace(0, len(actions) - 1, args.samples_per_episode, dtype=np.int64)
        )
        camera_frames = [
            _decode_selected(
                args.dataset_root
                / "videos"
                / "chunk-000"
                / camera
                / f"episode_{episode_index:06d}.mp4",
                selected,
                args.image_size,
            )
            for camera in _CAMERAS
        ]

        for local_index, frame_index in enumerate(selected):
            chunk, valid = build_action_chunk(actions, int(frame_index), args.action_horizon)
            all_images.append(
                np.stack([frames[local_index] for frames in camera_frames], axis=0)
            )
            all_states.append(states[frame_index])
            all_actions.append(chunk)
            all_action_valid.append(valid)
            prompts.append(str(tasks[task_index]["task"]))
            episode_indices.append(episode_index)
            frame_indices.append(int(frame_index))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        images=np.stack(all_images),
        image_keys=np.asarray(["base_0_rgb", "wrist_0_rgb"]),
        state=np.stack(all_states).astype(np.float32),
        actions=np.stack(all_actions).astype(np.float32),
        action_valid=np.stack(all_action_valid),
        prompts=np.asarray(prompts),
        episode_index=np.asarray(episode_indices, dtype=np.int64),
        frame_index=np.asarray(frame_indices, dtype=np.int64),
    )
    print(f"Wrote {len(all_images)} real LIBERO samples to {args.output}")


def _read_jsonl(path: Path) -> dict[int, dict[str, object]]:
    records: dict[int, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        records[int(record["task_index"])] = record
    return records


def _decode_selected(path: Path, selected: np.ndarray, size: int) -> np.ndarray:
    requested = set(selected.tolist())
    decoded: dict[int, np.ndarray] = {}
    with av.open(str(path)) as container:
        for frame_index, frame in enumerate(container.decode(video=0)):
            if frame_index not in requested:
                continue
            rgb = frame.to_ndarray(format="rgb24")
            resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
            decoded[frame_index] = np.moveaxis(resized, -1, 0)
            if len(decoded) == len(requested):
                break
    missing = requested - decoded.keys()
    if missing:
        raise ValueError(f"video {path} is missing requested frames: {sorted(missing)}")
    return np.stack([decoded[int(index)] for index in selected])


if __name__ == "__main__":
    main()

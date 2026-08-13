"""Convert complete teacher episodes into the compact training NPZ contract."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from pi0_minimal.teacher_data import (
    load_teacher_episode_training_samples,
    select_teacher_episode_dirs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-failures", action="store_true")
    parser.add_argument("--successes-per-task", type=int)
    parser.add_argument("--expected-task-ids", type=int, nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode_dirs = select_teacher_episode_dirs(
        args.input_root,
        include_failures=args.include_failures,
        successes_per_task=args.successes_per_task,
        expected_task_ids=args.expected_task_ids,
    )
    samples = [
        load_teacher_episode_training_samples(
            episode_dir,
            require_success=not args.include_failures,
        )
        for episode_dir in episode_dirs
    ]
    episode_ids = [int(sample["episode_index"][0]) for sample in samples]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("teacher task/init episode identities must be unique")
    image_keys = samples[0]["image_keys"]
    if any(not np.array_equal(sample["image_keys"], image_keys) for sample in samples):
        raise ValueError("teacher episodes disagree on image keys")
    payload = {
        name: np.concatenate([sample[name] for sample in samples], axis=0)
        for name in (
            "images",
            "state",
            "actions",
            "action_valid",
            "prompts",
            "episode_index",
            "frame_index",
            "task_index",
        )
    }
    payload["image_keys"] = image_keys
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, **payload)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, args.output)
    print(
        f"wrote {len(payload['state'])} queries from {len(samples)} teacher episodes "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()

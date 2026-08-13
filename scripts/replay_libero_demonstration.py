"""Replay one official LIBERO HDF5 demonstration as an integration gate."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import h5py
import imageio.v2 as imageio
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from pi0_minimal.data import validate_libero_actions
from pi0_minimal.libero_adapter import (
    libero_actions_to_training_space,
    training_actions_to_libero_space,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--demo-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.demo_index < 0 or args.max_steps <= 0 or args.resolution <= 0:
        raise ValueError("demo-index must be non-negative and step limits must be positive")

    suite_type = benchmark.get_benchmark_dict()[args.task_suite]
    suite = suite_type()
    if not 0 <= args.task_id < suite.n_tasks:
        raise ValueError(f"task-id must lie in [0,{suite.n_tasks - 1}]")
    task = suite.get_task(args.task_id)
    demonstration = _load_demonstration(args.dataset, args.demo_index)
    _validate_dataset_task(demonstration, task)

    training_actions = libero_actions_to_training_space(demonstration["actions"])
    replay_actions = training_actions_to_libero_space(training_actions)
    roundtrip_max_error = float(
        np.max(np.abs(replay_actions - demonstration["actions"]))
    )
    if roundtrip_max_error > 1e-6:
        raise ValueError(
            f"action conversion round trip changed expert actions by {roundtrip_max_error}"
        )

    env = _make_env(task, args.resolution, args.seed)
    frames: list[np.ndarray] = []
    executed_actions: list[np.ndarray] = []
    rewards: list[float] = []
    success = False
    exception_message = ""
    started = time.perf_counter()
    try:
        env.reset()
        observation = env.set_init_state(demonstration["states"][0])
        limit = min(args.max_steps, len(replay_actions))
        for action in replay_actions[:limit]:
            frames.append(_video_frame(observation))
            observation, reward, done, _info = env.step(action)
            executed_actions.append(action.copy())
            rewards.append(float(reward))
            if done or reward > 0.0:
                success = True
                break
    except Exception as error:  # noqa: BLE001 - preserve simulator failures.
        exception_message = f"{type(error).__name__}: {error}"
    finally:
        env.close()
    elapsed_seconds = time.perf_counter() - started

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.output_dir / (
        f"task{args.task_id:02d}_demo{args.demo_index:02d}_"
        f"{'success' if success else 'failure'}.mp4"
    )
    if frames:
        imageio.mimwrite(video_path, frames, fps=20)
    trace_path = args.output_dir / "trace.npz"
    np.savez_compressed(
        trace_path,
        actions=np.asarray(executed_actions, dtype=np.float32).reshape(-1, 7),
        rewards=np.asarray(rewards, dtype=np.float32),
    )
    result: dict[str, Any] = {
        "task_suite": args.task_suite,
        "task_id": args.task_id,
        "task_description": str(task.language),
        "dataset": str(args.dataset),
        "demo_index": args.demo_index,
        "source_steps": len(demonstration["actions"]),
        "executed_steps": len(executed_actions),
        "source_final_reward": float(demonstration["rewards"][-1]),
        "source_final_done": bool(demonstration["dones"][-1]),
        "roundtrip_max_error": roundtrip_max_error,
        "success": success,
        "completed_without_exception": not exception_message,
        "exception": exception_message,
        "elapsed_seconds": elapsed_seconds,
        "video": str(video_path) if frames else None,
        "trace": str(trace_path),
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    if exception_message:
        raise RuntimeError(exception_message)
    if not success:
        raise RuntimeError("expert demonstration replay did not reach task success")


def _load_demonstration(path: Path, demo_index: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    key = f"demo_{demo_index}"
    with h5py.File(path, "r") as dataset:
        if "data" not in dataset or key not in dataset["data"]:
            raise KeyError(f"dataset does not contain data/{key}")
        data = dataset["data"]
        demo = data[key]
        actions = validate_libero_actions(np.asarray(demo["actions"]))
        states = np.asarray(demo["states"])
        rewards = np.asarray(demo["rewards"])
        dones = np.asarray(demo["dones"])
        metadata = {
            "bddl_file_name": str(data.attrs.get("bddl_file_name", "")),
            "problem_info": str(data.attrs.get("problem_info", "")),
        }
    if (
        not np.issubdtype(states.dtype, np.floating)
        or states.ndim != 2
        or len(states) != len(actions)
        or not np.isfinite(states).all()
    ):
        raise ValueError("demonstration states must be finite floating point [T,S]")
    if rewards.shape != (len(actions),) or dones.shape != (len(actions),):
        raise ValueError("demonstration rewards and dones must match its action length")
    if not np.isfinite(rewards).all():
        raise ValueError("demonstration rewards must be finite")
    return {
        "actions": actions,
        "states": states.astype(np.float64, copy=False),
        "rewards": rewards.astype(np.float32, copy=False),
        "dones": dones.astype(np.bool_, copy=False),
        **metadata,
    }


def _validate_dataset_task(demonstration: dict[str, Any], task: Any) -> None:
    bddl_name = Path(str(demonstration["bddl_file_name"])).name
    if bddl_name and bddl_name != task.bddl_file:
        raise ValueError(
            f"dataset BDDL {bddl_name!r} does not match task {task.bddl_file!r}"
        )
    problem_info = str(demonstration["problem_info"])
    if problem_info:
        language = str(json.loads(problem_info).get("language_instruction", ""))
        if language and language != str(task.language):
            raise ValueError("dataset language instruction does not match benchmark task")


def _make_env(task: Any, resolution: int, seed: int) -> OffScreenRenderEnv:
    bddl_file = (
        Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env


def _video_frame(observation: dict[str, Any]) -> np.ndarray:
    frame = np.asarray(observation["agentview_image"])[::-1, ::-1]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


if __name__ == "__main__":
    main()

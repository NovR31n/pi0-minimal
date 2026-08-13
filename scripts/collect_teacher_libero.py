"""Collect resumable OpenPI teacher trajectories for offline distillation."""

from __future__ import annotations

import argparse
import collections
import json
import os
import socket
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy

from pi0_minimal.teacher_data import (
    build_teacher_query,
    is_teacher_episode_complete,
    save_teacher_episode,
    teacher_state_from_observation,
)

_DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-ids", type=int, nargs="+", required=True)
    parser.add_argument("--init-state-indices", type=int, nargs="+", required=True)
    parser.add_argument("--student-horizon", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--teacher-image-size", type=int, default=224)
    parser.add_argument("--eval-seed-offset", type=int, default=1000)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)
    _require_server(args.host, args.port, args.connect_timeout)
    client = WebsocketClientPolicy(args.host, args.port)
    server_metadata = _json_safe(client.get_server_metadata())
    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    completed = successes = 0
    for task_id in args.task_ids:
        if not 0 <= task_id < suite.n_tasks:
            raise ValueError(f"task ID {task_id} is outside {args.task_suite}")
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        for init_state_index in args.init_state_indices:
            if not 0 <= init_state_index < len(initial_states):
                raise ValueError(
                    f"init state {init_state_index} is unavailable for task {task_id}"
                )
            episode_dir = (
                args.output_dir / f"task{task_id:02d}" / f"init{init_state_index:02d}"
            )
            if args.skip_existing and is_teacher_episode_complete(episode_dir):
                print(f"SKIP task={task_id} init={init_state_index}", flush=True)
                continue
            try:
                result = _collect_episode(
                    args,
                    client,
                    server_metadata,
                    task,
                    initial_states,
                    task_id=task_id,
                    init_state_index=init_state_index,
                    output_dir=episode_dir,
                )
            except Exception as error:
                _write_infrastructure_error(
                    episode_dir,
                    task_id=task_id,
                    init_state_index=init_state_index,
                    error=error,
                )
                raise RuntimeError(
                    "teacher collection aborted on infrastructure error; "
                    "the episode was not counted as a model failure"
                ) from error
            completed += 1
            successes += int(result["success"])
            print(
                f"DONE task={task_id} init={init_state_index} "
                f"success={result['success']} queries={result['teacher_queries']} "
                f"steps={result['episode_steps']}",
                flush=True,
            )
    print(
        f"teacher collection complete: episodes={completed} successes={successes}",
        flush=True,
    )


def _collect_episode(
    args: argparse.Namespace,
    client: WebsocketClientPolicy,
    server_metadata: Any,
    task: Any,
    initial_states: np.ndarray,
    *,
    task_id: int,
    init_state_index: int,
    output_dir: Path,
) -> dict[str, Any]:
    eval_seed = args.eval_seed_offset + init_state_index
    np.random.seed(eval_seed)
    env = _make_env(task, args.resolution, eval_seed)
    action_plan: collections.deque[np.ndarray] = collections.deque()
    queries: list[dict[str, np.ndarray]] = []
    query_steps: list[int] = []
    executed_actions: list[np.ndarray] = []
    end_effector_positions: list[np.ndarray] = []
    inference_times: list[float] = []
    frames: list[np.ndarray] = []
    success = False
    episode_steps = 0
    prompt = str(task.language)
    started = time.perf_counter()
    try:
        env.reset()
        observation = env.set_init_state(initial_states[init_state_index])
        for _ in range(args.wait_steps):
            observation, _reward, success, _info = env.step(_DUMMY_ACTION)
            if success:
                break
        while episode_steps < args.max_steps and not success:
            base_image, wrist_image = _teacher_images(
                observation,
                args.teacher_image_size,
            )
            frames.append(base_image)
            if not action_plan:
                element = {
                    "observation/image": base_image,
                    "observation/wrist_image": wrist_image,
                    "observation/state": teacher_state_from_observation(observation),
                    "prompt": prompt,
                }
                inference_started = time.perf_counter()
                response = client.infer(element)
                inference_times.append(time.perf_counter() - inference_started)
                if "actions" not in response:
                    raise KeyError("teacher response does not contain 'actions'")
                query = build_teacher_query(
                    observation,
                    base_image,
                    wrist_image,
                    response["actions"],
                    student_horizon=args.student_horizon,
                    replan_steps=args.replan_steps,
                )
                queries.append(query)
                query_steps.append(episode_steps)
                action_plan.extend(
                    query["raw_teacher_actions"][: args.replan_steps].copy()
                )
            action = action_plan.popleft()
            observation, _reward, success, _info = env.step(action.tolist())
            executed_actions.append(action.copy())
            end_effector_positions.append(
                np.asarray(observation["robot0_eef_pos"], dtype=np.float32).copy()
            )
            episode_steps += 1
    finally:
        env.close()

    if not queries:
        raise RuntimeError("teacher episode produced no policy queries")
    raw_lengths = {len(query["raw_teacher_actions"]) for query in queries}
    if len(raw_lengths) != 1:
        raise RuntimeError("teacher returned inconsistent action horizons")
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / (
        f"task{task_id:02d}_init{init_state_index:02d}_"
        f"{'success' if success else 'failure'}.mp4"
    )
    imageio.mimwrite(video_path, frames, fps=10)
    arrays = {
        "base_images": np.stack([query["base_image"] for query in queries]),
        "wrist_images": np.stack([query["wrist_image"] for query in queries]),
        "student_states": np.stack([query["student_state"] for query in queries]),
        "teacher_states": np.stack([query["teacher_state"] for query in queries]),
        "raw_teacher_actions": np.stack(
            [query["raw_teacher_actions"] for query in queries]
        ),
        "target_actions": np.stack([query["target_actions"] for query in queries]),
        "target_valid": np.stack([query["target_valid"] for query in queries]),
        "query_steps": np.asarray(query_steps, dtype=np.int32),
        "executed_actions": np.asarray(executed_actions, dtype=np.float32).reshape(-1, 7),
        "end_effector_positions": np.asarray(
            end_effector_positions, dtype=np.float32
        ).reshape(-1, 3),
        "inference_times": np.asarray(inference_times, dtype=np.float64),
    }
    elapsed_seconds = time.perf_counter() - started
    result: dict[str, Any] = {
        "model_type": "official_pi05_libero_teacher",
        "task_suite": args.task_suite,
        "task_id": task_id,
        "task_description": prompt,
        "init_state_index": init_state_index,
        "seed": eval_seed,
        "success": bool(success),
        "completed_without_exception": True,
        "infrastructure_error": False,
        "episode_steps": episode_steps,
        "max_steps": args.max_steps,
        "wait_steps": args.wait_steps,
        "replan_steps": args.replan_steps,
        "student_horizon": args.student_horizon,
        "teacher_action_horizon": raw_lengths.pop(),
        "teacher_queries": len(queries),
        "first_inference_seconds": inference_times[0],
        "mean_inference_seconds": float(np.mean(inference_times)),
        "elapsed_seconds": elapsed_seconds,
        "video": str(video_path),
        "teacher_episode": str(output_dir / "teacher_episode.npz"),
        "server_metadata": server_metadata,
    }
    save_teacher_episode(output_dir, metadata=result, arrays=arrays)
    return result


def _teacher_images(
    observation: dict[str, Any],
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    base = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(
        observation["robot0_eye_in_hand_image"][::-1, ::-1]
    )
    return (
        image_tools.convert_to_uint8(image_tools.resize_with_pad(base, size, size)),
        image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, size, size)),
    )


def _make_env(task: Any, resolution: int, seed: int) -> OffScreenRenderEnv:
    task_bddl_file = (
        Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env


def _require_server(host: str, port: int, timeout: float) -> None:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
    except OSError as error:
        raise ConnectionError(
            f"official policy server is unavailable at {host}:{port}"
        ) from error


def _write_infrastructure_error(
    output_dir: Path,
    *,
    task_id: int,
    init_state_index: int,
    error: Exception,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_id": task_id,
        "init_state_index": init_state_index,
        "success": None,
        "completed_without_exception": False,
        "infrastructure_error": True,
        "exception": f"{type(error).__name__}: {error}",
    }
    temp = output_dir / ".infrastructure_error.json.tmp"
    final = output_dir / "infrastructure_error.json"
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, final)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return repr(value)


def _validate_args(args: argparse.Namespace) -> None:
    if (
        args.port <= 0
        or args.student_horizon <= 0
        or args.replan_steps <= 0
        or args.max_steps <= 0
        or args.wait_steps < 0
        or args.resolution <= 0
        or args.teacher_image_size <= 0
        or args.connect_timeout <= 0
    ):
        raise ValueError("teacher collection arguments must be positive")
    if args.replan_steps > args.student_horizon:
        raise ValueError("replanning interval must not exceed the student horizon")


if __name__ == "__main__":
    main()

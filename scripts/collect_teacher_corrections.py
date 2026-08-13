"""Collect official-teacher recovery trajectories from student failure states."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from collect_teacher_libero import (
    _DUMMY_ACTION,
    _json_safe,
    _make_env,
    _require_server,
    _teacher_images,
)
from libero.libero import benchmark
from replay_libero_demonstration import (
    _load_demonstration,
    _validate_dataset_task,
)

from pi0_minimal.libero_adapter import (
    libero_simulator_state,
    restore_libero_controller_state,
)
from pi0_minimal.teacher_client import LongInferenceWebsocketClientPolicy
from pi0_minimal.teacher_data import (
    build_teacher_query,
    is_teacher_episode_complete,
    save_teacher_episode,
    teacher_state_from_observation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--student-horizon", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--teacher-image-size", type=int, default=224)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not manifest.get("seeds"):
        raise ValueError("correction manifest is empty or unsupported")
    _require_server(args.host, args.port, args.connect_timeout)
    client = LongInferenceWebsocketClientPolicy(args.host, args.port)
    server_metadata = _json_safe(client.get_server_metadata())
    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    completed = successes = 0
    for seed in manifest["seeds"]:
        task_id = int(seed["task_id"])
        demo_index = int(seed["demo_index"])
        recovery_step = int(seed["recovery_step"])
        if not 0 <= task_id < suite.n_tasks:
            raise ValueError(f"task ID {task_id} is outside {args.task_suite}")
        output_dir = (
            args.output_dir
            / f"task{task_id:02d}"
            / f"demo{demo_index:02d}_step{recovery_step:03d}"
        )
        if args.skip_existing and is_teacher_episode_complete(output_dir):
            print(
                f"SKIP task={task_id} demo={demo_index} step={recovery_step}",
                flush=True,
            )
            continue
        task = suite.get_task(task_id)
        try:
            result = _collect_correction(
                args,
                client,
                server_metadata,
                task,
                seed,
                output_dir,
            )
        except Exception as error:
            _write_infrastructure_error(output_dir, seed=seed, error=error)
            raise RuntimeError(
                "teacher correction collection aborted on infrastructure error; "
                "the seed was not counted as a model failure"
            ) from error
        completed += 1
        successes += int(result["success"])
        print(
            f"DONE task={task_id} demo={demo_index} step={recovery_step} "
            f"success={result['success']} recovery_steps={result['episode_steps']}",
            flush=True,
        )
    print(
        f"teacher correction complete: episodes={completed} successes={successes}",
        flush=True,
    )


def _collect_correction(
    args: argparse.Namespace,
    client: LongInferenceWebsocketClientPolicy,
    server_metadata: Any,
    task: Any,
    seed: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    task_id = int(seed["task_id"])
    demo_index = int(seed["demo_index"])
    recovery_step = int(seed["recovery_step"])
    dataset_path = Path(seed["source_initial_state_dataset"])
    demonstration = _load_demonstration(dataset_path, demo_index)
    _validate_dataset_task(demonstration, task)
    trace_path = Path(seed["source_trace"])
    if _sha256(trace_path) != seed["source_trace_sha256"]:
        raise ValueError(f"correction source trace hash changed: {trace_path}")
    with np.load(trace_path) as trace:
        simulator_states = trace["simulator_states_before"]
        controller_states = trace["controller_states_before"]
        if not 0 <= recovery_step < len(simulator_states):
            raise ValueError("recovery step is outside the source trace")
        simulator_state = simulator_states[recovery_step].copy()
        controller_state = controller_states[recovery_step].copy()

    environment = _make_env(task, args.resolution, int(seed["source_seed"]))
    prompt = str(task.language)
    action_plan: collections.deque[np.ndarray] = collections.deque()
    queries: list[dict[str, np.ndarray]] = []
    query_steps: list[int] = []
    executed_actions: list[np.ndarray] = []
    end_effector_positions: list[np.ndarray] = []
    inference_times: list[float] = []
    frames: list[np.ndarray] = []
    success = False
    episode_steps = 0
    started = time.perf_counter()
    try:
        environment.reset()
        observation = environment.set_init_state(demonstration["states"][0])
        for _ in range(int(seed["source_wait_steps"])):
            observation, _reward, _success, _info = environment.step(_DUMMY_ACTION)
        observation = environment.regenerate_obs_from_state(simulator_state)
        restore_libero_controller_state(environment, controller_state)
        restore_error = float(
            np.max(np.abs(libero_simulator_state(environment) - simulator_state))
        )
        if restore_error > 1e-12:
            raise RuntimeError(
                f"correction seed state restoration error is {restore_error}"
            )
        if environment.check_success():
            raise ValueError("correction seed is already a successful task state")

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
                query_steps.append(recovery_step + episode_steps)
                action_plan.extend(
                    query["raw_teacher_actions"][: args.replan_steps].copy()
                )
            action = action_plan.popleft()
            observation, _reward, success, _info = environment.step(action.tolist())
            executed_actions.append(action.copy())
            end_effector_positions.append(
                np.asarray(
                    observation["robot0_eef_pos"],
                    dtype=np.float32,
                ).copy()
            )
            episode_steps += 1
    finally:
        environment.close()

    if not queries:
        raise RuntimeError("teacher correction produced no policy queries")
    raw_lengths = {len(query["raw_teacher_actions"]) for query in queries}
    if len(raw_lengths) != 1:
        raise RuntimeError("teacher returned inconsistent action horizons")
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / (
        f"task{task_id:02d}_demo{demo_index:02d}_step{recovery_step:03d}_"
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
            end_effector_positions,
            dtype=np.float32,
        ).reshape(-1, 3),
        "inference_times": np.asarray(inference_times, dtype=np.float64),
        "source_simulator_state": simulator_state,
        "source_controller_state": controller_state,
    }
    elapsed_seconds = time.perf_counter() - started
    result: dict[str, Any] = {
        "model_type": "official_pi05_libero_teacher_correction",
        "task_suite": args.task_suite,
        "task_id": task_id,
        "task_description": prompt,
        "init_state_index": demo_index,
        "source_initial_state": "official_demonstration",
        "source_recovery_step": recovery_step,
        "source_selection_reason": seed["selection_reason"],
        "source_result": seed["source_result"],
        "source_trace": str(trace_path),
        "source_trace_sha256": seed["source_trace_sha256"],
        "source_gripper_switch_count": seed["gripper_switch_count"],
        "restored_simulator_state_max_abs_error": restore_error,
        "success": bool(success),
        "completed_without_exception": True,
        "infrastructure_error": False,
        "episode_steps": episode_steps,
        "max_steps": args.max_steps,
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


def _write_infrastructure_error(
    output_dir: Path,
    *,
    seed: dict[str, Any],
    error: Exception,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_id": seed["task_id"],
        "init_state_index": seed["demo_index"],
        "source_recovery_step": seed["recovery_step"],
        "success": None,
        "completed_without_exception": False,
        "infrastructure_error": True,
        "exception": f"{type(error).__name__}: {error}",
    }
    temporary = output_dir / ".infrastructure_error.json.tmp"
    final = output_dir / "infrastructure_error.json"
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, final)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_args(args: argparse.Namespace) -> None:
    if (
        args.port <= 0
        or args.student_horizon <= 0
        or args.replan_steps <= 0
        or args.max_steps <= 0
        or args.resolution <= 0
        or args.teacher_image_size <= 0
        or args.connect_timeout <= 0
    ):
        raise ValueError("teacher correction arguments must be positive")
    if args.replan_steps > args.student_horizon:
        raise ValueError("replanning interval must not exceed the student horizon")


if __name__ == "__main__":
    main()

"""Evaluate one loaded checkpoint over a task/initial-state rollout matrix."""

from __future__ import annotations

import argparse
import collections
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import h5py
import imageio.v2 as imageio
import numpy as np
import torch
from libero.libero import benchmark
from rollout_libero_single import (
    _DUMMY_ACTION,
    _autocast_context,
    _dtype_from_name,
    _images_from_observation,
    _load_trusted_initial_states,
    _make_env,
)

from pi0_minimal.data import NormalizationStats
from pi0_minimal.libero_adapter import (
    DebouncedBipolarGripper,
    libero_actions_to_training_space,
    libero_controller_state,
    libero_numeric_observation,
    libero_simulator_state,
    libero_state_from_observation,
    safe_libero_action_chunk,
    stack_libero_observation_trace,
)
from pi0_minimal.metrics import (
    action_smoothness_metrics,
    runtime_metrics,
    trajectory_metrics,
)
from pi0_minimal.model_spec import load_and_validate_model_spec
from pi0_minimal.models import (
    FrozenPaliGemmaBackbone,
    build_autoregressive_policy,
    build_flow_policy,
)
from pi0_minimal.training import load_training_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument(
        "--demonstration-root",
        type=Path,
        help=(
            "optional directory of task-matched *_demo.hdf5 files; when set, "
            "init-state indices refer to demonstration indices"
        ),
    )
    parser.add_argument("--task-ids", type=int, nargs="+", required=True)
    parser.add_argument("--init-state-indices", type=int, nargs="+", required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--eval-seed-offset", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=2)
    parser.add_argument("--gripper-debounce-steps", type=int, default=0)
    parser.add_argument("--gripper-threshold", type=float, default=0.5)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if (
        args.max_steps <= 0
        or args.replan_steps <= 0
        or args.resolution <= 0
        or args.wait_steps < 0
        or args.gripper_debounce_steps < 0
        or not 0.0 < args.gripper_threshold <= 1.0
    ):
        raise ValueError("rollout step counts and resolution are invalid")

    spec = load_and_validate_model_spec(args.config)
    normalization = NormalizationStats.load(args.normalization)
    policy = (
        build_flow_policy(spec, device=args.device)
        if "flow" in spec
        else build_autoregressive_policy(spec, device=args.device)
    )
    restored = load_training_checkpoint(
        args.checkpoint,
        policy=policy,
        map_location=args.device,
    )
    policy.eval()
    observation_spec = spec["observation"]
    backbone_spec = spec["backbone"]
    backbone = FrozenPaliGemmaBackbone.from_pretrained(
        str(backbone_spec["model_id"]),
        revision=str(backbone_spec["revision"]),
        device=args.device,
        compute_dtype=_dtype_from_name(spec["compute_dtype"]),
        expected_image_views=len(observation_spec["image_keys"]),
        expected_output_dim=int(backbone_spec["output_dim"]),
    )
    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    completed = failed = 0
    for task_id in args.task_ids:
        if not 0 <= task_id < suite.n_tasks:
            raise ValueError(f"task ID {task_id} is outside the suite")
        task = suite.get_task(task_id)
        if args.demonstration_root is None:
            initial_states = _load_trusted_initial_states(task)
            initial_state_source = "benchmark_evaluation"
            initial_state_dataset = None
        else:
            initial_states, initial_state_dataset = (
                _load_demonstration_initial_states(args.demonstration_root, task)
            )
            initial_state_source = "official_demonstration"
        prompt = str(task.language)
        prompt_ids, prompt_valid = backbone.tokenize_prompts(
            [prompt],
            max_length=int(observation_spec["max_prompt_tokens"]),
        )
        for init_state_index in args.init_state_indices:
            if not 0 <= init_state_index < len(initial_states):
                raise ValueError(
                    f"init state {init_state_index} is unavailable for task {task_id}"
                )
            output_dir = (
                args.output_dir
                / f"task{task_id:02d}"
                / f"init{init_state_index:02d}"
            )
            if args.skip_existing and (output_dir / "result.json").exists():
                print(
                    f"SKIP task={task_id} init={init_state_index}",
                    flush=True,
                )
                continue
            result = _run_episode(
                args,
                spec,
                normalization,
                policy,
                restored,
                backbone,
                task,
                initial_states,
                prompt,
                prompt_ids,
                prompt_valid,
                task_id=task_id,
                init_state_index=init_state_index,
                initial_state_source=initial_state_source,
                initial_state_dataset=initial_state_dataset,
                output_dir=output_dir,
            )
            completed += 1
            failed += int(not result["completed_without_exception"])
            print(
                f"DONE task={task_id} init={init_state_index} "
                f"success={result['success']} steps={result['episode_steps']}",
                flush=True,
            )
    print(f"matrix complete: episodes={completed}, exceptions={failed}")
    if failed:
        raise RuntimeError(f"{failed} rollout episodes raised exceptions")


def _run_episode(
    args: argparse.Namespace,
    spec: dict[str, Any],
    normalization: NormalizationStats,
    policy: Any,
    restored: dict[str, Any],
    backbone: FrozenPaliGemmaBackbone,
    task: Any,
    initial_states: np.ndarray,
    prompt: str,
    prompt_ids: torch.Tensor,
    prompt_valid: torch.Tensor,
    *,
    task_id: int,
    init_state_index: int,
    initial_state_source: str,
    initial_state_dataset: Path | None,
    output_dir: Path,
) -> dict[str, Any]:
    eval_seed = args.eval_seed_offset + init_state_index
    torch.manual_seed(eval_seed)
    np.random.seed(eval_seed)
    env = _make_env(task, args.resolution, eval_seed)
    action_generator = torch.Generator(device=args.device).manual_seed(eval_seed)
    action_plan: collections.deque[np.ndarray] = collections.deque()
    frames: list[np.ndarray] = []
    executed_actions: list[np.ndarray] = []
    end_effector_positions: list[np.ndarray] = []
    simulator_states_before: list[np.ndarray] = []
    simulator_states_after: list[np.ndarray] = []
    controller_states_before: list[np.ndarray] = []
    controller_states_after: list[np.ndarray] = []
    observations_before: list[dict[str, np.ndarray]] = []
    observations_after: list[dict[str, np.ndarray]] = []
    successes_after_action: list[bool] = []
    inference_times: list[float] = []
    normalized_clip_count = environment_clip_count = 0
    gripper_filter = (
        DebouncedBipolarGripper(
            threshold=args.gripper_threshold,
            debounce_steps=args.gripper_debounce_steps,
        )
        if args.gripper_debounce_steps
        else None
    )
    success = False
    exception_message = ""
    episode_steps = 0
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
        torch.cuda.synchronize(args.device)
    started = time.perf_counter()
    try:
        env.reset()
        observation = env.set_init_state(initial_states[init_state_index])
        for _ in range(args.wait_steps):
            observation, _reward, success, _info = env.step(_DUMMY_ACTION)
            if success:
                break
        while episode_steps < args.max_steps and not success:
            base_image, wrist_image = _images_from_observation(
                observation,
                int(spec["observation"]["image_height"]),
            )
            frames.append(np.moveaxis(base_image, 0, -1))
            if not action_plan:
                inference_started = time.perf_counter()
                condition = backbone.encode_numpy(
                    np.stack((base_image, wrist_image))[None],
                    np.ones((1, 2), dtype=np.bool_),
                    prompt_ids.numpy(),
                    prompt_valid.numpy(),
                )
                raw_state = libero_state_from_observation(observation)
                state = torch.from_numpy(
                    normalization.state.normalize(raw_state[None])
                ).to(args.device)
                action_valid = torch.ones(
                    (
                        1,
                        int(spec["action"]["horizon"]),
                        int(spec["action"]["dim"]),
                    ),
                    dtype=torch.bool,
                    device=args.device,
                )
                with _autocast_context(args.device):
                    normalized_actions = policy.sample(
                        condition,
                        state,
                        action_valid,
                        generator=action_generator,
                    )
                safe_chunk = safe_libero_action_chunk(
                    normalized_actions[0].cpu().numpy(),
                    normalization.action,
                )
                normalized_clip_count += safe_chunk.normalized_clip_count
                environment_clip_count += safe_chunk.environment_clip_count
                action_plan.extend(safe_chunk.values[: args.replan_steps])
                inference_times.append(time.perf_counter() - inference_started)
            action = action_plan.popleft()
            if gripper_filter is not None:
                action = gripper_filter.apply(action)
            simulator_state_before = libero_simulator_state(env)
            controller_state_before = libero_controller_state(env)
            observation_before = libero_numeric_observation(observation)
            observation, _reward, success, _info = env.step(action)
            simulator_state_after = libero_simulator_state(env)
            controller_state_after = libero_controller_state(env)
            observation_after = libero_numeric_observation(observation)
            executed_actions.append(action.copy())
            end_effector_positions.append(
                np.asarray(observation["robot0_eef_pos"], dtype=np.float32).copy()
            )
            simulator_states_before.append(simulator_state_before)
            simulator_states_after.append(simulator_state_after)
            controller_states_before.append(controller_state_before)
            controller_states_after.append(controller_state_after)
            observations_before.append(observation_before)
            observations_after.append(observation_after)
            successes_after_action.append(bool(success))
            episode_steps += 1
    except Exception as error:  # noqa: BLE001
        exception_message = f"{type(error).__name__}: {error}"
    finally:
        env.close()
    elapsed_seconds = time.perf_counter() - started

    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / (
        f"task{task_id:02d}_init{init_state_index:02d}_"
        f"{'success' if success else 'failure'}.mp4"
    )
    if frames:
        imageio.mimwrite(video_path, frames, fps=10)
    action_trace = np.asarray(executed_actions, dtype=np.float32).reshape(-1, 7)
    normalized_action_trace = normalization.action.normalize(
        libero_actions_to_training_space(action_trace)
    )
    position_trace = np.asarray(
        end_effector_positions,
        dtype=np.float32,
    ).reshape(-1, 3)
    transition_count = len(action_trace)
    transition_fields = (
        simulator_states_before,
        simulator_states_after,
        controller_states_before,
        controller_states_after,
        observations_before,
        observations_after,
        successes_after_action,
    )
    if any(len(field) != transition_count for field in transition_fields):
        raise RuntimeError("rich LIBERO trace fields are not action-aligned")
    simulator_before_trace = (
        np.stack(simulator_states_before)
        if simulator_states_before
        else np.empty((0, 0), dtype=np.float64)
    )
    simulator_after_trace = (
        np.stack(simulator_states_after)
        if simulator_states_after
        else np.empty((0, 0), dtype=np.float64)
    )
    controller_before_trace = (
        np.stack(controller_states_before)
        if controller_states_before
        else np.empty((0, 12), dtype=np.float64)
    )
    controller_after_trace = (
        np.stack(controller_states_after)
        if controller_states_after
        else np.empty((0, 12), dtype=np.float64)
    )
    trace_payload: dict[str, np.ndarray] = {
        "trace_schema_version": np.asarray(3, dtype=np.int64),
        "actions": action_trace,
        "normalized_actions": normalized_action_trace,
        "end_effector_positions": position_trace,
        "inference_times": np.asarray(inference_times, dtype=np.float64),
        "simulator_states_before": simulator_before_trace,
        "simulator_states_after": simulator_after_trace,
        "controller_states_before": controller_before_trace,
        "controller_states_after": controller_after_trace,
        "successes_after_action": np.asarray(
            successes_after_action,
            dtype=np.bool_,
        ),
    }
    trace_payload.update(
        stack_libero_observation_trace(observations_before, prefix="before__")
    )
    trace_payload.update(
        stack_libero_observation_trace(observations_after, prefix="after__")
    )
    trace_path = output_dir / "trace.npz"
    np.savez_compressed(trace_path, **trace_payload)
    result: dict[str, Any] = {
        "model_type": "flow" if "flow" in spec else "autoregressive",
        "config": str(args.config),
        "task_suite": args.task_suite,
        "task_id": task_id,
        "task_description": prompt,
        "init_state_index": init_state_index,
        "initial_state_source": initial_state_source,
        "initial_state_dataset": (
            str(initial_state_dataset) if initial_state_dataset else None
        ),
        "training_seed": args.training_seed,
        "seed": eval_seed,
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": restored["step"],
        "success": bool(success),
        "completed_without_exception": not exception_message,
        "exception": exception_message,
        "episode_steps": episode_steps,
        "trace_schema_version": 3,
        "transition_count": transition_count,
        "simulator_state_dim": (
            int(simulator_before_trace.shape[1]) if transition_count else None
        ),
        "numeric_observation_keys": (
            list(observations_before[0]) if observations_before else []
        ),
        "max_steps": args.max_steps,
        "wait_steps": args.wait_steps,
        "replan_steps": args.replan_steps,
        "gripper_debounce_steps": args.gripper_debounce_steps,
        "gripper_threshold": args.gripper_threshold,
        "inference_calls": len(inference_times),
        "first_inference_seconds": inference_times[0] if inference_times else None,
        "mean_inference_seconds": (
            float(np.mean(inference_times)) if inference_times else None
        ),
        "normalized_clip_count": normalized_clip_count,
        "environment_clip_count": environment_clip_count,
        "elapsed_seconds": elapsed_seconds,
        "video": str(video_path) if frames else None,
        "trace": str(trace_path),
    }
    if len(action_trace):
        result["action_smoothness"] = action_smoothness_metrics(
            normalized_action_trace
        ).to_dict()
        result["trajectory_metrics"] = trajectory_metrics(
            position_trace,
            normalized_action_trace,
        ).to_dict()
    if inference_times:
        result["runtime_metrics"] = runtime_metrics(inference_times).to_dict()
    if args.device.startswith("cuda"):
        result["cuda_peak_allocated_mib"] = (
            torch.cuda.max_memory_allocated(args.device) / 2**20
        )
        result["cuda_peak_reserved_mib"] = (
            torch.cuda.max_memory_reserved(args.device) / 2**20
        )
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _load_demonstration_initial_states(
    root: Path,
    task: Any,
) -> tuple[np.ndarray, Path]:
    dataset_path = root / f"{Path(task.bddl_file).stem}_demo.hdf5"
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    with h5py.File(dataset_path, "r") as dataset:
        if "data" not in dataset:
            raise KeyError(f"{dataset_path} does not contain a data group")
        data = dataset["data"]
        bddl_name = Path(str(data.attrs.get("bddl_file_name", ""))).name
        if bddl_name and bddl_name != task.bddl_file:
            raise ValueError(
                f"dataset BDDL {bddl_name!r} does not match {task.bddl_file!r}"
            )
        keys = sorted(
            (key for key in data if key.startswith("demo_")),
            key=lambda key: int(key.removeprefix("demo_")),
        )
        expected_keys = [f"demo_{index}" for index in range(len(keys))]
        if keys != expected_keys:
            raise ValueError("demonstration keys must be zero-based and contiguous")
        states = []
        for key in keys:
            if "states" not in data[key] or len(data[key]["states"]) == 0:
                raise ValueError(f"{dataset_path} data/{key} has no states")
            states.append(np.asarray(data[key]["states"][0], dtype=np.float64))
    if not states:
        raise ValueError(f"{dataset_path} contains no demonstrations")
    initial_states = np.stack(states)
    if initial_states.ndim != 2 or not np.isfinite(initial_states).all():
        raise ValueError("demonstration initial states must be finite [N,S]")
    return initial_states, dataset_path


if __name__ == "__main__":
    main()

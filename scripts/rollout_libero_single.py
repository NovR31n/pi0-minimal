"""Run one closed-loop LIBERO episode with a trained compact policy."""

from __future__ import annotations

import argparse
import collections
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

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

_DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/model_flow_tiny.toml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-index", type=int, default=14)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=2)
    parser.add_argument("--gripper-debounce-steps", type=int, default=0)
    parser.add_argument("--gripper-threshold", type=float, default=0.5)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--training-seed", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
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
    suite_type = benchmark.get_benchmark_dict()[args.task_suite]
    suite = suite_type()
    if not 0 <= args.task_id < suite.n_tasks:
        raise ValueError(f"task-id must lie in [0,{suite.n_tasks - 1}]")
    task = suite.get_task(args.task_id)
    initial_states = _load_trusted_initial_states(task)
    if not 0 <= args.init_state_index < len(initial_states):
        raise ValueError(
            f"init-state-index must lie in [0,{len(initial_states) - 1}]"
        )
    env = _make_env(task, args.resolution, args.seed)
    prompt = str(task.language)
    prompt_ids, prompt_valid = backbone.tokenize_prompts(
        [prompt],
        max_length=int(observation_spec["max_prompt_tokens"]),
    )
    action_generator = torch.Generator(device=args.device).manual_seed(args.seed)
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
    normalized_clip_count = 0
    environment_clip_count = 0
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
        observation = env.set_init_state(initial_states[args.init_state_index])
        for _ in range(args.wait_steps):
            observation, _reward, success, _info = env.step(_DUMMY_ACTION)
            if success:
                break
        while episode_steps < args.max_steps and not success:
            base_image, wrist_image = _images_from_observation(
                observation,
                int(observation_spec["image_height"]),
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
                normalized_state = normalization.state.normalize(raw_state[None])
                state = torch.from_numpy(normalized_state).to(args.device)
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
    except Exception as error:  # noqa: BLE001 - simulator failures must enter the artifact.
        exception_message = f"{type(error).__name__}: {error}"
    finally:
        env.close()
    elapsed_seconds = time.perf_counter() - started

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.output_dir / (
        f"task{args.task_id:02d}_init{args.init_state_index:02d}_"
        f"{'success' if success else 'failure'}.mp4"
    )
    if frames:
        imageio.mimwrite(video_path, frames, fps=10)
    trace_path = args.output_dir / "trace.npz"
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
    np.savez_compressed(trace_path, **trace_payload)
    result: dict[str, Any] = {
        "model_type": "flow" if "flow" in spec else "autoregressive",
        "config": str(args.config),
        "task_suite": args.task_suite,
        "task_id": args.task_id,
        "task_description": prompt,
        "init_state_index": args.init_state_index,
        "training_seed": args.training_seed,
        "seed": args.seed,
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
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    if exception_message:
        raise RuntimeError(exception_message)


def _make_env(task: Any, resolution: int, seed: int) -> OffScreenRenderEnv:
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env


def _load_trusted_initial_states(task: Any) -> np.ndarray:
    """Load repository-owned LIBERO state tensors under PyTorch 2.6+."""

    path = (
        Path(get_libero_path("init_states"))
        / task.problem_folder
        / task.init_states_file
    )
    # LIBERO's checked-in files contain NumPy objects and predate the
    # weights_only default. Never use this path for user-supplied artifacts.
    states = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(states, torch.Tensor):
        states = states.detach().cpu().numpy()
    array = np.asarray(states)
    if not np.issubdtype(array.dtype, np.floating) or array.ndim != 2:
        raise ValueError(
            "LIBERO initial states must be a floating-point rank-2 array"
        )
    if not np.isfinite(array).all():
        raise ValueError("LIBERO initial states must be finite")
    return array


def _images_from_observation(
    observation: Mapping[str, Any],
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    images = []
    for key in ("agentview_image", "robot0_eye_in_hand_image"):
        image = np.ascontiguousarray(np.asarray(observation[key])[::-1, ::-1])
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        images.append(np.moveaxis(image, -1, 0))
    return images[0], images[1]


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.max_steps,
        args.replan_steps,
        args.resolution,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("max-steps, replan-steps, and resolution must be positive")
    if args.wait_steps < 0:
        raise ValueError("wait-steps must be non-negative")
    if args.gripper_debounce_steps < 0:
        raise ValueError("gripper-debounce-steps must be non-negative")
    if not 0.0 < args.gripper_threshold <= 1.0:
        raise ValueError("gripper-threshold must lie in (0,1]")


def _autocast_context(device: str) -> torch.autocast:
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    return torch.autocast(
        device_type=device_type,
        dtype=torch.bfloat16,
        enabled=device_type == "cuda",
    )


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported compute dtype: {name}")


if __name__ == "__main__":
    main()

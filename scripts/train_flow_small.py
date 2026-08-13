"""Train a compact policy on an episode-disjoint real LIBERO subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pi0_minimal.data import (
    EpisodeSplit,
    NormalizationStats,
    merge_demo_and_teacher_data,
    source_balanced_sampling_probabilities,
    teacher_indices_after_demonstrations,
    three_source_balanced_sampling_probabilities,
)
from pi0_minimal.metrics import action_prediction_metrics
from pi0_minimal.model_spec import load_and_validate_model_spec
from pi0_minimal.models import (
    AutoregressivePolicy,
    ConditionMemory,
    FlowPolicy,
    FrozenPaliGemmaBackbone,
    build_autoregressive_policy,
    build_flow_policy,
    sample_paper_flow_time,
)
from pi0_minimal.training import load_training_checkpoint, save_training_checkpoint


@dataclass(frozen=True, slots=True)
class SegmentedConditionMemory:
    """Index one or more condition caches without copying their full tensors."""

    segments: tuple[ConditionMemory, ...]

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("condition memory must contain at least one segment")
        value_shape = self.segments[0].values.shape[1:]
        valid_shape = self.segments[0].valid.shape[1:]
        if any(
            segment.values.shape[1:] != value_shape
            or segment.valid.shape[1:] != valid_shape
            for segment in self.segments
        ):
            raise ValueError("condition cache segments must have matching shapes")

    @property
    def sample_count(self) -> int:
        return sum(segment.values.shape[0] for segment in self.segments)

    def select(self, indices: np.ndarray) -> ConditionMemory:
        selected = np.asarray(indices)
        if selected.ndim != 1 or not np.issubdtype(selected.dtype, np.integer):
            raise ValueError("condition indices must be an integer vector")
        if len(selected) == 0:
            raise ValueError("condition selection must not be empty")
        if selected.min() < 0 or selected.max() >= self.sample_count:
            raise IndexError("condition index is outside the segmented cache")
        first = self.segments[0]
        values = torch.empty(
            (len(selected), *first.values.shape[1:]), dtype=first.values.dtype
        )
        valid = torch.empty(
            (len(selected), *first.valid.shape[1:]), dtype=first.valid.dtype
        )
        offset = 0
        for segment in self.segments:
            stop = offset + segment.values.shape[0]
            mask = (selected >= offset) & (selected < stop)
            if mask.any():
                output_indices = torch.from_numpy(np.flatnonzero(mask))
                local_indices = torch.from_numpy(
                    (selected[mask] - offset).astype(np.int64, copy=False)
                )
                values[output_indices] = segment.values[local_indices]
                valid[output_indices] = segment.valid[local_indices]
            offset = stop
        return ConditionMemory(values, valid)


@dataclass(frozen=True, slots=True)
class CachedDataset:
    condition: SegmentedConditionMemory
    state: torch.Tensor
    actions: torch.Tensor
    action_valid: torch.Tensor
    episode_index: np.ndarray

    def __post_init__(self) -> None:
        size = self.state.shape[0]
        if (
            self.condition.sample_count != size
            or self.actions.shape[0] != size
            or self.action_valid.shape[0] != size
            or self.episode_index.shape != (size,)
        ):
            raise ValueError("cached dataset fields must share the sample dimension")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/model_flow_tiny.toml"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--teacher-data", type=Path)
    parser.add_argument("--teacher-sampling-fraction", type=float, default=0.5)
    parser.add_argument("--correction-data", type=Path)
    parser.add_argument("--correction-sampling-fraction", type=float, default=0.25)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--condition-batch-size", type=int, default=4)
    parser.add_argument("--condition-cache", type=Path)
    parser.add_argument("--teacher-condition-cache", type=Path)
    parser.add_argument("--correction-condition-cache", type=Path)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--generation-eval-samples", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--state-noise-std", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--snapshot-step", type=int, action="append", default=[])
    parser.add_argument(
        "--episode-range-weight",
        action="append",
        default=[],
        metavar="START:STOP:WEIGHT",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    spec = load_and_validate_model_spec(args.config)
    split = EpisodeSplit.load(args.split)
    normalization = NormalizationStats.load(args.normalization)
    if normalization.training_split_fingerprint != split.fingerprint:
        raise ValueError("normalization statistics do not match the requested split")
    demonstration_data = _load_and_validate_npz(args.data, spec)
    demonstration_sha256 = _sha256(args.data)
    demonstration_train, validation_indices = _indices_from_split(
        demonstration_data["episode_index"], split
    )
    demonstration_probabilities = _train_sampling_probabilities(
        demonstration_data["episode_index"],
        demonstration_train,
        args.episode_range_weight,
    )
    teacher_sha256: str | None = None
    teacher_train_samples = 0
    teacher_data: dict[str, np.ndarray] | None = None
    if args.teacher_data is not None:
        teacher_data = _load_and_validate_npz(
            args.teacher_data, spec, require_task_index=True
        )
        teacher_sha256 = _sha256(args.teacher_data)
        teacher_indices = teacher_indices_after_demonstrations(
            demonstration_data,
            teacher_data,
        )
        teacher_train_samples = len(teacher_indices)
        correction_sha256: str | None = None
        correction_train_samples = 0
        correction_data: dict[str, np.ndarray] | None = None
        if args.correction_data is not None:
            correction_data = _load_and_validate_npz(
                args.correction_data, spec, require_task_index=True
            )
            correction_sha256 = _sha256(args.correction_data)
            demonstration_teacher, _ = merge_demo_and_teacher_data(
                demonstration_data,
                teacher_data,
            )
            correction_indices = teacher_indices_after_demonstrations(
                demonstration_teacher,
                correction_data,
            )
            correction_train_samples = len(correction_indices)
            train_indices = np.concatenate(
                (demonstration_train, teacher_indices, correction_indices)
            )
            train_sampling_probabilities = (
                three_source_balanced_sampling_probabilities(
                    demonstration_probabilities,
                    demonstration_count=len(demonstration_train),
                    teacher_count=teacher_train_samples,
                    correction_count=correction_train_samples,
                    teacher_fraction=args.teacher_sampling_fraction,
                    correction_fraction=args.correction_sampling_fraction,
                    teacher_task_indices=teacher_data["task_index"],
                    correction_task_indices=correction_data["task_index"],
                )
            )
            data_sha256 = _combined_sha256(
                demonstration_sha256,
                teacher_sha256,
                correction_sha256,
            )
        else:
            train_indices = np.concatenate((demonstration_train, teacher_indices))
            train_sampling_probabilities = source_balanced_sampling_probabilities(
                demonstration_probabilities,
                demonstration_count=len(demonstration_train),
                teacher_count=teacher_train_samples,
                teacher_fraction=args.teacher_sampling_fraction,
                teacher_task_indices=teacher_data["task_index"],
            )
            data_sha256 = _combined_sha256(demonstration_sha256, teacher_sha256)
    else:
        train_indices = demonstration_train
        train_sampling_probabilities = demonstration_probabilities
        data_sha256 = demonstration_sha256
        correction_sha256 = None
        correction_train_samples = 0
        correction_data = None

    demonstration_cached = _encode_conditions(
        demonstration_data,
        normalization.state.normalize(demonstration_data["state"]),
        normalization.action.normalize(demonstration_data["actions"]),
        spec,
        device=args.device,
        batch_size=args.condition_batch_size,
        cache_path=args.condition_cache,
        data_sha256=demonstration_sha256,
    )
    if teacher_data is not None and teacher_sha256 is not None:
        teacher_cached = _encode_conditions(
            teacher_data,
            normalization.state.normalize(teacher_data["state"]),
            normalization.action.normalize(teacher_data["actions"]),
            spec,
            device=args.device,
            batch_size=args.condition_batch_size,
            cache_path=args.teacher_condition_cache,
            data_sha256=teacher_sha256,
        )
        cached = _concatenate_cached_datasets(
            demonstration_cached,
            teacher_cached,
        )
        del demonstration_cached, teacher_cached
        if correction_data is not None and correction_sha256 is not None:
            correction_cached = _encode_conditions(
                correction_data,
                normalization.state.normalize(correction_data["state"]),
                normalization.action.normalize(correction_data["actions"]),
                spec,
                device=args.device,
                batch_size=args.condition_batch_size,
                cache_path=args.correction_condition_cache,
                data_sha256=correction_sha256,
            )
            cached = _concatenate_cached_datasets(cached, correction_cached)
            del correction_cached
    else:
        cached = demonstration_cached
    policy = _build_policy(spec, args.device)
    initialization_step: int | None = None
    if args.init_checkpoint is not None:
        initialized = load_training_checkpoint(
            args.init_checkpoint,
            policy=policy,
            map_location=args.device,
        )
        initialization_step = int(initialized["step"])
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: max(0.1, 1.0 - step / args.steps),
    )
    flow_generator = torch.Generator(device=args.device).manual_seed(args.seed)
    batch_rng = np.random.default_rng(args.seed)
    checkpoint_metadata = {
        "config": str(args.config),
        "data": str(args.data),
        "data_sha256": data_sha256,
        "demonstration_data_sha256": demonstration_sha256,
        "teacher_data": None if args.teacher_data is None else str(args.teacher_data),
        "teacher_data_sha256": teacher_sha256,
        "teacher_sampling_fraction": (
            args.teacher_sampling_fraction if args.teacher_data is not None else 0.0
        ),
        "teacher_task_balanced": args.teacher_data is not None,
        "correction_data": (
            None if args.correction_data is None else str(args.correction_data)
        ),
        "correction_data_sha256": correction_sha256,
        "correction_sampling_fraction": (
            args.correction_sampling_fraction
            if args.correction_data is not None
            else 0.0
        ),
        "correction_task_balanced": args.correction_data is not None,
        "demonstration_train_samples": len(demonstration_train),
        "teacher_train_samples": teacher_train_samples,
        "correction_train_samples": correction_train_samples,
        "split_fingerprint": split.fingerprint,
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "state_noise_std": args.state_noise_std,
        "generation_eval_samples": args.generation_eval_samples,
        "train_samples": len(train_indices),
        "validation_samples": len(validation_indices),
        "train_episodes": sorted(set(cached.episode_index[train_indices].tolist())),
        "validation_episodes": sorted(
            set(cached.episode_index[validation_indices].tolist())
        ),
        "init_checkpoint": (
            None if args.init_checkpoint is None else str(args.init_checkpoint)
        ),
        "init_checkpoint_step": initialization_step,
        "episode_range_weight": list(args.episode_range_weight),
        "snapshot_steps": sorted(set(args.snapshot_step)),
    }
    start_step = 0
    best_validation_loss = math.inf
    best_generation_action_mae = math.inf
    history: list[dict[str, Any]] = []
    if args.resume is not None:
        restored = load_training_checkpoint(
            args.resume,
            policy=policy,
            optimizer=optimizer,
            scheduler=scheduler,
            flow_generator=flow_generator,
            map_location=args.device,
        )
        _validate_resume_metadata(restored["metadata"], checkpoint_metadata)
        start_step = int(restored["step"])
        best_validation_loss = float(restored["best_validation_loss"])
        batch_rng.bit_generator.state = restored["batch_rng_state"]
        history = list(restored["metadata"].get("history", []))
        best_generation_action_mae = _best_generation_action_mae(history)
        if start_step > args.steps:
            raise ValueError("resume checkpoint is beyond the requested steps")

    fixed_validation = _fixed_validation_problem(
        cached,
        validation_indices,
        seed=args.seed + 1,
        spec=spec,
    )
    generation_indices, generation_noise = _fixed_generation_problem(
        cached,
        validation_indices,
        sample_count=args.generation_eval_samples,
        seed=args.seed + 2,
        spec=spec,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    initial_validation_loss = _evaluate(
        policy,
        cached,
        validation_indices,
        fixed_validation,
        args.batch_size,
        args.device,
    )
    initial_generation_metrics = _evaluate_generated_actions(
        policy,
        cached,
        generation_indices,
        generation_noise,
        args.batch_size,
        args.device,
    )
    if start_step == 0:
        best_validation_loss = initial_validation_loss
        best_generation_action_mae = float(
            initial_generation_metrics["validation_normalized_action_mae"]
        )
        history.append(
            {
                "step": 0,
                "validation_loss": initial_validation_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **initial_generation_metrics,
            }
        )
        initial_metadata = checkpoint_metadata | {
            "best_generation_action_mae": best_generation_action_mae,
            "history": history,
        }
        for checkpoint_name in ("best.pt", "best_generation.pt"):
            save_training_checkpoint(
                args.output_dir / checkpoint_name,
                policy=policy,
                optimizer=optimizer,
                scheduler=scheduler,
                step=0,
                best_validation_loss=best_validation_loss,
                flow_generator=flow_generator,
                batch_rng_state=batch_rng.bit_generator.state,
                metadata=initial_metadata,
            )

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
        torch.cuda.synchronize(args.device)
    start_time = time.perf_counter()
    policy.train()
    for step in range(start_step + 1, args.steps + 1):
        indices = batch_rng.choice(
            train_indices,
            size=args.batch_size,
            replace=True,
            p=train_sampling_probabilities,
        )
        condition, state, actions, action_valid = _torch_batch(
            cached,
            indices,
            args.device,
        )
        if args.state_noise_std > 0.0:
            state = state + torch.randn(
                state.shape,
                dtype=state.dtype,
                device=state.device,
                generator=flow_generator,
            ) * args.state_noise_std
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(args.device):
            output = _training_step(
                policy,
                condition,
                state,
                actions,
                action_valid,
                flow_generator=flow_generator,
            )
        output.loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(),
            max_norm=args.gradient_clip,
            error_if_nonfinite=True,
        )
        optimizer.step()
        scheduler.step()

        if (
            step == 1
            or step % args.eval_every == 0
            or step == args.steps
            or step in args.snapshot_step
        ):
            validation_loss = _evaluate(
                policy,
                cached,
                validation_indices,
                fixed_validation,
                args.batch_size,
                args.device,
            )
            generation_metrics = _evaluate_generated_actions(
                policy,
                cached,
                generation_indices,
                generation_noise,
                args.batch_size,
                args.device,
            )
            event = {
                "step": step,
                "training_loss": float(output.loss.detach().cpu()),
                "validation_loss": validation_loss,
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "learning_rate": optimizer.param_groups[0]["lr"],
                **generation_metrics,
            }
            history.append(event)
            print(json.dumps(event), flush=True)
            improved = validation_loss < best_validation_loss
            generation_action_mae = float(
                generation_metrics["validation_normalized_action_mae"]
            )
            generation_improved = (
                generation_action_mae < best_generation_action_mae
            )
            best_validation_loss = min(best_validation_loss, validation_loss)
            best_generation_action_mae = min(
                best_generation_action_mae,
                generation_action_mae,
            )
            metadata = checkpoint_metadata | {
                "best_generation_action_mae": best_generation_action_mae,
                "history": history,
            }
            save_training_checkpoint(
                args.output_dir / "latest.pt",
                policy=policy,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                best_validation_loss=best_validation_loss,
                flow_generator=flow_generator,
                batch_rng_state=batch_rng.bit_generator.state,
                metadata=metadata,
            )
            if step in args.snapshot_step:
                save_training_checkpoint(
                    args.output_dir / f"step_{step:05d}.pt",
                    policy=policy,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    best_validation_loss=best_validation_loss,
                    flow_generator=flow_generator,
                    batch_rng_state=batch_rng.bit_generator.state,
                    metadata=metadata,
                )
            if improved:
                save_training_checkpoint(
                    args.output_dir / "best.pt",
                    policy=policy,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    best_validation_loss=best_validation_loss,
                    flow_generator=flow_generator,
                    batch_rng_state=batch_rng.bit_generator.state,
                    metadata=metadata,
                )
            if generation_improved:
                save_training_checkpoint(
                    args.output_dir / "best_generation.pt",
                    policy=policy,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    best_validation_loss=best_validation_loss,
                    flow_generator=flow_generator,
                    batch_rng_state=batch_rng.bit_generator.state,
                    metadata=metadata,
                )
            policy.train()
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
    elapsed_seconds = time.perf_counter() - start_time

    restore_verified = _verify_latest_checkpoint(
        args.output_dir / "latest.pt",
        policy,
        spec,
        args,
    )
    sample_summary = _sample_validation_action(
        args.output_dir / "best_generation.pt",
        cached,
        validation_indices,
        normalization,
        spec,
        args,
    )
    result: dict[str, Any] = checkpoint_metadata | {
        "start_step": start_step,
        "final_step": args.steps,
        "initial_validation_loss": initial_validation_loss,
        "best_validation_loss": best_validation_loss,
        "best_generation_action_mae": best_generation_action_mae,
        "final_validation_loss": float(history[-1]["validation_loss"]),
        "elapsed_seconds": elapsed_seconds,
        "restore_verified": restore_verified,
        "sample": sample_summary,
        "history": history,
    }
    if args.device.startswith("cuda"):
        result["cuda_peak_allocated_mib"] = (
            torch.cuda.max_memory_allocated(args.device) / 2**20
        )
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "steps": args.steps,
        "batch-size": args.batch_size,
        "condition-batch-size": args.condition_batch_size,
        "eval-every": args.eval_every,
        "generation-eval-samples": args.generation_eval_samples,
        "learning-rate": args.learning_rate,
        "gradient-clip": args.gradient_clip,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"these arguments must be positive: {invalid}")
    if args.weight_decay < 0:
        raise ValueError("weight-decay must be non-negative")
    if args.state_noise_std < 0:
        raise ValueError("state-noise-std must be non-negative")
    if args.resume is not None and args.init_checkpoint is not None:
        raise ValueError("resume and init-checkpoint are mutually exclusive")
    if args.teacher_data is not None and not 0.0 < args.teacher_sampling_fraction < 1.0:
        raise ValueError("teacher-sampling-fraction must lie strictly between 0 and 1")
    if args.teacher_condition_cache is not None and args.teacher_data is None:
        raise ValueError("teacher-condition-cache requires teacher-data")
    if args.correction_data is not None and args.teacher_data is None:
        raise ValueError("correction-data requires teacher-data")
    if args.correction_data is not None:
        if not 0.0 < args.correction_sampling_fraction < 1.0:
            raise ValueError(
                "correction-sampling-fraction must lie strictly between zero and one"
            )
        if args.teacher_sampling_fraction + args.correction_sampling_fraction >= 1.0:
            raise ValueError(
                "teacher and correction sampling fractions must sum to less than one"
            )
    if args.correction_condition_cache is not None and args.correction_data is None:
        raise ValueError("correction-condition-cache requires correction-data")
    invalid_snapshots = sorted(
        {step for step in args.snapshot_step if step <= 0 or step > args.steps}
    )
    if invalid_snapshots:
        raise ValueError(f"snapshot steps must lie within training: {invalid_snapshots}")


def _train_sampling_probabilities(
    episode_indices: np.ndarray,
    train_indices: np.ndarray,
    range_weights: list[str],
) -> np.ndarray | None:
    if not range_weights:
        return None
    weights = np.ones(len(train_indices), dtype=np.float64)
    train_episodes = episode_indices[train_indices]
    for specification in range_weights:
        parts = specification.split(":")
        if len(parts) != 3:
            raise ValueError(
                "episode-range-weight must use START:STOP:WEIGHT"
            )
        try:
            start, stop = (int(parts[0]), int(parts[1]))
            weight = float(parts[2])
        except ValueError as error:
            raise ValueError(
                "episode-range-weight must contain numeric values"
            ) from error
        if start > stop or not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("episode-range-weight values are invalid")
        selected = (train_episodes >= start) & (train_episodes <= stop)
        if not selected.any():
            raise ValueError(
                f"episode-range-weight {specification} selects no training samples"
            )
        weights[selected] *= weight
    return weights / weights.sum()


def _best_generation_action_mae(history: list[dict[str, Any]]) -> float:
    candidates = (
        float(event["validation_normalized_action_mae"])
        for event in history
        if "validation_normalized_action_mae" in event
    )
    return min(candidates, default=math.inf)


def _load_and_validate_npz(
    path: Path,
    spec: dict[str, Any],
    *,
    require_task_index: bool = False,
) -> dict[str, np.ndarray]:
    required = {
        "images",
        "image_keys",
        "state",
        "actions",
        "action_valid",
        "prompts",
        "episode_index",
    }
    with np.load(path) as payload:
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"training NPZ is missing fields: {sorted(missing)}")
        arrays = {key: payload[key].copy() for key in required}
        if "task_index" in payload.files:
            arrays["task_index"] = payload["task_index"].copy()
    observation = spec["observation"]
    action = spec["action"]
    sample_count = len(arrays["state"])
    expected_images = (
        sample_count,
        len(observation["image_keys"]),
        observation["image_channels"],
        observation["image_height"],
        observation["image_width"],
    )
    if arrays["images"].dtype != np.uint8 or arrays["images"].shape != expected_images:
        raise ValueError(f"images must have uint8 shape {expected_images}")
    if arrays["image_keys"].tolist() != observation["image_keys"]:
        raise ValueError("NPZ image keys do not match the model specification")
    if arrays["state"].shape != (sample_count, observation["state_dim"]):
        raise ValueError("state shape does not match the model specification")
    expected_actions = (sample_count, action["horizon"], action["dim"])
    if arrays["actions"].shape != expected_actions:
        raise ValueError(f"actions must have shape {expected_actions}")
    if arrays["action_valid"].dtype != np.bool_:
        raise TypeError("action_valid must be boolean")
    if arrays["action_valid"].shape != expected_actions:
        raise ValueError("action_valid shape must match actions")
    if "task_index" in arrays:
        if arrays["task_index"].shape != (sample_count,) or not np.issubdtype(
            arrays["task_index"].dtype, np.integer
        ):
            raise ValueError("task_index must be an integer vector over samples")
    elif require_task_index:
        raise ValueError("teacher training NPZ must contain task_index")
    return arrays


def _indices_from_split(
    episode_indices: np.ndarray,
    split: EpisodeSplit,
) -> tuple[np.ndarray, np.ndarray]:
    train_ids = {int(item.episode_id.removeprefix("episode_")) for item in split.train}
    validation_ids = {
        int(item.episode_id.removeprefix("episode_")) for item in split.validation
    }
    train = np.flatnonzero(np.isin(episode_indices, list(train_ids)))
    validation = np.flatnonzero(np.isin(episode_indices, list(validation_ids)))
    unknown = np.flatnonzero(
        ~np.isin(episode_indices, list(train_ids | validation_ids))
    )
    if len(unknown):
        raise ValueError("training NPZ contains episodes absent from the split manifest")
    if not len(train) or not len(validation):
        raise ValueError("training NPZ must contain both train and validation episodes")
    if set(episode_indices[train]) & set(episode_indices[validation]):
        raise ValueError("an episode cannot appear in both train and validation")
    return train, validation


def _encode_conditions(
    raw: dict[str, np.ndarray],
    normalized_state: np.ndarray,
    normalized_actions: np.ndarray,
    spec: dict[str, Any],
    *,
    device: str,
    batch_size: int,
    cache_path: Path | None,
    data_sha256: str,
) -> CachedDataset:
    observation = spec["observation"]
    backbone_spec = spec["backbone"]
    cache_metadata = {
        "schema_version": 1,
        "data_sha256": data_sha256,
        "sample_count": len(raw["state"]),
        "model_id": str(backbone_spec["model_id"]),
        "revision": str(backbone_spec["revision"]),
        "image_keys": list(observation["image_keys"]),
        "max_prompt_tokens": int(observation["max_prompt_tokens"]),
    }
    if cache_path is not None and cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("metadata") != cache_metadata:
            raise ValueError("condition cache metadata does not match this run")
        condition = SegmentedConditionMemory(
            (ConditionMemory(payload["values"], payload["valid"]),)
        )
        print(f"loaded condition cache: {cache_path}", flush=True)
        return CachedDataset(
            condition,
            torch.from_numpy(normalized_state),
            torch.from_numpy(normalized_actions),
            torch.from_numpy(raw["action_valid"]),
            raw["episode_index"],
        )
    backbone = FrozenPaliGemmaBackbone.from_pretrained(
        str(backbone_spec["model_id"]),
        revision=str(backbone_spec["revision"]),
        device=device,
        compute_dtype=_dtype_from_name(spec["compute_dtype"]),
        expected_image_views=len(observation["image_keys"]),
        expected_output_dim=int(backbone_spec["output_dim"]),
    )
    condition_values: torch.Tensor | None = None
    condition_valid: torch.Tensor | None = None
    prompts = raw["prompts"].tolist()
    for start in range(0, len(prompts), batch_size):
        stop = min(start + batch_size, len(prompts))
        prompt_ids, prompt_valid = backbone.tokenize_prompts(
            prompts[start:stop],
            max_length=int(observation["max_prompt_tokens"]),
        )
        encoded = backbone.encode_numpy(
            raw["images"][start:stop],
            np.ones(
                (stop - start, len(observation["image_keys"])),
                dtype=np.bool_,
            ),
            prompt_ids.numpy(),
            prompt_valid.numpy(),
        )
        encoded_values = encoded.values.detach().cpu()
        encoded_valid = encoded.valid.detach().cpu()
        if condition_values is None or condition_valid is None:
            condition_values = torch.empty(
                (len(prompts), *encoded_values.shape[1:]),
                dtype=encoded_values.dtype,
            )
            condition_valid = torch.empty(
                (len(prompts), encoded_valid.shape[1]),
                dtype=torch.bool,
            )
        condition_values[start:stop].copy_(encoded_values)
        condition_valid[start:stop].copy_(encoded_valid)
        print(f"encoded conditions {stop}/{len(prompts)}", flush=True)
    del backbone
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    if condition_values is None or condition_valid is None:
        raise ValueError("cannot encode an empty condition dataset")
    condition = ConditionMemory(condition_values, condition_valid)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(f".{cache_path.name}.tmp")
        torch.save(
            {
                "metadata": cache_metadata,
                "values": condition.values,
                "valid": condition.valid,
            },
            temporary,
        )
        os.replace(temporary, cache_path)
        print(f"saved condition cache: {cache_path}", flush=True)
    return CachedDataset(
        SegmentedConditionMemory((condition,)),
        torch.from_numpy(normalized_state),
        torch.from_numpy(normalized_actions),
        torch.from_numpy(raw["action_valid"]),
        raw["episode_index"],
    )


def _concatenate_cached_datasets(
    demonstration: CachedDataset,
    teacher: CachedDataset,
) -> CachedDataset:
    """Concatenate separately cached sources without re-encoding demonstrations."""

    return CachedDataset(
        SegmentedConditionMemory(
            (*demonstration.condition.segments, *teacher.condition.segments)
        ),
        torch.cat((demonstration.state, teacher.state)),
        torch.cat((demonstration.actions, teacher.actions)),
        torch.cat((demonstration.action_valid, teacher.action_valid)),
        np.concatenate((demonstration.episode_index, teacher.episode_index)),
    )


def _fixed_validation_problem(
    cached: CachedDataset,
    indices: np.ndarray,
    *,
    seed: int,
    spec: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if "flow" not in spec:
        return None
    generator = torch.Generator().manual_seed(seed)
    actions = cached.actions[indices]
    noise = torch.randn(actions.shape, generator=generator, dtype=torch.float32)
    flow = spec["flow"]
    flow_time = sample_paper_flow_time(
        len(indices),
        device="cpu",
        generator=generator,
        beta_alpha=float(flow["beta_alpha"]),
        beta_beta=float(flow["beta_beta"]),
        cutoff=float(flow["cutoff"]),
    )
    return noise, flow_time


def _fixed_generation_problem(
    cached: CachedDataset,
    validation_indices: np.ndarray,
    *,
    sample_count: int,
    seed: int,
    spec: dict[str, Any],
) -> tuple[np.ndarray, torch.Tensor | None]:
    generator = np.random.default_rng(seed)
    count = min(sample_count, len(validation_indices))
    selected = np.sort(
        generator.choice(validation_indices, size=count, replace=False)
    )
    if "flow" not in spec:
        return selected, None
    torch_generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(
        cached.actions[selected].shape,
        generator=torch_generator,
        dtype=torch.float32,
    )
    return selected, noise


def _torch_batch(
    cached: CachedDataset,
    indices: np.ndarray,
    device: str,
    *,
    condition_indices: np.ndarray | None = None,
) -> tuple[ConditionMemory, torch.Tensor, torch.Tensor, torch.Tensor]:
    tensor_indices = torch.as_tensor(indices, dtype=torch.int64)
    selected_condition_indices = np.asarray(
        indices if condition_indices is None else condition_indices,
        dtype=np.int64,
    )
    if selected_condition_indices.shape != tuple(tensor_indices.shape):
        raise ValueError("condition indices must match the batch indices")
    selected_condition = cached.condition.select(selected_condition_indices)
    condition = ConditionMemory(
        selected_condition.values.to(device),
        selected_condition.valid.to(device),
    )
    return (
        condition,
        cached.state[tensor_indices].to(device),
        cached.actions[tensor_indices].to(device),
        cached.action_valid[tensor_indices].to(device),
    )


@torch.no_grad()
def _evaluate(
    policy: FlowPolicy | AutoregressivePolicy,
    cached: CachedDataset,
    indices: np.ndarray,
    fixed_problem: tuple[torch.Tensor, torch.Tensor] | None,
    batch_size: int,
    device: str,
) -> float:
    policy.eval()
    total_squared_error = 0.0
    total_valid = 0
    noise = flow_time = None
    if fixed_problem is not None:
        noise, flow_time = fixed_problem
    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices))
        batch_indices = indices[start:stop]
        condition, state, actions, action_valid = _torch_batch(
            cached,
            batch_indices,
            device,
        )
        with _autocast_context(device):
            output = _training_step(
                policy,
                condition,
                state,
                actions,
                action_valid,
                noise=None if noise is None else noise[start:stop].to(device),
                flow_time=(
                    None
                    if flow_time is None
                    else flow_time[start:stop].to(device)
                ),
            )
        valid_count = int(action_valid.sum().cpu())
        total_squared_error += float(output.loss.cpu()) * valid_count
        total_valid += valid_count
    return total_squared_error / total_valid


@torch.no_grad()
def _evaluate_generated_actions(
    policy: FlowPolicy | AutoregressivePolicy,
    cached: CachedDataset,
    indices: np.ndarray,
    initial_noise: torch.Tensor | None,
    batch_size: int,
    device: str,
    *,
    condition_indices: np.ndarray | None = None,
) -> dict[str, float | int | list[float | int | None]]:
    policy.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices))
        condition, state, actions, action_valid = _torch_batch(
            cached,
            indices[start:stop],
            device,
            condition_indices=(
                None
                if condition_indices is None
                else condition_indices[start:stop]
            ),
        )
        with _autocast_context(device):
            if isinstance(policy, FlowPolicy):
                if initial_noise is None:
                    raise ValueError("flow generation evaluation requires fixed noise")
                generated = policy.sample(
                    condition,
                    state,
                    action_valid,
                    initial_noise=initial_noise[start:stop].to(device),
                )
            else:
                if initial_noise is not None:
                    raise ValueError("AR generation evaluation does not accept noise")
                generated = policy.sample(condition, state, action_valid)
        predictions.append(generated.cpu().numpy())
        targets.append(actions.cpu().numpy())
        masks.append(action_valid.cpu().numpy())
    metrics = action_prediction_metrics(
        np.concatenate(predictions),
        np.concatenate(targets),
        np.concatenate(masks),
    ).to_dict()
    return {f"validation_{key}": value for key, value in metrics.items()}


def _verify_latest_checkpoint(
    checkpoint: Path,
    trained_policy: FlowPolicy | AutoregressivePolicy,
    spec: dict[str, Any],
    args: argparse.Namespace,
) -> bool:
    restored_policy = _build_policy(spec, args.device)
    optimizer = torch.optim.AdamW(
        restored_policy.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: max(0.1, 1.0 - step / args.steps),
    )
    generator = torch.Generator(device=args.device)
    restored = load_training_checkpoint(
        checkpoint,
        policy=restored_policy,
        optimizer=optimizer,
        scheduler=scheduler,
        flow_generator=generator,
        map_location=args.device,
    )
    if restored["step"] != args.steps:
        raise RuntimeError("latest checkpoint did not restore the final step")
    exact = all(
        torch.equal(first, second)
        for first, second in zip(
            trained_policy.state_dict().values(),
            restored_policy.state_dict().values(),
            strict=True,
        )
    )
    if not exact:
        raise RuntimeError("latest checkpoint policy parameters changed after restore")
    return True


@torch.no_grad()
def _sample_validation_action(
    checkpoint: Path,
    cached: CachedDataset,
    validation_indices: np.ndarray,
    normalization: NormalizationStats,
    spec: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    policy = _build_policy(spec, args.device)
    load_training_checkpoint(checkpoint, policy=policy, map_location=args.device)
    policy.eval()
    index = np.asarray([validation_indices[0]])
    condition, state, _actions, action_valid = _torch_batch(cached, index, args.device)
    with _autocast_context(args.device):
        if isinstance(policy, FlowPolicy):
            initial_noise = torch.randn(
                action_valid.shape,
                generator=torch.Generator(device=args.device).manual_seed(
                    args.seed + 2
                ),
                dtype=torch.float32,
                device=args.device,
            )
            normalized = policy.sample(
                condition,
                state,
                action_valid,
                initial_noise=initial_noise,
            )
        else:
            normalized = policy.sample(condition, state, action_valid)
    normalized_np = normalized.cpu().numpy()
    raw = normalization.action.inverse(normalized_np)
    return {
        "episode_index": int(cached.episode_index[index[0]]),
        "normalized_shape": list(normalized_np.shape),
        "finite": bool(np.isfinite(raw).all()),
        "normalized_min": float(normalized_np[action_valid.cpu().numpy()].min()),
        "normalized_max": float(normalized_np[action_valid.cpu().numpy()].max()),
        "raw_min": float(raw[action_valid.cpu().numpy()].min()),
        "raw_max": float(raw[action_valid.cpu().numpy()].max()),
    }


def _validate_resume_metadata(
    restored: dict[str, Any],
    current: dict[str, Any],
) -> None:
    immutable = (
        "data_sha256",
        "teacher_data_sha256",
        "teacher_sampling_fraction",
        "teacher_task_balanced",
        "correction_data_sha256",
        "correction_sampling_fraction",
        "correction_task_balanced",
        "snapshot_steps",
        "split_fingerprint",
        "seed",
        "steps",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "gradient_clip",
        "generation_eval_samples",
    )
    changed = [key for key in immutable if restored.get(key) != current.get(key)]
    if changed:
        raise ValueError(f"resume checkpoint metadata mismatch: {changed}")


def _build_policy(
    spec: dict[str, Any],
    device: str,
) -> FlowPolicy | AutoregressivePolicy:
    if "flow" in spec:
        return build_flow_policy(spec, device=device)
    return build_autoregressive_policy(spec, device=device)


def _training_step(
    policy: FlowPolicy | AutoregressivePolicy,
    condition: ConditionMemory,
    state: torch.Tensor,
    actions: torch.Tensor,
    action_valid: torch.Tensor,
    *,
    flow_generator: torch.Generator | None = None,
    noise: torch.Tensor | None = None,
    flow_time: torch.Tensor | None = None,
):
    if isinstance(policy, FlowPolicy):
        return policy.training_step(
            condition,
            state,
            actions,
            action_valid,
            generator=flow_generator,
            noise=noise,
            flow_time=flow_time,
        )
    if noise is not None or flow_time is not None:
        raise ValueError("AR training does not accept a flow noise problem")
    return policy.training_step(condition, state, actions, action_valid)


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _combined_sha256(*digests: str) -> str:
    return hashlib.sha256(":".join(digests).encode()).hexdigest()


if __name__ == "__main__":
    main()

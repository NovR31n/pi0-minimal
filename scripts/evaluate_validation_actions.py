"""Evaluate deterministic generated actions from existing checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from train_flow_small import (
    _build_policy,
    _encode_conditions,
    _evaluate_generated_actions,
    _fixed_generation_problem,
    _indices_from_split,
    _load_and_validate_npz,
    _sha256,
)

from pi0_minimal.data import EpisodeSplit, NormalizationStats
from pi0_minimal.model_spec import load_and_validate_model_spec
from pi0_minimal.training import load_training_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--condition-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--condition-batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--evaluate-prompt-shuffle", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.sample_count <= 0
        or args.batch_size <= 0
        or args.condition_batch_size <= 0
    ):
        raise ValueError("sample and batch sizes must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    spec = load_and_validate_model_spec(args.config)
    split = EpisodeSplit.load(args.split)
    normalization = NormalizationStats.load(args.normalization)
    if normalization.training_split_fingerprint != split.fingerprint:
        raise ValueError("normalization statistics do not match the split")
    raw = _load_and_validate_npz(args.data, spec)
    data_sha256 = _sha256(args.data)
    normalized_state = normalization.state.normalize(raw["state"])
    normalized_actions = normalization.action.normalize(raw["actions"])
    _train_indices, validation_indices = _indices_from_split(
        raw["episode_index"],
        split,
    )
    cached = _encode_conditions(
        raw,
        normalized_state,
        normalized_actions,
        spec,
        device=args.device,
        batch_size=args.condition_batch_size,
        cache_path=args.condition_cache,
        data_sha256=data_sha256,
    )
    selected, initial_noise = _fixed_generation_problem(
        cached,
        validation_indices,
        sample_count=args.sample_count,
        seed=args.seed,
        spec=spec,
    )
    condition_generator = np.random.default_rng(args.seed + 1)
    shuffled_condition_indices = condition_generator.permutation(selected)
    prompt_shuffled_cached = None
    prompt_shuffle_mapping: dict[str, str] = {}
    if args.evaluate_prompt_shuffle:
        selected_prompts = raw["prompts"][selected].tolist()
        unique_prompts = sorted(set(selected_prompts))
        if len(unique_prompts) < 2:
            raise ValueError("prompt shuffling requires at least two unique prompts")
        prompt_shuffle_mapping = {
            prompt: unique_prompts[(index + 1) % len(unique_prompts)]
            for index, prompt in enumerate(unique_prompts)
        }
        shuffled_raw = {
            "images": raw["images"][selected],
            "image_keys": raw["image_keys"],
            "state": raw["state"][selected],
            "actions": raw["actions"][selected],
            "action_valid": raw["action_valid"][selected],
            "prompts": np.asarray(
                [prompt_shuffle_mapping[prompt] for prompt in selected_prompts],
                dtype=raw["prompts"].dtype,
            ),
            "episode_index": raw["episode_index"][selected],
        }
        prompt_shuffled_cached = _encode_conditions(
            shuffled_raw,
            normalized_state[selected],
            normalized_actions[selected],
            spec,
            device=args.device,
            batch_size=args.condition_batch_size,
            cache_path=None,
            data_sha256=data_sha256,
        )

    evaluations: list[dict[str, Any]] = []
    for checkpoint in args.checkpoints:
        policy = _build_policy(spec, args.device)
        restored = load_training_checkpoint(
            checkpoint,
            policy=policy,
            map_location=args.device,
        )
        metrics = _evaluate_generated_actions(
            policy,
            cached,
            selected,
            initial_noise,
            args.batch_size,
            args.device,
        )
        shuffled_metrics = _evaluate_generated_actions(
            policy,
            cached,
            selected,
            initial_noise,
            args.batch_size,
            args.device,
            condition_indices=shuffled_condition_indices,
        )
        shuffled_action_mae = float(
            shuffled_metrics["validation_normalized_action_mae"]
        )
        action_mae = float(metrics["validation_normalized_action_mae"])
        prompt_shuffle_fields: dict[str, float] = {}
        if prompt_shuffled_cached is not None:
            prompt_shuffled_metrics = _evaluate_generated_actions(
                policy,
                prompt_shuffled_cached,
                np.arange(len(selected)),
                initial_noise,
                args.batch_size,
                args.device,
            )
            prompt_shuffled_action_mae = float(
                prompt_shuffled_metrics["validation_normalized_action_mae"]
            )
            prompt_shuffle_fields = {
                "prompt_shuffled_action_mae": prompt_shuffled_action_mae,
                "prompt_shuffle_mae_increase": (
                    prompt_shuffled_action_mae - action_mae
                ),
            }
        evaluation = {
            "checkpoint": str(checkpoint),
            "checkpoint_step": restored["step"],
            "training_seed": restored["metadata"].get("seed"),
            "native_best_validation_loss": restored["best_validation_loss"],
            "shuffled_condition_action_mae": shuffled_action_mae,
            "condition_shuffle_mae_increase": shuffled_action_mae - action_mae,
            **prompt_shuffle_fields,
            **metrics,
        }
        evaluations.append(evaluation)
        print(json.dumps(evaluation), flush=True)
        del policy
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    result = {
        "model_type": "flow" if "flow" in spec else "autoregressive",
        "config": str(args.config),
        "data": str(args.data),
        "data_sha256": data_sha256,
        "split_fingerprint": split.fingerprint,
        "selection_seed": args.seed,
        "prompt_shuffle_mapping": prompt_shuffle_mapping,
        "requested_sample_count": args.sample_count,
        "evaluated_samples": len(selected),
        "evaluated_episodes": sorted(
            set(cached.episode_index[selected].tolist())
        ),
        "evaluations": evaluations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

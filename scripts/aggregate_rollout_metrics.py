"""Aggregate rollout result/trace pairs into episode and summary tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from pi0_minimal.data import NormalizationStats
from pi0_minimal.metrics import (
    action_smoothness_metrics,
    bootstrap_mean_interval,
    paired_difference_interval,
    runtime_metrics,
    trajectory_metrics,
    wilson_interval,
)

_CONTINUOUS_FIELDS = (
    "first_difference_rms",
    "second_difference_rms",
    "high_frequency_energy_ratio",
    "path_length",
    "path_efficiency",
    "stagnation_ratio",
    "action_saturation_ratio",
    "mean_inference_seconds",
    "stable_mean_inference_seconds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    normalization = NormalizationStats.load(args.normalization)
    rows = [_episode_row(path, normalization) for path in args.results]
    rows.sort(
        key=lambda row: (
            row["model_type"],
            row["task_suite"],
            row["task_id"],
            row["training_seed"],
            row["seed"],
            row["init_state_index"],
        )
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = _summarize(
        rows,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def _episode_row(
    result_path: Path,
    normalization: NormalizationStats,
) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    trace_path = result_path.parent / "trace.npz"
    with np.load(trace_path) as trace:
        actions = trace["actions"]
        normalized_actions = (
            trace["normalized_actions"]
            if "normalized_actions" in trace.files
            else normalization.action.normalize(actions)
        )
        smoothness = action_smoothness_metrics(normalized_actions)
        runtime = runtime_metrics(trace["inference_times"])
        trajectory = (
            trajectory_metrics(
                trace["end_effector_positions"],
                normalized_actions,
            )
            if "end_effector_positions" in trace.files
            else None
        )
    experiment_id = result_path.parent.name
    model_type = result.get("model_type")
    if model_type is None:
        model_type = (
            "flow"
            if experiment_id.startswith("FM-")
            else "autoregressive"
            if experiment_id.startswith("AR-")
            else "unknown"
        )
    return {
        "experiment_id": experiment_id,
        "model_type": model_type,
        "task_suite": result["task_suite"],
        "task_id": int(result["task_id"]),
        "init_state_index": int(result["init_state_index"]),
        "training_seed": (
            ""
            if result.get("training_seed") is None
            else int(result["training_seed"])
        ),
        "seed": int(result["seed"]),
        "success": int(bool(result["success"])),
        "episode_steps": int(result["episode_steps"]),
        "first_difference_rms": smoothness.first_difference_rms,
        "second_difference_rms": smoothness.second_difference_rms,
        "high_frequency_energy_ratio": smoothness.high_frequency_energy_ratio,
        "path_length": "" if trajectory is None else trajectory.path_length,
        "path_efficiency": "" if trajectory is None else trajectory.path_efficiency,
        "stagnation_ratio": "" if trajectory is None else trajectory.stagnation_ratio,
        "action_saturation_ratio": (
            "" if trajectory is None else trajectory.action_saturation_ratio
        ),
        "gripper_switch_count": (
            "" if trajectory is None else trajectory.gripper_switch_count
        ),
        "first_inference_seconds": runtime.first_seconds,
        "mean_inference_seconds": runtime.mean_seconds,
        "stable_mean_inference_seconds": runtime.stable_mean_seconds,
        "p95_inference_seconds": runtime.p95_seconds,
        "normalized_clip_count": int(result["normalized_clip_count"]),
        "environment_clip_count": int(result["environment_clip_count"]),
        "cuda_peak_allocated_mib": result.get("cuda_peak_allocated_mib", ""),
    }


def _summarize(
    rows: list[dict[str, Any]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model_type"])].append(row)
    models = {}
    for model, model_rows in sorted(grouped.items()):
        successes = sum(int(row["success"]) for row in model_rows)
        lower, upper = wilson_interval(successes, len(model_rows))
        metric_summary = {}
        for field in _CONTINUOUS_FIELDS:
            values = [
                float(row[field])
                for row in model_rows
                if row[field] != ""
            ]
            if values:
                interval = bootstrap_mean_interval(
                    values,
                    resamples=resamples,
                    seed=seed,
                )
                metric_summary[field] = interval.to_dict() | {
                    "count": len(values),
                    "standard_deviation": float(np.std(values, ddof=1))
                    if len(values) > 1
                    else 0.0,
                    "median": float(np.median(values)),
                }
        models[model] = {
            "episodes": len(model_rows),
            "successes": successes,
            "success_rate": successes / len(model_rows),
            "success_wilson_95": [lower, upper],
            "metrics": metric_summary,
        }
    return {
        "episodes": len(rows),
        "bootstrap_seed": seed,
        "bootstrap_resamples": resamples,
        "models": models,
        "paired_differences": _paired_summaries(
            grouped,
            resamples=resamples,
            seed=seed,
        ),
    }


def _paired_summaries(
    grouped: dict[str, list[dict[str, Any]]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    if len(grouped) != 2:
        return {}
    first_name, second_name = sorted(grouped)
    key = lambda row: (
        row["task_suite"],
        row["task_id"],
        row["training_seed"],
        row["seed"],
        row["init_state_index"],
    )
    first = {key(row): row for row in grouped[first_name]}
    second = {key(row): row for row in grouped[second_name]}
    common = sorted(first.keys() & second.keys())
    summaries = {}
    for field in ("success", *_CONTINUOUS_FIELDS):
        pairs = [
            (float(first[item][field]), float(second[item][field]))
            for item in common
            if first[item][field] != "" and second[item][field] != ""
        ]
        if pairs:
            interval = paired_difference_interval(
                [pair[0] for pair in pairs],
                [pair[1] for pair in pairs],
                resamples=resamples,
                seed=seed,
            )
            summaries[field] = interval.to_dict() | {"pairs": len(pairs)}
    return {
        "contrast": f"{first_name} - {second_name}",
        "pairing_keys": [
            "task_suite",
            "task_id",
            "training_seed",
            "seed",
            "init_state_index",
        ],
        "metrics": summaries,
    }


if __name__ == "__main__":
    main()

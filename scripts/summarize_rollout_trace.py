"""Compute versioned metrics from one saved rollout trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pi0_minimal.data import NormalizationStats
from pi0_minimal.metrics import (
    action_smoothness_metrics,
    runtime_metrics,
    trajectory_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--normalization", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.trace) as trace:
        required = {"actions", "inference_times"}
        missing = required - set(trace.files)
        if missing:
            raise ValueError(f"trace is missing fields: {sorted(missing)}")
        actions = trace["actions"]
        if "normalized_actions" in trace.files:
            normalized_actions = trace["normalized_actions"]
        elif args.normalization is not None:
            normalized_actions = NormalizationStats.load(
                args.normalization
            ).action.normalize(actions)
        else:
            raise ValueError(
                "legacy traces require --normalization for action metrics"
            )
        result = {
            "trace": str(args.trace),
            "steps": len(actions),
            "action_smoothness": action_smoothness_metrics(
                normalized_actions
            ).to_dict(),
            "runtime_metrics": runtime_metrics(
                trace["inference_times"]
            ).to_dict(),
        }
        if "end_effector_positions" in trace.files:
            result["trajectory_metrics"] = trajectory_metrics(
                trace["end_effector_positions"],
                normalized_actions,
            ).to_dict()
        else:
            result["trajectory_metrics"] = None
            result["trajectory_metrics_unavailable_reason"] = (
                "legacy trace has no end_effector_positions"
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

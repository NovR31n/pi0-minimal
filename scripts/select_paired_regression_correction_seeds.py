"""Select correction seeds where a candidate regressed against a comparator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from pi0_minimal.correction_data import select_recovery_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--comparator-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-ids", type=int, nargs="+", required=True)
    parser.add_argument("--per-task", type=int, default=50)
    parser.add_argument("--minimum-step", type=int, default=20)
    parser.add_argument("--pre-roll-steps", type=int, default=5)
    parser.add_argument("--burst-window", type=int, default=20)
    parser.add_argument("--minimum-burst-switches", type=int, default=4)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_results(root: Path) -> dict[tuple[int, int], tuple[Path, dict[str, Any]]]:
    rows: dict[tuple[int, int], tuple[Path, dict[str, Any]]] = {}
    for path in sorted(root.glob("task*/init*/result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        key = (int(result["task_id"]), int(result["init_state_index"]))
        if key in rows:
            raise ValueError(f"duplicate task/init identity under {root}: {key}")
        rows[key] = (path, result)
    if not rows:
        raise ValueError(f"no result records under {root}")
    return rows


def main() -> None:
    args = parse_args()
    if args.per_task <= 0:
        raise ValueError("per-task must be positive")
    requested_tasks = sorted(set(args.task_ids))
    source = _load_results(args.source_root)
    comparator = _load_results(args.comparator_root)
    if source.keys() != comparator.keys():
        raise ValueError("source and comparator episode identities do not match")

    candidates: dict[int, list[dict[str, Any]]] = {
        task_id: [] for task_id in requested_tasks
    }
    paired_counts = {
        str(task_id): {
            "both_success": 0,
            "source_only": 0,
            "comparator_only": 0,
            "both_fail": 0,
        }
        for task_id in requested_tasks
    }
    for key in sorted(source):
        task_id, init_id = key
        if task_id not in candidates:
            continue
        source_path, source_result = source[key]
        comparator_path, comparator_result = comparator[key]
        for label, result, path in (
            ("source", source_result, source_path),
            ("comparator", comparator_result, comparator_path),
        ):
            if not result.get("completed_without_exception", False) or result.get(
                "exception"
            ):
                raise ValueError(f"{label} result has an exception: {path}")

        source_success = bool(source_result["success"])
        comparator_success = bool(comparator_result["success"])
        if source_success and comparator_success:
            paired_counts[str(task_id)]["both_success"] += 1
            continue
        if source_success:
            paired_counts[str(task_id)]["source_only"] += 1
            continue
        if not comparator_success:
            paired_counts[str(task_id)]["both_fail"] += 1
            continue
        paired_counts[str(task_id)]["comparator_only"] += 1

        if source_result.get("initial_state_source") != "official_demonstration":
            raise ValueError(f"refusing non-demonstration source: {source_path}")
        if int(source_result.get("trace_schema_version", 0)) < 3:
            raise ValueError(f"source lacks schema-v3 state: {source_path}")
        trace_path = source_path.parent / "trace.npz"
        with np.load(trace_path) as trace:
            required = {
                "actions",
                "simulator_states_before",
                "controller_states_before",
            }
            missing = required.difference(trace.files)
            if missing:
                raise ValueError(f"{trace_path} is missing fields: {sorted(missing)}")
            selection = select_recovery_step(
                trace["actions"],
                minimum_step=args.minimum_step,
                pre_roll_steps=args.pre_roll_steps,
                burst_window=args.burst_window,
                minimum_burst_switches=args.minimum_burst_switches,
            )
        candidates[task_id].append(
            {
                "task_id": task_id,
                "demo_index": init_id,
                "recovery_step": selection.step,
                "selection_reason": selection.reason,
                "gripper_switch_count": selection.gripper_switch_count,
                "switch_burst_start": selection.burst_start,
                "source_episode_steps": int(source_result["episode_steps"]),
                "source_result": str(source_path.resolve()),
                "source_result_sha256": _sha256(source_path),
                "source_trace": str(trace_path.resolve()),
                "source_trace_sha256": _sha256(trace_path),
                "comparator_result": str(comparator_path.resolve()),
                "comparator_result_sha256": _sha256(comparator_path),
                "source_initial_state_dataset": source_result["initial_state_dataset"],
                "source_wait_steps": int(source_result["wait_steps"]),
                "source_seed": int(source_result["seed"]),
            }
        )

    selected: list[dict[str, Any]] = []
    selected_counts: dict[str, int] = {}
    reason_counts: dict[str, dict[str, int]] = {}
    for task_id in requested_tasks:
        ordered = sorted(
            candidates[task_id],
            key=lambda item: (
                item["selection_reason"] != "before_gripper_switch_burst",
                -item["gripper_switch_count"],
                item["demo_index"],
            ),
        )
        chosen = ordered[: args.per_task]
        selected.extend(chosen)
        selected_counts[str(task_id)] = len(chosen)
        counts: dict[str, int] = {}
        for item in chosen:
            reason = str(item["selection_reason"])
            counts[reason] = counts.get(reason, 0) + 1
        reason_counts[str(task_id)] = counts

    payload = {
        "schema_version": 1,
        "source_root": str(args.source_root.resolve()),
        "comparator_root": str(args.comparator_root.resolve()),
        "selection_policy": {
            "source_failed_comparator_succeeded_only": True,
            "required_initial_state_source": "official_demonstration",
            "minimum_step": args.minimum_step,
            "pre_roll_steps": args.pre_roll_steps,
            "burst_window": args.burst_window,
            "minimum_burst_switches": args.minimum_burst_switches,
            "per_task": args.per_task,
        },
        "paired_counts": paired_counts,
        "selected_seed_count": len(selected),
        "selected_per_task": selected_counts,
        "selection_reasons_per_task": reason_counts,
        "seeds": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "paired_counts": paired_counts,
                "selected_seed_count": len(selected),
                "selected_per_task": selected_counts,
                "selection_reasons_per_task": reason_counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

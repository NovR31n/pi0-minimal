"""Select auditable teacher takeover seeds from schema-v3 student failures."""

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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-ids", type=int, nargs="+", required=True)
    parser.add_argument("--per-task", type=int, default=5)
    parser.add_argument("--minimum-step", type=int, default=20)
    parser.add_argument("--pre-roll-steps", type=int, default=5)
    parser.add_argument("--burst-window", type=int, default=20)
    parser.add_argument("--minimum-burst-switches", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.per_task <= 0:
        raise ValueError("per-task must be positive")
    requested_tasks = set(args.task_ids)
    candidates: dict[int, list[dict[str, Any]]] = {
        task_id: [] for task_id in requested_tasks
    }
    audited = failures = 0
    for result_path in sorted(args.source_root.glob("task*/init*/result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        task_id = int(result["task_id"])
        if task_id not in requested_tasks:
            continue
        audited += 1
        if result.get("initial_state_source") != "official_demonstration":
            raise ValueError(
                f"refusing non-demonstration correction source: {result_path}"
            )
        if int(result.get("trace_schema_version", 0)) < 3:
            raise ValueError(f"correction source lacks schema-v3 state: {result_path}")
        if not result.get("completed_without_exception", False):
            raise ValueError(f"correction source raised an exception: {result_path}")
        if result.get("success", False):
            continue
        failures += 1
        trace_path = result_path.parent / "trace.npz"
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
                "demo_index": int(result["init_state_index"]),
                "recovery_step": selection.step,
                "selection_reason": selection.reason,
                "gripper_switch_count": selection.gripper_switch_count,
                "switch_burst_start": selection.burst_start,
                "source_episode_steps": int(result["episode_steps"]),
                "source_result": str(result_path.resolve()),
                "source_trace": str(trace_path.resolve()),
                "source_trace_sha256": _sha256(trace_path),
                "source_initial_state_dataset": result["initial_state_dataset"],
                "source_wait_steps": int(result["wait_steps"]),
                "source_seed": int(result["seed"]),
            }
        )

    selected: list[dict[str, Any]] = []
    per_task_counts: dict[str, int] = {}
    for task_id in sorted(requested_tasks):
        ordered = sorted(
            candidates[task_id],
            key=lambda item: (-item["gripper_switch_count"], item["demo_index"]),
        )
        chosen = ordered[: args.per_task]
        selected.extend(chosen)
        per_task_counts[str(task_id)] = len(chosen)
    payload = {
        "schema_version": 1,
        "source_root": str(args.source_root.resolve()),
        "selection_policy": {
            "failed_episodes_only": True,
            "required_initial_state_source": "official_demonstration",
            "minimum_step": args.minimum_step,
            "pre_roll_steps": args.pre_roll_steps,
            "burst_window": args.burst_window,
            "minimum_burst_switches": args.minimum_burst_switches,
            "per_task": args.per_task,
        },
        "audited_episodes": audited,
        "failed_episodes": failures,
        "selected_seed_count": len(selected),
        "selected_per_task": per_task_counts,
        "seeds": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "audited_episodes",
        "failed_episodes",
        "selected_seed_count",
        "selected_per_task",
    )}, indent=2))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()

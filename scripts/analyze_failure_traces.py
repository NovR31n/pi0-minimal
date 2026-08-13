"""Analyze saved rollout traces with conservative motion/gripper proxies."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from pi0_minimal.failure_analysis import FailureProxyThresholds, diagnose_trace

_REPORT_FLAGS = (
    "no_close_command",
    "no_reopen_after_close",
    "excessive_gripper_switching",
    "severe_gripper_switching",
    "low_endpoint_displacement",
    "low_path_efficiency",
    "high_stagnation",
    "no_proxy_detected",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--representatives-per-task", type=int, default=5)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--expected-failures", type=int)
    return parser.parse_args()


def _actual_video(result_path: Path, result: dict[str, Any]) -> str | None:
    videos = sorted(result_path.parent.glob("*.mp4"))
    if videos:
        return str(videos[0])
    declared = result.get("video")
    return None if declared is None else str(declared)


def _severity_score(diagnosis: dict[str, Any]) -> float:
    flags = set(diagnosis["proxy_flags"])
    trajectory = diagnosis["trajectory_metrics"]
    score = float(trajectory["gripper_switch_count"])
    score += 100.0 if "severe_gripper_switching" in flags else 0.0
    score += 60.0 if "no_close_command" in flags else 0.0
    score += 40.0 if "no_reopen_after_close" in flags else 0.0
    score += 30.0 if "low_endpoint_displacement" in flags else 0.0
    score += 20.0 if "low_path_efficiency" in flags else 0.0
    score += 10.0 if "high_stagnation" in flags else 0.0
    return score


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "maximum": float(np.max(array)),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Student-v2 formal failure-trace analysis",
        "",
        "This report uses motion and gripper proxy signals only. The saved trace",
        "schema has no object pose or contact state, so grasp, drop, transport, and",
        "placement stages remain unknown until video review or richer rollouts are",
        "available.",
        "",
        "## Integrity",
        "",
        f"- Episodes: {report['episodes']}",
        f"- Successes: {report['successes']}",
        f"- Failures: {report['failures']}",
        f"- Exceptions: {report['exceptions']}",
        "",
        "## Failure proxies by task",
        "",
        "| Task | Failures | No close | No reopen | Switches >=10 | Switches >=24 | Low endpoint | Low efficiency | Stagnation | No proxy |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task in report["tasks"]:
        counts = task["failure_proxy_counts"]
        lines.append(
            f"| {task['task_id']} | {task['failures']} | "
            f"{counts['no_close_command']} | {counts['no_reopen_after_close']} | "
            f"{counts['excessive_gripper_switching']} | "
            f"{counts['severe_gripper_switching']} | "
            f"{counts['low_endpoint_displacement']} | "
            f"{counts['low_path_efficiency']} | {counts['high_stagnation']} | "
            f"{counts['no_proxy_detected']} |"
        )
    lines.extend(
        [
            "",
            "Flags are non-exclusive. `No proxy` means that the available motion and",
            "gripper signals did not isolate a symptom; it does not mean the policy",
            "executed correctly.",
            "",
            "## Outcome-level trace distributions",
            "",
        ]
    )
    for outcome in ("success", "failure"):
        metrics = report["outcome_metric_quantiles"][outcome]
        lines.append(f"### {outcome.title()}")
        lines.append("")
        lines.append("| Metric | P25 | Median | P75 |")
        lines.append("|---|---:|---:|---:|")
        for name, values in metrics.items():
            lines.append(
                f"| {name} | {values['p25']:.4f} | {values['median']:.4f} | "
                f"{values['p75']:.4f} |"
            )
        lines.append("")
    lines.extend(["## Representative failure videos", ""])
    for task_id, representatives in report["representatives"].items():
        lines.append(f"### Task {task_id}")
        lines.append("")
        for row in representatives:
            flags = ", ".join(row["diagnosis"]["proxy_flags"])
            lines.append(
                f"- init {row['init_state_index']:02d}: `{row['video']}`; "
                f"primary `{row['diagnosis']['primary_proxy']}`; flags: {flags}"
            )
        lines.append("")
    lines.extend(
        [
            "## Next instrumentation requirement",
            "",
            "Future diagnostic rollouts must save task-relevant object poses, gripper",
            "opening, and available contact signals at every step. Only those signals",
            "can support semantic phase labels such as failed grasp, drop, or failed",
            "placement without relying solely on manual video review.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.representatives_per_task <= 0:
        raise ValueError("representatives-per-task must be positive")
    result_paths = sorted(args.results_root.glob("task*/init*/result.json"))
    if not result_paths:
        raise ValueError(f"no result.json files under {args.results_root}")

    thresholds = FailureProxyThresholds()
    episodes: list[dict[str, Any]] = []
    keys: set[tuple[int, int]] = set()
    for result_path in result_paths:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        key = (int(result["task_id"]), int(result["init_state_index"]))
        if key in keys:
            raise ValueError(f"duplicate task/init key: {key}")
        keys.add(key)
        trace_path = result_path.parent / "trace.npz"
        with np.load(trace_path) as trace:
            required = {"normalized_actions", "end_effector_positions"}
            missing = required - set(trace.files)
            if missing:
                raise ValueError(f"{trace_path} is missing fields: {sorted(missing)}")
            diagnosis = diagnose_trace(
                trace["normalized_actions"],
                trace["end_effector_positions"],
                thresholds=thresholds,
            )
        episodes.append(
            {
                "task_id": key[0],
                "init_state_index": key[1],
                "success": bool(result["success"]),
                "completed_without_exception": bool(
                    result["completed_without_exception"]
                ),
                "episode_steps": int(result["episode_steps"]),
                "result": str(result_path),
                "trace": str(trace_path),
                "video": _actual_video(result_path, result),
                "diagnosis": diagnosis,
            }
        )

    failures = [row for row in episodes if not row["success"]]
    exceptions = sum(not row["completed_without_exception"] for row in episodes)
    if args.expected_episodes is not None and len(episodes) != args.expected_episodes:
        raise ValueError(
            f"expected {args.expected_episodes} episodes, found {len(episodes)}"
        )
    if args.expected_failures is not None and len(failures) != args.expected_failures:
        raise ValueError(
            f"expected {args.expected_failures} failures, found {len(failures)}"
        )
    if exceptions:
        raise ValueError(f"analysis input contains {exceptions} infrastructure exceptions")

    tasks = []
    representatives: dict[str, list[dict[str, Any]]] = {}
    for task_id in sorted({row["task_id"] for row in episodes}):
        task_rows = [row for row in episodes if row["task_id"] == task_id]
        task_failures = [row for row in task_rows if not row["success"]]
        counter = Counter(
            flag
            for row in task_failures
            for flag in row["diagnosis"]["proxy_flags"]
        )
        tasks.append(
            {
                "task_id": task_id,
                "episodes": len(task_rows),
                "successes": sum(row["success"] for row in task_rows),
                "failures": len(task_failures),
                "failure_proxy_counts": {
                    flag: counter.get(flag, 0) for flag in _REPORT_FLAGS
                },
            }
        )
        ranked = sorted(
            task_failures,
            key=lambda row: (
                -_severity_score(row["diagnosis"]),
                row["init_state_index"],
            ),
        )
        representatives[str(task_id)] = ranked[: args.representatives_per_task]

    outcome_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for outcome, success in (("success", True), ("failure", False)):
        rows = [row for row in episodes if row["success"] is success]
        outcome_metrics[outcome] = {}
        for metric in (
            "gripper_switch_count",
            "path_efficiency",
            "endpoint_displacement",
            "stagnation_ratio",
            "action_saturation_ratio",
        ):
            values = [
                float(row["diagnosis"]["trajectory_metrics"][metric])
                for row in rows
            ]
            outcome_metrics[outcome][metric] = _quantiles(values)

    failure_counter = Counter(
        flag for row in failures for flag in row["diagnosis"]["proxy_flags"]
    )
    success_counter = Counter(
        flag
        for row in episodes
        if row["success"]
        for flag in row["diagnosis"]["proxy_flags"]
    )
    report = {
        "schema_version": 1,
        "results_root": str(args.results_root),
        "limitations": [
            "Trace proxies cannot determine object contact, grasp, drop, or placement stage.",
            "All semantic failure stages require video review or richer object-state traces.",
            "Proxy flags are non-exclusive and are not causal labels.",
        ],
        "thresholds": thresholds.to_dict(),
        "episodes": len(episodes),
        "successes": len(episodes) - len(failures),
        "failures": len(failures),
        "exceptions": exceptions,
        "failure_proxy_counts": {
            flag: failure_counter.get(flag, 0) for flag in _REPORT_FLAGS
        },
        "success_control_proxy_counts": {
            flag: success_counter.get(flag, 0) for flag in _REPORT_FLAGS
        },
        "tasks": tasks,
        "outcome_metric_quantiles": outcome_metrics,
        "representatives": representatives,
        "failure_episodes": failures,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "episodes": report["episodes"],
                "successes": report["successes"],
                "failures": report["failures"],
                "exceptions": report["exceptions"],
                "failure_proxy_counts": report["failure_proxy_counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

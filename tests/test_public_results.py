from __future__ import annotations

import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_public_result_summary_is_internally_consistent() -> None:
    summary_path = PROJECT_ROOT / "results" / "student_v3_formal_500_summary.json"
    summary = json.loads(summary_path.read_text())
    result = summary["student_v3"]
    paired = summary["paired_comparison_to_student_v2"]

    assert summary["protocol"]["euler_steps"] == 20
    assert result["successes"] + result["failures"] == result["episodes"] == 500
    assert result["success_rate"] == result["successes"] / result["episodes"]
    assert sum(
        paired[key]
        for key in ("both_success", "student_v3_only", "student_v2_only", "both_fail")
    ) == 500
    assert paired["both_success"] + paired["student_v3_only"] == result["successes"]
    assert paired["both_success"] + paired["student_v2_only"] == paired["student_v2_successes"]
    assert paired["success_gain"] == paired["student_v3_only"] - paired["student_v2_only"]


def test_public_per_task_rows_reproduce_aggregate_result() -> None:
    csv_path = PROJECT_ROOT / "results" / "student_v3_per_task.csv"
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert [int(row["task_id"]) for row in rows] == list(range(10))
    assert sum(int(row["episodes"]) for row in rows) == 500
    assert sum(int(row["student_v2_successes"]) for row in rows) == 249
    assert sum(int(row["student_v3_successes"]) for row in rows) == 319
    assert sum(int(row["student_v3_only"]) for row in rows) == 123
    assert sum(int(row["student_v2_only"]) for row in rows) == 53
    for row in rows:
        episodes = int(row["episodes"])
        v2_successes = int(row["student_v2_successes"])
        v3_successes = int(row["student_v3_successes"])
        assert float(row["student_v2_rate"]) == v2_successes / episodes
        assert float(row["student_v3_rate"]) == v3_successes / episodes
        assert int(row["difference_successes"]) == v3_successes - v2_successes

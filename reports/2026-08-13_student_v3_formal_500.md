# Student-v3 formal 500-episode evaluation

## Decision

The frozen Student-v3 step-10K checkpoint completed the formal LIBERO Spatial
matrix at **319/500 (63.8%)**, with zero rollout exceptions. The Wilson 95%
confidence interval is 59.5%--67.9%. Student-v2 scored 249/500 (49.8%) on the
identical 500 episode identities, so Student-v3 improves the formal total by 70
successes and 14.0 percentage points.

The paired table is 196 both successful, 123 Student-v3-only successes, 53
Student-v2-only successes, and 128 both failed. The net paired gain is 70 and the
two-sided exact McNemar p-value is `1.371e-7`. Student-v3 therefore clearly
advances over Student-v2 overall.

## Protocol and integrity

- Checkpoint: `artifacts/STUDENT-V3-CORRECTION-SUCCESS39-10K-S7-015/step_10000.pt`.
- Matrix: LIBERO Spatial tasks 0--9, initial-state indices 0--49.
- Protocol: training seed 7, evaluation seed offset 1000, 220 maximum steps,
  10 wait steps, 5-step replanning, 20-step Euler action sampling, 256-pixel
  observations, CUDA.
- Result directory: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018`.
- All 500 `(task_id, init_state_index)` identities are unique and complete.
- All records use checkpoint step 10K and the frozen rollout protocol; all report
  `completed_without_exception=true` with an empty exception field.
- The 100 development and 90 untouched-validation episodes staged into the formal
  directory are byte-identical to their source `result.json` records. The other
  310 episodes were newly evaluated under the same command with recovery enabled.
- The runner exited with code 0 and released the GPU after completion.

## Per-task results

| Task | Student-v3 | Student-v2 | Difference | V3-only | V2-only | Exact paired p |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 36/50 | 32/50 | +4 | 8 | 4 | 0.3877 |
| 1 | 31/50 | 19/50 | +12 | 17 | 5 | 0.01690 |
| 2 | 38/50 | 19/50 | +19 | 22 | 3 | 0.000157 |
| 3 | 44/50 | 37/50 | +7 | 10 | 3 | 0.09229 |
| 4 | 27/50 | 25/50 | +2 | 7 | 5 | 0.7744 |
| 5 | 29/50 | 26/50 | +3 | 11 | 8 | 0.6476 |
| 6 | 37/50 | 20/50 | +17 | 19 | 2 | 0.000221 |
| 7 | 20/50 | 29/50 | -9 | 3 | 12 | 0.03516 |
| 8 | 26/50 | 27/50 | -1 | 7 | 8 | 1.0000 |
| 9 | 31/50 | 15/50 | +16 | 19 | 3 | 0.000855 |

The largest gains are tasks 2, 6, 9, and 1. Task 7 is a statistically detectable
paired regression and task 8 is slightly lower. These task-level regressions are
the priority for the next failure and correction audit even though the aggregate
model clearly improves.

## Next task

Audit the 181 Student-v3 failures and their traces, beginning with the 12 task-7
episodes where Student-v2 succeeded but Student-v3 failed and the eight analogous
task-8 losses. Build a provenance-preserving candidate manifest before launching
any new teacher correction collection. Keep the formal checkpoint and all 500
results immutable.

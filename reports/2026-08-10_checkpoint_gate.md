# Student-v1 checkpoint development gate — 2026-08-10

## Purpose

Select the strongest retained Student-v1 continuation checkpoint before
designing Student-v2. This is a development gate, not a new formal success
rate. The completed 10K formal matrix supplies its matching subset without a
redundant rollout.

## Frozen protocol

- checkpoints: continuation steps 2K, 5K, and 10K;
- task suite: LIBERO Spatial, task IDs 0--9;
- development initial-state indices: 10--19 for every task;
- trials: 10 per task and 100 per checkpoint;
- evaluation seeds: 1010--1019;
- `replan=5`, maximum 220 steps, wait 10 steps, resolution 256;
- identical normalization, deterministic Flow noise, gripper mapping, and
  success predicate;
- 10K results are read from the completed 500-episode matrix.

The development panel is now fixed and must not be changed after observing
2K or 5K outcomes. Formal Student-v2 reporting will continue to use the full
50-state-per-task matrix and will disclose that indices 10--19 were used for
checkpoint and design decisions.

## Results

All three checkpoints have complete results for the 100 declared development
episodes. The 2K and 5K runs each contain 100 result JSON files, 100 traces,
and 100 videos; both matrix logs terminate with `episodes=100, exceptions=0`.

| Task | 2K | 5K | 10K |
|---:|---:|---:|---:|
| 0 | 2/10 | 2/10 | 7/10 |
| 1 | 2/10 | 3/10 | 4/10 |
| 2 | 0/10 | 1/10 | 1/10 |
| 3 | 4/10 | 6/10 | 6/10 |
| 4 | 5/10 | 3/10 | 7/10 |
| 5 | 2/10 | 2/10 | 3/10 |
| 6 | 0/10 | 0/10 | 0/10 |
| 7 | 2/10 | 5/10 | 5/10 |
| 8 | 6/10 | 6/10 | 7/10 |
| 9 | 1/10 | 2/10 | 0/10 |
| **Total** | **24/100** | **30/100** | **40/100** |

The two-sided 95% Wilson intervals are 16.7%--33.2% for 2K, 21.9%--39.6%
for 5K, and 30.9%--49.8% for 10K. These are episode success rates, not the
fraction of task IDs that were solved.

Episode-paired comparisons are:

| Contrast | Both succeed | Later gains | Later losses | Both fail | Net | Exact McNemar p |
|:---|---:|---:|---:|---:|---:|---:|
| 2K to 5K | 11 | 19 | 13 | 57 | +6 | 0.3771 |
| 2K to 10K | 15 | 25 | 9 | 51 | +16 | 0.0090 |
| 5K to 10K | 18 | 22 | 12 | 48 | +10 | 0.1214 |

The 2K and 5K log SHA-256 values are, respectively,
`76a84c53689f22764875b6eee9b9c3179379b278d1613d343e56402b24b0225a`
and `5f6518b067dfdab0c06a6bd75f7b1bfb8a69c95d57dd6dc6cb48aa2ad14c93af`.

## Decision

The frozen selection rule chooses the checkpoint with the highest total
development successes. The selected Student-v1 checkpoint is therefore 10K
at 40/100. Its improvement over 2K is statistically clear on this panel; its
10-point advantage over 5K is directionally useful but not individually
significant at the 5% level, so the checkpoint ordering should not be treated
as a precise estimate of a monotonic training curve.

The gate selects the Student-v1 training duration; it does not change model
capacity. Student-v2 matched controls will still initialize both arms from the
same accepted demonstration-only student so that the effect of all-task
demonstrations and teacher data can be separated.

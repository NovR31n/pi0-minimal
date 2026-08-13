# Student-v3 checkpoint gate and untouched validation plan

## Frozen development panel

- Model family: `STUDENT-V3-CORRECTION-SUCCESS39-10K-S7-015`.
- Candidate checkpoints: step 2K, step 5K, best-generation step 9K, and step 10K.
- Panel: LIBERO Spatial tasks 0--9, initial-state indices 10--19.
- Protocol: training seed 7, evaluation seed offset 1000, 220 maximum steps,
  10 wait steps, 5-step replanning, 256-pixel observations, CUDA.
- Selection rule: choose the checkpoint with the greatest total success count over
  all 100 frozen episodes. No per-task tuning or post-validation reselection.

## Results

| Checkpoint | Total | Per-task successes (tasks 0--9) |
| --- | ---: | --- |
| step 2K | 53/100 | 7, 4, 7, 8, 6, 4, 7, 5, 1, 4 |
| step 5K | 56/100 | 6, 2, 6, 10, 7, 6, 7, 4, 3, 5 |
| best-generation step 9K | 61/100 | 8, 3, 9, 8, 5, 7, 7, 4, 4, 6 |
| step 10K | **64/100** | 7, 7, 6, 10, 5, 6, 8, 5, 7, 3 |

All 400 result records completed with zero rollout exceptions. The frozen rule
therefore selects step 10K.

## Paired interpretation

Against Student-v2 on the identical 100 episode identities, step 10K scored
64/100 versus 47/100: 34 both succeeded, 30 only Student-v3 succeeded, 13 only
Student-v2 succeeded, and 23 both failed. The two-sided exact McNemar p-value is
0.01372.

The step-10K comparisons within Student-v3 were:

| Comparator | Both success | 10K gain | 10K loss | Both fail | Net | Exact p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| step 2K | 39 | 25 | 14 | 22 | +11 | 0.10813 |
| step 5K | 42 | 22 | 14 | 22 | +8 | 0.24299 |
| step 9K | 47 | 17 | 14 | 22 | +3 | 0.72010 |

The development panel supports the frozen choice but is not an unbiased estimate
of the chosen model. It must not be reported as the final success rate.

## Untouched validation gate

Evaluate the frozen step-10K checkpoint on tasks 0--9 and initial-state indices
21--29, using the same rollout protocol. These 90 states were not used for
Student-v3 checkpoint selection. The advancement rule is fixed before launch:

1. all 90 episodes must be complete with zero rollout exceptions; and
2. Student-v3 must achieve at least 39/90, the existing Student-v2 score on the
   exact same episode panel.

If both conditions pass, keep step 10K frozen and proceed directly to the formal
500-episode evaluation. Otherwise stop and diagnose rather than selecting another
checkpoint from the untouched results.

## Untouched validation result

The frozen step-10K checkpoint completed all 90 episodes with zero rollout
exceptions and scored **59/90 (65.6%)**, passing the predeclared 39/90 floor by
20 successes. Per-task successes for tasks 0--9 were `7, 6, 9, 8, 6, 4, 7, 3,
2, 7`.

Student-v2 scored 39/90 on the identical episode identities. The paired table was
30 both successful, 29 Student-v3-only successes, 9 Student-v2-only successes,
and 22 both failed, for a net gain of 20. The two-sided exact McNemar p-value was
0.001658. The advancement gate therefore passed without checkpoint reselection.

The formal 500-episode evaluation uses the frozen step-10K checkpoint and the same
rollout protocol. It reuses the exact 100 development and 90 untouched-validation
episodes already completed under that checkpoint, audits all 190 identities before
launch, and runs the remaining 310 episodes with `--skip-existing` recovery.

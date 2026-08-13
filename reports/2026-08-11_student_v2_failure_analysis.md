# Student-v2 formal failure-trace analysis

This report uses motion and gripper proxy signals only. The saved trace
schema has no object pose or contact state, so grasp, drop, transport, and
placement stages remain unknown until video review or richer rollouts are
available.

## Integrity

- Episodes: 500
- Successes: 249
- Failures: 251
- Exceptions: 0

## Failure proxies by task

| Task | Failures | No close | No reopen | Switches >=10 | Switches >=24 | Low endpoint | Low efficiency | Stagnation | No proxy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 18 | 0 | 0 | 16 | 6 | 5 | 12 | 0 | 2 |
| 1 | 31 | 0 | 0 | 20 | 5 | 0 | 21 | 0 | 2 |
| 2 | 31 | 0 | 0 | 22 | 5 | 0 | 5 | 0 | 8 |
| 3 | 13 | 0 | 0 | 8 | 2 | 0 | 0 | 0 | 5 |
| 4 | 25 | 0 | 0 | 21 | 3 | 5 | 1 | 2 | 3 |
| 5 | 24 | 0 | 0 | 19 | 6 | 6 | 19 | 0 | 1 |
| 6 | 30 | 0 | 0 | 26 | 20 | 0 | 0 | 2 | 4 |
| 7 | 21 | 0 | 0 | 16 | 6 | 8 | 2 | 11 | 2 |
| 8 | 23 | 0 | 0 | 18 | 4 | 2 | 8 | 0 | 1 |
| 9 | 35 | 0 | 0 | 23 | 8 | 5 | 4 | 10 | 9 |

Flags are non-exclusive. `No proxy` means that the available motion and
gripper signals did not isolate a symptom; it does not mean the policy
executed correctly.

## Outcome-level trace distributions

### Success

| Metric | P25 | Median | P75 |
|---|---:|---:|---:|
| gripper_switch_count | 1.0000 | 1.0000 | 3.0000 |
| path_efficiency | 0.3993 | 0.4438 | 0.4937 |
| endpoint_displacement | 0.3834 | 0.4225 | 0.4352 |
| stagnation_ratio | 0.0000 | 0.0000 | 0.0000 |
| action_saturation_ratio | 0.0721 | 0.0810 | 0.0919 |

### Failure

| Metric | P25 | Median | P75 |
|---|---:|---:|---:|
| gripper_switch_count | 10.0000 | 17.0000 | 24.0000 |
| path_efficiency | 0.2743 | 0.3118 | 0.3661 |
| endpoint_displacement | 0.3369 | 0.3731 | 0.4019 |
| stagnation_ratio | 0.0000 | 0.0000 | 0.0000 |
| action_saturation_ratio | 0.0545 | 0.0623 | 0.0711 |

## Representative failure videos

### Task 0

- init 45: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task00/init45/task00_init45_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_endpoint_displacement, low_path_efficiency
- init 23: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task00/init23/task00_init23_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_endpoint_displacement, low_path_efficiency
- init 21: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task00/init21/task00_init21_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_path_efficiency
- init 24: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task00/init24/task00_init24_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_path_efficiency
- init 14: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task00/init14/task00_init14_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_path_efficiency

### Task 1

- init 22: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task01/init22/task01_init22_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_path_efficiency
- init 19: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task01/init19/task01_init19_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching
- init 28: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task01/init28/task01_init28_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching
- init 05: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task01/init05/task01_init05_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching
- init 49: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task01/init49/task01_init49_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching

### Task 2

- init 43: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task02/init43/task02_init43_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_path_efficiency
- init 21: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task02/init21/task02_init21_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching
- init 42: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task02/init42/task02_init42_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching
- init 26: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task02/init26/task02_init26_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching
- init 20: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task02/init20/task02_init20_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching

### Task 3

- init 16: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task03/init16/task03_init16_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching
- init 47: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task03/init47/task03_init47_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching
- init 06: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task03/init06/task03_init06_failure.mp4`; primary `excessive_gripper_switching`; flags: excessive_gripper_switching
- init 25: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task03/init25/task03_init25_failure.mp4`; primary `excessive_gripper_switching`; flags: excessive_gripper_switching
- init 27: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task03/init27/task03_init27_failure.mp4`; primary `excessive_gripper_switching`; flags: excessive_gripper_switching

### Task 4

- init 43: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task04/init43/task04_init43_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_endpoint_displacement
- init 17: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task04/init17/task04_init17_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching
- init 42: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task04/init42/task04_init42_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching
- init 00: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task04/init00/task04_init00_failure.mp4`; primary `excessive_gripper_switching`; flags: excessive_gripper_switching, low_endpoint_displacement, high_stagnation
- init 34: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task04/init34/task04_init34_failure.mp4`; primary `excessive_gripper_switching`; flags: excessive_gripper_switching, low_endpoint_displacement

### Task 5

- init 30: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task05/init30/task05_init30_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_endpoint_displacement, low_path_efficiency
- init 44: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task05/init44/task05_init44_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_endpoint_displacement, low_path_efficiency
- init 08: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task05/init08/task05_init08_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_endpoint_displacement, low_path_efficiency
- init 28: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task05/init28/task05_init28_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_path_efficiency
- init 32: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task05/init32/task05_init32_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching

### Task 6

- init 33: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task06/init33/task06_init33_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, high_stagnation
- init 14: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task06/init14/task06_init14_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching
- init 42: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task06/init42/task06_init42_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, high_stagnation
- init 17: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task06/init17/task06_init17_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching
- init 23: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task06/init23/task06_init23_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching

### Task 7

- init 34: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task07/init34/task07_init34_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_endpoint_displacement, high_stagnation
- init 41: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task07/init41/task07_init41_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_endpoint_displacement, high_stagnation
- init 27: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task07/init27/task07_init27_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_endpoint_displacement, high_stagnation
- init 22: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task07/init22/task07_init22_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_endpoint_displacement, high_stagnation
- init 37: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task07/init37/task07_init37_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_endpoint_displacement, high_stagnation

### Task 8

- init 28: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task08/init28/task08_init28_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_endpoint_displacement, low_path_efficiency
- init 49: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task08/init49/task08_init49_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching
- init 31: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task08/init31/task08_init31_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching
- init 45: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task08/init45/task08_init45_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching
- init 15: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task08/init15/task08_init15_failure.mp4`; primary `excessive_gripper_switching`; flags: excessive_gripper_switching, low_endpoint_displacement, low_path_efficiency

### Task 9

- init 44: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task09/init44/task09_init44_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_endpoint_displacement, high_stagnation
- init 25: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task09/init25/task09_init25_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_endpoint_displacement, high_stagnation
- init 20: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task09/init20/task09_init20_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_endpoint_displacement, high_stagnation
- init 32: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task09/init32/task09_init32_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_endpoint_displacement, high_stagnation
- init 19: `results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/task09/init19/task09_init19_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, high_stagnation

## Next instrumentation requirement

Future diagnostic rollouts must save task-relevant object poses, gripper
opening, and available contact signals at every step. Only those signals
can support semantic phase labels such as failed grasp, drop, or failed
placement without relying solely on manual video review.

## Manual review of severe representatives

Contact sheets sampled every two seconds were inspected for task/init pairs
0/45, 2/43, 4/43, 6/33, 8/28, and 9/44. In all six cases the target bowl was
not visibly lifted into a stable transport phase. The gripper command traces
begin switching at approximately steps 39--50, near the first approach to the
target, and then repeatedly reverse. This small selected sample supports a
pre-grasp alignment/closure-instability hypothesis; it does not establish the
failure distribution of all 251 failed episodes.

The next diagnostic gate keeps the Student-v2 10K checkpoint and all other
inference settings fixed and tests gripper debounce values 3 and 5 on tasks
0--9 crossed with initial states 21, 28, and 43. The no-debounce baseline on
these 30 pairs is 14/30. A candidate advances only if it reaches at least
17/30, loses no more than three of the 14 baseline successes, produces a net
paired gain of at least three, and reduces median gripper switches on the 16
baseline failures. This panel was chosen using the formal traces and is a
diagnostic set, not an unbiased success-rate estimate. Any passing setting
must still be validated on the full matched protocol.

## Gripper debounce diagnostic result

Both variants completed all 30 declared pairs with zero exceptions.

| Setting | Successes | Paired gains | Paired losses | Median switches on the 16 baseline failures |
|---|---:|---:|---:|---:|
| No debounce | 14/30 | -- | -- | 24.0 |
| Debounce 3 | 15/30 | 1 | 0 | 2.5 |
| Debounce 5 | 13/30 | 1 | 2 | 2.0 |

Debounce 3 rescued only task 8/init 28. Debounce 5 rescued task 2/init 21
but lost task 4/init 28 and task 5/init 43. Neither variant reached the
predeclared 17/30 success threshold or net gain of three. The large reduction
in switch count without corresponding task recovery shows that repeated
gripper switching is mainly a downstream symptom: suppressing it does not fix
the initial grasp geometry or recovery behavior. No debounce setting advances
to expanded evaluation. The next iteration proceeds with richer state
instrumentation and teacher correction data.

The debounce 3 and debounce 5 evaluation-log SHA-256 values are respectively
`3963b9b86e93e5ab3c8441531c1b04c6a9aad0c9c75c386e4a135ce2210fd29f`
and `1481833a42da6325d999be0962a6bd21e203d25d418723ae0bd70666d2139b52`.

# Student-v3 formal failure-trace analysis

This report uses motion and gripper proxy signals only. The saved trace
schema has no object pose or contact state, so grasp, drop, transport, and
placement stages remain unknown until video review or richer rollouts are
available.

## Integrity

- Episodes: 500
- Successes: 319
- Failures: 181
- Exceptions: 0

## Failure proxies by task

| Task | Failures | No close | No reopen | Switches >=10 | Switches >=24 | Low endpoint | Low efficiency | Stagnation | No proxy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 14 | 0 | 0 | 11 | 5 | 1 | 5 | 0 | 3 |
| 1 | 19 | 0 | 0 | 13 | 3 | 0 | 6 | 0 | 4 |
| 2 | 12 | 0 | 0 | 11 | 3 | 4 | 6 | 0 | 0 |
| 3 | 6 | 0 | 0 | 4 | 3 | 0 | 3 | 0 | 0 |
| 4 | 23 | 0 | 0 | 19 | 8 | 5 | 2 | 6 | 4 |
| 5 | 21 | 0 | 0 | 20 | 4 | 8 | 14 | 0 | 1 |
| 6 | 13 | 0 | 0 | 9 | 2 | 1 | 4 | 0 | 1 |
| 7 | 30 | 0 | 0 | 19 | 13 | 14 | 14 | 9 | 3 |
| 8 | 24 | 0 | 0 | 17 | 10 | 10 | 15 | 0 | 2 |
| 9 | 19 | 0 | 0 | 14 | 7 | 1 | 17 | 0 | 1 |

Flags are non-exclusive. `No proxy` means that the available motion and
gripper signals did not isolate a symptom; it does not mean the policy
executed correctly.

## Outcome-level trace distributions

### Success

| Metric | P25 | Median | P75 |
|---|---:|---:|---:|
| gripper_switch_count | 1.0000 | 1.0000 | 3.0000 |
| path_efficiency | 0.3964 | 0.4332 | 0.4980 |
| endpoint_displacement | 0.3954 | 0.4240 | 0.4362 |
| stagnation_ratio | 0.0000 | 0.0000 | 0.0000 |
| action_saturation_ratio | 0.0778 | 0.0893 | 0.0993 |

### Failure

| Metric | P25 | Median | P75 |
|---|---:|---:|---:|
| gripper_switch_count | 10.0000 | 19.0000 | 26.0000 |
| path_efficiency | 0.2511 | 0.2812 | 0.3208 |
| endpoint_displacement | 0.3142 | 0.3514 | 0.3838 |
| stagnation_ratio | 0.0000 | 0.0000 | 0.0000 |
| action_saturation_ratio | 0.0649 | 0.0714 | 0.0805 |

## Representative failure videos

### Task 0

- init 06: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task00/init06/task00_init06_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_endpoint_displacement, low_path_efficiency
- init 07: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task00/init07/task00_init07_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_path_efficiency
- init 23: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task00/init23/task00_init23_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching
- init 01: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task00/init01/task00_init01_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching
- init 45: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task00/init45/task00_init45_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching

### Task 1

- init 49: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task01/init49/task01_init49_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching
- init 41: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task01/init41/task01_init41_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching
- init 28: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task01/init28/task01_init28_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching
- init 21: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task01/init21/task01_init21_failure.mp4`; primary `excessive_gripper_switching`; flags: excessive_gripper_switching, low_path_efficiency
- init 47: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task01/init47/task01_init47_failure.mp4`; primary `excessive_gripper_switching`; flags: excessive_gripper_switching, low_path_efficiency

### Task 2

- init 44: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task02/init44/task02_init44_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_endpoint_displacement, low_path_efficiency
- init 37: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task02/init37/task02_init37_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_path_efficiency
- init 45: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task02/init45/task02_init45_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_path_efficiency
- init 32: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task02/init32/task02_init32_failure.mp4`; primary `excessive_gripper_switching`; flags: excessive_gripper_switching, low_endpoint_displacement, low_path_efficiency
- init 11: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task02/init11/task02_init11_failure.mp4`; primary `excessive_gripper_switching`; flags: excessive_gripper_switching, low_endpoint_displacement, low_path_efficiency

### Task 3

- init 47: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task03/init47/task03_init47_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching
- init 38: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task03/init38/task03_init38_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching
- init 03: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task03/init03/task03_init03_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching
- init 33: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task03/init33/task03_init33_failure.mp4`; primary `excessive_gripper_switching`; flags: excessive_gripper_switching, low_path_efficiency
- init 44: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task03/init44/task03_init44_failure.mp4`; primary `low_path_efficiency`; flags: low_path_efficiency

### Task 4

- init 08: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task04/init08/task04_init08_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_endpoint_displacement, high_stagnation
- init 46: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task04/init46/task04_init46_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_endpoint_displacement, high_stagnation
- init 00: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task04/init00/task04_init00_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_endpoint_displacement, high_stagnation
- init 45: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task04/init45/task04_init45_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_endpoint_displacement
- init 17: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task04/init17/task04_init17_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, high_stagnation

### Task 5

- init 32: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task05/init32/task05_init32_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_endpoint_displacement, low_path_efficiency
- init 40: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task05/init40/task05_init40_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_endpoint_displacement, low_path_efficiency
- init 30: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task05/init30/task05_init30_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_path_efficiency
- init 44: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task05/init44/task05_init44_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching
- init 34: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task05/init34/task05_init34_failure.mp4`; primary `excessive_gripper_switching`; flags: excessive_gripper_switching, low_endpoint_displacement, low_path_efficiency

### Task 6

- init 18: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task06/init18/task06_init18_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching
- init 36: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task06/init36/task06_init36_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching
- init 34: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task06/init34/task06_init34_failure.mp4`; primary `excessive_gripper_switching`; flags: excessive_gripper_switching, low_endpoint_displacement, low_path_efficiency
- init 20: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task06/init20/task06_init20_failure.mp4`; primary `low_path_efficiency`; flags: low_path_efficiency
- init 32: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task06/init32/task06_init32_failure.mp4`; primary `low_path_efficiency`; flags: low_path_efficiency

### Task 7

- init 10: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task07/init10/task07_init10_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_endpoint_displacement, low_path_efficiency
- init 26: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task07/init26/task07_init26_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_endpoint_displacement, low_path_efficiency
- init 21: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task07/init21/task07_init21_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_endpoint_displacement, low_path_efficiency
- init 03: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task07/init03/task07_init03_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_endpoint_displacement, low_path_efficiency
- init 28: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task07/init28/task07_init28_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_endpoint_displacement, high_stagnation

### Task 8

- init 13: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task08/init13/task08_init13_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_endpoint_displacement, low_path_efficiency
- init 38: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task08/init38/task08_init38_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_endpoint_displacement, low_path_efficiency
- init 32: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task08/init32/task08_init32_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_endpoint_displacement, low_path_efficiency
- init 05: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task08/init05/task08_init05_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_endpoint_displacement, low_path_efficiency
- init 04: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task08/init04/task08_init04_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_endpoint_displacement, low_path_efficiency

### Task 9

- init 12: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task09/init12/task09_init12_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_endpoint_displacement, low_path_efficiency
- init 19: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task09/init19/task09_init19_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_path_efficiency
- init 35: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task09/init35/task09_init35_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_path_efficiency
- init 37: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task09/init37/task09_init37_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_path_efficiency
- init 05: `results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018/task09/init05/task09_init05_failure.mp4`; primary `severe_gripper_switching`; flags: excessive_gripper_switching, severe_gripper_switching, low_path_efficiency

## Next instrumentation requirement

Future diagnostic rollouts must save task-relevant object poses, gripper
opening, and available contact signals at every step. Only those signals
can support semantic phase labels such as failed grasp, drop, or failed
placement without relying solely on manual video review.

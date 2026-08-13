# Balanced official-teacher distillation — 2026-08-06

## Stage outcome

The balanced-teacher data stage and the formal Student-v1 10K continuation
run completed. A paired ten-task gate improved from 3/10 for the unchanged
demonstration-only student to 7/10 for Student-v1. Both students subsequently
completed the identical 500-episode matrix with zero exceptions.

The unchanged student scored 108/500 = 21.6%, while Student-v1 scored
214/500 = 42.8%. Distillation therefore added 106 net paired successes and
improved the primary success rate by 21.2 percentage points. The official
teacher remains substantially stronger at 491/500 = 98.2%.

## Balanced teacher collection

- Official teacher: `pi05_libero` served by OpenPI with CUDA JAX.
- Protocol: LIBERO Spatial, `replan=5`, maximum 220 steps.
- Initial pass: 100 episodes covering task IDs 0–9 and initial-state indices
  0–9.
- Initial-pass result: 96 successes, 4 model failures, and zero infrastructure
  errors.
- Directed replacements: task 3 / init 10 and task 4 / init 10–12 all
  succeeded.
- Final raw artifact: 104 complete episodes, 100 successes, 4 failures, and
  exactly 10 successful trajectories per task.
- Raw artifact size: approximately 268 MiB.
- Artifact directory:
  `artifacts/TEACHER-COLLECT-BALANCED100-001`.

The conversion tool now skips completed model-failure episodes by default and
hard-fails if an expected task has fewer than the requested number of
successes. The formal conversion selected exactly ten successes for every
task.

Converted teacher data:

- path: `data/distillation/teacher_balanced100_v1.npz`;
- episodes: 100;
- teacher-query samples: 2,107;
- tensor shapes: images `(2107,2,3,224,224)`, states `(2107,7)`, actions
  `(2107,10,7)`;
- finite state/action values: true;
- canonical action range: `[-1,1]`;
- SHA-256:
  `2a3a8fb677028d76aceb9b4e2130a917f4b000adee90ca76a5155fef5e21e34c`.

## Resource-safe formal training

The accepted demonstration condition cache is approximately 31 GiB and the
new teacher cache is approximately 4.7 GiB. Concatenating both caches into a
third tensor would exceed the server's 62 GiB RAM budget. Student-v1 therefore
uses segmented condition-cache indexing: mixed batches are gathered from the
two source tensors without copying the full caches.

Additional formal safeguards:

- demonstration and teacher sources each receive exactly 50% sampling mass;
- teacher sampling mass is equal across all ten tasks, independent of query
  count or trajectory duration;
- validation remains the original held-out demonstration split;
- initialization at continuation step 0 participates in both best-loss and
  best-generation checkpoint selection;
- checkpoints at 2K, 5K, and 10K are retained independently;
- the original official teacher weights are never loaded into the student.

Full-scale 100-step preflight:

- demonstration training samples: 12,352;
- teacher training samples: 2,107;
- validation samples: 1,344;
- batch size: 16;
- restore verification: passed;
- training CUDA peak: approximately 1,399 MiB;
- elapsed training time after cache setup: 34.29 seconds.

## Formal Student-v1 10K run

Experiment: `STUDENT-V1-DISTILL-BAL100-10K-S7-001`.

- initialization: accepted demonstration-only Flow student checkpoint at its
  step 10,000;
- continuation steps: 10,000;
- learning rate: `3e-5` with linear decay;
- batch size: 16;
- state noise: 0.05;
- seed: 7;
- restore verification: passed;
- elapsed training time after cache loading: 899.30 seconds;
- training CUDA peak: 1,396.66 MiB.

The held-out demonstration action MAE improved from `0.203824` before
continuation to `0.193397` at 10K, an improvement of approximately 5.1%.
The Flow validation loss regressed from `0.181767` to `0.213562`; therefore
loss-best remains the initialization while generation-best is the 10K
checkpoint. Closed-loop evaluation, not either offline metric alone, decides
acceptance.

Hashes:

- `best_generation.pt`:
  `73cab3048a38ef3ee396a281354d2555a48b981b6f28f8565b210e86e753533f`;
- 2K snapshot:
  `180adeb519e595ff3ba957137e22bc7dcbc0722b03f19fa2f90b78907b3a6e6c`;
- 5K snapshot:
  `efa055842795bf36b262aef64890934b34e1a8d5157c73053183c84fd18ed2c9`;
- 10K snapshot:
  `d8ccc9746eabc27578400d787235236fa610d392dbbbd42943de03db44d6d925`;
- result JSON:
  `f6a81582220138b0b402989ddad5b48fdd4ec37f6a3cc0f7f883cf8ffa1d9ba8`.

## Paired ten-task gate

Both students used the same task IDs, initial-state index 0, evaluation seeds,
maximum steps, resolution, success predicate, and `replan=5`.

| Task | Original student | Distilled Student-v1 |
|---:|:---:|:---:|
| 0 | pass | pass |
| 1 | fail | fail |
| 2 | fail | fail |
| 3 | pass | pass |
| 4 | fail | pass |
| 5 | fail | pass |
| 6 | fail | pass |
| 7 | fail | pass |
| 8 | pass | fail |
| 9 | fail | pass |
| **Total** | **3/10** | **7/10** |

Student-v1 gained five paired successes and regressed on one state, for a net
gain of four. Tasks 5, 6, and 9 were absent from the original demonstration
student's training set and all succeeded in the Student-v1 gate.

## Formal three-way evaluation

The unchanged student and Student-v1 each completed all 500 episodes with zero
exceptions. Both evaluated initial-state indices 0--49 for every Spatial task,
using `replan=5`, a 220-step limit, and evaluation seeds 1000--1049. The clean
official `pi05_libero` baseline used the same task and initial-state matrix.

| Task | Original student | Student-v1 | Official teacher |
|---:|---:|---:|---:|
| 0 | 32/50 = 64% | 37/50 = 74% | 50/50 = 100% |
| 1 | 0/50 = 0% | 13/50 = 26% | 50/50 = 100% |
| 2 | 0/50 = 0% | 9/50 = 18% | 50/50 = 100% |
| 3 | 40/50 = 80% | 35/50 = 70% | 49/50 = 98% |
| 4 | 12/50 = 24% | 30/50 = 60% | 47/50 = 94% |
| 5 | 0/50 = 0% | 21/50 = 42% | 48/50 = 96% |
| 6 | 0/50 = 0% | 8/50 = 16% | 49/50 = 98% |
| 7 | 12/50 = 24% | 27/50 = 54% | 50/50 = 100% |
| 8 | 12/50 = 24% | 30/50 = 60% | 50/50 = 100% |
| 9 | 0/50 = 0% | 4/50 = 8% | 48/50 = 96% |
| **Total** | **108/500 = 21.6%** | **214/500 = 42.8%** | **491/500 = 98.2%** |

Two-sided 95% Wilson intervals are 18.2%--25.4% for the original student,
38.5%--47.2% for Student-v1, and 96.6%--99.1% for the official teacher.
Student-v1 improved nine tasks and regressed only task 3, from 80% to 70%.

The episode-paired contingency table for the two students is:

- both succeed: 81;
- Student-v1 succeeds and the original fails: 133;
- original succeeds and Student-v1 fails: 27;
- both fail: 259.

The exact two-sided McNemar p-value is approximately `4.98e-18`. On the five
tasks present in the original student's demonstration training set (0, 3, 4,
7, and 8), success improved from 108/250 = 43.2% to 159/250 = 63.6%. On the
other five tasks, Student-v1 established 55/250 = 22.0% success from an
original 0/250.

For the 81 initial states where both students succeeded, Student-v1 reduced
mean completion length from 111.7 to 98.7 steps. Its mean first-difference
action RMS decreased from 0.1741 to 0.1461, second-difference RMS from 0.2552
to 0.2028, and high-frequency energy ratio from 0.0326 to 0.0217. Stable
inference latency remained effectively unchanged at approximately 0.277
seconds per call.

Each student artifact contains 500 result JSON files, 500 traces, and 500
videos. Both matrix logs terminate with `episodes=500, exceptions=0`.
The log SHA-256 values are:

- original student:
  `ea7149507aabb7965d311457a25c82e87e13491f65d197c34248de144296bbd6`;
- Student-v1:
  `1d3daad4e0324fe9246448f31deabfcbeb3594e042d5fa7b7ecf7117ec6e05a1`.

The primary milestone of improving the existing student by at least ten
percentage points is satisfied. The 85% formal target is not: Student-v1 is
42.2 percentage points below it and 55.4 points below the official teacher.
Stage 9 still requires closed-loop screening of the retained 2K and 5K
snapshots before selecting the initialization and data mix for Student-v2.

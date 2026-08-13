# Official-teacher distillation smoke test — 2026-08-06

## Outcome

The first end-to-end distillation smoke test passed. The official
`pi05_libero` policy produced one successful, auditable teacher trajectory;
that trajectory was converted to the compact Flow student's existing action
contract; a 100-step source-balanced continuation run completed; and the
resulting student checkpoint completed one independent closed-loop rollout.

This is a functional gate, not a success-rate measurement. Only one task and
one initial state were used for the teacher and rollout smoke tests.

## Teacher collection

- Teacher: official `pi05_libero` OpenPI policy served over its WebSocket API.
- Suite/task/state: LIBERO Spatial, task 0, initial-state index 0.
- Result: success in 77 environment steps and 16 teacher queries.
- Stored observations: base RGB, wrist RGB, 7D student state, and 8D official
  teacher state.
- Stored actions: original official action chunks and separately canonicalized
  10x7 student targets.
- Atomicity: `_COMPLETE` is written only after all arrays, metadata, and video
  have been saved; incomplete and infrastructure-error episodes are not model
  failures.
- Episode artifact:
  `artifacts/TEACHER-COLLECT-SMOKE-001/task00/init00`
- `teacher_episode.npz` SHA-256:
  `76bb88ecc9e9957733e45ff886630105929360009df2ecd403edfcc373ef5511`
- `result.json` SHA-256:
  `492e89c806d162fe2688c95ac6880e3f99054faa26adb68c4e77fd98e27253f3`

The first contract check correctly rejected small official-policy excursions
outside `[-1, 1]`. The collector was changed to preserve those raw values for
audit and execution while clipping only the canonical student target to the
LIBERO student action domain. The raw range was approximately
`[-1.00722, 1.00692]`; the canonical target range was approximately
`[-0.98499, 1.0]`.

The converted training file is
`data/distillation/teacher_smoke_task0_init0.npz`. It contains 16 query-level
samples with images `(16, 2, 3, 224, 224)`, states `(16, 7)`, actions
`(16, 10, 7)`, action-valid masks, prompts, and disjoint synthetic episode IDs.
Its image keys exactly match the student specification: `base_0_rgb` and
`wrist_0_rgb`.

## Student-v1 training smoke test

- Initialization: accepted demonstration-only Flow checkpoint at step 10,000.
- Demonstration data: 16 train samples from episode 0.
- Teacher data: 16 train samples from the teacher smoke trajectory.
- Sampling: 50% probability mass per source; teacher samples are train-only.
- Validation: the original held-out demonstration episode 14 remained
  authoritative and was not mixed with teacher samples.
- Training: 100 steps, batch size 4, learning rate `3e-5`, seed 7.
- Resume/initialization verification: passed.
- Runtime after frozen-condition caching: 34.09 seconds.
- Training peak CUDA allocation: 1376.02 MiB.

Validation results:

| Metric | Before continuation | Best/final |
|---|---:|---:|
| Flow validation loss | 0.165850 | best 0.165209 at step 60 |
| Normalized action MAE | 0.191500 | best 0.178439 at step 80 |
| Prediction clip ratio | 0.092664 | final 0.063707 |
| Gripper accuracy | 0.986486 | final 0.979730 |

The best normalized action MAE improved by about 6.8% in this tiny smoke run.
The final validation loss was 0.172566, so the best checkpoints rather than the
last checkpoint must be retained.

- Teacher training file SHA-256:
  `e99fe8528a5ad03b82cc15c65468355c57d7f5cdc837cfa498a629907f75f650`
- Best-generation checkpoint SHA-256:
  `b841aea99d044c5b697bfac0eb0ad5282a44190c32233865bcea2ba3c6d8fc24`
- Artifact directory:
  `artifacts/STUDENT-V1-DISTILL-SMOKE-001`

## Independent closed-loop smoke rollout

The step-80 `best_generation.pt` checkpoint completed task 0 / initial state 0
successfully:

- success: true;
- environment steps: 77 of 220;
- inference calls: 16;
- stable mean inference: 0.2801 seconds;
- mean inference: 0.2993 seconds;
- first inference: 0.5868 seconds;
- peak CUDA allocation: 5854.72 MiB;
- video:
  `artifacts/STUDENT-V1-DISTILL-SMOKE-001/rollout_task0_init0/task00_init00_success.mp4`;
- rollout result SHA-256:
  `a875c011a6093a7a24650bee2839135d1461398fffb6ca59afbf03f659a25481`.

The matching teacher also took 77 steps, but that coincidence is not evidence
of behavioral equivalence. A paired multi-task evaluation is still required.

## Gate decision

The implementation gate passed: collection is resumable, raw teacher actions
remain auditable, converted tensors satisfy the student contract, mixed-source
training runs within a 4090 budget, checkpoint restoration is verified, and an
independent rollout succeeds.

Do not report a distilled success rate yet. The next controlled stage is to
collect balanced successful teacher trajectories across all ten LIBERO Spatial
tasks, then run a longer Student-v1 experiment and compare it against the
unchanged demonstration-only student under the same task, initial-state,
replanning, and success protocol.

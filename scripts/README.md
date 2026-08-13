# Entry points

- `prepare_libero_data.py`: freeze the episode split and training statistics.
- `create_libero_smoke_subset.py`: extract a real two-camera smoke subset.
- `smoke_paligemma_backbone.py`: load the pinned gated checkpoint and validate
  condition-memory shape, finiteness, latency, freezing, and CUDA memory.
- `smoke_flow_stack.py`: run the real frozen PaliGemma condition encoder plus
  trainable state projection, action/time embedding, and eight-layer flow
  expert through one CUDA forward/backward memory probe.
- `overfit_flow_single_batch.py`: optimize one fixed flow problem from a real
  LIBERO sample and require a documented order-of-magnitude loss reduction.
- `overfit_ar_single_batch.py`: apply the same real-sample overfit gate to the
  matched continuous autoregressive baseline.
- `train_flow_small.py`: train either committed generation stack on an
  episode-disjoint real LIBERO subset, evaluate validation loss, and atomically
  save resumable `latest.pt`, loss-selected `best.pt`, and generated-action-
  selected `best_generation.pt` checkpoints.
- `evaluate_validation_actions.py`: load one or more existing checkpoints and
  compare their deterministic generated action chunks with a fixed held-out
  subset without updating model weights.
- `rollout_libero_single.py`: load either trained policy directly into one
  off-screen LIBERO environment and record a closed-loop episode, trace, and
  replay video.
- `replay_libero_demonstration.py`: restore one official LIBERO HDF5 simulator
  state and replay its expert actions through the committed gripper conversion
  boundary as an environment/action-contract gate.
- `collect_teacher_libero.py`: query a separately running official OpenPI
  `pi05_libero` server and save one atomic, resumable distillation artifact per
  task/initial-state episode. Infrastructure errors abort collection and are
  never counted as model failures.
- `select_teacher_correction_seeds.py`: select failed schema-v3 demonstration
  rollouts only, choose a takeover step before dense gripper switching, and
  bind every seed to its source trace SHA-256 in a JSON manifest.
- `collect_teacher_corrections.py`: restore each manifest seed and let the
  official teacher take over, saving atomic teacher episodes with source-state
  provenance. Install the client from an OpenPI checkout into the project
  environment first with
  `uv pip install --python .venv/bin/python "$OPENPI_ROOT/packages/openpi-client"`.
- `prepare_teacher_distillation_data.py`: gather complete teacher episode
  artifacts into the validated compact training NPZ contract; successful
  episodes are required by default.
- `prepare_student_v2_all10_data.sh`: build and validate the fixed 434-episode,
  ten-task Student-v2 demonstration NPZ with 32 samples per episode and an
  atomic final rename.
- `rollout_libero_matrix.py`: load a checkpoint and frozen backbone once, then
  run a serial task/initial-state matrix with a fresh environment per episode.
  New rollouts use trace schema v3, which action-aligns pre/post MuJoCo state,
  OSC controller goals, numeric object/robot observations, and success flags.
  `--demonstration-root` switches the seed source from benchmark evaluation
  states to task-matched official demonstration starts for correction-data
  collection without reusing the formal evaluation initial-state matrix.
- `run_student_v1_checkpoint_gate.sh`: sequentially screen the retained 2K and
  5K Student-v1 checkpoints on the frozen ten-task development matrix; the 10K
  comparison is reused from its completed formal matrix.
- `run_student_v2_training.sh`: run either arm of the matched all-ten-task
  Student-v2 demonstration-only versus balanced-teacher experiment, with a
  separate 100-step preflight and 10K formal mode.
- `summarize_rollout_trace.py`: compute versioned action, trajectory (when
  available), and runtime metrics from one rollout trace.
- `aggregate_rollout_metrics.py`: generate an episode CSV, per-model confidence
  intervals, and paired model differences from matched rollout result/trace
  pairs.
- `analyze_failure_traces.py`: audit a rollout matrix with conservative motion
  and gripper proxy flags, write machine-readable JSON, and index
  representative failure videos in Markdown. Object-contact, grasp, drop, and
  placement stages remain unresolved for the historical formal matrix because
  its older traces do not contain object poses or contacts. Schema-v3 rollouts
  add object poses but still do not claim contact semantics without validation.

Example failure audit:

```bash
.venv/bin/python scripts/analyze_failure_traces.py \
  --results-root results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001 \
  --output-json results/STUDENT-V2-ALL10-DISTILL-10K-S7-R5-500-001/failure_analysis.json \
  --output-markdown reports/2026-08-11_student_v2_failure_analysis.md \
  --expected-episodes 500 \
  --expected-failures 251
```

Teacher-correction collection should create a fresh environment, run the same
settling steps, restore the saved physical/controller state, and then let the
teacher take over. Restored states are valid recovery seeds; they are not
described as exact counterfactual replays of the student's next action because
robosuite also maintains transient controller and observable caches.

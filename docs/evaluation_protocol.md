# Unified evaluation and statistical protocol

## Unit of observation

A rollout is identified by model type, training seed, task ID, environment
seed, and initial-state index. Flow and AR must use paired
training-seed/task/environment-seed/init tuples.
An episode is never discarded because it fails or reaches the step limit.

Metrics are first computed per episode. Report task-level summaries before a
macro average across tasks, so tasks with more recorded steps cannot dominate.
Training-seed results remain visible; rollout seeds are not treated as
independent training replicates.

## Action smoothness

For executed actions transformed back into the frozen training normalization
space, `a[t,d]`, excluding padded scalars:

- first-difference RMS: `sqrt(mean((a[t]-a[t-1])²))`;
- second-difference RMS: `sqrt(mean((a[t]-2a[t-1]+a[t-2])²))`;
- high-frequency energy ratio: after subtracting each dimension's episode
  mean, sum real-FFT power at frequencies at or above 0.25 cycles/sample and
  divide by total non-DC power.

Report the aggregate over all valid scalar differences plus per-dimension
values. A lower value means smoother under that metric; it does not imply
higher task success.

## Trajectory and action events

- path length: sum of Euclidean end-effector position increments;
- endpoint displacement: distance between first and last recorded positions;
- path efficiency: endpoint displacement divided by path length, or zero for a
  stationary path;
- stagnation ratio: fraction of position increments no greater than `1e-4 m`;
- action saturation: fraction of executed normalized action scalars with
  magnitude at least `0.999`;
- gripper switches: sign changes after ignoring values in the `[-0.1,0.1]`
  deadband.

## Runtime and resources

Chunk latency includes image preprocessing, frozen PaliGemma encoding, state
projection, and complete action generation. Report first call separately from
the mean over all later calls, plus overall mean, median, p95, total calls,
peak CUDA allocated memory, and peak CUDA reserved memory. Use the same GPU,
precision, horizon, replan interval, and synchronization boundary.

## Frozen fair-comparison boundary

The primary comparison fixes:

- PaliGemma model ID and full revision;
- two image views, prompt, state representation, normalization, and masks;
- train/validation episodes and data fingerprint;
- 43.94M trainable-parameter capacity (difference 0.0085%);
- action horizon 10, action width 7, optimizer, schedule, batch size, training
  steps, training seeds, task IDs, environment seeds, and initial states;
- deterministic primary generation (flow uses fixed-seed Gaussian initial
  noise; AR uses distribution means);
- rollout maximum steps, wait steps, replan interval, image resolution, and
  action safety clipping.

Only the generation mechanism, its native objective, and its required
sequential integration/generation differ. Hyperparameter tuning, if performed,
uses an equal declared trial budget and never formal evaluation rollouts.

## Statistical reporting

Success is reported with two-sided 95% Wilson intervals. Continuous episode
metrics report count, mean, standard deviation, median, and a deterministic
10,000-resample 95% percentile-bootstrap interval. Primary model contrasts use
paired task/seed/init differences and paired bootstrap intervals; an interval
containing zero is described as no stable observed difference. Per-task and
per-training-seed tables accompany macro summaries. No conclusion is based on
one smoke checkpoint, one task, or one seed.

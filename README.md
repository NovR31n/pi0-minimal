# pi0-minimal

An independent, compact π0-style vision-language-action policy for controlled
flow-matching and autoregressive experiments on LIBERO.

The project implements the policy, data contracts, training loop, closed-loop
rollout tooling, teacher-distillation pipeline, checkpoint gates, and failure
auditing without importing OpenPI's π0 model, flow loss, or action sampler.

## Main result

The frozen Student-v3 checkpoint reached **319/500 (63.8%)** on LIBERO
Spatial, compared with **249/500 (49.8%)** for Student-v2 on the same 500
task/state identities. All 500 Student-v3 rollouts completed without an
exception.

| Policy | Success | Rate |
| --- | ---: | ---: |
| Direct-video compact student | 108/500 | 21.6% |
| Student-v1, balanced teacher distillation | 214/500 | 42.8% |
| Student-v2, all-task balanced distillation | 249/500 | 49.8% |
| **Student-v3, recovery-data continuation** | **319/500** | **63.8%** |

Student-v3 improved by 70 successes and 14.0 percentage points over
Student-v2. The paired outcomes were 196 both-success, 123 Student-v3-only,
53 Student-v2-only, and 128 both-fail; the two-sided exact McNemar p-value was
`1.371e-7`. The Wilson 95% confidence interval for Student-v3 was
59.5%–67.9%.

See the [formal evaluation report](reports/2026-08-13_student_v3_formal_500.md)
and [checkpoint-gate report](reports/2026-08-13_student_v3_checkpoint_gate.md)
for the full protocol and per-task results.

## Model

The compact Flow policy uses:

- two 224 × 224 RGB views and a language instruction;
- a frozen `google/paligemma-3b-pt-224` condition encoder;
- one projected 7D robot-state token;
- an eight-layer, width-512 action expert with bidirectional self-attention
  and cross-attention to cached vision-language context;
- ten 7D action tokens trained with a masked flow-matching objective;
- ten Euler integration steps at inference time.

The trainable compact policy is approximately 44M parameters. A matched
continuous autoregressive baseline shares the observation encoder, state
conditioning, action horizon, decoder scale, and evaluation protocol.

```text
base RGB ─┐
wrist RGB ├─ frozen PaliGemma ─ context projection ─┐
language ─┘                                         ├─ action expert ─ 10 × 7 actions
robot state ─────────────── state projection ───────┘
```

The detailed tensor and attention contracts are in
[`docs/architecture`](docs/architecture/).

## Training progression

Student-v3 continued from the accepted Student-v2 checkpoint with three
independently balanced sources:

- 50% all-task LIBERO demonstrations;
- 25% balanced successful trajectories from the official teacher;
- 25% successful teacher recoveries from student-induced states.

The recovery dataset contained 39 successful trajectories and 785 teacher
queries. Teacher failures and infrastructure errors were excluded from the
student training index. Source/task probabilities were explicit rather than
being determined by raw trajectory length.

Large frozen condition caches are kept as separate segments. Mixed batches
gather from those segments without concatenating another full cache, keeping
the continuation run within a 62 GiB host-memory budget. Peak trainable-model
CUDA allocation was approximately 1.4 GiB; closed-loop inference, including
the frozen backbone, used approximately 5.9 GiB.

## Evaluation discipline

The reported result follows a staged protocol:

1. retain 2K, 5K, 9K best-generation, and 10K checkpoints;
2. choose one checkpoint on a frozen 100-episode development panel;
3. require that checkpoint to pass a predeclared 90-episode gate without
   reselection;
4. evaluate the frozen checkpoint on the complete 500-episode matrix;
5. preserve task/state identities, exceptions, traces, videos, configuration,
   checkpoint step, and hashes in the private artifact store.

The development panel is not reported as the final success rate. Offline
action MAE is treated as a health metric rather than a closed-loop checkpoint
selector.

## Repository layout

```text
configs/              Model, data, task, and experiment configurations
docs/architecture/    Architecture, tensor, mask, and objective specifications
reports/              Versioned experiment reports and failure summaries
scripts/              Data, training, rollout, aggregation, and audit tools
src/pi0_minimal/      Independent model and data implementation
tests/                CPU unit and integration tests
```

Datasets, gated weights, checkpoints, cached condition tensors, complete
rollout traces, and videos are intentionally excluded from Git.

## Setup

Python 3.11 and [uv](https://docs.astral.sh/uv/) are recommended.

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check .
```

For LIBERO rollout dependencies:

```bash
uv sync --extra dev --extra libero
export PYTHONPATH=/path/to/LIBERO
export MUJOCO_GL=egl
```

Unit tests use a deterministic dummy condition encoder and do not require
PaliGemma access, LIBERO, CUDA, datasets, or checkpoints. Real training and
rollout require acceptance of the gated PaliGemma license and a separately
installed LIBERO checkout.

## Reproducibility

Core entry points include:

- `scripts/prepare_libero_data.py` for episode-disjoint splits and statistics;
- `scripts/train_flow_small.py` for resumable Flow/AR training;
- `scripts/rollout_libero_matrix.py` for serial closed-loop matrices;
- `scripts/aggregate_rollout_metrics.py` for confidence intervals and paired
  comparisons;
- `scripts/analyze_failure_traces.py` for conservative failure proxies;
- `scripts/collect_teacher_corrections.py` and
  `scripts/prepare_teacher_distillation_data.py` for provenance-preserving
  teacher recovery data.

The shell launchers document the exact accepted experiment configurations.
They derive the project directory from their own location; external dataset,
cache, and OpenPI client locations are supplied through environment variables.

## Limitations

- Results are from LIBERO simulation, not a physical robot.
- The formal Student-v3 result uses one training seed.
- The frozen PaliGemma backbone and official teacher are pretrained external
  components; this repository independently implements the compact student.
- Task 7 regressed relative to Student-v2 despite the aggregate gain.
- Artifact hashes are recorded in the reports, but licensed datasets and
  checkpoints cannot be redistributed here.

## References

- [π0: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164)
- [OpenPI](https://github.com/Physical-Intelligence/openpi)
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [PaliGemma model card](https://huggingface.co/google/paligemma-3b-pt-224)

This is an independent research implementation and is not affiliated with
Physical Intelligence, Google, or the LIBERO authors.

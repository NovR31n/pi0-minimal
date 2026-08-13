# P1-05: simplification decisions

## Goal

The project is an independent compact π0-style reproduction designed for one
24GB RTX 4090 and a controlled flow-matching versus autoregressive comparison.
It is not a claim to reproduce Physical Intelligence's data scale or full
3.3B-parameter training recipe.

## Decision table

| Area | Full π0 | Compact implementation | Reason and consequence |
|---|---|---|---|
| VLM | PaliGemma 3B, trained with policy | PaliGemma 3B frozen initially | fits 24GB and isolates action-generator comparison |
| Expert interaction | two experts inside one shared-attention transformer | small decoder cross-attends to cached VLM/state memory | much simpler independent implementation; not weight-compatible with π0 |
| Action expert | ~300M, width 1024, depth 18 | ~40–50M, width 512, depth 8 | faster training and matched AR baseline |
| Cameras | embodiment-dependent, often 3 | LIBERO base + wrist, 2 views | uses available informative views without dummy tokens |
| State/action width | padded to 32 in OpenPI default | native LIBERO dimension 7 | removes unused padded outputs |
| Horizon | 50 in paper | 10 | matches available LIBERO/OpenPI experience and lowers memory |
| Prompt length | 48 in OpenPI π0 | 64 | covers LIBERO instructions with small overhead |
| Flow steps | 10 | 10 | preserved for primary experiment |
| Time distribution | shifted Beta | shifted Beta default; uniform only as ablation | preserves paper training emphasis |
| Training | large cross-embodiment pre-training/post-training | LIBERO supervised training/fine-tuning | conclusions apply only to this benchmark |
| Precision | mixed precision implementation-dependent | BF16 modules, FP32 noise/loss/normalization | 4090 memory safety and numerical stability |
| Backbone tuning | trainable/full or LoRA | frozen milestone first; LoRA optional later | main comparison must work before adding another variable |

## Fair flow-versus-AR boundary

The future autoregressive baseline must share:

- identical frozen `ConditionEncoder`;
- identical context/state projections;
- action-expert width, depth, heads, FFN, and parameter budget as closely as
  possible;
- the same normalized data, action horizon, tasks, seeds, optimizer budget,
  and LIBERO execution protocol.

Only the action-generation mechanism changes:

- flow: bidirectional noisy-action tokens, velocity loss, iterative Euler
  sampling;
- AR: shifted previous-action inputs, causal action-token mask, continuous
  next-action likelihood or regression target.

Discrete action bins are not the primary AR baseline because quantization would
confound the action-smoothness comparison. Any discrete AR experiment must be
labeled as a separate secondary ablation.

## Claims that are allowed after implementation

If all P1–P7 gates pass, the project may claim:

- an independent compact π0-style implementation;
- independent flow-matching objective and sampler;
- a matched continuous autoregressive baseline;
- controlled empirical comparison on the reported LIBERO subset.

It may not claim:

- reproduction of π0's original large-scale pre-training;
- weight-compatible reimplementation of the 3.3B π0 architecture;
- reproduction of Physical Intelligence's private cross-embodiment results;
- that flow matching is smoother or better unless formal experiments support
  the statement.

## Resource gates

Before loading PaliGemma:

1. record acceptance of the model license and verified checkpoint access;
2. run an inference-only memory probe;
3. verify frozen-backbone plus one compact expert forward/backward remains
   below 22GB peak memory;
4. reduce expert layers or use activation checkpointing before changing the
   scientific protocol;
5. never silently substitute a different VLM in only one experimental arm.

## Deferred features

The following are deliberately outside the first implementation:

- full π0 two-expert weight routing;
- full VLM fine-tuning;
- LoRA on the VLM;
- Heun or adaptive ODE solvers;
- action horizons above 10;
- more than two camera views;
- LIBERO suites beyond the selected Spatial tasks;
- real-robot deployment.

Each deferred feature requires a new experiment ID and may not be folded into
the primary comparison without updating this specification.

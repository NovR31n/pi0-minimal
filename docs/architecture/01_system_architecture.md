# P1-01: system architecture

## Purpose

This document defines the first independently implementable compact π0-style
policy. It preserves the paper's multimodal conditioning, state conditioning,
bidirectional action-chunk modeling, flow-matching objective, and iterative
sampling. It deliberately replaces the full two-expert shared-attention
transformer with a smaller action decoder that cross-attends to cached
vision-language context.

Primary references:

- π0 paper: <https://arxiv.org/abs/2410.24164>
- Physical Intelligence paper PDF:
  <https://www.physicalintelligence.company/download/pi0.pdf>
- OpenPI reference commit inspected for this specification: `3e0325a`
- PaliGemma model card:
  <https://huggingface.co/google/paligemma-3b-pt-224>

The PaliGemma checkpoint is gated by Google's usage license. Formal training
must not begin until model access and license acceptance are recorded. Unit
tests use a dummy condition encoder and require no gated weights.

## End-to-end data flow

```text
LIBERO observation at control step t
│
├── base RGB image ─┐
├── wrist RGB image ├── shared PaliGemma vision-language prefix
└── language prompt ┘          │
                               ├── context tokens C [B, S_c, 2048]
robot state q_t [B, 7] ────────┴── condition projection / state projection
                                           │
                                           ▼
                              cached memory M [B, S_c + 1, 512]
                                           │
Gaussian noise ε [B, 10, 7]                 │
expert action A [B, 10, 7]                  │
flow time τ [B]                             │
          │                                 │
          ├── x_τ = τA + (1-τ)ε             │
          └── action/time embedding         │
                        │                    │
                        ▼                    │
             action tokens X [B, 10, 512]   │
                        │                    │
                        └──── action expert decoder
                              - full self-attention across 10 action tokens
                              - cross-attention to cached memory M
                              - 8 layers, width 512
                                           │
                                           ▼
                              velocity v_θ [B, 10, 7]
                                           │
                 training ─────────────────┴──────────── inference
                 target A-ε                              10 Euler steps
                 masked MSE                              ε → action chunk
```

## Modules

### 1. `ConditionEncoder`

Inputs:

- two RGB views at 224 × 224;
- tokenized language prompt and padding mask.

Output:

- contextual prefix tokens `context`;
- Boolean `context_valid_mask`.

The implementation backend is PaliGemma 3B at 224 resolution. Each image view
uses the shared vision tower. Image and prompt tokens form a vision-language
prefix. The backbone is frozen in the first project milestone so the 24GB GPU
trains only the compact policy layers.

The interface, rather than a specific Transformers class, is the contract:

```text
ConditionEncoder(images, prompt_ids, prompt_mask)
    -> context [B, S_c, 2048], context_valid [B, S_c]
```

A deterministic dummy encoder with the same contract is required for CPU unit
tests.

### 2. `ConditionProjector`

Projects `context` from 2048 to 512 dimensions. The normalized robot state is
projected independently into one 512-dimensional state token. Concatenation
forms:

```text
memory = [project(context), project(state)]
```

The condition encoder does not depend on state or actions. The state token does
not depend on actions. Therefore `memory` is computed once and cached for all
flow integration steps.

### 3. `ActionTimeEmbedding`

For every noisy action token:

1. linearly project the 7-dimensional noisy action to width 512;
2. encode scalar `τ` with a 128-dimensional sinusoidal embedding;
3. concatenate action and repeated time embeddings;
4. apply a two-layer SwiGLU MLP and output width 512.

The same `τ` is used for every action token in one sample.
Before the first decoder block, a learned horizon-position embedding is added
to each valid action token. Without this explicit position signal, a standard
attention decoder would be permutation-equivariant and could not distinguish
the order of otherwise identical action tokens.

### 4. `FlowActionExpert`

The action expert contains eight pre-norm decoder blocks:

- bidirectional self-attention among all valid action tokens;
- cross-attention from action tokens to valid memory tokens;
- SwiGLU feed-forward network;
- residual connections and RMSNorm.

Configuration:

```text
width=512, layers=8, heads=8, head_dim=64, ffn_dim=2048,
position_embedding=learned
```

There is no causal relationship within the flow action chunk. Every action
token can use every other action token, matching π0's full attention within the
action block.

### 5. `VelocityHead`

A linear layer maps each final action token from width 512 to 7. The result is
the velocity field for all ten actions simultaneously.

### 6. `FlowPolicy`

Training:

```text
observation, expert action chunk
    -> encode condition once
    -> sample ε and τ
    -> create x_τ and target A-ε
    -> predict velocity
    -> masked mean squared error
```

Inference:

```text
observation
    -> encode and cache condition once
    -> sample x_0 ~ N(0, I)
    -> integrate velocity from τ=0 to τ=1
    -> return normalized action chunk
    -> inverse-normalize before LIBERO execution
```

## Attention dependencies

| Query block | May attend to vision/language | May attend to state | May attend to actions |
|---|---:|---:|---:|
| Vision/language prefix | yes | no | no |
| State token | no | itself only | no |
| Flow action tokens | yes | yes | all valid action tokens |

This is equivalent to the causal dependency direction of π0's three blocks,
but it is implemented as a frozen condition encoder plus a cross-attention
decoder instead of a shared two-expert transformer.

## Control interface

The policy predicts ten normalized 7-dimensional actions. LIBERO may execute
fewer than ten actions before replanning; execution length is a runtime control
parameter and is not baked into the model. Both flow and autoregressive models
must use the same action horizon and replanning protocol.

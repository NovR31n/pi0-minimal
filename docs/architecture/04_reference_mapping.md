# P1-04: paper and OpenPI reference mapping

This file records references for verification. Production modules in
`src/pi0_minimal` must not import OpenPI's π0 model, loss, or sampler.

Reference repository state inspected on 2026-07-26:

```text
repository: https://github.com/Physical-Intelligence/openpi
commit: 3e0325a
```

| Concept | π0 paper | OpenPI reference | Independent target |
|---|---|---|---|
| VLM backbone | PaliGemma 3B | `src/openpi/models/pi0.py`, `PaliGemma.img/llm` | `ConditionEncoder` adapter |
| Image resolution | 224 × 224 | `src/openpi/models/model.py`, `IMAGE_RESOLUTION` | 224 × 224 |
| Reference image slots | multiple cameras | `model.py`, three `IMAGE_KEYS` | two valid LIBERO views |
| Prompt length | task-dependent | `Pi0Config.max_token_len=48` for π0 | 64 |
| State token | linear projection | `Pi0.state_proj` | independent `StateProjector` |
| Action input | action + sinusoidal time MLP | `Pi0.action_in_proj`, `action_time_mlp_*` | independent `ActionTimeEmbedding` |
| Expert size | width 1024, depth 18, ~300M | `gemma_300m` | width 512, depth 8, ~40–50M |
| Prefix/action dependency | three blockwise causal blocks | `make_attn_mask`, `embed_prefix/suffix` | condition encoder + cached memory |
| Action-to-action attention | full bidirectional | suffix mask: first action starts block; rest share block | full self-attention |
| Flow interpolation | `τA+(1-τ)ε` | reverse-time equivalent | paper convention |
| Flow target | `A-ε` | reverse-time `ε-A` | paper convention |
| Time sampling | shifted Beta, noisy times emphasized | `Beta(1.5,1)*0.999+0.001` in reverse time | `0.999*(1-Beta(1.5,1))` |
| Sampling | forward Euler, 10 steps | reverse-time Euler, default 10 | forward Euler, default 10 |
| Action horizon | 50 in paper tasks | default 50; π0.5 LIBERO config 10 | 10 |
| Action dimension | embodiment-dependent | padded default 32 | native LIBERO 7 |
| KV caching | prefix cached | `sample_actions` caches prefix | condition memory cached |

## Important discrepancy: paper time versus code time

The paper and OpenPI use opposite time directions. This is an intentional
notation change in OpenPI, documented in `Pi0.sample_actions`. The independent
implementation follows the paper direction consistently. See
`03_flow_matching.md`.

## OpenPI classes used only as references

- `openpi.models.pi0_config.Pi0Config`
- `openpi.models.pi0.Pi0`
- `openpi.models.pi0.make_attn_mask`
- `openpi.models.pi0.posemb_sincos`
- `openpi.models.model.Observation`
- `openpi.models.model.preprocess_observation`
- `openpi.models.gemma.Config`

No code is copied from these classes. Equivalent behavior will be established
through independently written unit tests and small numerical reference cases.

## Backend decision

The independent implementation uses PyTorch because:

- the 4090 environment already verifies CUDA-enabled PyTorch;
- isolated module/unit testing is simpler for the project scope;
- activation checkpointing, frozen-backbone inference, and profiling are
  available without coupling to OpenPI's JAX training stack;
- the flow and autoregressive models can share the same PyTorch condition
  encoder and action-expert building blocks.

OpenPI remains a read-only JAX/PyTorch behavior reference.

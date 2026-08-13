# P1-02: tensor specification

## Symbols

| Symbol | Meaning | Initial value |
|---|---|---:|
| `B` | batch size | configurable |
| `V` | valid camera views | 2 |
| `H_img`, `W_img` | image height and width | 224, 224 |
| `L` | padded prompt length | 64 |
| `S_v` | image tokens per view | backbone-defined, expected 256 |
| `S_c` | total vision-language context tokens | `V*S_v + L` before any backend-specific special tokens |
| `D_c` | condition width | 2048 |
| `D_m` | compact action-expert width | 512 |
| `S` | robot-state dimension | 7 |
| `H` | action horizon | 10 |
| `D_a` | action dimension | 7 |
| `T_e` | time-embedding width | 128 |

`S_v` and `S_c` must be obtained from the loaded backbone at runtime and
asserted; implementation code must not silently assume 256 image tokens.

## External batch contract

| Field | Shape | dtype | Range/meaning | Validity |
|---|---|---|---|---|
| `images` | `[B,V,3,224,224]` | `uint8` | RGB `[0,255]` before processor | controlled by `image_valid` |
| `image_valid` | `[B,V]` | `bool` | valid camera view | `True` means usable |
| `prompt_ids` | `[B,L]` | `int64` | PaliGemma token IDs | controlled by `prompt_valid` |
| `prompt_valid` | `[B,L]` | `bool` | non-padding prompt tokens | at least one valid token |
| `state` | `[B,7]` | `float32` | quantile-normalized proprioception | finite; normally clipped to `[-1,1]` |
| `actions` | `[B,10,7]` | `float32` | quantile-normalized expert actions | controlled by `action_valid` |
| `action_valid` | `[B,10,7]` | `bool` | valid horizon and dimension entries | at least one valid element/sample |

The dataset layer owns conversion from LIBERO image layout `[H,W,3]` to the
PyTorch layout `[3,H,W]`. The policy rejects ambiguous layouts.

## Backbone and memory tensors

| Tensor | Shape | dtype | Notes |
|---|---|---|---|
| processed pixels | `[B*V,3,224,224]` | `bfloat16` or `float32` | processor-normalized |
| context | `[B,S_c,2048]` | `bfloat16` | frozen backbone output |
| context valid | `[B,S_c]` | `bool` | image and prompt padding combined |
| projected context | `[B,S_c,512]` | `bfloat16` | trainable projection |
| state token | `[B,1,512]` | `bfloat16` | trainable state projection |
| memory | `[B,S_c+1,512]` | `bfloat16` | cached for sampling |
| memory valid | `[B,S_c+1]` | `bool` | state token always valid |

The frozen backbone may run in `bfloat16`, but normalization statistics and
input validation remain `float32`.

## Flow tensors

| Tensor | Shape | dtype | Definition |
|---|---|---|---|
| expert chunk `A` | `[B,10,7]` | `float32` | normalized data action |
| noise `ε` | `[B,10,7]` | `float32` | standard normal |
| flow time `τ` | `[B]` | `float32` | paper convention, `[0,1)` in training |
| expanded time | `[B,1,1]` | `float32` | broadcast helper only |
| noisy action `x_τ` | `[B,10,7]` | `float32` | `τA+(1-τ)ε` |
| target velocity `u` | `[B,10,7]` | `float32` | `A-ε` |
| action/time tokens | `[B,10,512]` | `bfloat16` | expert input |
| learned action positions | `[10,512]` | `bfloat16` | added once before the first decoder block |
| predicted velocity `v_θ` | `[B,10,7]` | `float32` | cast before loss/update |
| squared error | `[B,10,7]` | `float32` | `(v_θ-u)^2` |

Random noise and flow time are generated in `float32`, even when the expert
runs in `bfloat16`.

## Masks

### Context mask

`context_valid[b,j]` is true only when context token `j` corresponds to a
valid image token or non-padding prompt token. Invalid keys cannot be attended
to. Invalid query outputs are ignored.

### Flow action self-attention mask

The flow model uses full bidirectional attention across valid action positions:

```text
flow_self_mask[b,i,j] =
    horizon_valid[b,i] AND horizon_valid[b,j]
```

There is no triangular causal mask.

### Cross-attention mask

Every valid action query can attend to every valid memory key:

```text
cross_mask[b,i,j] =
    horizon_valid[b,i] AND memory_valid[b,j]
```

### Loss mask

```text
loss =
    sum((v_θ-u)^2 * action_valid)
    / max(sum(action_valid), 1)
```

The denominator counts valid scalar action elements, not batches or horizons.
An all-false mask for any sample is a data error and must be rejected before
the forward pass.

## Runtime assertions

The implementation must reject:

- image keys or camera count different from the configuration;
- image resolution/layout mismatch after preprocessing;
- prompt IDs without an equally shaped prompt mask;
- state/action dimensions different from 7;
- action horizon different from 10 unless a different committed config is used;
- NaN or infinity in state, actions, loss, or predicted velocity;
- inconsistent devices within one batch;
- an all-false action mask.

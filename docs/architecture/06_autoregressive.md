# P4-01: matched continuous autoregressive baseline

## Primary representation

The primary autoregressive baseline models each normalized 7D action as a
diagonal Gaussian. It does not discretize actions. Quantization would introduce
an extra smoothness and accuracy variable into the flow-versus-AR comparison.

For horizon position `i`, the model input is:

```text
i = 0: learned continuous BOS action
i > 0: action A[i-1]
```

The target is `A[i]`. A lower-triangular self-attention mask allows position
`i` to use only shifted inputs at positions `0..i`; therefore teacher forcing
cannot expose the current or future target.

The prediction head returns:

```text
mean:       [B,10,7] float32
log_scale:  [B,10,7] float32, clamped to [-5,2]
```

Training minimizes diagonal-Gaussian negative log likelihood over valid scalar
action elements. Evaluation uses the deterministic mean. Stochastic Gaussian
sampling remains an explicit secondary option and must not be mixed into the
primary comparison.

## Generation order

Generation begins with an all-zero output buffer. For positions `0..9`, the
policy runs the causal decoder, takes the current-position mean, writes that
action into the buffer, and advances one position. The initial implementation
does not use a KV cache; latency therefore includes ten decoder calls and is
reported as an intrinsic cost of this simple AR baseline.

## Matched boundary

Flow and AR share:

- the exact frozen PaliGemma revision and prompt/image preprocessing;
- condition and 7D robot-state projection;
- 512 model width, 8 layers, 8 heads, 2048 FFN width, and learned positions;
- normalized continuous `[B,10,7]` action data and scalar validity mask;
- task split, optimizer budget, seeds, checkpoints, and LIBERO adapter.

Only the generation mechanism differs:

- flow uses noisy actions, continuous flow time, bidirectional action
  attention, velocity MSE, and ten Euler steps;
- AR uses shifted previous actions, causal attention, Gaussian NLL, and ten
  sequential next-action predictions.

The AR embedding uses a learned 128D constant mode vector in the same two-layer
SwiGLU topology as the flow time fusion. Together with the doubled
mean/log-scale output head, its trainable parameter count must remain within
one percent of the flow stack. Exact counts are recorded by P4-06.

The committed counts are 43,940,871 trainable parameters for flow and
43,944,597 for AR. AR is larger by 3,726 parameters (0.0085%), which satisfies
the one-percent matching boundary.

## Interpretation limits

This baseline measures continuous autoregressive factorization, not discrete
token modeling. A failed 10-step non-cached AR rollout does not establish that
all autoregressive policies are slow or weak. Any later discretized or
KV-cached model is a separate experiment.

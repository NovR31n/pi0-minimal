# P1-03: flow-matching convention

## Chosen direction

This project follows the notation in the π0 paper:

- `τ=0`: pure Gaussian noise;
- `τ=1`: expert action data;
- integration proceeds forward from 0 to 1.

Let:

- `A ~ p_data(A | o)` be a normalized expert action chunk;
- `ε ~ N(0,I)` have the same shape as `A`;
- `τ ∈ [0,1]` be the flow time.

The linear conditional probability path is:

```text
x_τ = τ A + (1-τ) ε
```

Its time derivative is constant:

```text
u_τ = d x_τ / dτ = A - ε
```

The network predicts:

```text
v_θ(x_τ, o, τ)
```

and is trained with masked conditional flow matching:

```text
L(θ) =
    sum(mask * ||v_θ(x_τ,o,τ) - (A-ε)||²)
    / sum(mask)
```

Loss accumulation and the final reduction use `float32`.

## Flow-time sampling

The paper emphasizes noisier, lower `τ` values. The committed default is:

```text
z ~ Beta(1.5, 1.0)
s = 0.999
τ = s * (1-z)
```

Therefore `τ ∈ [0,s]` and the density emphasizes values near zero. A uniform
distribution is allowed only through an explicit ablation configuration.

Tests must inject `ε` and `τ` directly instead of relying on random sampling.

## Euler inference

For `N` integration steps:

```text
Δτ = 1/N
x_0 ~ N(0,I)

for k in 0,...,N-1:
    τ_k = k/N
    x_{k+1} = x_k + Δτ * v_θ(x_k,o,τ_k)

predicted normalized action = x_N
```

The implementation default is `N=10`, matching the π0 paper. The committed
formal LIBERO configuration explicitly overrides it to `N=20`. The same cached
condition memory is reused at every integration step. No stochastic noise is
added after initialization.

The sampler accepts an optional caller-provided initial noise tensor. Formal
latency and sampling-step ablations reuse fixed noise seeds.

## Analytical sanity checks

The following tests are mandatory before training:

1. `τ=0` gives `x_τ=ε`.
2. `τ=1` gives `x_τ=A` for the interpolation helper, even though the training
   sampler normally caps `τ` below 1.
3. Target velocity is exactly `A-ε`.
4. A perfect constant velocity oracle integrated with any positive number of
   Euler steps returns `A` up to floating-point error:

   ```text
   x_0=ε
   x_N=ε + N*(1/N)*(A-ε)=A
   ```

5. Masked loss is unchanged when invalid padded targets are perturbed.
6. Fixed observation, weights, noise, and seed produce deterministic sampling.

## Relationship to OpenPI's code convention

OpenPI reference commit `3e0325a` uses the reverse, diffusion-style time
variable:

```text
t=1: noise
t=0: data
x_t = t ε + (1-t) A
target = ε-A
dt = -1/N
```

The mapping is:

```text
t = 1-τ
v_openpi = -v_paper
dt_openpi = -dτ
```

The paths and final action distribution are mathematically equivalent. The
independent implementation must not mix the paper interpolation with OpenPI's
target sign or negative integration step.

## Normalization boundary

Flow matching operates only in normalized action space. Dataset statistics
map each LIBERO action dimension approximately to `[-1,1]` using committed
training-split quantiles. The policy returns normalized actions; inverse
normalization happens exactly once at the environment boundary.

Clipping predicted normalized actions during training is forbidden. Optional
inference clipping must be separately configured and reported.

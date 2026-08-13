# Environment baseline

Verified on the project GPU host on 2026-07-26:

| Component | Version or capacity |
|---|---|
| Operating system | Ubuntu 22.04.4 LTS |
| GPU | NVIDIA GeForce RTX 4090, 24,564 MiB |
| NVIDIA driver | 570.172.08 |
| Driver CUDA compatibility | 12.8 |
| CPU | 16 vCPU, AMD EPYC 7542 |
| RAM | 62 GiB |
| Project Python | 3.11 |
| uv | 0.9.11 |
| Reference OpenPI Python | 3.11.14 |
| Reference OpenPI JAX | 0.5.3 |
| Reference OpenPI PyTorch | 2.7.1+cu126 |
| Reference OpenPI Transformers | 4.53.2 |

The independent project will use its own `.venv`. The pre-existing conda
environment named `uv_envs` uses Python 3.10 and is not the project
environment.

The committed runtime is now locked by `uv.lock`. NumPy is pinned to 1.26.4
because the verified LIBERO stack uses Gym 0.25, robosuite 1.4.1, and MuJoCo
2.3.7; NumPy 2.4 is incompatible with the required Numba release.

Install the model/test environment with:

```bash
uv sync --extra dev
```

Install the additional closed-loop simulator dependencies with:

```bash
uv sync --extra dev --extra libero
```

Point `PYTHONPATH` at a checked-out LIBERO source tree before direct off-screen
rollout:

```bash
export PYTHONPATH=/path/to/LIBERO
export MUJOCO_GL=egl
```

The independent policy, flow objective, sampler, training loop, action safety
adapter, and rollout loop do not import OpenPI's policy implementation or
client. Only the general-purpose LIBERO benchmark source and pretrained
PaliGemma weights are reused.

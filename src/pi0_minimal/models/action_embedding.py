"""Noisy-action and continuous flow-time embeddings."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class ActionTokens:
    """Embedded action chunk and per-horizon validity mask."""

    values: torch.Tensor
    valid: torch.Tensor

    def __post_init__(self) -> None:
        if self.values.ndim != 3:
            raise ValueError(f"action tokens must have shape [B,H,D], got {self.values.shape}")
        if self.valid.dtype != torch.bool or self.valid.shape != self.values.shape[:2]:
            raise ValueError("action token valid must have bool shape [B,H]")
        if self.values.device != self.valid.device:
            raise ValueError("action token values and valid mask must share a device")
        if not torch.isfinite(self.values).all():
            raise ValueError("action token values must be finite")


class ActionTimeEmbedding(nn.Module):
    """Fuse masked noisy actions with one sinusoidal flow time per sample."""

    def __init__(
        self,
        *,
        action_dim: int = 7,
        model_dim: int = 512,
        time_dim: int = 128,
        max_period: float = 10_000.0,
    ) -> None:
        super().__init__()
        if action_dim <= 0 or model_dim <= 0:
            raise ValueError("action_dim and model_dim must be positive")
        if time_dim <= 0 or time_dim % 2:
            raise ValueError("time_dim must be a positive even integer")
        if max_period <= 1.0:
            raise ValueError("max_period must be greater than one")
        self.action_dim = action_dim
        self.model_dim = model_dim
        self.time_dim = time_dim
        self.max_period = max_period
        self.action_projection = nn.Linear(action_dim, model_dim)
        self.fusion_gate_value = nn.Linear(model_dim + time_dim, 2 * model_dim)
        self.fusion_output = nn.Linear(model_dim, model_dim)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        flow_time: torch.Tensor,
        action_valid: torch.Tensor | None = None,
    ) -> ActionTokens:
        self._validate_inputs(noisy_actions, flow_time, action_valid)
        if action_valid is None:
            action_valid = torch.ones_like(noisy_actions, dtype=torch.bool)
        horizon_valid = action_valid.any(dim=-1)

        compute_dtype = self.action_projection.weight.dtype
        masked_actions = noisy_actions.masked_fill(~action_valid, 0.0)
        action_features = self.action_projection(masked_actions.to(compute_dtype))
        time_features = sinusoidal_time_embedding(
            flow_time,
            self.time_dim,
            max_period=self.max_period,
        ).to(compute_dtype)
        expanded_time = time_features.unsqueeze(1).expand(
            noisy_actions.shape[0],
            noisy_actions.shape[1],
            self.time_dim,
        )
        gate, value = self.fusion_gate_value(
            torch.cat((action_features, expanded_time), dim=-1)
        ).chunk(2, dim=-1)
        tokens = self.fusion_output(F.silu(gate) * value)
        tokens = tokens.masked_fill(~horizon_valid.unsqueeze(-1), 0.0)
        return ActionTokens(tokens, horizon_valid)

    def _validate_inputs(
        self,
        noisy_actions: torch.Tensor,
        flow_time: torch.Tensor,
        action_valid: torch.Tensor | None,
    ) -> None:
        if noisy_actions.dtype != torch.float32 or noisy_actions.ndim != 3:
            raise TypeError("noisy_actions must be float32 [B,H,D]")
        if noisy_actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"noisy action width must be {self.action_dim}, "
                f"got {noisy_actions.shape[-1]}"
            )
        if not torch.isfinite(noisy_actions).all():
            raise ValueError("noisy_actions must contain only finite values")
        if flow_time.dtype != torch.float32 or flow_time.ndim != 1:
            raise TypeError("flow_time must be float32 [B]")
        if flow_time.shape[0] != noisy_actions.shape[0]:
            raise ValueError("flow_time and noisy_actions must share the batch dimension")
        if flow_time.device != noisy_actions.device:
            raise ValueError("flow_time and noisy_actions must share a device")
        if not torch.isfinite(flow_time).all():
            raise ValueError("flow_time must contain only finite values")
        if torch.any((flow_time < 0.0) | (flow_time > 1.0)):
            raise ValueError("flow_time must lie in [0,1]")
        if action_valid is None:
            return
        if action_valid.dtype != torch.bool or action_valid.shape != noisy_actions.shape:
            raise ValueError("action_valid must be bool with the same shape as noisy_actions")
        if action_valid.device != noisy_actions.device:
            raise ValueError("action_valid and noisy_actions must share a device")
        if not action_valid.reshape(action_valid.shape[0], -1).any(dim=1).all():
            raise ValueError("every sample must contain at least one valid action element")


def sinusoidal_time_embedding(
    flow_time: torch.Tensor,
    embedding_dim: int,
    *,
    max_period: float = 10_000.0,
) -> torch.Tensor:
    """Encode scalar paper-convention flow time in deterministic FP32."""

    if flow_time.dtype != torch.float32 or flow_time.ndim != 1:
        raise TypeError("flow_time must be float32 [B]")
    if embedding_dim <= 0 or embedding_dim % 2:
        raise ValueError("embedding_dim must be a positive even integer")
    if max_period <= 1.0:
        raise ValueError("max_period must be greater than one")
    half_dim = embedding_dim // 2
    exponent = torch.arange(
        half_dim,
        dtype=torch.float32,
        device=flow_time.device,
    )
    exponent = exponent / max(half_dim - 1, 1)
    frequencies = torch.exp(-math.log(max_period) * exponent)
    angles = flow_time[:, None] * frequencies[None, :] * (2.0 * math.pi)
    return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)

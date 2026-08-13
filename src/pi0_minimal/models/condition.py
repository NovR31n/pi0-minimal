"""Trainable projection and fusion of frozen context with robot state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from pi0_minimal.models.backbone import ConditionMemory


@dataclass(frozen=True, slots=True)
class PolicyMemory:
    """Projected context followed by one always-valid robot-state token."""

    values: torch.Tensor
    valid: torch.Tensor

    def __post_init__(self) -> None:
        if self.values.ndim != 3:
            raise ValueError(f"memory values must have shape [B,S,D], got {self.values.shape}")
        if self.valid.dtype != torch.bool or self.valid.shape != self.values.shape[:2]:
            raise ValueError("memory valid must have bool shape [B,S]")
        if self.values.device != self.valid.device:
            raise ValueError("memory values and valid mask must share a device")
        if not torch.isfinite(self.values).all():
            raise ValueError("memory values must be finite")


class ConditionProjector(nn.Module):
    """Project frozen VLM tokens and append a masked robot-state token."""

    def __init__(
        self,
        *,
        condition_dim: int = 2048,
        state_dim: int = 7,
        model_dim: int = 512,
    ) -> None:
        super().__init__()
        if condition_dim <= 0 or state_dim <= 0 or model_dim <= 0:
            raise ValueError("condition_dim, state_dim, and model_dim must be positive")
        self.condition_dim = condition_dim
        self.state_dim = state_dim
        self.model_dim = model_dim
        self.condition_projection = nn.Linear(condition_dim, model_dim)
        self.state_projection = nn.Linear(state_dim, model_dim)

    def forward(
        self,
        condition: ConditionMemory,
        state: torch.Tensor,
        state_valid: torch.Tensor | None = None,
    ) -> PolicyMemory:
        self._validate_inputs(condition, state, state_valid)
        if state_valid is None:
            state_valid = torch.ones_like(state, dtype=torch.bool)

        compute_dtype = self.condition_projection.weight.dtype
        projected_context = self.condition_projection(condition.values.to(compute_dtype))
        projected_context = projected_context.masked_fill(
            ~condition.valid.unsqueeze(-1),
            0.0,
        )
        masked_state = state.masked_fill(~state_valid, 0.0).to(compute_dtype)
        state_token = self.state_projection(masked_state).unsqueeze(1)

        values = torch.cat((projected_context, state_token), dim=1)
        state_token_valid = torch.ones(
            (state.shape[0], 1),
            dtype=torch.bool,
            device=state.device,
        )
        valid = torch.cat((condition.valid, state_token_valid), dim=1)
        return PolicyMemory(values, valid)

    def encode_numpy(
        self,
        condition: ConditionMemory,
        state: np.ndarray,
        state_valid: np.ndarray | None = None,
    ) -> PolicyMemory:
        """Bridge normalized NumPy state into the projector's existing device."""

        device = condition.values.device
        state_tensor = torch.from_numpy(state).to(device)
        valid_tensor = None
        if state_valid is not None:
            valid_tensor = torch.from_numpy(state_valid).to(device)
        return self(condition, state_tensor, valid_tensor)

    def _validate_inputs(
        self,
        condition: ConditionMemory,
        state: torch.Tensor,
        state_valid: torch.Tensor | None,
    ) -> None:
        if condition.values.shape[-1] != self.condition_dim:
            raise ValueError(
                f"condition width must be {self.condition_dim}, "
                f"got {condition.values.shape[-1]}"
            )
        if state.dtype != torch.float32 or state.ndim != 2:
            raise TypeError("state must be float32 [B,S]")
        expected_state = (condition.values.shape[0], self.state_dim)
        if state.shape != expected_state:
            raise ValueError(f"state must have shape {expected_state}, got {state.shape}")
        if state.device != condition.values.device or condition.valid.device != state.device:
            raise ValueError("condition values, condition mask, and state must share a device")
        if not torch.isfinite(state).all():
            raise ValueError("state must contain only finite values")
        if state_valid is None:
            return
        if state_valid.dtype != torch.bool or state_valid.shape != state.shape:
            raise ValueError("state_valid must be bool with the same shape as state")
        if state_valid.device != state.device:
            raise ValueError("state and state_valid must share a device")
        if not state_valid.any(dim=1).all():
            raise ValueError("every sample must contain at least one valid state dimension")

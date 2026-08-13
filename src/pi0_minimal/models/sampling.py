"""Forward Euler sampling from Gaussian noise to normalized action chunks."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from pi0_minimal.models.action_embedding import ActionTimeEmbedding
from pi0_minimal.models.condition import PolicyMemory
from pi0_minimal.models.expert import FlowActionExpert

VelocityFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class ConditionedVelocityField(nn.Module):
    """Compose action/time embedding and the expert around cached condition memory."""

    def __init__(
        self,
        action_embedding: ActionTimeEmbedding,
        expert: FlowActionExpert,
    ) -> None:
        super().__init__()
        if action_embedding.model_dim != expert.model_dim:
            raise ValueError("action embedding and expert model widths must match")
        if action_embedding.action_dim != expert.action_dim:
            raise ValueError("action embedding and expert action widths must match")
        self.action_embedding = action_embedding
        self.expert = expert

    def forward(
        self,
        noisy_actions: torch.Tensor,
        flow_time: torch.Tensor,
        action_valid: torch.Tensor,
        memory: PolicyMemory,
    ) -> torch.Tensor:
        tokens = self.action_embedding(noisy_actions, flow_time, action_valid)
        return self.expert(tokens, memory)


def euler_integrate(
    velocity_function: VelocityFunction,
    initial_noise: torch.Tensor,
    action_valid: torch.Tensor,
    *,
    num_steps: int,
) -> torch.Tensor:
    """Integrate dx/dτ=v(x,τ) from τ=0 to 1 with positive FP32 steps."""

    _validate_initial_state(initial_noise, action_valid)
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    actions = initial_noise.masked_fill(~action_valid, 0.0)
    step_size = 1.0 / num_steps
    batch_size = actions.shape[0]
    for step in range(num_steps):
        flow_time = torch.full(
            (batch_size,),
            step * step_size,
            dtype=torch.float32,
            device=actions.device,
        )
        velocity = velocity_function(actions, flow_time)
        _validate_velocity(velocity, actions)
        velocity = velocity.masked_fill(~action_valid, 0.0)
        actions = actions + step_size * velocity
        actions = actions.masked_fill(~action_valid, 0.0)
    return actions


@torch.no_grad()
def sample_actions_euler(
    velocity_field: ConditionedVelocityField,
    memory: PolicyMemory,
    action_valid: torch.Tensor,
    *,
    num_steps: int = 10,
    initial_noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one normalized action chunk while reusing cached memory."""

    if action_valid.dtype != torch.bool or action_valid.ndim != 3:
        raise ValueError("action_valid must have bool shape [B,H,D]")
    expected_shape = (
        memory.values.shape[0],
        action_valid.shape[1],
        velocity_field.expert.action_dim,
    )
    if action_valid.shape != expected_shape:
        raise ValueError(f"action_valid must have shape {expected_shape}")
    if action_valid.device != memory.values.device:
        raise ValueError("action_valid and memory must share a device")
    if initial_noise is None:
        initial_noise = torch.randn(
            expected_shape,
            dtype=torch.float32,
            device=memory.values.device,
            generator=generator,
        )

    def predict_velocity(
        noisy_actions: torch.Tensor,
        flow_time: torch.Tensor,
    ) -> torch.Tensor:
        return velocity_field(
            noisy_actions,
            flow_time,
            action_valid,
            memory,
        )

    return euler_integrate(
        predict_velocity,
        initial_noise,
        action_valid,
        num_steps=num_steps,
    )


def _validate_initial_state(
    initial_noise: torch.Tensor,
    action_valid: torch.Tensor,
) -> None:
    if initial_noise.dtype != torch.float32 or initial_noise.ndim != 3:
        raise TypeError("initial_noise must be float32 [B,H,D]")
    if not torch.isfinite(initial_noise).all():
        raise ValueError("initial_noise must contain only finite values")
    if action_valid.dtype != torch.bool or action_valid.shape != initial_noise.shape:
        raise ValueError("action_valid must have bool shape [B,H,D]")
    if action_valid.device != initial_noise.device:
        raise ValueError("initial_noise and action_valid must share a device")
    if not action_valid.reshape(initial_noise.shape[0], -1).any(dim=1).all():
        raise ValueError("every sample must contain at least one valid action element")


def _validate_velocity(
    velocity: torch.Tensor,
    actions: torch.Tensor,
) -> None:
    if velocity.dtype != torch.float32 or velocity.shape != actions.shape:
        raise ValueError("velocity function must return float32 with the action shape")
    if velocity.device != actions.device:
        raise ValueError("velocity and actions must share a device")
    if not torch.isfinite(velocity).all():
        raise ValueError("velocity must contain only finite values")

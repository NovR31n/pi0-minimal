"""Unified compact flow policy over precomputed frozen condition memory."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from pi0_minimal.models.action_embedding import ActionTimeEmbedding
from pi0_minimal.models.backbone import ConditionMemory
from pi0_minimal.models.condition import ConditionProjector, PolicyMemory
from pi0_minimal.models.expert import FlowActionExpert
from pi0_minimal.models.flow_matching import (
    FlowMatchingBatch,
    build_flow_matching_batch,
    masked_flow_matching_loss,
)
from pi0_minimal.models.sampling import (
    ConditionedVelocityField,
    sample_actions_euler,
)


@dataclass(frozen=True, slots=True)
class FlowTrainingOutput:
    """Loss and inspectable tensors for one conditional flow training step."""

    loss: torch.Tensor
    predicted_velocity: torch.Tensor
    flow_batch: FlowMatchingBatch

    def __post_init__(self) -> None:
        if self.loss.dtype != torch.float32 or self.loss.ndim != 0:
            raise ValueError("loss must be a scalar float32 tensor")
        if not torch.isfinite(self.loss):
            raise ValueError("loss must be finite")
        if (
            self.predicted_velocity.dtype != torch.float32
            or self.predicted_velocity.shape != self.flow_batch.target_velocity.shape
        ):
            raise ValueError("predicted velocity must match the FP32 flow target")
        if not torch.isfinite(self.predicted_velocity).all():
            raise ValueError("predicted velocity must be finite")


class FlowPolicy(nn.Module):
    """Compose state/context projection, flow target, expert, and Euler sampler."""

    def __init__(
        self,
        condition_projector: ConditionProjector,
        action_embedding: ActionTimeEmbedding,
        expert: FlowActionExpert,
        *,
        beta_alpha: float = 1.5,
        beta_beta: float = 1.0,
        cutoff: float = 0.999,
        num_euler_steps: int = 10,
        smoothness_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if condition_projector.model_dim != action_embedding.model_dim:
            raise ValueError("condition and action projection widths must match")
        if action_embedding.model_dim != expert.model_dim:
            raise ValueError("action embedding and expert widths must match")
        if action_embedding.action_dim != expert.action_dim:
            raise ValueError("action embedding and expert action widths must match")
        if beta_alpha <= 0.0 or beta_beta != 1.0:
            raise ValueError("paper flow configuration requires alpha>0 and beta=1")
        if not 0.0 < cutoff < 1.0:
            raise ValueError("flow cutoff must lie strictly between zero and one")
        if num_euler_steps <= 0:
            raise ValueError("num_euler_steps must be positive")
        if smoothness_weight < 0.0:
            raise ValueError("smoothness_weight must be non-negative")
        self.condition_projector = condition_projector
        self.velocity_field = ConditionedVelocityField(action_embedding, expert)
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta
        self.cutoff = cutoff
        self.num_euler_steps = num_euler_steps
        self.smoothness_weight = smoothness_weight

    def encode_memory(
        self,
        condition: ConditionMemory,
        state: torch.Tensor,
        state_valid: torch.Tensor | None = None,
    ) -> PolicyMemory:
        return self.condition_projector(condition, state, state_valid)

    def training_step(
        self,
        condition: ConditionMemory,
        state: torch.Tensor,
        expert_actions: torch.Tensor,
        action_valid: torch.Tensor,
        *,
        state_valid: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        flow_time: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> FlowTrainingOutput:
        memory = self.encode_memory(condition, state, state_valid)
        flow_batch = build_flow_matching_batch(
            expert_actions,
            action_valid,
            noise=noise,
            flow_time=flow_time,
            generator=generator,
            beta_alpha=self.beta_alpha,
            beta_beta=self.beta_beta,
            cutoff=self.cutoff,
        )
        predicted_velocity = self.velocity_field(
            flow_batch.noisy_actions,
            flow_batch.flow_time,
            action_valid,
            memory,
        )
        loss = masked_flow_matching_loss(
            predicted_velocity,
            flow_batch.target_velocity,
            action_valid,
        )
        if self.smoothness_weight:
            time = flow_batch.flow_time[:, None, None]
            predicted_clean = (
                flow_batch.noisy_actions + (1.0 - time) * predicted_velocity
            )
            pair_valid = action_valid[:, 1:] & action_valid[:, :-1]
            predicted_delta = predicted_clean[:, 1:] - predicted_clean[:, :-1]
            target_delta = expert_actions[:, 1:] - expert_actions[:, :-1]
            valid = pair_valid.to(torch.float32)
            smoothness_loss = (
                (predicted_delta - target_delta).square() * valid
            ).sum() / valid.sum().clamp_min(1.0)
            loss = loss + self.smoothness_weight * smoothness_loss
        return FlowTrainingOutput(loss, predicted_velocity, flow_batch)

    @torch.no_grad()
    def sample(
        self,
        condition: ConditionMemory,
        state: torch.Tensor,
        action_valid: torch.Tensor,
        *,
        state_valid: torch.Tensor | None = None,
        initial_noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        memory = self.encode_memory(condition, state, state_valid)
        return sample_actions_euler(
            self.velocity_field,
            memory,
            action_valid,
            num_steps=self.num_euler_steps if num_steps is None else num_steps,
            initial_noise=initial_noise,
            generator=generator,
        )

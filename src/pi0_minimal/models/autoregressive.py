"""Matched continuous autoregressive action policy."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from pi0_minimal.models.action_embedding import ActionTokens
from pi0_minimal.models.attention import build_causal_attention_masks
from pi0_minimal.models.backbone import ConditionMemory
from pi0_minimal.models.condition import ConditionProjector, PolicyMemory
from pi0_minimal.models.expert import (
    FlowDecoderBlock,
    RMSNorm,
    _zero_invalid_queries,
)


class AutoregressiveActionEmbedding(nn.Module):
    """Embed BOS followed by shifted previous continuous actions."""

    def __init__(
        self,
        *,
        action_dim: int = 7,
        model_dim: int = 512,
        mode_dim: int = 128,
    ) -> None:
        super().__init__()
        if action_dim <= 0 or model_dim <= 0 or mode_dim <= 0:
            raise ValueError("action_dim, model_dim, and mode_dim must be positive")
        self.action_dim = action_dim
        self.model_dim = model_dim
        self.mode_dim = mode_dim
        self.bos_action = nn.Parameter(torch.zeros(action_dim))
        self.mode_embedding = nn.Parameter(torch.empty(mode_dim))
        nn.init.normal_(self.mode_embedding, mean=0.0, std=mode_dim**-0.5)
        self.action_projection = nn.Linear(action_dim, model_dim)
        self.fusion_gate_value = nn.Linear(model_dim + mode_dim, 2 * model_dim)
        self.fusion_output = nn.Linear(model_dim, model_dim)

    def forward(
        self,
        actions: torch.Tensor,
        action_valid: torch.Tensor,
    ) -> ActionTokens:
        self._validate_inputs(actions, action_valid)
        horizon_valid = action_valid.any(dim=-1)
        previous = torch.zeros_like(actions)
        previous_valid = torch.zeros_like(action_valid)
        previous[:, 0] = self.bos_action.float()
        previous_valid[:, 0] = True
        if actions.shape[1] > 1:
            previous[:, 1:] = actions[:, :-1]
            previous_valid[:, 1:] = action_valid[:, :-1]
        previous = previous.masked_fill(~previous_valid, 0.0)

        compute_dtype = self.action_projection.weight.dtype
        action_features = self.action_projection(previous.to(compute_dtype))
        mode = self.mode_embedding.to(compute_dtype).view(1, 1, -1)
        mode = mode.expand(actions.shape[0], actions.shape[1], self.mode_dim)
        gate, value = self.fusion_gate_value(
            torch.cat((action_features, mode), dim=-1)
        ).chunk(2, dim=-1)
        tokens = self.fusion_output(F.silu(gate) * value)
        return ActionTokens(
            tokens.masked_fill(~horizon_valid.unsqueeze(-1), 0.0),
            horizon_valid,
        )

    def _validate_inputs(
        self,
        actions: torch.Tensor,
        action_valid: torch.Tensor,
    ) -> None:
        if (
            actions.dtype != torch.float32
            or actions.ndim != 3
            or actions.shape[-1] != self.action_dim
        ):
            raise TypeError(
                f"actions must be float32 [B,H,{self.action_dim}]"
            )
        if not torch.isfinite(actions).all():
            raise ValueError("actions must contain only finite values")
        if action_valid.dtype != torch.bool or action_valid.shape != actions.shape:
            raise ValueError("action_valid must be bool with the same shape as actions")
        if action_valid.device != actions.device:
            raise ValueError("actions and action_valid must share a device")
        if not action_valid.reshape(actions.shape[0], -1).any(dim=1).all():
            raise ValueError("every sample must contain at least one valid action element")


@dataclass(frozen=True, slots=True)
class ActionDistribution:
    mean: torch.Tensor
    log_scale: torch.Tensor
    valid: torch.Tensor

    def __post_init__(self) -> None:
        if (
            self.mean.dtype != torch.float32
            or self.log_scale.dtype != torch.float32
            or self.mean.shape != self.log_scale.shape
            or self.mean.ndim != 3
        ):
            raise ValueError("mean and log_scale must be matching float32 [B,H,D]")
        if self.valid.dtype != torch.bool or self.valid.shape != self.mean.shape:
            raise ValueError("distribution valid must match the action tensors")
        if not (
            self.mean.device == self.log_scale.device == self.valid.device
        ):
            raise ValueError("distribution tensors must share a device")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(
            self.log_scale
        ).all():
            raise ValueError("distribution parameters must be finite")


class AutoregressiveActionExpert(nn.Module):
    """Causal decoder predicting a diagonal Gaussian per action step."""

    def __init__(
        self,
        *,
        model_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        action_dim: int = 7,
        max_horizon: int = 10,
        dropout: float = 0.0,
        log_scale_min: float = -5.0,
        log_scale_max: float = 2.0,
    ) -> None:
        super().__init__()
        if num_layers <= 0 or action_dim <= 0 or max_horizon <= 0:
            raise ValueError("num_layers, action_dim, and max_horizon must be positive")
        if not log_scale_min < log_scale_max:
            raise ValueError("log-scale bounds must be increasing")
        self.model_dim = model_dim
        self.action_dim = action_dim
        self.max_horizon = max_horizon
        self.log_scale_min = log_scale_min
        self.log_scale_max = log_scale_max
        self.position_embedding = nn.Parameter(torch.empty(max_horizon, model_dim))
        nn.init.normal_(self.position_embedding, mean=0.0, std=model_dim**-0.5)
        self.blocks = nn.ModuleList(
            [
                FlowDecoderBlock(
                    model_dim=model_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(model_dim)
        self.distribution_head = nn.Linear(model_dim, 2 * action_dim)

    def forward(
        self,
        action_tokens: ActionTokens,
        memory: PolicyMemory,
        scalar_valid: torch.Tensor,
    ) -> ActionDistribution:
        self._validate_inputs(action_tokens, memory, scalar_valid)
        compute_dtype = self.position_embedding.dtype
        horizon = action_tokens.values.shape[1]
        values = action_tokens.values.to(compute_dtype)
        values = values + self.position_embedding[:horizon].unsqueeze(0)
        values = _zero_invalid_queries(values, action_tokens.valid)
        memory_values = memory.values.to(compute_dtype)
        memory_values = memory_values.masked_fill(~memory.valid.unsqueeze(-1), 0.0)
        masks = build_causal_attention_masks(action_tokens.valid, memory.valid)
        for block in self.blocks:
            values = block(values, memory_values, masks, action_tokens.valid)
        parameters = self.distribution_head(self.final_norm(values)).float()
        mean, log_scale = parameters.chunk(2, dim=-1)
        mean = mean.masked_fill(~scalar_valid, 0.0)
        log_scale = log_scale.clamp(
            min=self.log_scale_min,
            max=self.log_scale_max,
        ).masked_fill(~scalar_valid, 0.0)
        return ActionDistribution(mean, log_scale, scalar_valid)

    def _validate_inputs(
        self,
        tokens: ActionTokens,
        memory: PolicyMemory,
        scalar_valid: torch.Tensor,
    ) -> None:
        if tokens.values.shape[-1] != self.model_dim:
            raise ValueError("action token width does not match the expert")
        if memory.values.shape[-1] != self.model_dim:
            raise ValueError("memory width does not match the expert")
        if tokens.values.shape[0] != memory.values.shape[0]:
            raise ValueError("action tokens and memory must share the batch dimension")
        if tokens.values.shape[1] > self.max_horizon:
            raise ValueError("action horizon exceeds the configured maximum")
        expected_valid = (*tokens.valid.shape, self.action_dim)
        if scalar_valid.dtype != torch.bool or scalar_valid.shape != expected_valid:
            raise ValueError(f"scalar_valid must have bool shape {expected_valid}")
        devices = {
            tokens.values.device,
            tokens.valid.device,
            memory.values.device,
            memory.valid.device,
            scalar_valid.device,
            self.position_embedding.device,
        }
        if len(devices) != 1:
            raise ValueError("expert inputs and parameters must share a device")


@dataclass(frozen=True, slots=True)
class AutoregressiveTrainingOutput:
    loss: torch.Tensor
    distribution: ActionDistribution

    def __post_init__(self) -> None:
        if self.loss.dtype != torch.float32 or self.loss.ndim != 0:
            raise ValueError("loss must be scalar float32")
        if not torch.isfinite(self.loss):
            raise ValueError("loss must be finite")


class AutoregressivePolicy(nn.Module):
    """Shared condition projection plus causal continuous action generation."""

    def __init__(
        self,
        condition_projector: ConditionProjector,
        action_embedding: AutoregressiveActionEmbedding,
        expert: AutoregressiveActionExpert,
        *,
        mean_mse_weight: float = 0.0,
        log_scale_regularization: float = 0.0,
    ) -> None:
        super().__init__()
        if condition_projector.model_dim != action_embedding.model_dim:
            raise ValueError("condition and action projection widths must match")
        if action_embedding.model_dim != expert.model_dim:
            raise ValueError("action embedding and expert widths must match")
        if action_embedding.action_dim != expert.action_dim:
            raise ValueError("action embedding and expert action widths must match")
        if mean_mse_weight < 0.0 or log_scale_regularization < 0.0:
            raise ValueError("AR loss regularization weights must be non-negative")
        self.condition_projector = condition_projector
        self.action_embedding = action_embedding
        self.expert = expert
        self.mean_mse_weight = mean_mse_weight
        self.log_scale_regularization = log_scale_regularization

    def encode_memory(
        self,
        condition: ConditionMemory,
        state: torch.Tensor,
        state_valid: torch.Tensor | None = None,
    ) -> PolicyMemory:
        return self.condition_projector(condition, state, state_valid)

    def distribution(
        self,
        condition: ConditionMemory,
        state: torch.Tensor,
        actions: torch.Tensor,
        action_valid: torch.Tensor,
        *,
        state_valid: torch.Tensor | None = None,
    ) -> ActionDistribution:
        memory = self.encode_memory(condition, state, state_valid)
        tokens = self.action_embedding(actions, action_valid)
        return self.expert(tokens, memory, action_valid)

    def training_step(
        self,
        condition: ConditionMemory,
        state: torch.Tensor,
        expert_actions: torch.Tensor,
        action_valid: torch.Tensor,
        *,
        state_valid: torch.Tensor | None = None,
    ) -> AutoregressiveTrainingOutput:
        distribution = self.distribution(
            condition,
            state,
            expert_actions,
            action_valid,
            state_valid=state_valid,
        )
        loss = masked_gaussian_nll(
            distribution,
            expert_actions,
            action_valid,
        )
        valid = action_valid.to(torch.float32)
        denominator = valid.sum()
        if self.mean_mse_weight:
            mean_mse = (
                (distribution.mean - expert_actions).square() * valid
            ).sum() / denominator
            loss = loss + self.mean_mse_weight * mean_mse
        if self.log_scale_regularization:
            scale_penalty = (
                distribution.log_scale.square() * valid
            ).sum() / denominator
            loss = loss + self.log_scale_regularization * scale_penalty
        return AutoregressiveTrainingOutput(loss, distribution)

    @torch.no_grad()
    def sample(
        self,
        condition: ConditionMemory,
        state: torch.Tensor,
        action_valid: torch.Tensor,
        *,
        state_valid: torch.Tensor | None = None,
        stochastic: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if action_valid.dtype != torch.bool or action_valid.ndim != 3:
            raise ValueError("action_valid must have bool shape [B,H,D]")
        if action_valid.shape[-1] != self.expert.action_dim:
            raise ValueError("action_valid width does not match the expert")
        actions = torch.zeros(
            action_valid.shape,
            dtype=torch.float32,
            device=action_valid.device,
        )
        memory = self.encode_memory(condition, state, state_valid)
        for step in range(action_valid.shape[1]):
            tokens = self.action_embedding(actions, action_valid)
            distribution = self.expert(tokens, memory, action_valid)
            next_action = distribution.mean[:, step]
            if stochastic:
                noise = torch.randn(
                    next_action.shape,
                    dtype=torch.float32,
                    device=next_action.device,
                    generator=generator,
                )
                next_action = next_action + distribution.log_scale[
                    :, step
                ].exp() * noise
            actions[:, step] = next_action.masked_fill(
                ~action_valid[:, step],
                0.0,
            )
        return actions


def masked_gaussian_nll(
    distribution: ActionDistribution,
    target: torch.Tensor,
    action_valid: torch.Tensor,
) -> torch.Tensor:
    """Return scalar-element masked diagonal-Gaussian negative log likelihood."""

    if target.dtype != torch.float32 or target.shape != distribution.mean.shape:
        raise ValueError("target must be float32 and match the distribution")
    if action_valid.dtype != torch.bool or action_valid.shape != target.shape:
        raise ValueError("action_valid must be bool and match the target")
    if not torch.equal(distribution.valid, action_valid):
        raise ValueError("action_valid must match the distribution validity mask")
    if not (
        target.device == action_valid.device == distribution.mean.device
    ):
        raise ValueError("loss tensors must share a device")
    if not torch.isfinite(target).all():
        raise ValueError("target must be finite")
    if not action_valid.reshape(target.shape[0], -1).any(dim=1).all():
        raise ValueError("every sample must contain a valid action element")
    inverse_scale = torch.exp(-distribution.log_scale)
    standardized = (target - distribution.mean) * inverse_scale
    nll = (
        0.5 * standardized.square()
        + distribution.log_scale
        + 0.5 * math.log(2.0 * math.pi)
    )
    valid = action_valid.to(torch.float32)
    return (nll * valid).sum() / valid.sum()

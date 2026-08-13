"""Compact conditioned action decoder for flow velocity prediction."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from pi0_minimal.models.action_embedding import ActionTokens
from pi0_minimal.models.attention import AttentionMasks, build_attention_masks
from pi0_minimal.models.condition import PolicyMemory


class RMSNorm(nn.Module):
    """RMS normalization with FP32 variance accumulation."""

    def __init__(self, dim: int, *, eps: float = 1e-6) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("RMSNorm dim must be positive")
        if eps <= 0.0:
            raise ValueError("RMSNorm eps must be positive")
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        variance = values.float().square().mean(dim=-1, keepdim=True)
        normalized = values * torch.rsqrt(variance + self.eps).to(values.dtype)
        return normalized * self.weight


class MultiHeadAttention(nn.Module):
    """Independent multi-head attention with an explicit allowed-pair mask."""

    def __init__(
        self,
        *,
        model_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if model_dim <= 0 or num_heads <= 0 or model_dim % num_heads:
            raise ValueError("model_dim must be positive and divisible by num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.dropout = dropout
        self.query = nn.Linear(model_dim, model_dim, bias=False)
        self.key = nn.Linear(model_dim, model_dim, bias=False)
        self.value = nn.Linear(model_dim, model_dim, bias=False)
        self.output = nn.Linear(model_dim, model_dim, bias=False)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        allowed: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(queries, keys, values, allowed)
        query = self._split_heads(self.query(queries))
        key = self._split_heads(self.key(keys))
        value = self._split_heads(self.value(values))
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed.unsqueeze(1),
            dropout_p=self.dropout if self.training else 0.0,
        )
        merged = attended.transpose(1, 2).reshape(
            queries.shape[0],
            queries.shape[1],
            self.model_dim,
        )
        return self.output(merged)

    def _split_heads(self, values: torch.Tensor) -> torch.Tensor:
        return values.reshape(
            values.shape[0],
            values.shape[1],
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

    def _validate_inputs(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        allowed: torch.Tensor,
    ) -> None:
        if queries.ndim != 3 or keys.ndim != 3 or values.ndim != 3:
            raise ValueError("attention inputs must have shape [B,S,D]")
        if keys.shape != values.shape:
            raise ValueError("attention keys and values must have identical shapes")
        if (
            queries.shape[0] != keys.shape[0]
            or queries.shape[-1] != self.model_dim
            or keys.shape[-1] != self.model_dim
        ):
            raise ValueError("attention batch or model dimensions are inconsistent")
        expected_mask = (queries.shape[0], queries.shape[1], keys.shape[1])
        if allowed.dtype != torch.bool or allowed.shape != expected_mask:
            raise ValueError(f"allowed mask must have bool shape {expected_mask}")
        if not (
            queries.device == keys.device == values.device == allowed.device
        ):
            raise ValueError("attention inputs and mask must share a device")


class SwiGLUFeedForward(nn.Module):
    """Three-projection SwiGLU feed-forward network."""

    def __init__(self, *, model_dim: int, ffn_dim: int) -> None:
        super().__init__()
        if model_dim <= 0 or ffn_dim < model_dim:
            raise ValueError("ffn_dim must be at least model_dim")
        self.gate = nn.Linear(model_dim, ffn_dim, bias=False)
        self.value = nn.Linear(model_dim, ffn_dim, bias=False)
        self.output = nn.Linear(ffn_dim, model_dim, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.output(F.silu(self.gate(values)) * self.value(values))


class FlowDecoderBlock(nn.Module):
    """One pre-norm self-attention, cross-attention, and SwiGLU block."""

    def __init__(
        self,
        *,
        model_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_norm = RMSNorm(model_dim)
        self.self_attention = MultiHeadAttention(
            model_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.cross_norm = RMSNorm(model_dim)
        self.cross_attention = MultiHeadAttention(
            model_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.ffn_norm = RMSNorm(model_dim)
        self.feed_forward = SwiGLUFeedForward(
            model_dim=model_dim,
            ffn_dim=ffn_dim,
        )

    def forward(
        self,
        action_values: torch.Tensor,
        memory_values: torch.Tensor,
        masks: AttentionMasks,
        action_valid: torch.Tensor,
    ) -> torch.Tensor:
        normalized_actions = self.self_norm(action_values)
        action_values = action_values + self.self_attention(
            normalized_actions,
            normalized_actions,
            normalized_actions,
            masks.self_allowed,
        )
        action_values = _zero_invalid_queries(action_values, action_valid)
        action_values = action_values + self.cross_attention(
            self.cross_norm(action_values),
            memory_values,
            memory_values,
            masks.cross_allowed,
        )
        action_values = _zero_invalid_queries(action_values, action_valid)
        action_values = action_values + self.feed_forward(self.ffn_norm(action_values))
        return _zero_invalid_queries(action_values, action_valid)


class FlowActionExpert(nn.Module):
    """Eight-layer compact decoder predicting FP32 flow velocity."""

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
    ) -> None:
        super().__init__()
        if num_layers <= 0 or action_dim <= 0 or max_horizon <= 0:
            raise ValueError("num_layers, action_dim, and max_horizon must be positive")
        self.model_dim = model_dim
        self.action_dim = action_dim
        self.max_horizon = max_horizon
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
        self.velocity_head = nn.Linear(model_dim, action_dim)

    def forward(
        self,
        action_tokens: ActionTokens,
        memory: PolicyMemory,
    ) -> torch.Tensor:
        self._validate_inputs(action_tokens, memory)
        compute_dtype = self.position_embedding.dtype
        horizon = action_tokens.values.shape[1]
        action_values = action_tokens.values.to(compute_dtype)
        action_values = action_values + self.position_embedding[:horizon].unsqueeze(0)
        action_values = _zero_invalid_queries(action_values, action_tokens.valid)
        memory_values = memory.values.to(compute_dtype)
        memory_values = memory_values.masked_fill(~memory.valid.unsqueeze(-1), 0.0)
        masks = build_attention_masks(action_tokens.valid, memory.valid)

        for block in self.blocks:
            action_values = block(
                action_values,
                memory_values,
                masks,
                action_tokens.valid,
            )
        action_values = self.final_norm(action_values)
        velocity = self.velocity_head(action_values).float()
        return velocity.masked_fill(~action_tokens.valid.unsqueeze(-1), 0.0)

    def _validate_inputs(
        self,
        action_tokens: ActionTokens,
        memory: PolicyMemory,
    ) -> None:
        if action_tokens.values.shape[-1] != self.model_dim:
            raise ValueError(
                f"action token width must be {self.model_dim}, "
                f"got {action_tokens.values.shape[-1]}"
            )
        if memory.values.shape[-1] != self.model_dim:
            raise ValueError(
                f"memory width must be {self.model_dim}, got {memory.values.shape[-1]}"
            )
        if action_tokens.values.shape[0] != memory.values.shape[0]:
            raise ValueError("action tokens and memory must share the batch dimension")
        if action_tokens.values.shape[1] > self.max_horizon:
            raise ValueError(
                f"action horizon exceeds configured maximum {self.max_horizon}"
            )
        devices = {
            action_tokens.values.device,
            action_tokens.valid.device,
            memory.values.device,
            memory.valid.device,
            self.position_embedding.device,
        }
        if len(devices) != 1:
            raise ValueError("expert inputs and parameters must share a device")
        if not action_tokens.valid.any(dim=1).all():
            raise ValueError("every sample must contain at least one valid action token")
        if not memory.valid.any(dim=1).all():
            raise ValueError("every sample must contain at least one valid memory token")


def _zero_invalid_queries(
    values: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    return values.masked_fill(~valid.unsqueeze(-1), 0.0)

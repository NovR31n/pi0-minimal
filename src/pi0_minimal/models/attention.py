"""Explicit bidirectional action and cross-attention masks."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class AttentionMasks:
    """Allowed attention pairs for action self-attention and condition memory."""

    self_allowed: torch.Tensor
    cross_allowed: torch.Tensor

    def __post_init__(self) -> None:
        if self.self_allowed.dtype != torch.bool or self.self_allowed.ndim != 3:
            raise ValueError("self_allowed must have bool shape [B,H,H]")
        batch_size, query_length, key_length = self.self_allowed.shape
        if query_length != key_length:
            raise ValueError("self_allowed must be square over action positions")
        expected_cross_prefix = (batch_size, query_length)
        if (
            self.cross_allowed.dtype != torch.bool
            or self.cross_allowed.ndim != 3
            or self.cross_allowed.shape[:2] != expected_cross_prefix
        ):
            raise ValueError("cross_allowed must have bool shape [B,H,M]")
        if self.self_allowed.device != self.cross_allowed.device:
            raise ValueError("attention masks must share a device")


def build_attention_masks(
    action_valid: torch.Tensor,
    memory_valid: torch.Tensor,
) -> AttentionMasks:
    """Build the exact non-causal masks specified for the flow decoder."""

    if action_valid.dtype != torch.bool or action_valid.ndim != 2:
        raise ValueError("action_valid must have bool shape [B,H]")
    if memory_valid.dtype != torch.bool or memory_valid.ndim != 2:
        raise ValueError("memory_valid must have bool shape [B,M]")
    if action_valid.shape[0] != memory_valid.shape[0]:
        raise ValueError("action and memory masks must share the batch dimension")
    if action_valid.device != memory_valid.device:
        raise ValueError("action and memory masks must share a device")
    if not action_valid.any(dim=1).all():
        raise ValueError("every sample must contain at least one valid action token")
    if not memory_valid.any(dim=1).all():
        raise ValueError("every sample must contain at least one valid memory token")

    action_queries = action_valid.unsqueeze(-1)
    self_allowed = action_queries & action_valid.unsqueeze(1)
    cross_allowed = action_queries & memory_valid.unsqueeze(1)
    return AttentionMasks(self_allowed, cross_allowed)


def build_causal_attention_masks(
    action_valid: torch.Tensor,
    memory_valid: torch.Tensor,
) -> AttentionMasks:
    """Build causal action self-attention and unrestricted condition cross-attention."""

    masks = build_attention_masks(action_valid, memory_valid)
    horizon = action_valid.shape[1]
    causal = torch.ones(
        (horizon, horizon),
        dtype=torch.bool,
        device=action_valid.device,
    ).tril()
    return AttentionMasks(masks.self_allowed & causal.unsqueeze(0), masks.cross_allowed)

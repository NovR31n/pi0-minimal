"""Paper-direction conditional flow-matching targets and masked loss."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class FlowMatchingBatch:
    """One injected or randomly sampled conditional flow-matching problem."""

    noisy_actions: torch.Tensor
    target_velocity: torch.Tensor
    flow_time: torch.Tensor
    noise: torch.Tensor
    action_valid: torch.Tensor

    def __post_init__(self) -> None:
        action_shape = self.noisy_actions.shape
        for name, value in (
            ("noisy_actions", self.noisy_actions),
            ("target_velocity", self.target_velocity),
            ("noise", self.noise),
        ):
            if value.dtype != torch.float32 or value.ndim != 3:
                raise TypeError(f"{name} must be float32 [B,H,D]")
            if value.shape != action_shape:
                raise ValueError("flow action tensors must have identical shapes")
            if value.device != self.noisy_actions.device:
                raise ValueError("flow action tensors must share a device")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain only finite values")
        if (
            self.flow_time.dtype != torch.float32
            or self.flow_time.shape != (action_shape[0],)
        ):
            raise ValueError("flow_time must have float32 shape [B]")
        if self.flow_time.device != self.noisy_actions.device:
            raise ValueError("flow_time and actions must share a device")
        if not torch.isfinite(self.flow_time).all():
            raise ValueError("flow_time must contain only finite values")
        if torch.any((self.flow_time < 0.0) | (self.flow_time > 1.0)):
            raise ValueError("flow_time must lie in [0,1]")
        if (
            self.action_valid.dtype != torch.bool
            or self.action_valid.shape != action_shape
        ):
            raise ValueError("action_valid must have bool shape [B,H,D]")
        if self.action_valid.device != self.noisy_actions.device:
            raise ValueError("action_valid and actions must share a device")
        if not self.action_valid.reshape(action_shape[0], -1).any(dim=1).all():
            raise ValueError("every sample must contain at least one valid action element")


def sample_paper_flow_time(
    batch_size: int,
    *,
    device: torch.device | str,
    generator: torch.Generator | None = None,
    beta_alpha: float = 1.5,
    beta_beta: float = 1.0,
    cutoff: float = 0.999,
) -> torch.Tensor:
    """Sample τ=cutoff*(1-z), z~Beta(alpha,1), using an injectable RNG."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if beta_alpha <= 0.0 or beta_beta != 1.0:
        raise ValueError("paper flow-time sampler requires alpha>0 and beta=1")
    if not 0.0 < cutoff < 1.0:
        raise ValueError("cutoff must lie strictly between zero and one")
    uniform = torch.rand(
        (batch_size,),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    beta_sample = uniform.pow(1.0 / beta_alpha)
    return cutoff * (1.0 - beta_sample)


def interpolate_actions(
    expert_actions: torch.Tensor,
    noise: torch.Tensor,
    flow_time: torch.Tensor,
) -> torch.Tensor:
    """Return x_τ=τA+(1-τ)ε in FP32."""

    _validate_action_pair(expert_actions, noise)
    _validate_flow_time(flow_time, expert_actions)
    expanded_time = flow_time[:, None, None]
    return expanded_time * expert_actions + (1.0 - expanded_time) * noise


def target_velocity(
    expert_actions: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Return the paper-direction constant target A-ε."""

    _validate_action_pair(expert_actions, noise)
    return expert_actions - noise


def build_flow_matching_batch(
    expert_actions: torch.Tensor,
    action_valid: torch.Tensor,
    *,
    noise: torch.Tensor | None = None,
    flow_time: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    beta_alpha: float = 1.5,
    beta_beta: float = 1.0,
    cutoff: float = 0.999,
) -> FlowMatchingBatch:
    """Construct noisy actions and targets, allowing exact test injection."""

    _validate_actions_and_mask(expert_actions, action_valid)
    if noise is None:
        noise = torch.randn(
            expert_actions.shape,
            dtype=torch.float32,
            device=expert_actions.device,
            generator=generator,
        )
    if flow_time is None:
        flow_time = sample_paper_flow_time(
            expert_actions.shape[0],
            device=expert_actions.device,
            generator=generator,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
            cutoff=cutoff,
        )
    noisy_actions = interpolate_actions(expert_actions, noise, flow_time)
    velocity = target_velocity(expert_actions, noise)
    return FlowMatchingBatch(
        noisy_actions=noisy_actions,
        target_velocity=velocity,
        flow_time=flow_time,
        noise=noise,
        action_valid=action_valid,
    )


def masked_flow_matching_loss(
    predicted_velocity: torch.Tensor,
    target: torch.Tensor,
    action_valid: torch.Tensor,
) -> torch.Tensor:
    """Return scalar-element masked MSE with FP32 accumulation."""

    _validate_action_pair(predicted_velocity, target)
    _validate_actions_and_mask(target, action_valid)
    squared_error = (predicted_velocity.float() - target.float()).square()
    valid = action_valid.to(torch.float32)
    return (squared_error * valid).sum() / valid.sum()


def _validate_action_pair(
    first: torch.Tensor,
    second: torch.Tensor,
) -> None:
    for name, value in (("first action tensor", first), ("second action tensor", second)):
        if value.dtype != torch.float32 or value.ndim != 3:
            raise TypeError(f"{name} must be float32 [B,H,D]")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")
    if first.shape != second.shape:
        raise ValueError("action tensors must have identical shapes")
    if first.device != second.device:
        raise ValueError("action tensors must share a device")


def _validate_flow_time(
    flow_time: torch.Tensor,
    actions: torch.Tensor,
) -> None:
    if flow_time.dtype != torch.float32 or flow_time.shape != (actions.shape[0],):
        raise ValueError("flow_time must have float32 shape [B]")
    if flow_time.device != actions.device:
        raise ValueError("flow_time and actions must share a device")
    if not torch.isfinite(flow_time).all():
        raise ValueError("flow_time must contain only finite values")
    if torch.any((flow_time < 0.0) | (flow_time > 1.0)):
        raise ValueError("flow_time must lie in [0,1]")


def _validate_actions_and_mask(
    actions: torch.Tensor,
    action_valid: torch.Tensor,
) -> None:
    if actions.dtype != torch.float32 or actions.ndim != 3:
        raise TypeError("actions must be float32 [B,H,D]")
    if not torch.isfinite(actions).all():
        raise ValueError("actions must contain only finite values")
    if action_valid.dtype != torch.bool or action_valid.shape != actions.shape:
        raise ValueError("action_valid must have bool shape [B,H,D]")
    if action_valid.device != actions.device:
        raise ValueError("actions and action_valid must share a device")
    if not action_valid.reshape(actions.shape[0], -1).any(dim=1).all():
        raise ValueError("every sample must contain at least one valid action element")

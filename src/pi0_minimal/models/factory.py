"""Construct the committed flow policy from a validated model specification."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from pi0_minimal.models.action_embedding import ActionTimeEmbedding
from pi0_minimal.models.autoregressive import (
    AutoregressiveActionEmbedding,
    AutoregressiveActionExpert,
    AutoregressivePolicy,
)
from pi0_minimal.models.condition import ConditionProjector
from pi0_minimal.models.expert import FlowActionExpert
from pi0_minimal.models.policy import FlowPolicy


def build_flow_policy(
    spec: Mapping[str, object],
    *,
    device: str | torch.device | None = None,
) -> FlowPolicy:
    """Build the trainable flow stack without loading the frozen backbone."""

    observation = _table(spec, "observation")
    action = _table(spec, "action")
    backbone = _table(spec, "backbone")
    expert = _table(spec, "action_expert")
    flow = _table(spec, "flow")
    policy = FlowPolicy(
        ConditionProjector(
            condition_dim=int(backbone["output_dim"]),
            state_dim=int(observation["state_dim"]),
            model_dim=int(expert["model_dim"]),
        ),
        ActionTimeEmbedding(
            action_dim=int(action["dim"]),
            model_dim=int(expert["model_dim"]),
            time_dim=int(expert["time_embedding_dim"]),
        ),
        FlowActionExpert(
            model_dim=int(expert["model_dim"]),
            num_layers=int(expert["num_layers"]),
            num_heads=int(expert["num_heads"]),
            ffn_dim=int(expert["ffn_dim"]),
            action_dim=int(action["dim"]),
            max_horizon=int(action["horizon"]),
            dropout=float(expert["dropout"]),
        ),
        beta_alpha=float(flow["beta_alpha"]),
        beta_beta=float(flow["beta_beta"]),
        cutoff=float(flow["cutoff"]),
        num_euler_steps=int(flow["num_euler_steps"]),
        smoothness_weight=float(flow.get("smoothness_weight", 0.0)),
    )
    return policy if device is None else policy.to(device)


def build_autoregressive_policy(
    spec: Mapping[str, object],
    *,
    device: str | torch.device | None = None,
) -> AutoregressivePolicy:
    """Build the matched continuous autoregressive action stack."""

    observation = _table(spec, "observation")
    action = _table(spec, "action")
    backbone = _table(spec, "backbone")
    expert = _table(spec, "action_expert")
    autoregressive = _table(spec, "autoregressive")
    policy = AutoregressivePolicy(
        ConditionProjector(
            condition_dim=int(backbone["output_dim"]),
            state_dim=int(observation["state_dim"]),
            model_dim=int(expert["model_dim"]),
        ),
        AutoregressiveActionEmbedding(
            action_dim=int(action["dim"]),
            model_dim=int(expert["model_dim"]),
            mode_dim=int(expert["time_embedding_dim"]),
        ),
        AutoregressiveActionExpert(
            model_dim=int(expert["model_dim"]),
            num_layers=int(expert["num_layers"]),
            num_heads=int(expert["num_heads"]),
            ffn_dim=int(expert["ffn_dim"]),
            action_dim=int(action["dim"]),
            max_horizon=int(action["horizon"]),
            dropout=float(expert["dropout"]),
            log_scale_min=float(autoregressive["log_scale_min"]),
            log_scale_max=float(autoregressive["log_scale_max"]),
        ),
        mean_mse_weight=float(autoregressive.get("mean_mse_weight", 0.0)),
        log_scale_regularization=float(
            autoregressive.get("log_scale_regularization", 0.0)
        ),
    )
    return policy if device is None else policy.to(device)


def _table(spec: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = spec[key]
    if not isinstance(value, Mapping):
        raise TypeError(f"model specification field {key!r} must be a table")
    return value

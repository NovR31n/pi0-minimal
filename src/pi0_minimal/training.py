"""Small, testable training utilities shared by flow experiments."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

CHECKPOINT_SCHEMA_VERSION = 1


def save_training_checkpoint(
    path: str | Path,
    *,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    best_validation_loss: float,
    flow_generator: torch.Generator,
    batch_rng_state: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    """Atomically save all state required for exact optimizer continuation."""

    if step < 0:
        raise ValueError("checkpoint step must be non-negative")
    if not torch.isfinite(torch.tensor(best_validation_loss)):
        raise ValueError("best validation loss must be finite")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "step": step,
            "best_validation_loss": best_validation_loss,
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "flow_generator_state": flow_generator.get_state(),
            "batch_rng_state": batch_rng_state,
            "metadata": metadata,
        },
        temporary,
    )
    os.replace(temporary, destination)


def load_training_checkpoint(
    path: str | Path,
    *,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    flow_generator: torch.Generator | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Restore a checkpoint and return its validated non-model training state."""

    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload must be a dictionary")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema: {payload.get('schema_version')}")
    required = {
        "step",
        "best_validation_loss",
        "policy",
        "optimizer",
        "scheduler",
        "flow_generator_state",
        "batch_rng_state",
        "metadata",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"checkpoint is missing fields: {sorted(missing)}")
    step = payload["step"]
    best_validation_loss = payload["best_validation_loss"]
    if not isinstance(step, int) or step < 0:
        raise ValueError("checkpoint step must be a non-negative integer")
    if not isinstance(best_validation_loss, float) or not torch.isfinite(
        torch.tensor(best_validation_loss)
    ):
        raise ValueError("checkpoint best validation loss must be finite")
    if not isinstance(payload["batch_rng_state"], dict):
        raise TypeError("checkpoint batch RNG state must be a dictionary")
    if not isinstance(payload["metadata"], dict):
        raise TypeError("checkpoint metadata must be a dictionary")

    policy.load_state_dict(payload["policy"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if flow_generator is not None:
        flow_generator.set_state(payload["flow_generator_state"].cpu())
    return {
        "step": step,
        "best_validation_loss": best_validation_loss,
        "batch_rng_state": payload["batch_rng_state"],
        "metadata": payload["metadata"],
    }

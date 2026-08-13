from pathlib import Path

import numpy as np
import pytest
import torch

from pi0_minimal.models import (
    ActionTimeEmbedding,
    ConditionMemory,
    ConditionProjector,
    FlowActionExpert,
    FlowPolicy,
)
from pi0_minimal.training import load_training_checkpoint, save_training_checkpoint


def _policy() -> FlowPolicy:
    return FlowPolicy(
        ConditionProjector(condition_dim=6, state_dim=3, model_dim=16),
        ActionTimeEmbedding(action_dim=2, model_dim=16, time_dim=8),
        FlowActionExpert(
            model_dim=16,
            num_layers=2,
            num_heads=4,
            ffn_dim=32,
            action_dim=2,
            max_horizon=4,
        ),
    )


def _optimizer_and_scheduler(
    policy: FlowPolicy,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=10,
    )
    return optimizer, scheduler


def test_checkpoint_restores_policy_optimizer_scheduler_and_rng(tmp_path: Path) -> None:
    torch.manual_seed(7)
    policy = _policy()
    optimizer, scheduler = _optimizer_and_scheduler(policy)
    condition = ConditionMemory(torch.randn(2, 5, 6), torch.ones(2, 5, dtype=torch.bool))
    state = torch.randn(2, 3)
    actions = torch.randn(2, 4, 2)
    valid = torch.ones_like(actions, dtype=torch.bool)
    flow_generator = torch.Generator().manual_seed(17)
    loss = policy.training_step(
        condition,
        state,
        actions,
        valid,
        generator=flow_generator,
    ).loss
    loss.backward()
    optimizer.step()
    scheduler.step()
    expected_generator_state = flow_generator.get_state().clone()
    expected_parameters = {
        name: parameter.detach().clone() for name, parameter in policy.named_parameters()
    }
    expected_optimizer = optimizer.state_dict()
    checkpoint = tmp_path / "latest.pt"

    save_training_checkpoint(
        checkpoint,
        policy=policy,
        optimizer=optimizer,
        scheduler=scheduler,
        step=1,
        best_validation_loss=0.25,
        flow_generator=flow_generator,
        batch_rng_state=np.random.default_rng(23).bit_generator.state,
        metadata={"split_fingerprint": "abc"},
    )

    restored = _policy()
    restored_optimizer, restored_scheduler = _optimizer_and_scheduler(restored)
    restored_generator = torch.Generator().manual_seed(999)
    state_payload = load_training_checkpoint(
        checkpoint,
        policy=restored,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        flow_generator=restored_generator,
    )

    assert state_payload["step"] == 1
    assert state_payload["best_validation_loss"] == 0.25
    assert state_payload["metadata"]["split_fingerprint"] == "abc"
    for name, parameter in restored.named_parameters():
        torch.testing.assert_close(parameter, expected_parameters[name])
    assert restored_optimizer.state_dict()["param_groups"] == expected_optimizer["param_groups"]
    assert restored_scheduler.last_epoch == scheduler.last_epoch
    torch.testing.assert_close(restored_generator.get_state(), expected_generator_state)


def test_checkpoint_rejects_unknown_schema(tmp_path: Path) -> None:
    checkpoint = tmp_path / "bad.pt"
    torch.save({"schema_version": 999}, checkpoint)

    with pytest.raises(ValueError, match="unsupported checkpoint schema"):
        load_training_checkpoint(checkpoint, policy=_policy())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_checkpoint_rng_state_survives_cuda_map_location(tmp_path: Path) -> None:
    policy = _policy().cuda()
    optimizer, scheduler = _optimizer_and_scheduler(policy)
    generator = torch.Generator(device="cuda").manual_seed(31)
    checkpoint = tmp_path / "cuda.pt"
    save_training_checkpoint(
        checkpoint,
        policy=policy,
        optimizer=optimizer,
        scheduler=scheduler,
        step=0,
        best_validation_loss=1.0,
        flow_generator=generator,
        batch_rng_state=np.random.default_rng(31).bit_generator.state,
        metadata={},
    )

    load_training_checkpoint(
        checkpoint,
        policy=policy,
        flow_generator=generator,
        map_location="cuda",
    )

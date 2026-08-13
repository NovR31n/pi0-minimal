import pytest
import torch

from pi0_minimal.models import (
    ActionTimeEmbedding,
    ConditionMemory,
    ConditionProjector,
    FlowActionExpert,
    FlowPolicy,
)


def _policy(*, smoothness_weight: float = 0.0) -> FlowPolicy:
    torch.manual_seed(7)
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
        num_euler_steps=4,
        smoothness_weight=smoothness_weight,
    )


def _inputs() -> tuple[
    ConditionMemory,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    torch.manual_seed(17)
    condition = ConditionMemory(
        torch.randn((2, 5, 6)),
        torch.tensor(
            [[True, True, True, False, False], [True, True, False, False, True]]
        ),
    )
    state = torch.randn((2, 3), dtype=torch.float32)
    actions = torch.randn((2, 4, 2), dtype=torch.float32)
    valid = torch.tensor(
        [
            [[True, True], [True, True], [True, True], [False, False]],
            [[True, True], [True, False], [False, False], [False, False]],
        ]
    )
    noise = torch.randn_like(actions)
    flow_time = torch.tensor([0.25, 0.75], dtype=torch.float32)
    return condition, state, actions, valid, noise, flow_time


def test_training_step_connects_exact_flow_target_to_expert_loss() -> None:
    condition, state, actions, valid, noise, flow_time = _inputs()
    policy = _policy()

    output = policy.training_step(
        condition,
        state,
        actions,
        valid,
        noise=noise,
        flow_time=flow_time,
    )

    assert output.loss.shape == ()
    assert output.loss.dtype == torch.float32
    assert torch.isfinite(output.loss)
    assert output.predicted_velocity.shape == actions.shape
    torch.testing.assert_close(output.flow_batch.target_velocity, actions - noise)
    assert torch.count_nonzero(output.predicted_velocity[~valid.any(dim=-1)]) == 0


def test_training_step_backpropagates_to_every_policy_parameter() -> None:
    condition, state, actions, valid, noise, flow_time = _inputs()
    policy = _policy()

    output = policy.training_step(
        condition,
        state,
        actions,
        valid,
        noise=noise,
        flow_time=flow_time,
    )
    output.loss.backward()

    assert all(parameter.grad is not None for parameter in policy.parameters())
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in policy.parameters()
    )


def test_smoothness_regularization_matches_masked_clean_action_delta_mse() -> None:
    condition, state, actions, valid, noise, flow_time = _inputs()
    baseline = _policy().eval()
    regularized = _policy(smoothness_weight=0.25).eval()
    regularized.load_state_dict(baseline.state_dict())

    first = baseline.training_step(
        condition,
        state,
        actions,
        valid,
        noise=noise,
        flow_time=flow_time,
    )
    second = regularized.training_step(
        condition,
        state,
        actions,
        valid,
        noise=noise,
        flow_time=flow_time,
    )
    time = first.flow_batch.flow_time[:, None, None]
    predicted_clean = (
        first.flow_batch.noisy_actions
        + (1.0 - time) * first.predicted_velocity
    )
    pair_valid = valid[:, 1:] & valid[:, :-1]
    squared_error = (
        (predicted_clean[:, 1:] - predicted_clean[:, :-1])
        - (actions[:, 1:] - actions[:, :-1])
    ).square()
    expected_penalty = squared_error[pair_valid].mean()

    torch.testing.assert_close(
        second.loss,
        first.loss + 0.25 * expected_penalty,
    )


def test_padding_changes_cannot_affect_training_loss() -> None:
    condition, state, actions, valid, noise, flow_time = _inputs()
    changed_actions = actions.masked_fill(~valid, 999.0)
    changed_noise = noise.masked_fill(~valid, -999.0)
    policy = _policy(smoothness_weight=0.25).eval()

    first = policy.training_step(
        condition,
        state,
        actions,
        valid,
        noise=noise,
        flow_time=flow_time,
    )
    second = policy.training_step(
        condition,
        state,
        changed_actions,
        valid,
        noise=changed_noise,
        flow_time=flow_time,
    )

    torch.testing.assert_close(first.loss, second.loss)
    torch.testing.assert_close(
        first.predicted_velocity[valid],
        second.predicted_velocity[valid],
    )


def test_policy_rejects_negative_smoothness_weight() -> None:
    with pytest.raises(ValueError, match="smoothness_weight"):
        _policy(smoothness_weight=-0.1)


def test_policy_sampling_is_deterministic_for_fixed_noise() -> None:
    condition, state, actions, valid, noise, _flow_time = _inputs()
    policy = _policy().eval()

    first = policy.sample(
        condition,
        state,
        valid,
        initial_noise=noise,
    )
    second = policy.sample(
        condition,
        state,
        valid,
        initial_noise=noise,
    )

    assert first.shape == actions.shape
    assert first.dtype == torch.float32
    assert torch.isfinite(first).all()
    assert not first.requires_grad
    torch.testing.assert_close(first, second)
    assert torch.count_nonzero(first[~valid]) == 0


def test_policy_supports_bfloat16_modules_with_fp32_loss_and_sampling() -> None:
    condition, state, actions, valid, noise, flow_time = _inputs()
    policy = _policy().to(torch.bfloat16)

    output = policy.training_step(
        condition,
        state,
        actions,
        valid,
        noise=noise,
        flow_time=flow_time,
    )
    sampled = policy.sample(
        condition,
        state,
        valid,
        initial_noise=noise,
        num_steps=2,
    )

    assert output.loss.dtype == torch.float32
    assert sampled.dtype == torch.float32
    assert torch.isfinite(output.loss)
    assert torch.isfinite(sampled).all()

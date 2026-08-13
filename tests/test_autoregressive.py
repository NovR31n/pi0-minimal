import math

import pytest
import torch

from pi0_minimal.models import (
    ActionDistribution,
    AutoregressiveActionEmbedding,
    AutoregressiveActionExpert,
    AutoregressivePolicy,
    ConditionMemory,
    ConditionProjector,
    masked_gaussian_nll,
)


def _policy(
    *,
    mean_mse_weight: float = 0.0,
    log_scale_regularization: float = 0.0,
) -> AutoregressivePolicy:
    torch.manual_seed(7)
    return AutoregressivePolicy(
        ConditionProjector(condition_dim=6, state_dim=3, model_dim=16),
        AutoregressiveActionEmbedding(
            action_dim=2,
            model_dim=16,
            mode_dim=8,
        ),
        AutoregressiveActionExpert(
            model_dim=16,
            num_layers=2,
            num_heads=4,
            ffn_dim=32,
            action_dim=2,
            max_horizon=4,
        ),
        mean_mse_weight=mean_mse_weight,
        log_scale_regularization=log_scale_regularization,
    )


def _inputs() -> tuple[
    ConditionMemory,
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
    return condition, state, actions, valid


def test_masked_gaussian_nll_matches_unit_normal_and_ignores_padding() -> None:
    mean = torch.zeros((1, 2, 2), dtype=torch.float32)
    log_scale = torch.zeros_like(mean)
    valid = torch.tensor([[[True, False], [True, True]]])
    target = torch.zeros_like(mean)
    distribution = ActionDistribution(mean, log_scale, valid)

    first = masked_gaussian_nll(distribution, target, valid)
    changed_target = target.masked_fill(~valid, 10_000.0)
    second = masked_gaussian_nll(distribution, changed_target, valid)

    torch.testing.assert_close(
        first,
        torch.tensor(0.5 * math.log(2.0 * math.pi)),
    )
    torch.testing.assert_close(first, second)


def test_teacher_forcing_is_shifted_and_causal_without_target_leakage() -> None:
    condition, state, actions, valid = _inputs()
    changed = actions.clone()
    changed[:, 1] += 100.0
    policy = _policy().eval()

    first = policy.distribution(condition, state, actions, valid)
    second = policy.distribution(condition, state, changed, valid)

    # Target A[1] is shifted into token 2. Predictions through position 1
    # therefore cannot observe it.
    torch.testing.assert_close(first.mean[:, :2], second.mean[:, :2])
    torch.testing.assert_close(first.log_scale[:, :2], second.log_scale[:, :2])
    assert not torch.allclose(first.mean[0, 2], second.mean[0, 2])


def test_training_step_backpropagates_to_every_policy_parameter() -> None:
    condition, state, actions, valid = _inputs()
    policy = _policy()

    output = policy.training_step(condition, state, actions, valid)
    output.loss.backward()

    assert output.loss.dtype == torch.float32
    assert torch.isfinite(output.loss)
    assert all(parameter.grad is not None for parameter in policy.parameters())
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in policy.parameters()
    )


def test_regularized_loss_adds_masked_mean_mse_and_log_scale_penalty() -> None:
    condition, state, actions, valid = _inputs()
    baseline = _policy().eval()
    regularized = _policy(
        mean_mse_weight=0.75,
        log_scale_regularization=0.25,
    ).eval()
    regularized.load_state_dict(baseline.state_dict())

    first = baseline.training_step(condition, state, actions, valid)
    second = regularized.training_step(condition, state, actions, valid)
    mean_mse = (first.distribution.mean[valid] - actions[valid]).square().mean()
    scale_penalty = first.distribution.log_scale[valid].square().mean()

    torch.testing.assert_close(
        second.loss,
        first.loss + 0.75 * mean_mse + 0.25 * scale_penalty,
    )


def test_padding_changes_cannot_affect_valid_predictions_or_loss() -> None:
    condition, state, actions, valid = _inputs()
    changed = actions.masked_fill(~valid, 999.0)
    policy = _policy(
        mean_mse_weight=0.75,
        log_scale_regularization=0.25,
    ).eval()

    first = policy.training_step(condition, state, actions, valid)
    second = policy.training_step(condition, state, changed, valid)

    torch.testing.assert_close(first.loss, second.loss)
    torch.testing.assert_close(
        first.distribution.mean[valid],
        second.distribution.mean[valid],
    )
    assert torch.count_nonzero(first.distribution.mean[~valid]) == 0


def test_policy_rejects_negative_regularization_weights() -> None:
    for kwargs in (
        {"mean_mse_weight": -0.1},
        {"log_scale_regularization": -0.1},
    ):
        with pytest.raises(ValueError, match="regularization"):
            _policy(**kwargs)


def test_sequential_generation_is_deterministic_finite_and_masked() -> None:
    condition, state, actions, valid = _inputs()
    policy = _policy().eval()

    first = policy.sample(condition, state, valid)
    second = policy.sample(condition, state, valid)

    assert first.shape == actions.shape
    assert first.dtype == torch.float32
    assert torch.isfinite(first).all()
    assert not first.requires_grad
    torch.testing.assert_close(first, second)
    assert torch.count_nonzero(first[~valid]) == 0


def test_policy_supports_bfloat16_modules_with_fp32_loss_and_sampling() -> None:
    condition, state, actions, valid = _inputs()
    policy = _policy().to(torch.bfloat16)

    output = policy.training_step(condition, state, actions, valid)
    sampled = policy.sample(condition, state, valid)

    assert output.loss.dtype == torch.float32
    assert sampled.dtype == torch.float32
    assert torch.isfinite(output.loss)
    assert torch.isfinite(sampled).all()

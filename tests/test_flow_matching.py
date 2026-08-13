import pytest
import torch

from pi0_minimal.models import (
    FlowMatchingBatch,
    build_flow_matching_batch,
    interpolate_actions,
    masked_flow_matching_loss,
    sample_paper_flow_time,
    target_velocity,
)


def _actions() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    expert = torch.tensor(
        [
            [[0.2, -0.1], [0.4, 0.5], [0.0, 0.0]],
            [[-0.3, 0.8], [0.1, -0.2], [0.6, 0.7]],
        ],
        dtype=torch.float32,
    )
    noise = torch.tensor(
        [
            [[-0.4, 0.3], [0.2, -0.7], [1.0, -1.0]],
            [[0.5, -0.6], [-0.9, 0.4], [0.2, -0.1]],
        ],
        dtype=torch.float32,
    )
    valid = torch.tensor(
        [
            [[True, True], [True, True], [False, False]],
            [[True, True], [True, False], [False, False]],
        ]
    )
    return expert, noise, valid


def test_interpolation_has_exact_noise_and_data_boundaries() -> None:
    expert, noise, _valid = _actions()

    at_noise = interpolate_actions(expert, noise, torch.zeros(2))
    at_data = interpolate_actions(expert, noise, torch.ones(2))

    torch.testing.assert_close(at_noise, noise)
    torch.testing.assert_close(at_data, expert)


def test_target_velocity_uses_paper_direction_a_minus_noise() -> None:
    expert, noise, _valid = _actions()

    velocity = target_velocity(expert, noise)

    torch.testing.assert_close(velocity, expert - noise)
    assert not torch.equal(velocity, noise - expert)


def test_injected_flow_batch_matches_analytical_path() -> None:
    expert, noise, valid = _actions()
    flow_time = torch.tensor([0.25, 0.75], dtype=torch.float32)

    batch = build_flow_matching_batch(
        expert,
        valid,
        noise=noise,
        flow_time=flow_time,
    )

    expected = flow_time[:, None, None] * expert + (1.0 - flow_time[:, None, None]) * noise
    torch.testing.assert_close(batch.noisy_actions, expected)
    torch.testing.assert_close(batch.target_velocity, expert - noise)
    torch.testing.assert_close(batch.flow_time, flow_time)
    torch.testing.assert_close(batch.action_valid, valid)


def test_masked_loss_counts_valid_scalar_elements_only() -> None:
    expert, noise, valid = _actions()
    target = target_velocity(expert, noise)
    prediction = target.clone()
    prediction[valid] += 2.0
    prediction[~valid] = 100.0

    loss = masked_flow_matching_loss(prediction, target, valid)

    torch.testing.assert_close(loss, torch.tensor(4.0))


def test_masked_loss_ignores_perturbed_padding_and_has_gradients() -> None:
    expert, noise, valid = _actions()
    target = target_velocity(expert, noise)
    prediction = torch.zeros_like(target, requires_grad=True)
    changed_target = target.masked_fill(~valid, -999.0)

    first = masked_flow_matching_loss(prediction, target, valid)
    second = masked_flow_matching_loss(prediction, changed_target, valid)
    torch.testing.assert_close(first, second)
    first.backward()

    assert prediction.grad is not None
    assert torch.count_nonzero(prediction.grad[~valid]) == 0
    assert torch.isfinite(prediction.grad).all()


def test_paper_time_sampler_is_deterministic_and_emphasizes_low_times() -> None:
    first_generator = torch.Generator().manual_seed(7)
    second_generator = torch.Generator().manual_seed(7)

    first = sample_paper_flow_time(
        20_000,
        device="cpu",
        generator=first_generator,
    )
    second = sample_paper_flow_time(
        20_000,
        device="cpu",
        generator=second_generator,
    )

    torch.testing.assert_close(first, second)
    assert torch.all((first >= 0.0) & (first <= 0.999))
    assert 0.38 < first.mean().item() < 0.42
    assert (first < 0.5).float().mean().item() > 0.5


def test_random_flow_batch_is_reproducible_with_fixed_generator() -> None:
    expert, _noise, valid = _actions()
    first_generator = torch.Generator().manual_seed(17)
    second_generator = torch.Generator().manual_seed(17)

    first = build_flow_matching_batch(expert, valid, generator=first_generator)
    second = build_flow_matching_batch(expert, valid, generator=second_generator)

    torch.testing.assert_close(first.noise, second.noise)
    torch.testing.assert_close(first.flow_time, second.flow_time)
    torch.testing.assert_close(first.noisy_actions, second.noisy_actions)


def test_flow_helpers_reject_invalid_inputs() -> None:
    expert, noise, valid = _actions()

    with pytest.raises(TypeError, match="float32"):
        interpolate_actions(expert.to(torch.float64), noise, torch.zeros(2))
    with pytest.raises(ValueError, match="identical shapes"):
        target_velocity(expert, noise[:, :2])
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        interpolate_actions(expert, noise, torch.tensor([-0.1, 1.1]))
    with pytest.raises(ValueError, match="at least one valid"):
        build_flow_matching_batch(expert, torch.zeros_like(valid))
    with pytest.raises(ValueError, match="beta=1"):
        sample_paper_flow_time(2, device="cpu", beta_beta=2.0)


def test_flow_batch_rejects_nonfinite_contract() -> None:
    expert, noise, valid = _actions()
    flow_time = torch.tensor([0.25, 0.75])

    with pytest.raises(ValueError, match="finite"):
        FlowMatchingBatch(
            noisy_actions=expert.masked_fill(expert == 0.2, torch.nan),
            target_velocity=expert - noise,
            flow_time=flow_time,
            noise=noise,
            action_valid=valid,
        )

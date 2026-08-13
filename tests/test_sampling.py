import pytest
import torch

from pi0_minimal.models import (
    ActionTimeEmbedding,
    ConditionedVelocityField,
    FlowActionExpert,
    PolicyMemory,
    euler_integrate,
    sample_actions_euler,
)


@pytest.mark.parametrize("num_steps", [1, 2, 10, 37])
def test_constant_oracle_recovers_expert_actions_for_any_step_count(
    num_steps: int,
) -> None:
    torch.manual_seed(7)
    noise = torch.randn((2, 4, 3))
    expert_actions = torch.randn((2, 4, 3))
    valid = torch.ones_like(noise, dtype=torch.bool)
    constant_velocity = expert_actions - noise

    sampled = euler_integrate(
        lambda _actions, _time: constant_velocity,
        noise,
        valid,
        num_steps=num_steps,
    )

    torch.testing.assert_close(sampled, expert_actions, atol=2e-6, rtol=2e-6)


def test_euler_uses_forward_times_and_positive_updates() -> None:
    initial = torch.zeros((1, 2, 1))
    valid = torch.ones_like(initial, dtype=torch.bool)
    observed_times: list[torch.Tensor] = []

    def velocity(actions: torch.Tensor, flow_time: torch.Tensor) -> torch.Tensor:
        observed_times.append(flow_time.clone())
        return torch.ones_like(actions)

    sampled = euler_integrate(velocity, initial, valid, num_steps=4)

    torch.testing.assert_close(sampled, torch.ones_like(sampled))
    torch.testing.assert_close(
        torch.stack(observed_times).flatten(),
        torch.tensor([0.0, 0.25, 0.5, 0.75]),
    )


def test_invalid_padding_is_zero_and_cannot_affect_valid_sampling() -> None:
    first_noise = torch.tensor([[[0.5], [1.0], [99.0]]])
    second_noise = torch.tensor([[[0.5], [1.0], [-99.0]]])
    valid = torch.tensor([[[True], [True], [False]]])

    def velocity(actions: torch.Tensor, _time: torch.Tensor) -> torch.Tensor:
        mean = actions[:, :2].mean(dim=1, keepdim=True)
        return mean.expand_as(actions)

    first = euler_integrate(velocity, first_noise, valid, num_steps=3)
    second = euler_integrate(velocity, second_noise, valid, num_steps=3)

    torch.testing.assert_close(first, second)
    assert first[0, 2, 0] == 0.0


def test_fixed_noise_produces_deterministic_conditioned_sampling() -> None:
    torch.manual_seed(17)
    embedding = ActionTimeEmbedding(action_dim=2, model_dim=8, time_dim=6)
    expert = FlowActionExpert(
        model_dim=8,
        num_layers=1,
        num_heads=2,
        ffn_dim=16,
        action_dim=2,
        max_horizon=3,
    )
    field = ConditionedVelocityField(embedding, expert).eval()
    memory = PolicyMemory(
        torch.randn((1, 4, 8)),
        torch.tensor([[True, True, False, True]]),
    )
    valid = torch.tensor([[[True, True], [True, True], [False, False]]])
    noise = torch.randn((1, 3, 2))

    first = sample_actions_euler(
        field,
        memory,
        valid,
        num_steps=4,
        initial_noise=noise,
    )
    second = sample_actions_euler(
        field,
        memory,
        valid,
        num_steps=4,
        initial_noise=noise,
    )

    torch.testing.assert_close(first, second)
    assert first.dtype == torch.float32
    assert torch.isfinite(first).all()
    assert not first.requires_grad


def test_generated_initial_noise_is_reproducible() -> None:
    torch.manual_seed(27)
    field = ConditionedVelocityField(
        ActionTimeEmbedding(action_dim=1, model_dim=4, time_dim=4),
        FlowActionExpert(
            model_dim=4,
            num_layers=1,
            num_heads=1,
            ffn_dim=8,
            action_dim=1,
            max_horizon=2,
        ),
    ).eval()
    memory = PolicyMemory(torch.randn((1, 2, 4)), torch.ones((1, 2), dtype=torch.bool))
    valid = torch.ones((1, 2, 1), dtype=torch.bool)

    first = sample_actions_euler(
        field,
        memory,
        valid,
        num_steps=2,
        generator=torch.Generator().manual_seed(37),
    )
    second = sample_actions_euler(
        field,
        memory,
        valid,
        num_steps=2,
        generator=torch.Generator().manual_seed(37),
    )

    torch.testing.assert_close(first, second)


def test_sampler_rejects_invalid_steps_noise_and_velocity() -> None:
    noise = torch.zeros((1, 2, 1))
    valid = torch.ones_like(noise, dtype=torch.bool)

    with pytest.raises(ValueError, match="positive"):
        euler_integrate(lambda actions, _time: actions, noise, valid, num_steps=0)
    with pytest.raises(TypeError, match="float32"):
        euler_integrate(
            lambda actions, _time: actions.float(),
            noise.to(torch.float64),
            valid,
            num_steps=1,
        )
    with pytest.raises(ValueError, match="action shape"):
        euler_integrate(
            lambda actions, _time: actions[..., :0],
            noise,
            valid,
            num_steps=1,
        )
    with pytest.raises(ValueError, match="finite"):
        euler_integrate(
            lambda actions, _time: actions.masked_fill(actions == 0, torch.nan),
            noise,
            valid,
            num_steps=1,
        )


def test_velocity_field_rejects_incompatible_components() -> None:
    embedding = ActionTimeEmbedding(action_dim=2, model_dim=8, time_dim=6)
    wrong_width = FlowActionExpert(
        model_dim=16,
        num_layers=1,
        num_heads=4,
        ffn_dim=32,
        action_dim=2,
        max_horizon=3,
    )

    with pytest.raises(ValueError, match="model widths"):
        ConditionedVelocityField(embedding, wrong_width)

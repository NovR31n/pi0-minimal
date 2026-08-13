import pytest
import torch

from pi0_minimal.models import ActionTokens, FlowActionExpert, PolicyMemory


def _inputs() -> tuple[ActionTokens, PolicyMemory]:
    torch.manual_seed(7)
    action_values = torch.randn((2, 4, 16))
    action_valid = torch.tensor([[True, True, True, False], [True, True, False, False]])
    memory_values = torch.randn((2, 5, 16))
    memory_valid = torch.tensor(
        [[True, True, True, False, False], [True, True, False, False, True]]
    )
    return (
        ActionTokens(action_values, action_valid),
        PolicyMemory(memory_values, memory_valid),
    )


def _expert() -> FlowActionExpert:
    torch.manual_seed(17)
    return FlowActionExpert(
        model_dim=16,
        num_layers=2,
        num_heads=4,
        ffn_dim=32,
        action_dim=3,
        max_horizon=4,
    )


def test_expert_returns_fp32_velocity_and_zeros_invalid_queries() -> None:
    action_tokens, memory = _inputs()

    velocity = _expert()(action_tokens, memory)

    assert velocity.shape == (2, 4, 3)
    assert velocity.dtype == torch.float32
    assert torch.isfinite(velocity).all()
    assert torch.count_nonzero(velocity[0, 3]) == 0
    assert torch.count_nonzero(velocity[1, 2:]) == 0


def test_invalid_action_and_memory_values_cannot_affect_valid_outputs() -> None:
    action_tokens, memory = _inputs()
    changed_actions = action_tokens.values.masked_fill(
        ~action_tokens.valid.unsqueeze(-1),
        99.0,
    )
    changed_memory = memory.values.masked_fill(~memory.valid.unsqueeze(-1), -99.0)
    expert = _expert().eval()

    baseline = expert(action_tokens, memory)
    changed = expert(
        ActionTokens(changed_actions, action_tokens.valid),
        PolicyMemory(changed_memory, memory.valid),
    )

    torch.testing.assert_close(baseline, changed)


def test_bidirectional_attention_allows_later_action_to_change_earlier_output() -> None:
    action_tokens, memory = _inputs()
    changed_values = action_tokens.values.clone()
    changed_values[:, 2] += 3.0
    expert = _expert().eval()

    baseline = expert(action_tokens, memory)
    changed = expert(ActionTokens(changed_values, action_tokens.valid), memory)

    assert not torch.allclose(baseline[0, 0], changed[0, 0])


def test_expert_is_deterministic_and_all_parameters_receive_gradients() -> None:
    action_tokens, memory = _inputs()
    expert = _expert()

    first = expert(action_tokens, memory)
    second = expert(action_tokens, memory)
    torch.testing.assert_close(first, second)
    first.square().mean().backward()

    assert all(parameter.grad is not None for parameter in expert.parameters())


def test_expert_supports_bfloat16_compute_with_fp32_output() -> None:
    action_tokens, memory = _inputs()
    expert = _expert().to(torch.bfloat16)

    velocity = expert(action_tokens, memory)

    assert velocity.dtype == torch.float32
    assert torch.isfinite(velocity).all()


def test_formal_expert_parameter_count_matches_compact_budget() -> None:
    expert = FlowActionExpert()
    parameters = sum(parameter.numel() for parameter in expert.parameters())

    assert 40_000_000 <= parameters <= 50_000_000


def test_expert_rejects_width_batch_and_horizon_mismatches() -> None:
    action_tokens, memory = _inputs()
    expert = _expert()

    with pytest.raises(ValueError, match="action token width"):
        expert(
            ActionTokens(torch.ones((2, 4, 8)), action_tokens.valid),
            memory,
        )
    with pytest.raises(ValueError, match="batch"):
        expert(action_tokens, PolicyMemory(memory.values[:1], memory.valid[:1]))
    with pytest.raises(ValueError, match="maximum"):
        expert(
            ActionTokens(
                torch.ones((2, 5, 16)),
                torch.ones((2, 5), dtype=torch.bool),
            ),
            memory,
        )

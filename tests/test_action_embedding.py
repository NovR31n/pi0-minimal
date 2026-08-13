import pytest
import torch

from pi0_minimal.models import (
    ActionTimeEmbedding,
    ActionTokens,
    sinusoidal_time_embedding,
)


def _embedding() -> ActionTimeEmbedding:
    torch.manual_seed(7)
    return ActionTimeEmbedding(action_dim=3, model_dim=8, time_dim=6)


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    actions = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3) / 10.0
    flow_time = torch.tensor([0.0, 0.75], dtype=torch.float32)
    valid = torch.tensor(
        [
            [[True, True, True], [True, True, True], [False, False, False], [False, False, False]],
            [[True, True, True], [True, False, True], [True, True, True], [False, False, False]],
        ]
    )
    return actions, flow_time, valid


def test_action_embedding_returns_masked_horizon_tokens() -> None:
    actions, flow_time, valid = _inputs()

    tokens = _embedding()(actions, flow_time, valid)

    assert tokens.values.shape == (2, 4, 8)
    torch.testing.assert_close(
        tokens.valid,
        torch.tensor([[True, True, False, False], [True, True, True, False]]),
    )
    assert torch.count_nonzero(tokens.values[0, 2:]) == 0
    assert torch.count_nonzero(tokens.values[1, 3]) == 0


def test_invalid_action_dimensions_do_not_change_tokens() -> None:
    actions, flow_time, valid = _inputs()
    changed = actions.masked_fill(~valid, 99.0)

    first = _embedding()(actions, flow_time, valid)
    second = _embedding()(changed, flow_time, valid)

    torch.testing.assert_close(first.values, second.values)


def test_time_embedding_is_deterministic_and_has_known_zero_boundary() -> None:
    flow_time = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)

    first = sinusoidal_time_embedding(flow_time, 8)
    second = sinusoidal_time_embedding(flow_time, 8)

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first[0, :4], torch.zeros(4))
    torch.testing.assert_close(first[0, 4:], torch.ones(4))
    assert not torch.equal(first[0], first[1])


def test_same_time_is_repeated_across_each_action_chunk() -> None:
    embedding = _embedding()
    actions = torch.zeros((2, 4, 3), dtype=torch.float32)
    flow_time = torch.tensor([0.25, 0.75], dtype=torch.float32)

    tokens = embedding(actions, flow_time)

    for horizon_index in range(1, 4):
        torch.testing.assert_close(tokens.values[:, 0], tokens.values[:, horizon_index])
    assert not torch.equal(tokens.values[0], tokens.values[1])


def test_embedding_follows_parameter_dtype_and_receives_gradients() -> None:
    embedding = _embedding().to(torch.bfloat16)
    actions, flow_time, valid = _inputs()

    tokens = embedding(actions, flow_time, valid)
    tokens.values.float().sum().backward()

    assert tokens.values.dtype == torch.bfloat16
    assert all(parameter.grad is not None for parameter in embedding.parameters())


def test_embedding_rejects_invalid_inputs() -> None:
    embedding = _embedding()
    actions, flow_time, valid = _inputs()

    with pytest.raises(TypeError, match="float32"):
        embedding(actions.to(torch.float64), flow_time, valid)
    with pytest.raises(ValueError, match="width"):
        embedding(actions[..., :2], flow_time, valid[..., :2])
    with pytest.raises(ValueError, match="batch"):
        embedding(actions, flow_time[:1], valid)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        embedding(actions, torch.tensor([-0.1, 1.1]), valid)
    with pytest.raises(ValueError, match="at least one valid"):
        embedding(actions, flow_time, torch.zeros_like(valid))


def test_action_tokens_reject_invalid_contract() -> None:
    values = torch.ones((2, 4, 8))
    valid = torch.ones((2, 4), dtype=torch.bool)

    with pytest.raises(ValueError, match="shape"):
        ActionTokens(values, valid[:, :3])
    with pytest.raises(ValueError, match="finite"):
        ActionTokens(values.masked_fill(values > 0, torch.nan), valid)

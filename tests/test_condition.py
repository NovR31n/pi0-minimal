import numpy as np
import pytest
import torch

from pi0_minimal.models import ConditionMemory, ConditionProjector, PolicyMemory


def _condition() -> ConditionMemory:
    values = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    valid = torch.tensor([[True, True, False], [True, False, False]])
    return ConditionMemory(values, valid)


def _projector() -> ConditionProjector:
    torch.manual_seed(7)
    return ConditionProjector(condition_dim=4, state_dim=3, model_dim=8)


def test_projector_appends_one_always_valid_state_token() -> None:
    memory = _projector()(
        _condition(),
        torch.tensor([[0.1, 0.2, 0.3], [-0.1, 0.0, 0.1]], dtype=torch.float32),
    )

    assert memory.values.shape == (2, 4, 8)
    assert memory.valid.shape == (2, 4)
    torch.testing.assert_close(memory.valid[:, :-1], _condition().valid)
    assert memory.valid[:, -1].all()
    assert torch.count_nonzero(memory.values[0, 2]) == 0
    assert torch.count_nonzero(memory.values[1, 1:3]) == 0


def test_state_dimension_mask_removes_invalid_values_before_projection() -> None:
    projector = _projector()
    state_valid = torch.tensor([[True, False, True], [True, True, False]])
    first_state = torch.tensor([[0.1, 99.0, 0.3], [-0.1, 0.2, -99.0]])
    second_state = torch.tensor([[0.1, -77.0, 0.3], [-0.1, 0.2, 88.0]])

    first = projector(_condition(), first_state, state_valid)
    second = projector(_condition(), second_state, state_valid)

    torch.testing.assert_close(first.values[:, -1], second.values[:, -1])


def test_only_projector_parameters_receive_gradients() -> None:
    projector = _projector()
    context_values = _condition().values.detach()
    condition = ConditionMemory(context_values, _condition().valid)
    state = torch.ones((2, 3), dtype=torch.float32)

    memory = projector(condition, state)
    memory.values.sum().backward()

    assert not context_values.requires_grad
    assert all(parameter.grad is not None for parameter in projector.parameters())


def test_numpy_bridge_preserves_contract() -> None:
    projector = _projector()
    state = np.ones((2, 3), dtype=np.float32)
    state_valid = np.array([[True, False, True], [True, True, True]])

    memory = projector.encode_numpy(_condition(), state, state_valid)

    assert memory.values.dtype == torch.float32
    assert memory.values.device.type == "cpu"


def test_projector_uses_its_parameter_dtype_for_memory() -> None:
    projector = _projector().to(torch.bfloat16)
    state = torch.ones((2, 3), dtype=torch.float32)

    memory = projector(_condition(), state)

    assert memory.values.dtype == torch.bfloat16


def test_projector_rejects_bad_state_and_condition_inputs() -> None:
    projector = _projector()
    condition = _condition()
    state = torch.ones((2, 3), dtype=torch.float32)

    with pytest.raises(TypeError, match="float32"):
        projector(condition, state.to(torch.float64))
    with pytest.raises(ValueError, match="shape"):
        projector(condition, state[:, :2])
    with pytest.raises(ValueError, match="finite"):
        projector(condition, state.masked_fill(torch.tensor([[True, False, False]] * 2), torch.nan))
    with pytest.raises(ValueError, match="at least one valid"):
        projector(condition, state, torch.zeros_like(state, dtype=torch.bool))
    with pytest.raises(ValueError, match="condition width"):
        projector(
            ConditionMemory(torch.ones((2, 3, 5)), condition.valid),
            state,
        )


def test_policy_memory_rejects_invalid_contract() -> None:
    values = torch.ones((2, 3, 4))
    valid = torch.ones((2, 3), dtype=torch.bool)

    with pytest.raises(ValueError, match="shape"):
        PolicyMemory(values, valid[:, :2])
    with pytest.raises(ValueError, match="finite"):
        PolicyMemory(values.masked_fill(values > 0, torch.nan), valid)

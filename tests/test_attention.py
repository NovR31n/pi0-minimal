import pytest
import torch

from pi0_minimal.models import (
    AttentionMasks,
    build_attention_masks,
    build_causal_attention_masks,
)


def test_flow_masks_are_bidirectional_and_respect_invalid_tokens() -> None:
    action_valid = torch.tensor([[True, True, False], [True, False, True]])
    memory_valid = torch.tensor([[True, False], [True, True]])

    masks = build_attention_masks(action_valid, memory_valid)

    torch.testing.assert_close(
        masks.self_allowed,
        torch.tensor(
            [
                [[True, True, False], [True, True, False], [False, False, False]],
                [[True, False, True], [False, False, False], [True, False, True]],
            ]
        ),
    )
    torch.testing.assert_close(
        masks.cross_allowed,
        torch.tensor(
            [
                [[True, False], [True, False], [False, False]],
                [[True, True], [False, False], [True, True]],
            ]
        ),
    )
    assert masks.self_allowed[0, 0, 1]
    assert masks.self_allowed[0, 1, 0]


def test_mask_builder_rejects_empty_or_inconsistent_inputs() -> None:
    action_valid = torch.tensor([[True, False]])
    memory_valid = torch.tensor([[True, False]])

    with pytest.raises(ValueError, match="at least one valid action"):
        build_attention_masks(torch.zeros_like(action_valid), memory_valid)
    with pytest.raises(ValueError, match="at least one valid memory"):
        build_attention_masks(action_valid, torch.zeros_like(memory_valid))
    with pytest.raises(ValueError, match="batch"):
        build_attention_masks(action_valid.repeat(2, 1), memory_valid)


def test_autoregressive_masks_are_strictly_causal_over_future_keys() -> None:
    action_valid = torch.tensor([[True, True, True, False]])
    memory_valid = torch.tensor([[True, False]])

    masks = build_causal_attention_masks(action_valid, memory_valid)

    torch.testing.assert_close(
        masks.self_allowed,
        torch.tensor(
            [
                [
                    [True, False, False, False],
                    [True, True, False, False],
                    [True, True, True, False],
                    [False, False, False, False],
                ]
            ]
        ),
    )
    torch.testing.assert_close(
        masks.cross_allowed,
        torch.tensor(
            [
                [
                    [True, False],
                    [True, False],
                    [True, False],
                    [False, False],
                ]
            ]
        ),
    )


def test_attention_masks_reject_invalid_contract() -> None:
    with pytest.raises(ValueError, match="square"):
        AttentionMasks(
            torch.ones((1, 2, 3), dtype=torch.bool),
            torch.ones((1, 2, 4), dtype=torch.bool),
        )

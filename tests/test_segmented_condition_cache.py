import numpy as np
import pytest
import torch

from pi0_minimal.models import ConditionMemory
from scripts.train_flow_small import (
    SegmentedConditionMemory,
    _best_generation_action_mae,
)


def _segment(start: int, count: int) -> ConditionMemory:
    values = torch.arange(start, start + count, dtype=torch.float32).reshape(
        count, 1, 1
    )
    valid = torch.ones((count, 1), dtype=torch.bool)
    return ConditionMemory(values, valid)


def test_segmented_condition_cache_preserves_mixed_index_order() -> None:
    cache = SegmentedConditionMemory((_segment(0, 2), _segment(2, 3)))
    selected = cache.select(np.asarray([3, 0, 4, 2, 3], dtype=np.int64))
    np.testing.assert_array_equal(
        selected.values[:, 0, 0].numpy(),
        np.asarray([3.0, 0.0, 4.0, 2.0, 3.0]),
    )
    assert selected.valid.all()


def test_segmented_condition_cache_rejects_out_of_range_index() -> None:
    cache = SegmentedConditionMemory((_segment(0, 2),))
    with pytest.raises(IndexError, match="outside"):
        cache.select(np.asarray([2], dtype=np.int64))


def test_initial_generation_metric_participates_in_best_checkpoint() -> None:
    history = [
        {"step": 0, "validation_normalized_action_mae": 0.2},
        {"step": 1, "validation_normalized_action_mae": 0.3},
    ]
    assert _best_generation_action_mae(history) == pytest.approx(0.2)

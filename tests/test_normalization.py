import json

import numpy as np
import pytest

from pi0_minimal.data import NormalizationStats, QuantileStats


def test_quantile_normalization_round_trip_inside_bounds() -> None:
    values = np.arange(60, dtype=np.float32).reshape(10, 2, 3)
    stats = QuantileStats.fit(values, lower_quantile=0.0, upper_quantile=1.0)

    restored = stats.inverse(stats.normalize(values))

    np.testing.assert_allclose(restored, values, rtol=0.0, atol=4e-6)
    assert stats.normalize(values).dtype == np.float32


def test_constant_feature_normalizes_to_zero_and_restores_constant() -> None:
    values = np.array([[3.0, 1.0], [3.0, 5.0]], dtype=np.float32)
    stats = QuantileStats.fit(values, lower_quantile=0.0, upper_quantile=1.0)

    normalized = stats.normalize(values)
    restored = stats.inverse(normalized)

    np.testing.assert_array_equal(normalized[:, 0], np.zeros(2, dtype=np.float32))
    np.testing.assert_array_equal(restored[:, 0], np.full(2, 3.0, dtype=np.float32))


def test_invalid_padded_actions_do_not_affect_statistics() -> None:
    actions = np.array([[[0.0, 10.0], [2.0, 20.0], [999.0, 999.0]]], dtype=np.float32)
    valid = np.ones_like(actions, dtype=np.bool_)
    valid[:, -1] = False

    stats = QuantileStats.fit(actions, valid, lower_quantile=0.0, upper_quantile=1.0)

    np.testing.assert_array_equal(stats.lower, np.array([0.0, 10.0]))
    np.testing.assert_array_equal(stats.upper, np.array([2.0, 20.0]))
    np.testing.assert_array_equal(stats.count, np.array([2, 2]))


def test_normalization_cache_round_trip(tmp_path) -> None:
    state = QuantileStats.fit(np.array([[0.0, 2.0], [1.0, 4.0]], dtype=np.float32))
    action = QuantileStats.fit(np.array([[[0.0], [1.0]]], dtype=np.float32))
    expected = NormalizationStats(state, action, "split-sha256")
    path = tmp_path / "normalization.json"

    expected.save(path)
    actual = NormalizationStats.load(path)

    assert actual.training_split_fingerprint == expected.training_split_fingerprint
    np.testing.assert_array_equal(actual.state.lower, expected.state.lower)
    assert json.loads(path.read_text())["schema_version"] == 1


def test_normalization_rejects_wrong_shape_and_empty_feature() -> None:
    stats = QuantileStats.fit(np.ones((2, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="last dimension"):
        stats.normalize(np.ones((2, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="no valid"):
        QuantileStats.fit(
            np.ones((2, 3), dtype=np.float32),
            np.zeros((2, 3), dtype=np.bool_),
        )

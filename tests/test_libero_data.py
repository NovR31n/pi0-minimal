import numpy as np
import pytest

from pi0_minimal.data import (
    build_action_chunk,
    compact_libero_state,
    libero_arrays_from_mapping,
    validate_libero_actions,
)


def test_compact_state_uses_end_effector_and_finger_opening_width() -> None:
    raw = np.array([[1, 2, 3, 4, 5, 6, 0.04, -0.04]], dtype=np.float32)

    compact = compact_libero_state(raw)

    np.testing.assert_allclose(compact, [[1, 2, 3, 4, 5, 6, 0.08]])
    assert compact.shape == (1, 7)
    assert compact.dtype == np.float32


def test_action_chunk_at_episode_boundary_is_zero_padded_and_masked() -> None:
    actions = np.arange(21, dtype=np.float32).reshape(3, 7)
    actions[:, -1] = np.array([-1.0, 1.0, -1.0])

    chunk, valid = build_action_chunk(actions, start=2, horizon=4)

    np.testing.assert_array_equal(chunk[0], actions[2])
    np.testing.assert_array_equal(chunk[1:], np.zeros((3, 7), dtype=np.float32))
    np.testing.assert_array_equal(valid[0], np.ones(7, dtype=np.bool_))
    np.testing.assert_array_equal(valid[1:], np.zeros((3, 7), dtype=np.bool_))


def test_action_chunk_rejects_invalid_time_boundaries() -> None:
    actions = np.zeros((3, 7), dtype=np.float32)

    with pytest.raises(IndexError, match="start"):
        build_action_chunk(actions, start=-1, horizon=2)
    with pytest.raises(IndexError, match="start"):
        build_action_chunk(actions, start=3, horizon=2)


def test_libero_adapter_rejects_bad_fields_and_gripper_range() -> None:
    with pytest.raises(ValueError, match="dimension 8"):
        compact_libero_state(np.zeros((2, 7), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        validate_libero_actions(np.zeros((2, 6), dtype=np.float32))

    actions = np.zeros((2, 7), dtype=np.float32)
    actions[0, -1] = 1.01
    with pytest.raises(ValueError, match="gripper"):
        validate_libero_actions(actions)


def test_libero_adapter_rejects_nonfinite_values() -> None:
    state = np.zeros((2, 8), dtype=np.float32)
    state[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        compact_libero_state(state)

    actions = np.zeros((2, 7), dtype=np.float32)
    actions[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        validate_libero_actions(actions)


def test_libero_mapping_rejects_missing_fields_and_length_mismatch() -> None:
    with pytest.raises(KeyError, match="action"):
        libero_arrays_from_mapping(
            {"observation.state": np.zeros((2, 8), dtype=np.float32)}
        )

    with pytest.raises(ValueError, match="same length"):
        libero_arrays_from_mapping(
            {
                "observation.state": np.zeros((2, 8), dtype=np.float32),
                "action": np.zeros((3, 7), dtype=np.float32),
            }
        )

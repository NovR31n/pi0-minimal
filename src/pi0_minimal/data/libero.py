"""LIBERO-specific conversion at the boundary of the model-neutral data API."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt


def libero_arrays_from_mapping(
    record: Mapping[str, object],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Extract and validate the two required numeric fields from one episode."""

    required = ("observation.state", "action")
    missing = [key for key in required if key not in record]
    if missing:
        raise KeyError(f"missing required LIBERO fields: {missing}")
    state = compact_libero_state(np.asarray(record["observation.state"]))
    actions = validate_libero_actions(np.asarray(record["action"]))
    if len(state) != len(actions):
        raise ValueError("LIBERO state and action sequences must have the same length")
    return state, actions


def compact_libero_state(raw_state: npt.NDArray[np.floating]) -> npt.NDArray[np.float32]:
    """Convert LIBERO's 6D end-effector + two fingers into 6D + opening width."""

    state = np.asarray(raw_state)
    if not np.issubdtype(state.dtype, np.floating):
        raise TypeError("raw LIBERO state must be floating point")
    if state.ndim < 1 or state.shape[-1] != 8:
        raise ValueError(f"raw LIBERO state must end in dimension 8, got {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError("raw LIBERO state must contain only finite values")
    opening_width = state[..., 6:7] - state[..., 7:8]
    return np.concatenate((state[..., :6], opening_width), axis=-1).astype(np.float32)


def validate_libero_actions(actions: npt.NDArray[np.floating]) -> npt.NDArray[np.float32]:
    """Validate raw 7D actions; the last component is the normalized gripper command."""

    array = np.asarray(actions)
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError("LIBERO actions must be floating point")
    if array.ndim != 2 or array.shape[1] != 7:
        raise ValueError(f"LIBERO actions must have shape [T,7], got {array.shape}")
    if len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("LIBERO actions must be non-empty and finite")
    if np.any(np.abs(array[:, 6]) > 1.0 + 1e-6):
        raise ValueError("LIBERO gripper commands must lie in [-1,1]")
    return array.astype(np.float32)


def build_action_chunk(
    actions: npt.NDArray[np.floating],
    start: int,
    horizon: int,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """Build a fixed horizon and zero-pad only after the episode boundary."""

    action_array = validate_libero_actions(actions)
    if not isinstance(start, int):
        raise TypeError("start must be an integer frame index")
    if not 0 <= start < len(action_array):
        raise IndexError(f"start must be in [0,{len(action_array) - 1}], got {start}")
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    chunk = np.zeros((horizon, action_array.shape[1]), dtype=np.float32)
    valid = np.zeros_like(chunk, dtype=np.bool_)
    available = min(horizon, len(action_array) - start)
    chunk[:available] = action_array[start : start + available]
    valid[:available] = True
    return chunk, valid

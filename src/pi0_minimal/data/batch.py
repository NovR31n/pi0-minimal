"""Validated NumPy batch contracts shared by all policy implementations."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

UInt8Array = npt.NDArray[np.uint8]
Int64Array = npt.NDArray[np.int64]
Float32Array = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class BatchSpec:
    """Exact external tensor dimensions for one committed model configuration."""

    image_keys: tuple[str, ...]
    image_height: int
    image_width: int
    image_channels: int
    max_prompt_tokens: int
    state_dim: int
    action_horizon: int
    action_dim: int

    @classmethod
    def from_model_spec(cls, model_spec: dict) -> "BatchSpec":
        observation = model_spec["observation"]
        action = model_spec["action"]
        return cls(
            image_keys=tuple(observation["image_keys"]),
            image_height=observation["image_height"],
            image_width=observation["image_width"],
            image_channels=observation["image_channels"],
            max_prompt_tokens=observation["max_prompt_tokens"],
            state_dim=observation["state_dim"],
            action_horizon=action["horizon"],
            action_dim=action["dim"],
        )


@dataclass(frozen=True, slots=True)
class ObservationBatch:
    """Canonical CPU observation batch before model-specific preprocessing."""

    image_keys: tuple[str, ...]
    images: UInt8Array
    image_valid: BoolArray
    prompt_ids: Int64Array
    prompt_valid: BoolArray
    state: Float32Array

    def __post_init__(self) -> None:
        _require_array("images", self.images, np.uint8, rank=5)
        _require_array("image_valid", self.image_valid, np.bool_, rank=2)
        _require_array("prompt_ids", self.prompt_ids, np.int64, rank=2)
        _require_array("prompt_valid", self.prompt_valid, np.bool_, rank=2)
        _require_array("state", self.state, np.float32, rank=2)

        batch_size, num_views = self.images.shape[:2]
        if len(self.image_keys) != num_views or len(set(self.image_keys)) != num_views:
            raise ValueError("image_keys must be unique and match the image view dimension")
        if self.image_valid.shape != (batch_size, num_views):
            raise ValueError("image_valid must have shape [B,V]")
        if self.prompt_ids.shape != self.prompt_valid.shape:
            raise ValueError("prompt_ids and prompt_valid must have identical shapes")
        if self.prompt_ids.shape[0] != batch_size or self.state.shape[0] != batch_size:
            raise ValueError("all observation fields must have the same batch dimension")
        if not np.all(self.image_valid.any(axis=1)):
            raise ValueError("every sample must contain at least one valid image")
        if not np.all(self.prompt_valid.any(axis=1)):
            raise ValueError("every sample must contain at least one valid prompt token")
        if np.any(self.prompt_ids[self.prompt_valid] < 0):
            raise ValueError("valid prompt token IDs must be non-negative")
        if not np.isfinite(self.state).all():
            raise ValueError("state must contain only finite values")

    @property
    def batch_size(self) -> int:
        return self.images.shape[0]

    def validate_against(self, spec: BatchSpec) -> None:
        expected_images = (
            self.batch_size,
            len(spec.image_keys),
            spec.image_channels,
            spec.image_height,
            spec.image_width,
        )
        if self.image_keys != spec.image_keys:
            raise ValueError(f"image_keys must equal {spec.image_keys}, got {self.image_keys}")
        if self.images.shape != expected_images:
            raise ValueError(f"images must have shape {expected_images}, got {self.images.shape}")
        expected_prompt = (self.batch_size, spec.max_prompt_tokens)
        if self.prompt_ids.shape != expected_prompt:
            raise ValueError(f"prompt tensors must have shape {expected_prompt}, got {self.prompt_ids.shape}")
        expected_state = (self.batch_size, spec.state_dim)
        if self.state.shape != expected_state:
            raise ValueError(f"state must have shape {expected_state}, got {self.state.shape}")


@dataclass(frozen=True, slots=True)
class ActionBatch:
    """Normalized or raw action chunk plus a scalar-element validity mask."""

    values: Float32Array
    valid: BoolArray

    def __post_init__(self) -> None:
        _require_array("action values", self.values, np.float32, rank=3)
        _require_array("action valid", self.valid, np.bool_, rank=3)
        if self.values.shape != self.valid.shape:
            raise ValueError("action values and valid mask must have identical shapes")
        if not np.isfinite(self.values).all():
            raise ValueError("action values must contain only finite values")
        if not np.all(self.valid.reshape(self.batch_size, -1).any(axis=1)):
            raise ValueError("every sample must contain at least one valid action element")

    @property
    def batch_size(self) -> int:
        return self.values.shape[0]

    def validate_against(self, spec: BatchSpec) -> None:
        expected = (self.batch_size, spec.action_horizon, spec.action_dim)
        if self.values.shape != expected:
            raise ValueError(f"actions must have shape {expected}, got {self.values.shape}")


@dataclass(frozen=True, slots=True)
class PolicyBatch:
    """One observation/action pair used by both flow and AR training."""

    observation: ObservationBatch
    action: ActionBatch

    def __post_init__(self) -> None:
        if self.observation.batch_size != self.action.batch_size:
            raise ValueError("observation and action batch dimensions must match")

    @property
    def batch_size(self) -> int:
        return self.observation.batch_size

    def validate_against(self, spec: BatchSpec) -> None:
        self.observation.validate_against(spec)
        self.action.validate_against(spec)


def _require_array(name: str, value: object, dtype: np.dtype, *, rank: int) -> None:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {np.dtype(dtype)}, got {value.dtype}")
    if value.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}, got shape {value.shape}")

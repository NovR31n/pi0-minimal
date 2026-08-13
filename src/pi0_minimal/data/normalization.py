"""Training-split quantile normalization with a versioned JSON cache."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating[Any]]
BoolArray = npt.NDArray[np.bool_]

_SCHEMA_VERSION = 1
_EPSILON = 1e-8


@dataclass(frozen=True, slots=True)
class QuantileStats:
    """Per-feature lower and upper quantiles used to map values to [-1, 1]."""

    lower: npt.NDArray[np.float64]
    upper: npt.NDArray[np.float64]
    count: npt.NDArray[np.int64]
    lower_quantile: float = 0.01
    upper_quantile: float = 0.99

    def __post_init__(self) -> None:
        for name, value, dtype in (
            ("lower", self.lower, np.float64),
            ("upper", self.upper, np.float64),
            ("count", self.count, np.int64),
        ):
            if not isinstance(value, np.ndarray) or value.ndim != 1 or value.dtype != dtype:
                raise TypeError(f"{name} must be a rank-1 {np.dtype(dtype)} NumPy array")
        if not (self.lower.shape == self.upper.shape == self.count.shape):
            raise ValueError("lower, upper, and count must have identical shapes")
        if self.lower.size == 0 or np.any(self.count <= 0):
            raise ValueError("every feature must contain at least one training value")
        if not np.isfinite(self.lower).all() or not np.isfinite(self.upper).all():
            raise ValueError("quantile bounds must be finite")
        if np.any(self.upper < self.lower):
            raise ValueError("upper bounds must not be lower than lower bounds")
        if not 0.0 <= self.lower_quantile < self.upper_quantile <= 1.0:
            raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")

    @property
    def feature_dim(self) -> int:
        return self.lower.size

    @classmethod
    def fit(
        cls,
        values: FloatArray,
        valid: BoolArray | None = None,
        *,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
    ) -> QuantileStats:
        array = _validate_values(values)
        mask = _validate_mask(valid, array.shape)
        lower = np.empty(array.shape[-1], dtype=np.float64)
        upper = np.empty_like(lower)
        count = np.empty(array.shape[-1], dtype=np.int64)

        flat_values = array.reshape(-1, array.shape[-1])
        flat_mask = mask.reshape(-1, array.shape[-1])
        for feature in range(array.shape[-1]):
            selected = flat_values[flat_mask[:, feature], feature]
            if selected.size == 0:
                raise ValueError(f"feature {feature} has no valid training values")
            lower[feature] = np.quantile(selected, lower_quantile)
            upper[feature] = np.quantile(selected, upper_quantile)
            count[feature] = selected.size
        return cls(lower, upper, count, lower_quantile, upper_quantile)

    def normalize(self, values: FloatArray, *, clip: bool = True) -> npt.NDArray[np.float32]:
        array = _validate_values(values, feature_dim=self.feature_dim)
        lower, scale, constant = self._parameters()
        normalized = 2.0 * (array.astype(np.float64) - lower) / scale - 1.0
        normalized[..., constant] = 0.0
        if clip:
            normalized = np.clip(normalized, -1.0, 1.0)
        return normalized.astype(np.float32)

    def inverse(self, values: FloatArray) -> npt.NDArray[np.float32]:
        array = _validate_values(values, feature_dim=self.feature_dim)
        lower, scale, constant = self._parameters()
        restored = (array.astype(np.float64) + 1.0) * scale / 2.0 + lower
        restored[..., constant] = lower[constant]
        return restored.astype(np.float32)

    def _parameters(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
        constant = np.abs(self.upper - self.lower) <= _EPSILON
        scale = np.where(constant, 1.0, self.upper - self.lower)
        return self.lower, scale, constant

    def to_dict(self) -> dict[str, object]:
        return {
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
            "count": self.count.tolist(),
            "lower_quantile": self.lower_quantile,
            "upper_quantile": self.upper_quantile,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> QuantileStats:
        return cls(
            lower=np.asarray(payload["lower"], dtype=np.float64),
            upper=np.asarray(payload["upper"], dtype=np.float64),
            count=np.asarray(payload["count"], dtype=np.int64),
            lower_quantile=float(payload["lower_quantile"]),
            upper_quantile=float(payload["upper_quantile"]),
        )


@dataclass(frozen=True, slots=True)
class NormalizationStats:
    """State/action statistics tied to one immutable training split."""

    state: QuantileStats
    action: QuantileStats
    training_split_fingerprint: str

    def __post_init__(self) -> None:
        if not self.training_split_fingerprint.strip():
            raise ValueError("training_split_fingerprint must not be empty")

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "training_split_fingerprint": self.training_split_fingerprint,
            "state": self.state.to_dict(),
            "action": self.action.to_dict(),
        }
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> NormalizationStats:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError(f"unsupported normalization schema: {payload.get('schema_version')}")
        return cls(
            state=QuantileStats.from_dict(payload["state"]),
            action=QuantileStats.from_dict(payload["action"]),
            training_split_fingerprint=str(payload["training_split_fingerprint"]),
        )


def _validate_values(
    values: FloatArray,
    *,
    feature_dim: int | None = None,
) -> FloatArray:
    if not isinstance(values, np.ndarray) or not np.issubdtype(values.dtype, np.floating):
        raise TypeError("values must be a floating-point NumPy array")
    if values.ndim < 2:
        raise ValueError("values must have at least two dimensions with features last")
    if values.shape[-1] == 0 or (feature_dim is not None and values.shape[-1] != feature_dim):
        raise ValueError(f"expected last dimension {feature_dim}, got {values.shape[-1]}")
    if not np.isfinite(values).all():
        raise ValueError("values must contain only finite entries")
    return values


def _validate_mask(valid: BoolArray | None, shape: tuple[int, ...]) -> BoolArray:
    if valid is None:
        return np.ones(shape, dtype=np.bool_)
    if not isinstance(valid, np.ndarray) or valid.dtype != np.bool_:
        raise TypeError("valid must be a boolean NumPy array")
    if valid.shape != shape:
        raise ValueError(f"valid must have shape {shape}, got {valid.shape}")
    return valid

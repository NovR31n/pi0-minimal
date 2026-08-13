"""Deterministic episode-level train/validation splitting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

_SCHEMA_VERSION = 1


@dataclass(frozen=True, order=True, slots=True)
class EpisodeRef:
    episode_id: str
    task_id: str

    def __post_init__(self) -> None:
        if not self.episode_id.strip() or not self.task_id.strip():
            raise ValueError("episode_id and task_id must not be empty")


@dataclass(frozen=True, slots=True)
class EpisodeSplit:
    """Disjoint episode IDs, stratified by task and stable across input order."""

    train: tuple[EpisodeRef, ...]
    validation: tuple[EpisodeRef, ...]
    seed: int
    validation_fraction: float

    def __post_init__(self) -> None:
        all_episodes = self.train + self.validation
        ids = [episode.episode_id for episode in all_episodes]
        if len(ids) != len(set(ids)):
            raise ValueError("episode IDs must be unique and train/validation must be disjoint")
        if not self.train:
            raise ValueError("training split must not be empty")

    @property
    def fingerprint(self) -> str:
        canonical = {
            "train": [(item.episode_id, item.task_id) for item in sorted(self.train)],
            "validation": [(item.episode_id, item.task_id) for item in sorted(self.validation)],
            "seed": self.seed,
            "validation_fraction": self.validation_fraction,
        }
        encoded = json.dumps(canonical, separators=(",", ":"), sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "seed": self.seed,
            "validation_fraction": self.validation_fraction,
            "fingerprint": self.fingerprint,
            "train": [_episode_to_dict(item) for item in self.train],
            "validation": [_episode_to_dict(item) for item in self.validation],
        }
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> EpisodeSplit:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError(f"unsupported split schema: {payload.get('schema_version')}")
        split = cls(
            train=tuple(EpisodeRef(**item) for item in payload["train"]),
            validation=tuple(EpisodeRef(**item) for item in payload["validation"]),
            seed=int(payload["seed"]),
            validation_fraction=float(payload["validation_fraction"]),
        )
        if payload.get("fingerprint") != split.fingerprint:
            raise ValueError("split manifest fingerprint does not match its contents")
        return split


def split_episodes(
    episodes: list[EpisodeRef] | tuple[EpisodeRef, ...],
    *,
    validation_fraction: float,
    seed: int,
) -> EpisodeSplit:
    """Split whole episodes within each task so frames can never leak."""

    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must satisfy 0 <= fraction < 1")
    if not episodes:
        raise ValueError("episodes must not be empty")
    ids = [episode.episode_id for episode in episodes]
    if len(ids) != len(set(ids)):
        raise ValueError("episode IDs must be globally unique")

    by_task: dict[str, list[EpisodeRef]] = {}
    for episode in episodes:
        by_task.setdefault(episode.task_id, []).append(episode)

    train: list[EpisodeRef] = []
    validation: list[EpisodeRef] = []
    for task_id in sorted(by_task):
        ranked = sorted(
            by_task[task_id],
            key=lambda item: (_rank(seed, item.episode_id), item.episode_id),
        )
        validation_count = round(len(ranked) * validation_fraction)
        if validation_fraction > 0.0 and len(ranked) > 1:
            validation_count = max(1, min(len(ranked) - 1, validation_count))
        else:
            validation_count = 0
        validation.extend(ranked[:validation_count])
        train.extend(ranked[validation_count:])

    return EpisodeSplit(tuple(sorted(train)), tuple(sorted(validation)), seed, validation_fraction)


def _rank(seed: int, episode_id: str) -> str:
    return hashlib.sha256(f"{seed}:{episode_id}".encode()).hexdigest()


def _episode_to_dict(episode: EpisodeRef) -> dict[str, str]:
    return {"episode_id": episode.episode_id, "task_id": episode.task_id}

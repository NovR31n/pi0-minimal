"""Freeze the real LIBERO split and training-only normalization statistics."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from pi0_minimal.data import (
    EpisodeRef,
    NormalizationStats,
    QuantileStats,
    libero_arrays_from_mapping,
    split_episodes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    split_config = config["split"]
    normalization = config["normalization"]
    episode_records = _read_jsonl(args.dataset_root / "meta" / "episodes.jsonl")
    episode_refs = [
        EpisodeRef(f"episode_{int(record['episode_index']):06d}", str(record["tasks"][0]))
        for record in episode_records
    ]
    split = split_episodes(
        episode_refs,
        validation_fraction=float(split_config["validation_fraction"]),
        seed=int(split_config["seed"]),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_path = args.output_dir / "split.json"
    split.save(split_path)

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for episode in split.train:
        episode_index = int(episode.episode_id.removeprefix("episode_"))
        parquet_path = (
            args.dataset_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        )
        table = pq.read_table(parquet_path, columns=["observation.state", "action"])
        episode_states, episode_actions = libero_arrays_from_mapping(
            {
                "observation.state": np.asarray(table["observation.state"].to_pylist()),
                "action": np.asarray(table["action"].to_pylist(), dtype=np.float32),
            }
        )
        states.append(episode_states)
        actions.append(episode_actions)

    stats = NormalizationStats(
        state=QuantileStats.fit(
            np.concatenate(states),
            lower_quantile=float(normalization["lower_quantile"]),
            upper_quantile=float(normalization["upper_quantile"]),
        ),
        action=QuantileStats.fit(
            np.concatenate(actions),
            lower_quantile=float(normalization["lower_quantile"]),
            upper_quantile=float(normalization["upper_quantile"]),
        ),
        training_split_fingerprint=split.fingerprint,
    )
    stats_path = args.output_dir / "normalization.json"
    stats.save(stats_path)
    print(
        f"Wrote {len(split.train)} train / {len(split.validation)} validation episodes; "
        f"fingerprint={split.fingerprint}"
    )
    print(f"Wrote {stats_path} from {sum(len(item) for item in actions)} training frames")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


if __name__ == "__main__":
    main()

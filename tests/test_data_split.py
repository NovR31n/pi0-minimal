import json

import pytest

from pi0_minimal.data import EpisodeRef, EpisodeSplit, split_episodes


def _episodes() -> list[EpisodeRef]:
    return [
        EpisodeRef(f"{task}-episode-{index}", task)
        for task in ("pick", "place")
        for index in range(5)
    ]


def test_split_is_deterministic_and_input_order_independent() -> None:
    forward = split_episodes(_episodes(), validation_fraction=0.2, seed=7)
    reverse = split_episodes(list(reversed(_episodes())), validation_fraction=0.2, seed=7)

    assert forward == reverse
    assert forward.fingerprint == reverse.fingerprint


def test_split_has_no_episode_leakage_and_stratifies_tasks() -> None:
    split = split_episodes(_episodes(), validation_fraction=0.2, seed=7)
    train_ids = {item.episode_id for item in split.train}
    validation_ids = {item.episode_id for item in split.validation}

    assert train_ids.isdisjoint(validation_ids)
    assert {item.task_id for item in split.train} == {"pick", "place"}
    assert {item.task_id for item in split.validation} == {"pick", "place"}
    assert len(split.train) == 8
    assert len(split.validation) == 2


def test_split_changes_with_seed() -> None:
    assert split_episodes(_episodes(), validation_fraction=0.4, seed=7) != split_episodes(
        _episodes(),
        validation_fraction=0.4,
        seed=17,
    )


def test_single_episode_task_remains_in_training() -> None:
    split = split_episodes([EpisodeRef("only", "rare")], validation_fraction=0.5, seed=7)

    assert split.train == (EpisodeRef("only", "rare"),)
    assert split.validation == ()


def test_duplicate_episode_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="globally unique"):
        split_episodes(
            [EpisodeRef("duplicate", "a"), EpisodeRef("duplicate", "b")],
            validation_fraction=0.2,
            seed=7,
        )


def test_split_manifest_records_fingerprint(tmp_path) -> None:
    split = split_episodes(_episodes(), validation_fraction=0.2, seed=7)
    path = tmp_path / "split.json"

    split.save(path)
    payload = json.loads(path.read_text())

    assert payload["schema_version"] == 1
    assert payload["fingerprint"] == split.fingerprint
    assert EpisodeSplit.load(path) == split


def test_modified_split_manifest_is_rejected(tmp_path) -> None:
    split = split_episodes(_episodes(), validation_fraction=0.2, seed=7)
    path = tmp_path / "split.json"
    split.save(path)
    payload = json.loads(path.read_text())
    payload["seed"] = 99
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="fingerprint"):
        EpisodeSplit.load(path)

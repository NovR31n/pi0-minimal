from pathlib import Path

import pytest

from pi0_minimal.experiment_registry import load_and_validate_registry


def test_project_registry_is_valid() -> None:
    project_root = Path(__file__).resolve().parents[1]

    registry = load_and_validate_registry(project_root / "configs" / "experiments.toml")

    assert registry["experiments"][0]["id"] == "P0-SCAFFOLD-001"


def test_duplicate_experiment_ids_are_rejected(tmp_path: Path) -> None:
    registry_path = tmp_path / "experiments.toml"
    registry_path.write_text(
        """
schema_version = 1

[[experiments]]
id = "DUPLICATE"
status = "planned"
purpose = "first"
code_commit = "abc"
config = "a.toml"
artifact_dir = "/tmp/a"
seeds = [7]

[[experiments]]
id = "DUPLICATE"
status = "planned"
purpose = "second"
code_commit = "def"
config = "b.toml"
artifact_dir = "/tmp/b"
seeds = [17]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate experiment id"):
        load_and_validate_registry(registry_path)


def test_non_list_experiments_are_rejected(tmp_path: Path) -> None:
    registry_path = tmp_path / "experiments.toml"
    registry_path.write_text(
        """
schema_version = 1
experiments = "not a list"
""",
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="experiments list"):
        load_and_validate_registry(registry_path)

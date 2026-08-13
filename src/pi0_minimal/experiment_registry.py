"""Validation helpers for the versioned experiment registry."""

import tomllib
from pathlib import Path

REQUIRED_EXPERIMENT_FIELDS = {
    "id",
    "status",
    "purpose",
    "code_commit",
    "config",
    "artifact_dir",
    "seeds",
}


def load_and_validate_registry(path: str | Path) -> dict:
    """Load a TOML experiment registry and reject incomplete or duplicate entries."""
    registry_path = Path(path)
    with registry_path.open("rb") as registry_file:
        registry = tomllib.load(registry_file)

    if registry.get("schema_version") != 1:
        raise ValueError("experiment registry schema_version must be 1")

    experiments = registry.get("experiments")
    if not isinstance(experiments, list):
        raise TypeError("experiment registry must contain an experiments list")

    seen_ids: set[str] = set()
    for index, experiment in enumerate(experiments):
        if not isinstance(experiment, dict):
            raise TypeError(f"experiment at index {index} must be a table")

        missing_fields = REQUIRED_EXPERIMENT_FIELDS - experiment.keys()
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"experiment at index {index} is missing fields: {missing}")

        experiment_id = experiment["id"]
        if not isinstance(experiment_id, str) or not experiment_id.strip():
            raise ValueError(f"experiment at index {index} has an invalid id")
        if experiment_id in seen_ids:
            raise ValueError(f"duplicate experiment id: {experiment_id}")
        seen_ids.add(experiment_id)

        if not isinstance(experiment["seeds"], list) or not all(isinstance(seed, int) for seed in experiment["seeds"]):
            raise ValueError(f"experiment {experiment_id} seeds must be a list of integers")

    return registry

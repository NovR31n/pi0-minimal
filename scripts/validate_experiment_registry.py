"""Validate the versioned experiment registry."""

from pathlib import Path

from pi0_minimal.experiment_registry import load_and_validate_registry


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    registry_path = project_root / "configs" / "experiments.toml"
    registry = load_and_validate_registry(registry_path)
    print(f"Validated {len(registry['experiments'])} experiment entries: {registry_path}")


if __name__ == "__main__":
    main()

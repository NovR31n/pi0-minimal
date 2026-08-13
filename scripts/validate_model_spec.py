"""Validate the committed compact flow-policy specification."""

from pathlib import Path

from pi0_minimal.model_spec import load_and_validate_model_spec


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    spec_path = project_root / "configs" / "model_flow_tiny.toml"
    spec = load_and_validate_model_spec(spec_path)
    action = spec["action"]
    expert = spec["action_expert"]
    print(
        "Validated compact flow specification: "
        f"H={action['horizon']}, D={action['dim']}, "
        f"width={expert['model_dim']}, layers={expert['num_layers']}"
    )


if __name__ == "__main__":
    main()

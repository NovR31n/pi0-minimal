import tomllib
from copy import deepcopy
from pathlib import Path

import pytest

from pi0_minimal.model_spec import load_and_validate_model_spec


def _project_spec_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "model_flow_tiny.toml"


def _ar_spec_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "model_ar_tiny.toml"


def test_committed_model_spec_is_valid() -> None:
    spec = load_and_validate_model_spec(_project_spec_path())

    assert spec["action"] == {"dim": 7, "horizon": 10}
    assert spec["action_expert"]["model_dim"] == 512
    assert spec["flow"]["convention"] == "data_at_one"


def test_committed_autoregressive_model_spec_is_valid() -> None:
    spec = load_and_validate_model_spec(_ar_spec_path())

    assert spec["action"] == {"dim": 7, "horizon": 10}
    assert spec["action_expert"]["model_dim"] == 512
    assert (
        spec["autoregressive"]["representation"]
        == "continuous_diagonal_gaussian"
    )


def test_inconsistent_attention_width_is_rejected(tmp_path: Path) -> None:
    with _project_spec_path().open("rb") as spec_file:
        spec = tomllib.load(spec_file)
    invalid_spec = deepcopy(spec)
    invalid_spec["action_expert"]["head_dim"] = 32

    invalid_path = tmp_path / "invalid.toml"
    _write_model_spec_for_test(invalid_path, invalid_spec)

    with pytest.raises(ValueError, match="num_heads"):
        load_and_validate_model_spec(invalid_path)


def test_unpinned_backbone_revision_is_rejected(tmp_path: Path) -> None:
    with _project_spec_path().open("rb") as spec_file:
        spec = tomllib.load(spec_file)
    invalid_spec = deepcopy(spec)
    invalid_spec["backbone"]["revision"] = "main"
    invalid_path = tmp_path / "invalid_revision.toml"
    _write_model_spec_for_test(invalid_path, invalid_spec)

    with pytest.raises(ValueError, match="40-character revision"):
        load_and_validate_model_spec(invalid_path)


@pytest.mark.parametrize(
    ("source", "field"),
    [
        ("flow", "smoothness_weight"),
        ("autoregressive", "mean_mse_weight"),
        ("autoregressive", "log_scale_regularization"),
    ],
)
def test_negative_loss_regularization_is_rejected(
    tmp_path: Path,
    source: str,
    field: str,
) -> None:
    spec_path = _project_spec_path() if source == "flow" else _ar_spec_path()
    contents = spec_path.read_text(encoding="utf-8")
    old_line = next(
        line for line in contents.splitlines() if line.startswith(f"{field} = ")
    )
    invalid_path = tmp_path / f"negative_{field}.toml"
    invalid_path.write_text(
        contents.replace(old_line, f"{field} = -0.1"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=field):
        load_and_validate_model_spec(invalid_path)


def _write_model_spec_for_test(path: Path, spec: dict) -> None:
    """Write the fixed test fixture subset without adding a TOML dependency."""
    path.write_text(
        f"""
schema_version = {spec["schema_version"]}
name = "{spec["name"]}"
backend = "{spec["backend"]}"
compute_dtype = "{spec["compute_dtype"]}"
loss_dtype = "{spec["loss_dtype"]}"

[observation]
image_keys = ["base_0_rgb", "wrist_0_rgb"]
image_height = {spec["observation"]["image_height"]}
image_width = {spec["observation"]["image_width"]}
image_channels = {spec["observation"]["image_channels"]}
max_prompt_tokens = {spec["observation"]["max_prompt_tokens"]}
state_dim = {spec["observation"]["state_dim"]}

[action]
dim = {spec["action"]["dim"]}
horizon = {spec["action"]["horizon"]}

[backbone]
model_id = "{spec["backbone"]["model_id"]}"
revision = "{spec["backbone"]["revision"]}"
output_dim = {spec["backbone"]["output_dim"]}
frozen = true
license_acceptance_required = true

[action_expert]
model_dim = {spec["action_expert"]["model_dim"]}
num_layers = {spec["action_expert"]["num_layers"]}
num_heads = {spec["action_expert"]["num_heads"]}
head_dim = {spec["action_expert"]["head_dim"]}
ffn_dim = {spec["action_expert"]["ffn_dim"]}
time_embedding_dim = {spec["action_expert"]["time_embedding_dim"]}
position_embedding = "{spec["action_expert"]["position_embedding"]}"
dropout = 0.0
activation = "swiglu"

[flow]
convention = "data_at_one"
time_distribution = "paper_shifted_beta"
beta_alpha = 1.5
beta_beta = 1.0
cutoff = 0.999
num_euler_steps = 10
""",
        encoding="utf-8",
    )

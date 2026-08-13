"""Load and validate the machine-readable compact policy specification."""

import tomllib
from pathlib import Path


def load_and_validate_model_spec(path: str | Path) -> dict:
    """Return a model specification after checking cross-field invariants."""
    spec_path = Path(path)
    with spec_path.open("rb") as spec_file:
        spec = tomllib.load(spec_file)

    if spec.get("schema_version") != 1:
        raise ValueError("model specification schema_version must be 1")
    if spec.get("backend") != "pytorch":
        raise ValueError("initial compact model backend must be pytorch")
    if spec.get("compute_dtype") != "bfloat16" or spec.get("loss_dtype") != "float32":
        raise ValueError("initial precision contract must use bfloat16 compute and float32 loss")

    observation = _required_table(spec, "observation")
    action = _required_table(spec, "action")
    backbone = _required_table(spec, "backbone")
    expert = _required_table(spec, "action_expert")
    flow = spec.get("flow")
    autoregressive = spec.get("autoregressive")
    if (flow is None) == (autoregressive is None):
        raise ValueError(
            "model specification must contain exactly one generation table"
        )

    if observation.get("image_keys") != ["base_0_rgb", "wrist_0_rgb"]:
        raise ValueError("initial specification requires base and wrist RGB views")
    if (observation.get("image_height"), observation.get("image_width"), observation.get("image_channels")) != (
        224,
        224,
        3,
    ):
        raise ValueError("initial image shape must be 3x224x224")
    if observation.get("state_dim") != 7:
        raise ValueError("initial LIBERO state dimension must be 7")
    if action.get("dim") != 7 or action.get("horizon") != 10:
        raise ValueError("initial LIBERO action contract must be [horizon=10, dim=7]")

    if not backbone.get("frozen"):
        raise ValueError("the first milestone requires a frozen condition backbone")
    if not backbone.get("license_acceptance_required"):
        raise ValueError("the PaliGemma access gate must remain explicit")
    revision = backbone.get("revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("the PaliGemma checkpoint must use a full 40-character revision")

    if expert.get("model_dim") != expert.get("num_heads") * expert.get("head_dim"):
        raise ValueError("action expert model_dim must equal num_heads * head_dim")
    if expert.get("num_layers", 0) < 1 or expert.get("ffn_dim", 0) < expert.get("model_dim", 0):
        raise ValueError("action expert depth and feed-forward dimensions are invalid")
    if expert.get("position_embedding") != "learned":
        raise ValueError("primary action expert must use learned action positions")
    if expert.get("dropout") != 0.0:
        raise ValueError("primary action expert dropout must be zero")

    if flow is not None:
        if not isinstance(flow, dict):
            raise TypeError("model specification [flow] must be a table")
        if flow.get("convention") != "data_at_one":
            raise ValueError(
                "independent flow implementation must follow data_at_one"
            )
        if flow.get("time_distribution") != "paper_shifted_beta":
            raise ValueError("primary flow configuration must use paper_shifted_beta")
        if flow.get("beta_alpha") != 1.5 or flow.get("beta_beta") != 1.0:
            raise ValueError(
                "primary shifted Beta parameters must be alpha=1.5, beta=1.0"
            )
        cutoff = flow.get("cutoff")
        if not isinstance(cutoff, float) or not 0.0 < cutoff < 1.0:
            raise ValueError("flow cutoff must lie strictly between zero and one")
        if flow.get("num_euler_steps", 0) < 1:
            raise ValueError("num_euler_steps must be positive")
        if flow.get("smoothness_weight", 0.0) < 0.0:
            raise ValueError("flow smoothness_weight must be non-negative")
    else:
        if not isinstance(autoregressive, dict):
            raise TypeError(
                "model specification [autoregressive] must be a table"
            )
        if (
            autoregressive.get("representation")
            != "continuous_diagonal_gaussian"
        ):
            raise ValueError("primary AR representation must be continuous Gaussian")
        if (
            autoregressive.get("teacher_forcing")
            != "shifted_previous_action"
        ):
            raise ValueError("primary AR teacher forcing must shift previous actions")
        if autoregressive.get("generation") != "deterministic_mean":
            raise ValueError("primary AR generation must use deterministic means")
        lower = autoregressive.get("log_scale_min")
        upper = autoregressive.get("log_scale_max")
        if (
            not isinstance(lower, float)
            or not isinstance(upper, float)
            or lower >= upper
        ):
            raise ValueError("AR log-scale bounds must be increasing floats")
        for key in ("mean_mse_weight", "log_scale_regularization"):
            if autoregressive.get(key, 0.0) < 0.0:
                raise ValueError(f"AR {key} must be non-negative")

    return spec


def _required_table(spec: dict, key: str) -> dict:
    value = spec.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"model specification requires a [{key}] table")
    return value

from pathlib import Path

import numpy as np
import pytest

from pi0_minimal.data import ActionBatch, BatchSpec, ObservationBatch, PolicyBatch
from pi0_minimal.model_spec import load_and_validate_model_spec


@pytest.fixture
def batch_spec() -> BatchSpec:
    project_root = Path(__file__).resolve().parents[1]
    model_spec = load_and_validate_model_spec(project_root / "configs" / "model_flow_tiny.toml")
    return BatchSpec.from_model_spec(model_spec)


def _valid_observation(spec: BatchSpec, batch_size: int = 2) -> ObservationBatch:
    images = np.zeros(
        (
            batch_size,
            len(spec.image_keys),
            spec.image_channels,
            spec.image_height,
            spec.image_width,
        ),
        dtype=np.uint8,
    )
    return ObservationBatch(
        image_keys=spec.image_keys,
        images=images,
        image_valid=np.ones((batch_size, len(spec.image_keys)), dtype=np.bool_),
        prompt_ids=np.zeros((batch_size, spec.max_prompt_tokens), dtype=np.int64),
        prompt_valid=np.ones((batch_size, spec.max_prompt_tokens), dtype=np.bool_),
        state=np.zeros((batch_size, spec.state_dim), dtype=np.float32),
    )


def _valid_action(spec: BatchSpec, batch_size: int = 2) -> ActionBatch:
    shape = (batch_size, spec.action_horizon, spec.action_dim)
    return ActionBatch(values=np.zeros(shape, dtype=np.float32), valid=np.ones(shape, dtype=np.bool_))


def test_policy_batch_matches_committed_spec(batch_spec: BatchSpec) -> None:
    batch = PolicyBatch(_valid_observation(batch_spec), _valid_action(batch_spec))

    batch.validate_against(batch_spec)

    assert batch.batch_size == 2


def test_channel_last_images_are_rejected(batch_spec: BatchSpec) -> None:
    observation = _valid_observation(batch_spec)
    channel_last = np.moveaxis(observation.images, 2, -1)
    invalid = ObservationBatch(
        image_keys=observation.image_keys,
        images=channel_last,
        image_valid=observation.image_valid,
        prompt_ids=observation.prompt_ids,
        prompt_valid=observation.prompt_valid,
        state=observation.state,
    )

    with pytest.raises(ValueError, match="images must have shape"):
        invalid.validate_against(batch_spec)


def test_incorrect_image_dtype_is_rejected(batch_spec: BatchSpec) -> None:
    observation = _valid_observation(batch_spec)

    with pytest.raises(TypeError, match="uint8"):
        ObservationBatch(
            image_keys=observation.image_keys,
            images=observation.images.astype(np.float32),
            image_valid=observation.image_valid,
            prompt_ids=observation.prompt_ids,
            prompt_valid=observation.prompt_valid,
            state=observation.state,
        )


def test_empty_prompt_is_rejected(batch_spec: BatchSpec) -> None:
    observation = _valid_observation(batch_spec)

    with pytest.raises(ValueError, match="prompt token"):
        ObservationBatch(
            image_keys=observation.image_keys,
            images=observation.images,
            image_valid=observation.image_valid,
            prompt_ids=observation.prompt_ids,
            prompt_valid=np.zeros_like(observation.prompt_valid),
            state=observation.state,
        )


def test_non_finite_state_and_actions_are_rejected(batch_spec: BatchSpec) -> None:
    observation = _valid_observation(batch_spec)
    invalid_state = observation.state.copy()
    invalid_state[0, 0] = np.nan
    with pytest.raises(ValueError, match="state"):
        ObservationBatch(
            image_keys=observation.image_keys,
            images=observation.images,
            image_valid=observation.image_valid,
            prompt_ids=observation.prompt_ids,
            prompt_valid=observation.prompt_valid,
            state=invalid_state,
        )

    action = _valid_action(batch_spec)
    invalid_actions = action.values.copy()
    invalid_actions[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="action values"):
        ActionBatch(values=invalid_actions, valid=action.valid)


def test_observation_action_batch_mismatch_is_rejected(batch_spec: BatchSpec) -> None:
    with pytest.raises(ValueError, match="batch dimensions"):
        PolicyBatch(_valid_observation(batch_spec, batch_size=2), _valid_action(batch_spec, batch_size=1))

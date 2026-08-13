from types import SimpleNamespace

import pytest
import torch
from torch import nn

from pi0_minimal.models import FrozenPaliGemmaBackbone


class _FakeLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_attention_mask: torch.Tensor | None = None

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
    ) -> SimpleNamespace:
        del position_ids, use_cache, return_dict
        self.last_attention_mask = attention_mask
        return SimpleNamespace(last_hidden_state=inputs_embeds + 1.0)


class _FakeCore(nn.Module):
    def __init__(self, hidden_size: int, image_tokens: int) -> None:
        super().__init__()
        self.image_projection = nn.Linear(3, hidden_size, bias=False)
        self.prompt_embedding = nn.Embedding(64, hidden_size)
        self.language_model = _FakeLanguageModel()
        self.image_tokens = image_tokens

    def get_image_features(self, pixels: torch.Tensor) -> torch.Tensor:
        channel_mean = pixels.mean(dim=(-1, -2))
        projected = self.image_projection(channel_mean)
        return projected[:, None, :].repeat(1, self.image_tokens, 1)

    def get_input_embeddings(self) -> nn.Module:
        return self.prompt_embedding


class _FakePaliGemma(nn.Module):
    def __init__(self, hidden_size: int = 16, image_tokens: int = 4) -> None:
        super().__init__()
        self.model = _FakeCore(hidden_size, image_tokens)
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(
                hidden_size=hidden_size,
                num_image_tokens=image_tokens,
            ),
            vision_config=SimpleNamespace(image_size=8),
        )


class _FakeTokenizer:
    def __call__(self, prompts: list[str], **kwargs: object) -> dict[str, torch.Tensor]:
        length = int(kwargs["max_length"])
        ids = torch.zeros((len(prompts), length), dtype=torch.int64)
        valid = torch.zeros_like(ids)
        for index, prompt in enumerate(prompts):
            used = min(len(prompt.split()) + 1, length)
            ids[index, :used] = torch.arange(1, used + 1)
            valid[index, :used] = 1
        return {"input_ids": ids, "attention_mask": valid}


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    images = torch.zeros((2, 2, 3, 8, 8), dtype=torch.uint8)
    images[:, 1] = 255
    image_valid = torch.tensor([[True, True], [True, False]])
    prompt_ids = torch.tensor([[1, 2, 0], [3, 0, 0]], dtype=torch.int64)
    prompt_valid = torch.tensor([[True, True, False], [True, False, False]])
    return images, image_valid, prompt_ids, prompt_valid


def _backbone() -> FrozenPaliGemmaBackbone:
    torch.manual_seed(7)
    return FrozenPaliGemmaBackbone(
        _FakePaliGemma(),
        expected_output_dim=16,
        compute_dtype=torch.float32,
    )


def test_backbone_returns_two_view_image_tokens_then_prompt_tokens() -> None:
    backbone = _backbone()

    memory = backbone(*_inputs())

    assert memory.values.shape == (2, 11, 16)
    assert memory.valid.shape == (2, 11)
    torch.testing.assert_close(memory.valid[0], torch.tensor([True] * 10 + [False]))
    torch.testing.assert_close(
        memory.valid[1],
        torch.tensor([True] * 4 + [False] * 4 + [True, False, False]),
    )
    assert torch.count_nonzero(memory.values[1, 4:8]) == 0


def test_backbone_uses_bidirectional_valid_key_mask() -> None:
    backbone = _backbone()

    memory = backbone(*_inputs())
    mask = backbone.model.model.language_model.last_attention_mask

    assert mask is not None
    assert mask.shape == (2, 1, 11, 11)
    assert torch.all(mask[0, 0, :, memory.valid[0]] == 0)
    assert torch.all(mask[0, 0, :, ~memory.valid[0]] == torch.finfo(torch.float32).min)


def test_backbone_is_frozen_deterministic_and_never_enters_train_mode() -> None:
    backbone = _backbone()
    inputs = _inputs()

    first = backbone(*inputs)
    backbone.train()
    second = backbone(*inputs)

    assert not backbone.training
    assert not backbone.model.training
    assert all(not parameter.requires_grad for parameter in backbone.parameters())
    assert not first.values.requires_grad
    torch.testing.assert_close(first.values, second.values)


def test_numpy_bridge_preserves_contract() -> None:
    backbone = _backbone()
    tensors = _inputs()

    memory = backbone.encode_numpy(*(tensor.numpy() for tensor in tensors))

    assert memory.values.device.type == "cpu"
    assert memory.values.dtype == torch.float32


def test_backbone_rejects_shape_mask_and_output_width_errors() -> None:
    images, image_valid, prompt_ids, prompt_valid = _inputs()
    backbone = _backbone()

    with pytest.raises(TypeError, match="uint8"):
        backbone(images.float(), image_valid, prompt_ids, prompt_valid)
    with pytest.raises(ValueError, match="at least one valid image"):
        backbone(images, torch.zeros_like(image_valid), prompt_ids, prompt_valid)
    with pytest.raises(ValueError, match="hidden size"):
        FrozenPaliGemmaBackbone(_FakePaliGemma(), expected_output_dim=2048)


def test_prompt_tokenizer_contract() -> None:
    backbone = FrozenPaliGemmaBackbone(
        _FakePaliGemma(),
        tokenizer=_FakeTokenizer(),
        expected_output_dim=16,
        compute_dtype=torch.float32,
    )

    ids, valid = backbone.tokenize_prompts(["pick up bowl", "place bowl"], max_length=5)

    assert ids.shape == (2, 5)
    assert ids.dtype == torch.int64
    assert valid.dtype == torch.bool
    with pytest.raises(ValueError, match="non-empty"):
        backbone.tokenize_prompts([""], max_length=5)


def test_adapter_runs_real_transformers_paligemma_modules() -> None:
    transformers = pytest.importorskip("transformers")
    text_config = transformers.GemmaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        num_image_tokens=4,
        pad_token_id=0,
    )
    vision_config = transformers.SiglipVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_channels=3,
        image_size=8,
        patch_size=4,
        projection_dim=16,
    )
    config = transformers.PaliGemmaConfig(
        vision_config=vision_config.to_dict(),
        text_config=text_config.to_dict(),
        image_token_index=63,
        projection_dim=16,
        hidden_size=16,
    )
    backbone = FrozenPaliGemmaBackbone(
        transformers.PaliGemmaForConditionalGeneration(config),
        expected_output_dim=16,
        compute_dtype=torch.float32,
    )
    images, image_valid, prompt_ids, prompt_valid = _inputs()

    memory = backbone(images[:1], image_valid[:1], prompt_ids[:1], prompt_valid[:1])

    assert memory.values.shape == (1, 11, 16)
    assert torch.isfinite(memory.values).all()

"""Frozen PaliGemma adapter producing model-neutral condition memory."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class ConditionMemory:
    """Contextualized condition tokens and their validity mask."""

    values: torch.Tensor
    valid: torch.Tensor

    def __post_init__(self) -> None:
        if self.values.ndim != 3:
            raise ValueError(f"condition values must have shape [B,S,D], got {self.values.shape}")
        if self.valid.dtype != torch.bool or self.valid.shape != self.values.shape[:2]:
            raise ValueError("condition valid must be bool [B,S]")
        # Validate large CPU caches in bounded chunks.  A single isfinite() over
        # a multi-gigabyte condition tensor materializes an equally shaped bool
        # tensor and can push an otherwise viable training run over host RAM.
        flattened = self.values.reshape(-1)
        finite_check_chunk_size = 1_048_576
        for start in range(0, flattened.numel(), finite_check_chunk_size):
            if not torch.isfinite(
                flattened[start : start + finite_check_chunk_size]
            ).all():
                raise ValueError("condition values must be finite")


class FrozenPaliGemmaBackbone(nn.Module):
    """Use PaliGemma vision/projector/Gemma blocks as a frozen condition encoder."""

    def __init__(
        self,
        model: nn.Module,
        *,
        tokenizer: Any | None = None,
        expected_image_views: int = 2,
        expected_output_dim: int = 2048,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        if expected_image_views <= 0:
            raise ValueError("expected_image_views must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.expected_image_views = expected_image_views
        self.compute_dtype = compute_dtype

        config = model.config
        self.output_dim = int(config.text_config.hidden_size)
        self.image_tokens_per_view = int(config.text_config.num_image_tokens)
        self.image_size = int(config.vision_config.image_size)
        if self.output_dim != expected_output_dim:
            raise ValueError(
                f"PaliGemma hidden size must be {expected_output_dim}, got {self.output_dim}"
            )
        if self.image_tokens_per_view <= 0:
            raise ValueError("PaliGemma must expose a positive num_image_tokens")

        self.model.requires_grad_(False)
        self.train(False)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        device: str | torch.device,
        compute_dtype: torch.dtype = torch.bfloat16,
        expected_image_views: int = 2,
        expected_output_dim: int = 2048,
        revision: str | None = None,
    ) -> FrozenPaliGemmaBackbone:
        """Load official weights; gated repositories require prior HF authentication."""

        from transformers import AutoTokenizer, PaliGemmaForConditionalGeneration

        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=compute_dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        model.to(device)
        return cls(
            model,
            tokenizer=tokenizer,
            expected_image_views=expected_image_views,
            expected_output_dim=expected_output_dim,
            compute_dtype=compute_dtype,
        )

    def train(self, mode: bool = True) -> FrozenPaliGemmaBackbone:
        """Keep the frozen backbone in evaluation mode when a parent policy trains."""

        del mode
        super().train(False)
        return self

    def tokenize_prompts(
        self,
        prompts: Sequence[str],
        *,
        max_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.tokenizer is None:
            raise RuntimeError("tokenizer is unavailable on this backbone instance")
        if not prompts or any(not prompt.strip() for prompt in prompts):
            raise ValueError("prompts must contain non-empty strings")
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        tokens = self.tokenizer(
            list(prompts),
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return tokens["input_ids"].to(torch.int64), tokens["attention_mask"].to(torch.bool)

    def forward(
        self,
        images: torch.Tensor,
        image_valid: torch.Tensor,
        prompt_ids: torch.Tensor,
        prompt_valid: torch.Tensor,
    ) -> ConditionMemory:
        self._validate_inputs(images, image_valid, prompt_ids, prompt_valid)
        device = next(self.model.parameters()).device
        images = images.to(device=device, dtype=self.compute_dtype)
        image_valid = image_valid.to(device=device)
        prompt_ids = prompt_ids.to(device=device)
        prompt_valid = prompt_valid.to(device=device)

        batch_size, image_views = images.shape[:2]
        flat_pixels = images.reshape(
            batch_size * image_views,
            images.shape[2],
            images.shape[3],
            images.shape[4],
        )
        flat_pixels = flat_pixels / 127.5 - 1.0

        with torch.no_grad():
            core = self.model.model
            image_features = core.get_image_features(flat_pixels)
            expected_image_shape = (
                batch_size * image_views,
                self.image_tokens_per_view,
                self.output_dim,
            )
            if image_features.shape != expected_image_shape:
                raise ValueError(
                    f"image features must have shape {expected_image_shape}, "
                    f"got {image_features.shape}"
                )
            image_features = image_features.reshape(
                batch_size,
                image_views * self.image_tokens_per_view,
                self.output_dim,
            )
            prompt_features = core.get_input_embeddings()(prompt_ids)
            if prompt_features.shape[-1] != self.output_dim:
                raise ValueError(
                    f"prompt embedding width must be {self.output_dim}, "
                    f"got {prompt_features.shape[-1]}"
                )
            values = torch.cat((image_features, prompt_features), dim=1)

            image_token_valid = image_valid.repeat_interleave(
                self.image_tokens_per_view,
                dim=1,
            )
            valid = torch.cat((image_token_valid, prompt_valid), dim=1)
            position_ids = valid.to(torch.int64).cumsum(dim=1).clamp_min(1)
            attention_mask = _full_attention_mask(valid, values.dtype)
            outputs = core.language_model(
                inputs_embeds=values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            )
            contextualized = outputs.last_hidden_state
            if contextualized.shape != values.shape:
                raise ValueError(
                    f"Gemma output must have shape {values.shape}, got {contextualized.shape}"
                )
            contextualized = contextualized.masked_fill(~valid.unsqueeze(-1), 0.0)

        return ConditionMemory(contextualized, valid)

    def encode_numpy(
        self,
        images: np.ndarray,
        image_valid: np.ndarray,
        prompt_ids: np.ndarray,
        prompt_valid: np.ndarray,
    ) -> ConditionMemory:
        """Bridge the model-neutral NumPy batch contract into PyTorch."""

        return self(
            torch.from_numpy(images),
            torch.from_numpy(image_valid),
            torch.from_numpy(prompt_ids),
            torch.from_numpy(prompt_valid),
        )

    def _validate_inputs(
        self,
        images: torch.Tensor,
        image_valid: torch.Tensor,
        prompt_ids: torch.Tensor,
        prompt_valid: torch.Tensor,
    ) -> None:
        if images.dtype != torch.uint8 or images.ndim != 5:
            raise TypeError("images must be uint8 [B,V,C,H,W]")
        batch_size, views, _channels, _height, _width = images.shape
        expected_images = (
            batch_size,
            self.expected_image_views,
            3,
            self.image_size,
            self.image_size,
        )
        if images.shape != expected_images:
            raise ValueError(f"images must have shape {expected_images}, got {images.shape}")
        if image_valid.dtype != torch.bool or image_valid.shape != (batch_size, views):
            raise ValueError("image_valid must be bool [B,V]")
        if not image_valid.any(dim=1).all():
            raise ValueError("every sample must contain at least one valid image")
        if prompt_ids.dtype != torch.int64 or prompt_ids.ndim != 2:
            raise TypeError("prompt_ids must be int64 [B,L]")
        if prompt_valid.dtype != torch.bool or prompt_valid.shape != prompt_ids.shape:
            raise ValueError("prompt_valid must be bool with the same shape as prompt_ids")
        if prompt_ids.shape[0] != batch_size:
            raise ValueError("images and prompts must share the batch dimension")
        if not prompt_valid.any(dim=1).all():
            raise ValueError("every sample must contain at least one valid prompt token")
        if torch.any(prompt_ids[prompt_valid] < 0):
            raise ValueError("valid prompt token IDs must be non-negative")


def _full_attention_mask(valid: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Additive full-attention mask: all queries see every valid condition key."""

    key_valid = valid[:, None, None, :]
    mask = torch.zeros(
        (valid.shape[0], 1, valid.shape[1], valid.shape[1]),
        dtype=dtype,
        device=valid.device,
    )
    return mask.masked_fill(~key_valid, torch.finfo(dtype).min)

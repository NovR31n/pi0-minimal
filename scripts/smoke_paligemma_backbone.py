"""Run the frozen PaliGemma adapter on the committed real LIBERO smoke subset."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from pi0_minimal.model_spec import load_and_validate_model_spec
from pi0_minimal.models import FrozenPaliGemmaBackbone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model_flow_tiny.toml"),
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    spec = load_and_validate_model_spec(args.config)
    observation = spec["observation"]
    backbone_config = spec["backbone"]
    dtype = _dtype_from_name(spec["compute_dtype"])

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    backbone = FrozenPaliGemmaBackbone.from_pretrained(
        str(backbone_config["model_id"]),
        revision=str(backbone_config["revision"]),
        device=args.device,
        compute_dtype=dtype,
        expected_image_views=len(observation["image_keys"]),
        expected_output_dim=int(backbone_config["output_dim"]),
    )

    with np.load(args.data) as smoke:
        available = len(smoke["images"])
        if args.batch_size > available:
            raise ValueError(f"batch-size {args.batch_size} exceeds {available} smoke samples")
        images = smoke["images"][: args.batch_size]
        prompts = smoke["prompts"][: args.batch_size].tolist()
    prompt_ids, prompt_valid = backbone.tokenize_prompts(
        prompts,
        max_length=int(observation["max_prompt_tokens"]),
    )
    image_valid = np.ones(images.shape[:2], dtype=np.bool_)

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
        torch.cuda.synchronize(args.device)
    start = time.perf_counter()
    memory = backbone.encode_numpy(
        images,
        image_valid,
        prompt_ids.numpy(),
        prompt_valid.numpy(),
    )
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    parameters = sum(parameter.numel() for parameter in backbone.parameters())
    trainable = sum(
        parameter.numel() for parameter in backbone.parameters() if parameter.requires_grad
    )
    result = {
        "model_id": backbone_config["model_id"],
        "revision": backbone_config["revision"],
        "batch_size": args.batch_size,
        "memory_shape": list(memory.values.shape),
        "memory_dtype": str(memory.values.dtype),
        "valid_tokens": memory.valid.sum(dim=1).tolist(),
        "parameters": parameters,
        "trainable_parameters": trainable,
        "forward_ms": elapsed_ms,
        "finite": bool(torch.isfinite(memory.values).all().item()),
        "detached": not memory.values.requires_grad,
    }
    if args.device.startswith("cuda"):
        result["cuda_allocated_mib"] = torch.cuda.memory_allocated(args.device) / 2**20
        result["cuda_peak_allocated_mib"] = (
            torch.cuda.max_memory_allocated(args.device) / 2**20
        )
    print(json.dumps(result, indent=2))


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported compute dtype: {name}")


if __name__ == "__main__":
    main()

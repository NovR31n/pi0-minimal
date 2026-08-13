"""Run the frozen PaliGemma plus trainable compact flow stack forward/backward."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from pi0_minimal.data import NormalizationStats
from pi0_minimal.model_spec import load_and_validate_model_spec
from pi0_minimal.models import (
    ActionTimeEmbedding,
    ConditionProjector,
    FlowActionExpert,
    FrozenPaliGemmaBackbone,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model_flow_tiny.toml"),
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    spec = load_and_validate_model_spec(args.config)
    observation = spec["observation"]
    action = spec["action"]
    backbone_config = spec["backbone"]
    expert_config = spec["action_expert"]
    dtype = _dtype_from_name(spec["compute_dtype"])

    backbone = FrozenPaliGemmaBackbone.from_pretrained(
        str(backbone_config["model_id"]),
        revision=str(backbone_config["revision"]),
        device=args.device,
        compute_dtype=dtype,
        expected_image_views=len(observation["image_keys"]),
        expected_output_dim=int(backbone_config["output_dim"]),
    )
    projector = ConditionProjector(
        condition_dim=int(backbone_config["output_dim"]),
        state_dim=int(observation["state_dim"]),
        model_dim=int(expert_config["model_dim"]),
    ).to(device=args.device, dtype=dtype)
    action_embedding = ActionTimeEmbedding(
        action_dim=int(action["dim"]),
        model_dim=int(expert_config["model_dim"]),
        time_dim=int(expert_config["time_embedding_dim"]),
    ).to(device=args.device, dtype=dtype)
    expert = FlowActionExpert(
        model_dim=int(expert_config["model_dim"]),
        num_layers=int(expert_config["num_layers"]),
        num_heads=int(expert_config["num_heads"]),
        ffn_dim=int(expert_config["ffn_dim"]),
        action_dim=int(action["dim"]),
        max_horizon=int(action["horizon"]),
        dropout=float(expert_config["dropout"]),
    ).to(device=args.device, dtype=dtype)

    normalization = NormalizationStats.load(args.normalization)
    with np.load(args.data) as smoke:
        available = len(smoke["images"])
        if args.batch_size > available:
            raise ValueError(f"batch-size {args.batch_size} exceeds {available} smoke samples")
        images = smoke["images"][: args.batch_size]
        prompts = smoke["prompts"][: args.batch_size].tolist()
        states = normalization.state.normalize(smoke["state"][: args.batch_size])
        action_valid = smoke["action_valid"][: args.batch_size]

    prompt_ids, prompt_valid = backbone.tokenize_prompts(
        prompts,
        max_length=int(observation["max_prompt_tokens"]),
    )
    image_valid = np.ones(images.shape[:2], dtype=np.bool_)
    state_valid = np.ones(states.shape, dtype=np.bool_)
    condition = backbone.encode_numpy(
        images,
        image_valid,
        prompt_ids.numpy(),
        prompt_valid.numpy(),
    )

    generator = torch.Generator(device=args.device)
    generator.manual_seed(args.seed)
    noisy_actions = torch.randn(
        (args.batch_size, int(action["horizon"]), int(action["dim"])),
        generator=generator,
        dtype=torch.float32,
        device=args.device,
    )
    flow_time = torch.full(
        (args.batch_size,),
        0.5,
        dtype=torch.float32,
        device=args.device,
    )
    action_valid_tensor = torch.from_numpy(action_valid).to(args.device)

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
        torch.cuda.synchronize(args.device)
    start_forward = time.perf_counter()
    memory = projector.encode_numpy(condition, states, state_valid)
    action_tokens = action_embedding(
        noisy_actions,
        flow_time,
        action_valid_tensor,
    )
    velocity = expert(action_tokens, memory)
    loss = (
        velocity.square() * action_valid_tensor
    ).sum() / action_valid_tensor.sum()
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
    forward_ms = (time.perf_counter() - start_forward) * 1000.0

    start_backward = time.perf_counter()
    loss.backward()
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
    backward_ms = (time.perf_counter() - start_backward) * 1000.0

    trainable_modules = (projector, action_embedding, expert)
    trainable_parameters = sum(
        parameter.numel()
        for module in trainable_modules
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    gradients = [
        parameter.grad
        for module in trainable_modules
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    result = {
        "model_id": backbone_config["model_id"],
        "revision": backbone_config["revision"],
        "batch_size": args.batch_size,
        "memory_shape": list(memory.values.shape),
        "action_token_shape": list(action_tokens.values.shape),
        "velocity_shape": list(velocity.shape),
        "velocity_dtype": str(velocity.dtype),
        "trainable_parameters": trainable_parameters,
        "loss": float(loss.detach().cpu()),
        "forward_ms_excluding_backbone": forward_ms,
        "backward_ms": backward_ms,
        "finite_velocity": bool(torch.isfinite(velocity).all().item()),
        "all_trainable_gradients_present": all(gradient is not None for gradient in gradients),
        "finite_gradients": all(
            gradient is not None and torch.isfinite(gradient).all().item()
            for gradient in gradients
        ),
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

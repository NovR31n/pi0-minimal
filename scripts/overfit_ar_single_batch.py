"""Overfit one real LIBERO sample with the continuous AR baseline."""

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
    AutoregressivePolicy,
    ConditionMemory,
    FrozenPaliGemmaBackbone,
    build_autoregressive_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/model_ar_tiny.toml")
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--acceptance-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.learning_rate <= 0.0 or args.gradient_clip <= 0.0:
        raise ValueError("steps, learning-rate, and gradient-clip must be positive")
    if not 0.0 < args.acceptance_ratio < 1.0:
        raise ValueError("acceptance-ratio must lie in (0,1)")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    spec = load_and_validate_model_spec(args.config)
    observation = spec["observation"]
    backbone_config = spec["backbone"]
    backbone = FrozenPaliGemmaBackbone.from_pretrained(
        str(backbone_config["model_id"]),
        revision=str(backbone_config["revision"]),
        device=args.device,
        compute_dtype=_dtype_from_name(spec["compute_dtype"]),
        expected_image_views=len(observation["image_keys"]),
        expected_output_dim=int(backbone_config["output_dim"]),
    )
    policy = build_autoregressive_policy(spec, device=args.device)
    policy.train()

    normalization = NormalizationStats.load(args.normalization)
    with np.load(args.data) as smoke:
        images = smoke["images"][:1]
        prompts = smoke["prompts"][:1].tolist()
        states = normalization.state.normalize(smoke["state"][:1])
        actions = normalization.action.normalize(smoke["actions"][:1])
        action_valid = smoke["action_valid"][:1]

    prompt_ids, prompt_valid = backbone.tokenize_prompts(
        prompts,
        max_length=int(observation["max_prompt_tokens"]),
    )
    condition = backbone.encode_numpy(
        images,
        np.ones(images.shape[:2], dtype=np.bool_),
        prompt_ids.numpy(),
        prompt_valid.numpy(),
    )
    del backbone
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    state = torch.from_numpy(states).to(args.device)
    state_valid = torch.ones_like(state, dtype=torch.bool)
    expert_actions = torch.from_numpy(actions).to(args.device)
    action_valid_tensor = torch.from_numpy(action_valid).to(args.device)

    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.learning_rate, weight_decay=0.0
    )
    history: list[dict[str, float | int]] = []
    initial_loss, initial_mae = _evaluate(
        policy,
        condition,
        state,
        state_valid,
        expert_actions,
        action_valid_tensor,
        device=args.device,
    )
    history.append(
        {"step": 0, "loss": initial_loss, "action_mean_mae": initial_mae}
    )

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
        torch.cuda.synchronize(args.device)
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(args.device):
            output = policy.training_step(
                condition,
                state,
                expert_actions,
                action_valid_tensor,
                state_valid=state_valid,
            )
        output.loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(),
            max_norm=args.gradient_clip,
            error_if_nonfinite=True,
        )
        optimizer.step()
        if step == 1 or step % 10 == 0 or step == args.steps:
            loss, mae = _evaluate(
                policy,
                condition,
                state,
                state_valid,
                expert_actions,
                action_valid_tensor,
                device=args.device,
            )
            history.append(
                {
                    "step": step,
                    "loss": loss,
                    "action_mean_mae": mae,
                    "gradient_norm": float(gradient_norm.detach().cpu()),
                }
            )
            print(json.dumps(history[-1]), flush=True)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
    elapsed_seconds = time.perf_counter() - start

    final_loss = float(history[-1]["loss"])
    final_mae = float(history[-1]["action_mean_mae"])
    mae_ratio = final_mae / initial_mae
    accepted = mae_ratio <= args.acceptance_ratio and final_loss < initial_loss
    result = {
        "seed": args.seed,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "gradient_clip": args.gradient_clip,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "initial_action_mean_mae": initial_mae,
        "final_action_mean_mae": final_mae,
        "mae_ratio": mae_ratio,
        "acceptance_ratio": args.acceptance_ratio,
        "elapsed_seconds": elapsed_seconds,
        "accepted": accepted,
        "history": history,
    }
    if args.device.startswith("cuda"):
        result["cuda_peak_allocated_mib"] = (
            torch.cuda.max_memory_allocated(args.device) / 2**20
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    torch.save(
        {
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "result": result,
            "config": str(args.config),
        },
        args.output_dir / "checkpoint.pt",
    )
    print(json.dumps(result, indent=2))
    if not accepted:
        raise RuntimeError(
            f"single-batch AR overfit failed: MAE ratio {mae_ratio:.6f} "
            f"exceeds {args.acceptance_ratio:.6f}"
        )


@torch.no_grad()
def _evaluate(
    policy: AutoregressivePolicy,
    condition: ConditionMemory,
    state: torch.Tensor,
    state_valid: torch.Tensor,
    expert_actions: torch.Tensor,
    action_valid: torch.Tensor,
    *,
    device: str,
) -> tuple[float, float]:
    policy.eval()
    with _autocast_context(device):
        output = policy.training_step(
            condition,
            state,
            expert_actions,
            action_valid,
            state_valid=state_valid,
        )
    policy.train()
    absolute_error = (output.distribution.mean - expert_actions).abs()
    mae = absolute_error[action_valid].mean()
    return float(output.loss.cpu()), float(mae.cpu())


def _autocast_context(device: str) -> torch.autocast:
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    return torch.autocast(
        device_type=device_type,
        dtype=torch.bfloat16,
        enabled=device_type == "cuda",
    )


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported compute dtype: {name}")


if __name__ == "__main__":
    main()

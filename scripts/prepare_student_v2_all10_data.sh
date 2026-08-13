#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
dataset_root=${LIBERO_DATASET_ROOT:?set LIBERO_DATASET_ROOT to the LeRobot dataset}
python_bin=${LIBERO_DATA_PYTHON:-${project_dir}/.venv/bin/python}
output_dir=${project_dir}/data/libero_spatial_v2
output_path=${output_dir}/formal_all10_all434_32.npz
partial_path=${output_dir}/formal_all10_all434_32.partial.npz

cd "${project_dir}"
mkdir -p "${output_dir}"

episode_ids=()
for ((episode_id = 0; episode_id < 434; episode_id++)); do
    episode_ids+=("${episode_id}")
done

PYTHONPATH=${project_dir}/src \
    "${python_bin}" \
    scripts/create_libero_smoke_subset.py \
    --dataset-root "${dataset_root}" \
    --output "${partial_path}" \
    --episodes "${episode_ids[@]}" \
    --samples-per-episode 32 \
    --image-size 224 \
    --action-horizon 10

.venv/bin/python - "${partial_path}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


path = Path(sys.argv[1])
with np.load(path, allow_pickle=False) as data:
    expected_samples = 434 * 32
    expected_shapes = {
        "images": (expected_samples, 2, 3, 224, 224),
        "state": (expected_samples, 7),
        "actions": (expected_samples, 10, 7),
        "action_valid": (expected_samples, 10, 7),
        "prompts": (expected_samples,),
        "episode_index": (expected_samples,),
        "frame_index": (expected_samples,),
    }
    actual_shapes = {key: data[key].shape for key in expected_shapes}
    if actual_shapes != expected_shapes:
        raise ValueError(f"unexpected Student-v2 data shapes: {actual_shapes}")
    if not np.isfinite(data["state"]).all() or not np.isfinite(data["actions"]).all():
        raise ValueError("Student-v2 state and action tensors must be finite")
    episode_ids, episode_counts = np.unique(data["episode_index"], return_counts=True)
    if not np.array_equal(episode_ids, np.arange(434)):
        raise ValueError("Student-v2 data must contain every episode ID from 0 through 433")
    if not np.array_equal(episode_counts, np.full(434, 32)):
        raise ValueError("Student-v2 data must contain exactly 32 samples per episode")
    if len(np.unique(data["prompts"])) != 10:
        raise ValueError("Student-v2 data must contain exactly ten task prompts")

print(f"validated {expected_samples} samples across 434 episodes and 10 prompts")
PY

mv "${partial_path}" "${output_path}"
sha256sum "${output_path}"

#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
checkpoint=artifacts/STUDENT-V3-CORRECTION-SUCCESS39-10K-S7-015/step_10000.pt
output_dir=results/STUDENT-V3-SUCCESS39-10K-S7-R5-DEV90-INIT21-29-017
task_ids=(0 1 2 3 4 5 6 7 8 9)
validation_inits=(21 22 23 24 25 26 27 28 29)

cd "${project_dir}"
if [[ ! -f "${checkpoint}" ]]; then
    echo "checkpoint is missing: ${checkpoint}" >&2
    exit 1
fi

mkdir -p "${output_dir}"
env \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    .venv/bin/python scripts/rollout_libero_matrix.py \
    --config configs/model_flow_tiny.toml \
    --checkpoint "${checkpoint}" \
    --normalization data/libero_spatial_v1/normalization.json \
    --output-dir "${output_dir}" \
    --task-suite libero_spatial \
    --task-ids "${task_ids[@]}" \
    --init-state-indices "${validation_inits[@]}" \
    --training-seed 7 \
    --eval-seed-offset 1000 \
    --max-steps 220 \
    --wait-steps 10 \
    --replan-steps 5 \
    --resolution 256 \
    --device cuda \
    --skip-existing \
    > "${output_dir}/eval.log" 2>&1

echo "completed STUDENT-V3-SUCCESS39-10K-S7-R5-DEV90-INIT21-29-017"

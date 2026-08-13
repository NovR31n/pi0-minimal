#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${project_dir}"

task_ids=(0 1 2 3 4 5 6 7 8 9)
development_inits=(10 11 12 13 14 15 16 17 18 19)

run_checkpoint() {
    local checkpoint_path=$1
    local output_dir=$2

    mkdir -p "${output_dir}"
    env \
        TOKENIZERS_PARALLELISM=false \
        PYTHONUNBUFFERED=1 \
        .venv/bin/python scripts/rollout_libero_matrix.py \
        --config configs/model_flow_tiny.toml \
        --checkpoint "${checkpoint_path}" \
        --normalization data/libero_spatial_v1/normalization.json \
        --output-dir "${output_dir}" \
        --task-suite libero_spatial \
        --task-ids "${task_ids[@]}" \
        --init-state-indices "${development_inits[@]}" \
        --training-seed 7 \
        --eval-seed-offset 1000 \
        --max-steps 220 \
        --wait-steps 10 \
        --replan-steps 5 \
        --resolution 256 \
        --device cuda \
        --skip-existing \
        > "${output_dir}/eval.log" 2>&1
}

run_checkpoint \
    artifacts/STUDENT-V1-DISTILL-BAL100-10K-S7-001/step_02000.pt \
    results/STUDENT-V1-DISTILL-BAL100-2K-S7-R5-DEV100-001

run_checkpoint \
    artifacts/STUDENT-V1-DISTILL-BAL100-10K-S7-001/step_05000.pt \
    results/STUDENT-V1-DISTILL-BAL100-5K-S7-R5-DEV100-001

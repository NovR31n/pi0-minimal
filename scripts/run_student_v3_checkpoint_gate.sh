#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
experiment_root=results/STUDENT-V3-SUCCESS39-CHECKPOINT-GATE-S7-R5-DEV400-016
task_ids=(0 1 2 3 4 5 6 7 8 9)
development_inits=(10 11 12 13 14 15 16 17 18 19)

run_checkpoint() {
    local checkpoint_name=$1
    local checkpoint_path=$2
    local output_dir=${experiment_root}/${checkpoint_name}

    if [[ ! -f "${checkpoint_path}" ]]; then
        echo "checkpoint is missing: ${checkpoint_path}" >&2
        exit 1
    fi
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

cd "${project_dir}"
mkdir -p "${experiment_root}"
run_checkpoint \
    step_02000 \
    artifacts/STUDENT-V3-CORRECTION-SUCCESS39-10K-S7-015/step_02000.pt
run_checkpoint \
    step_05000 \
    artifacts/STUDENT-V3-CORRECTION-SUCCESS39-10K-S7-015/step_05000.pt
run_checkpoint \
    best_generation_step_09000 \
    artifacts/STUDENT-V3-CORRECTION-SUCCESS39-10K-S7-015/best_generation.pt
run_checkpoint \
    step_10000 \
    artifacts/STUDENT-V3-CORRECTION-SUCCESS39-10K-S7-015/step_10000.pt

echo "completed STUDENT-V3-SUCCESS39-CHECKPOINT-GATE-S7-R5-DEV400-016"

#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cache_dir=${PI0_MINIMAL_CACHE_DIR:-${project_dir}/.cache/pi0_minimal}

if [[ $# -ne 1 ]]; then
    echo "usage: $0 {preflight|formal}" >&2
    exit 2
fi

run_kind=$1
case "${run_kind}" in
    preflight)
        experiment_id=STUDENT-V3-CORRECTION-SUCCESS39-PREFLIGHT-S7-014
        steps=100
        eval_every=100
        generation_eval_samples=32
        snapshot_args=()
        ;;
    formal)
        experiment_id=STUDENT-V3-CORRECTION-SUCCESS39-10K-S7-015
        steps=10000
        eval_every=1000
        generation_eval_samples=64
        snapshot_args=(
            --snapshot-step 2000
            --snapshot-step 5000
            --snapshot-step 10000
        )
        ;;
    *)
        echo "run kind must be preflight or formal" >&2
        exit 2
        ;;
esac

data_path=data/libero_spatial_v2/formal_all10_all434_32.npz
teacher_data=data/distillation/teacher_balanced100_v1.npz
correction_data=artifacts/TEACHER-CORRECTION-DATASET-V2-SUCCESS39-013/corrections_success39.npz
initial_checkpoint=artifacts/STUDENT-V2-ALL10-DISTILL-BAL100-10K-S7-001/best_generation.pt
condition_cache=${cache_dir}/all10_all434_32_conditions.pt
teacher_condition_cache=${cache_dir}/teacher_balanced100_v1_conditions.pt
correction_condition_cache=${cache_dir}/corrections_success39_conditions.pt
output_dir=artifacts/${experiment_id}

cd "${project_dir}"
for required_path in \
    "${data_path}" \
    "${teacher_data}" \
    "${correction_data}" \
    data/libero_spatial_v1/split.json \
    data/libero_spatial_v1/normalization.json \
    "${initial_checkpoint}" \
    "${condition_cache}" \
    "${teacher_condition_cache}"; do
    if [[ ! -f "${required_path}" ]]; then
        echo "required input is missing: ${required_path}" >&2
        exit 1
    fi
done
if [[ -e "${output_dir}/result.json" ]]; then
    echo "completed output already exists: ${output_dir}/result.json" >&2
    exit 1
fi

mkdir -p "${output_dir}"
env \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    .venv/bin/python scripts/train_flow_small.py \
    --config configs/model_flow_tiny.toml \
    --data "${data_path}" \
    --teacher-data "${teacher_data}" \
    --teacher-sampling-fraction 0.25 \
    --correction-data "${correction_data}" \
    --correction-sampling-fraction 0.25 \
    --split data/libero_spatial_v1/split.json \
    --normalization data/libero_spatial_v1/normalization.json \
    --output-dir "${output_dir}" \
    --steps "${steps}" \
    --batch-size 16 \
    --condition-batch-size 4 \
    --condition-cache "${condition_cache}" \
    --teacher-condition-cache "${teacher_condition_cache}" \
    --correction-condition-cache "${correction_condition_cache}" \
    --eval-every "${eval_every}" \
    --generation-eval-samples "${generation_eval_samples}" \
    --learning-rate 3e-5 \
    --weight-decay 1e-4 \
    --gradient-clip 1.0 \
    --state-noise-std 0.05 \
    --device cuda \
    --seed 7 \
    --init-checkpoint "${initial_checkpoint}" \
    "${snapshot_args[@]}" \
    > "${output_dir}/train.log" 2>&1

echo "completed ${experiment_id}"

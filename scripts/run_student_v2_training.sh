#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cache_dir=${PI0_MINIMAL_CACHE_DIR:-${project_dir}/.cache/pi0_minimal}

if [[ $# -ne 2 ]]; then
    echo "usage: $0 {preflight|formal} {demo|distill}" >&2
    exit 2
fi

run_kind=$1
training_arm=$2
case "${run_kind}" in
    preflight)
        steps=100
        eval_every=100
        snapshot_args=()
        ;;
    formal)
        steps=10000
        eval_every=1000
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

case "${training_arm}" in
    demo)
        arm_name=DEMO
        teacher_args=()
        ;;
    distill)
        arm_name=DISTILL-BAL100
        teacher_args=(
            --teacher-data data/distillation/teacher_balanced100_v1.npz
            --teacher-sampling-fraction 0.5
            --teacher-condition-cache "${cache_dir}/teacher_balanced100_v1_conditions.pt"
        )
        ;;
    *)
        echo "training arm must be demo or distill" >&2
        exit 2
        ;;
esac

data_path=data/libero_spatial_v2/formal_all10_all434_32.npz
condition_cache=${cache_dir}/all10_all434_32_conditions.pt
initial_checkpoint=artifacts/P6-FLOW-ENVTASKS03478-ALL214-64-S7-STATE-NOISE05-10K-001/best_generation.pt

if [[ "${run_kind}" == preflight ]]; then
    experiment_id=STUDENT-V2-ALL10-${arm_name}-PREFLIGHT-S7-001
else
    experiment_id=STUDENT-V2-ALL10-${arm_name}-10K-S7-001
fi
output_dir=artifacts/${experiment_id}

cd "${project_dir}"
for required_path in \
    "${data_path}" \
    data/libero_spatial_v1/split.json \
    data/libero_spatial_v1/normalization.json \
    "${initial_checkpoint}"; do
    if [[ ! -f "${required_path}" ]]; then
        echo "required input is missing: ${required_path}" >&2
        exit 1
    fi
done
if [[ "${training_arm}" == distill && ! -f data/distillation/teacher_balanced100_v1.npz ]]; then
    echo "teacher data is missing" >&2
    exit 1
fi
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
    --split data/libero_spatial_v1/split.json \
    --normalization data/libero_spatial_v1/normalization.json \
    --output-dir "${output_dir}" \
    --steps "${steps}" \
    --batch-size 16 \
    --condition-batch-size 4 \
    --condition-cache "${condition_cache}" \
    --eval-every "${eval_every}" \
    --generation-eval-samples 64 \
    --learning-rate 3e-5 \
    --weight-decay 1e-4 \
    --gradient-clip 1.0 \
    --state-noise-std 0.05 \
    --device cuda \
    --seed 7 \
    --init-checkpoint "${initial_checkpoint}" \
    "${snapshot_args[@]}" \
    "${teacher_args[@]}" \
    > "${output_dir}/train.log" 2>&1

echo "completed ${experiment_id}"

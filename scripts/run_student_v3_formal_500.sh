#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
checkpoint=artifacts/STUDENT-V3-CORRECTION-SUCCESS39-10K-S7-015/step_10000.pt
development_root=results/STUDENT-V3-SUCCESS39-CHECKPOINT-GATE-S7-R5-DEV400-016/step_10000
validation_root=results/STUDENT-V3-SUCCESS39-10K-S7-R5-DEV90-INIT21-29-017
output_dir=results/STUDENT-V3-SUCCESS39-10K-S7-R5-500-018
task_ids=(0 1 2 3 4 5 6 7 8 9)
formal_inits=({0..49})

cd "${project_dir}"
if [[ ! -f "${checkpoint}" ]]; then
    echo "checkpoint is missing: ${checkpoint}" >&2
    exit 1
fi

# Refuse to start the formal matrix unless the predeclared untouched gate passes.
.venv/bin/python - "${validation_root}" "${checkpoint}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
checkpoint = sys.argv[2]
rows = [json.loads(path.read_text()) for path in root.rglob("result.json")]
identities = {(int(row["task_id"]), int(row["init_state_index"])) for row in rows}
expected = {(task_id, init_id) for task_id in range(10) for init_id in range(21, 30)}
exceptions = [
    row for row in rows
    if row.get("exception") or not row.get("completed_without_exception", False)
]
successes = sum(bool(row.get("success")) for row in rows)

if len(rows) != 90 or identities != expected:
    raise SystemExit(
        f"untouched gate is incomplete: records={len(rows)} identities={len(identities)}"
    )
if exceptions:
    raise SystemExit(f"untouched gate has {len(exceptions)} exception records")
if any(row.get("checkpoint") != checkpoint for row in rows):
    raise SystemExit("untouched gate contains a mismatched checkpoint")
if successes < 39:
    raise SystemExit(f"untouched gate failed: successes={successes}/90, required>=39")
print(f"untouched gate passed: successes={successes}/90 exceptions=0")
PY

mkdir -p "${output_dir}"

stage_episode() {
    local source_root=$1
    local task_id=$2
    local init_id=$3
    local task_dir
    local init_dir
    printf -v task_dir 'task%02d' "${task_id}"
    printf -v init_dir 'init%02d' "${init_id}"
    local source=${source_root}/${task_dir}/${init_dir}
    local target=${output_dir}/${task_dir}/${init_dir}

    if [[ ! -f "${source}/result.json" ]]; then
        echo "reusable episode is missing: ${source}/result.json" >&2
        exit 1
    fi
    if [[ -e "${target}" ]]; then
        if [[ ! -f "${target}/result.json" ]] || ! cmp -s "${source}/result.json" "${target}/result.json"; then
            echo "existing staged episode does not match source: ${target}" >&2
            exit 1
        fi
        return
    fi
    mkdir -p "${output_dir}/${task_dir}"
    cp -a "${source}" "${target}"
    cmp -s "${source}/result.json" "${target}/result.json"
}

# Reuse only exact, previously completed step-10K episodes from the two frozen
# panels: 100 development episodes (init 10--19) and 90 untouched validation
# episodes (init 21--29). Init 20 remains new in the formal matrix.
for task_id in "${task_ids[@]}"; do
    for init_id in {10..19}; do
        stage_episode "${development_root}" "${task_id}" "${init_id}"
    done
    for init_id in {21..29}; do
        stage_episode "${validation_root}" "${task_id}" "${init_id}"
    done
done

.venv/bin/python - "${output_dir}" "${checkpoint}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
checkpoint = sys.argv[2]
rows = [json.loads(path.read_text()) for path in root.rglob("result.json")]
identities = {(int(row["task_id"]), int(row["init_state_index"])) for row in rows}
expected = {
    (task_id, init_id)
    for task_id in range(10)
    for init_id in (*range(10, 20), *range(21, 30))
}
if len(rows) != 190 or identities != expected:
    raise SystemExit(
        f"formal reuse audit failed: records={len(rows)} identities={len(identities)}"
    )
if any(
    row.get("exception")
    or not row.get("completed_without_exception", False)
    or row.get("checkpoint") != checkpoint
    for row in rows
):
    raise SystemExit("formal reuse audit found an invalid record")
print("formal reuse audit passed: 190 exact step-10K episodes staged")
PY

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
    --init-state-indices "${formal_inits[@]}" \
    --training-seed 7 \
    --eval-seed-offset 1000 \
    --max-steps 220 \
    --wait-steps 10 \
    --replan-steps 5 \
    --resolution 256 \
    --device cuda \
    --skip-existing \
    > "${output_dir}/eval.log" 2>&1

echo "completed STUDENT-V3-SUCCESS39-10K-S7-R5-500-018"

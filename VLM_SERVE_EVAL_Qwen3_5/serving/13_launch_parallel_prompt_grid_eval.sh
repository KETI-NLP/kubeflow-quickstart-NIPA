#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

NAMESPACE="${NAMESPACE:-gpu-workspace}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-/data/checkpoints/qwen3_5_9b_multimodal_sft_lr_1e-5}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/eval_results/qwen3_5_9b_multimodal_sft_lr_1e-5_prompt_grid_parallel}"
TEMPLATE_FILE="${TEMPLATE_FILE:-12_eval_single_checkpoint_prompt_job.yaml}"
CHECKPOINTS="${CHECKPOINTS:-checkpoint-300 checkpoint-500 checkpoint-600 checkpoint-700 checkpoint-780}"

if [[ ! -f "${TEMPLATE_FILE}" ]]; then
    echo "Template not found: ${TEMPLATE_FILE}" >&2
    exit 1
fi

echo "Launching prompt-grid eval jobs into ${OUTPUT_DIR}"

for checkpoint in ${CHECKPOINTS}; do
    job_name="qwen3-5-prompt-grid-${checkpoint}"
    echo "Applying ${job_name}"
    sed \
        -e "s|__CHECKPOINT_NAME__|${checkpoint}|g" \
        -e "s|/data/eval_results/qwen3_5_9b_multimodal_sft_lr_1e-5_prompt_grid_parallel|${OUTPUT_DIR}|g" \
        "${TEMPLATE_FILE}" | kubectl apply -f -
done

echo
echo "Active jobs:"
kubectl get jobs -n "${NAMESPACE}" | rg 'qwen3-5-prompt-grid-'

#!/bin/bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-gpu-workspace}"
POD_NAME="${POD_NAME:-pvc-upload-dummy}"
REMOTE_PARENT="${REMOTE_PARENT:-/data/checkpoints/qwen3_5_9b_multimodal_sft_lr_1e-5}"
REMOTE_CHECKPOINT="${REMOTE_CHECKPOINT:-${REMOTE_PARENT}/checkpoint-300}"
LOCAL_PARENT="${LOCAL_PARENT:-/workspace/local_models/qwen3_5_9b_multimodal_sft_lr_1e-5}"
LOCAL_CHECKPOINT="${LOCAL_CHECKPOINT:-${LOCAL_PARENT}/checkpoint-300}"

if [[ -z "${KUBECONFIG:-}" ]]; then
    DEFAULT_KUBECONFIG="/workspace/kubeflow-training-snippets/YOUR_WORKSPACE-h200-128-kubeconfig.yaml"
    if [[ -f "${DEFAULT_KUBECONFIG}" ]]; then
        export KUBECONFIG="${DEFAULT_KUBECONFIG}"
    fi
fi

echo "Checking PVC checkpoint in ${NAMESPACE}/${POD_NAME}:${REMOTE_CHECKPOINT}"
kubectl -n "${NAMESPACE}" exec "${POD_NAME}" -- bash -lc "test -d '${REMOTE_CHECKPOINT}' && find '${REMOTE_CHECKPOINT}' -maxdepth 1 -type f -printf '%f\t%s bytes\n' | sort"

echo
echo "Copying serving files to ${LOCAL_CHECKPOINT}"
mkdir -p "${LOCAL_PARENT}" "${LOCAL_CHECKPOINT}"
kubectl -n "${NAMESPACE}" exec "${POD_NAME}" -- bash -lc \
    "cd '${REMOTE_PARENT}' && tar cf - chat_template.jinja config.json generation_config.json processor_config.json tokenizer.json tokenizer_config.json" \
    | tar xf - -C "${LOCAL_PARENT}"
kubectl -n "${NAMESPACE}" exec "${POD_NAME}" -- bash -lc \
    "cd '${REMOTE_CHECKPOINT}' && tar cf - config.json generation_config.json model.safetensors" \
    | tar xf - -C "${LOCAL_CHECKPOINT}"

echo
echo "Downloaded files:"
find "${LOCAL_PARENT}" "${LOCAL_CHECKPOINT}" -maxdepth 1 -type f -printf '%p\t%s bytes\n' | sort

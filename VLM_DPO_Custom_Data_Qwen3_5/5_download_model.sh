#!/bin/bash

NAMESPACE="gpu-workspace"
POD_NAME="pvc-upload-vlm-dpo-qwen3-5"
LOCAL_DESTINATION="./trained_model_output"
REMOTE_PATH="/data/checkpoints/qwen3_5_9b_multimodal_sft_dpo"

echo "=========================================="
echo "1. Ensuring the PVC mount pod is running..."
echo "=========================================="
kubectl apply -f 2_upload_to_pvc_direct.yaml

echo -e "\nWaiting for the pod ($POD_NAME) to be ready..."
kubectl wait --for=condition=Ready pod/${POD_NAME} -n ${NAMESPACE} --timeout=300s

echo -e "\n=========================================="
echo "2. Downloading trained model from PVC..."
echo "=========================================="
mkdir -p ${LOCAL_DESTINATION}
kubectl cp ${NAMESPACE}/${POD_NAME}:${REMOTE_PATH} ${LOCAL_DESTINATION}/qwen3_5_9b_multimodal_sft_dpo

echo -e "\n=========================================="
echo "Download completed successfully!"
echo "The trained model checkpoints are saved locally at: ${LOCAL_DESTINATION}/qwen3_5_9b_multimodal_sft_dpo"
echo "=========================================="

echo -e "\n=========================================="
echo "3. Cleaning up dummy POD to free resources..."
echo "=========================================="
kubectl delete -f 2_upload_to_pvc_direct.yaml
echo "Cleanup completed."

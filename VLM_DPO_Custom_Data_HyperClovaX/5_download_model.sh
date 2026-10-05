#!/bin/bash

NAMESPACE="gpu-workspace"
POD_NAME="pvc-upload-vlm-dpo-hcx"
LOCAL_DESTINATION="./trained_model_output"
REMOTE_PATH="/data/result/vlm-dpo-hcx-think-32b"

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
kubectl cp ${NAMESPACE}/${POD_NAME}:${REMOTE_PATH} ${LOCAL_DESTINATION}/vlm-dpo-hcx-think-32b

echo -e "\n=========================================="
echo "Download completed successfully!"
echo "The trained model checkpoints are saved locally at: ${LOCAL_DESTINATION}/vlm-dpo-hcx-think-32b"
echo "=========================================="

echo -e "\n=========================================="
echo "3. Cleaning up dummy POD to free resources..."
echo "=========================================="
kubectl delete -f 2_upload_to_pvc_direct.yaml
echo "Cleanup completed."

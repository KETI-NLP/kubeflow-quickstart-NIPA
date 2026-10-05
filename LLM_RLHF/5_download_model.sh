#!/bin/bash

# Configuration
NAMESPACE="gpu-workspace"
POD_NAME="pvc-upload-dummy"
LOCAL_DESTINATION="./trained_model_output"
REMOTE_PATH="/data/result/llm-rlhf"

echo "=========================================="
echo "1. Ensuring the PVC mount pod is running..."
echo "=========================================="
kubectl apply -f 2_upload_to_pvc_direct.yaml

echo -e "\nWaiting for the pod ($POD_NAME) to be ready..."
kubectl wait --for=condition=Ready pod/${POD_NAME} -n ${NAMESPACE} --timeout=300s

echo -e "\n=========================================="
echo "2. Downloading trained model from PVC..."
echo "=========================================="
# Create local directory to hold the downloaded model
mkdir -p ${LOCAL_DESTINATION}

# Copy from pod's PVC mount path to local directory
# format: kubectl cp <namespace>/<pod-name>:<remote-path> <local-path>
kubectl cp ${NAMESPACE}/${POD_NAME}:${REMOTE_PATH} ${LOCAL_DESTINATION}/llm-rlhf

echo -e "\n=========================================="
echo "Download completed successfully! 🎉"
echo "The trained model checkpoints are saved locally at: ${LOCAL_DESTINATION}/llm-rlhf"
echo "=========================================="

echo -e "\n=========================================="
echo "3. Cleaning up dummy POD to free resources..."
echo "=========================================="
kubectl delete -f 2_upload_to_pvc_direct.yaml
echo "Cleanup completed."

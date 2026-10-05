#!/bin/bash

# Configuration
NAMESPACE="gpu-workspace"
POD_NAME="pvc-upload-dummy"
LOCAL_DATASET="rlhf_dataset"
REMOTE_PATH="/data/rlhf_dataset"

echo "=========================================="
echo "1. Deploying PVC mount dummy pod..."
echo "=========================================="
kubectl apply -f 2_upload_to_pvc_direct.yaml

echo -e "\nWaiting for the pod ($POD_NAME) to be ready..."
kubectl wait --for=condition=Ready pod/${POD_NAME} -n ${NAMESPACE} --timeout=300s

echo -e "\n=========================================="
echo "2. Uploading $LOCAL_DATASET to PVC..."
echo "=========================================="
# Copy local directory to the pod's PVC mount path
# format: kubectl cp <local-path> <namespace>/<pod-name>:<remote-path>
kubectl cp ${LOCAL_DATASET} ${NAMESPACE}/${POD_NAME}:${REMOTE_PATH}

echo -e "\n=========================================="
echo "Upload completed successfully! 🎉"
echo "Dataset is now securely stored at $REMOTE_PATH inside the persistent volume."
echo "You can now run 'kubectl apply -f 4_llm_pytorchjob.yaml' to start training!"
echo "=========================================="

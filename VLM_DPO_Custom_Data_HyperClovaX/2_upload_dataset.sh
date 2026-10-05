#!/bin/bash

NAMESPACE="gpu-workspace"
POD_NAME="pvc-upload-vlm-dpo-hcx"
LOCAL_DATASET="vlm_dpo_hcx_dataset"
REMOTE_PATH="/data/vlm_dpo_hcx_dataset"

echo "=========================================="
echo "1. Deploying PVC mount dummy pod..."
echo "=========================================="
kubectl apply -f 2_upload_to_pvc_direct.yaml

echo -e "\nWaiting for the pod ($POD_NAME) to be ready..."
kubectl wait --for=condition=Ready pod/${POD_NAME} -n ${NAMESPACE} --timeout=300s

echo -e "\n=========================================="
echo "2. Uploading $LOCAL_DATASET to PVC..."
echo "=========================================="
kubectl cp ${LOCAL_DATASET} ${NAMESPACE}/${POD_NAME}:${REMOTE_PATH}

echo -e "\n=========================================="
echo "Upload completed successfully!"
echo "Dataset is now stored at $REMOTE_PATH inside the persistent volume."
echo "=========================================="

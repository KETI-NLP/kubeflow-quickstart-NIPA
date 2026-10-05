#!/bin/bash
# ==============================================================================
# 1. Upload Conversion Scripts to PVC
# ==============================================================================
export KUBECONFIG="/workspace/kubeflow-training-snippets/YOUR_WORKSPACE-h200-128-kubeconfig.yaml"

NAMESPACE="gpu-workspace"
POD_NAME="pvc-upload-dummy"
PVC_MOUNT="/data"
CLOUD_DIR="$PVC_MOUNT/CLOUD_CONVERT_DATASET"
RAW_DATA_DIR="$CLOUD_DIR/raw_data"
SCRIPTS_DIR="$CLOUD_DIR/scripts"

echo "=========================================="
echo "Creating Directories inside PVC..."
echo "=========================================="
kubectl exec -n $NAMESPACE $POD_NAME -- mkdir -p $RAW_DATA_DIR
kubectl exec -n $NAMESPACE $POD_NAME -- mkdir -p $SCRIPTS_DIR
kubectl exec -n $NAMESPACE $POD_NAME -- mkdir -p $CLOUD_DIR/converted_data
kubectl exec -n $NAMESPACE $POD_NAME -- chmod -R 777 $CLOUD_DIR

echo "=========================================="
echo "Uploading Scripts into PVC..."
echo "=========================================="
# MLXP Download / Upload & Env
kubectl cp /workspace/kubeflow-training-snippets/DATAMANAGER_UPLOAD/download_from_mlxp.py $NAMESPACE/$POD_NAME:$SCRIPTS_DIR/
kubectl cp /workspace/kubeflow-training-snippets/DATAMANAGER_UPLOAD/upload_large_to_mlxp.py $NAMESPACE/$POD_NAME:$SCRIPTS_DIR/

if [ -f "/workspace/kubeflow-training-snippets/DATAMANAGER_UPLOAD/.env" ]; then
    kubectl cp /workspace/kubeflow-training-snippets/DATAMANAGER_UPLOAD/.env $NAMESPACE/$POD_NAME:$SCRIPTS_DIR/.env
else
    echo "⚠️ Warning: .env file not found in DATAMANAGER_UPLOAD! API Upload/Download might fail."
fi

# Convert Script
kubectl cp /workspace/2026_llm_data_generation/convert_to_llm_training_ready/convert_datasets.py $NAMESPACE/$POD_NAME:$SCRIPTS_DIR/

# Orchestrator
kubectl cp /workspace/kubeflow-training-snippets/CLOUD_CONVERT_DATASET/run_cloud_convert.sh $NAMESPACE/$POD_NAME:$SCRIPTS_DIR/

echo "=========================================="
echo "Scripts Upload completed! 🎉"
echo "You can now run:"
echo "  kubectl apply -f 2_convert_job.yaml"
echo "=========================================="

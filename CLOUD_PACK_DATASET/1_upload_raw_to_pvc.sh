#!/bin/bash
# ==============================================================================
# 1. Upload Raw Datasets & Scripts to PVC
# ==============================================================================
pip install "ncloud-mlx[data-manager]" huggingface_hub python-dotenv hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
export KUBECONFIG="/workspace/kubeflow-training-snippets/YOUR_WORKSPACE-h200-128-kubeconfig.yaml"

NAMESPACE="gpu-workspace"
POD_NAME="pvc-upload-dummy"
PVC_MOUNT="/data"
CLOUD_DIR="$PVC_MOUNT/CLOUD_PACK_DATASET"
RAW_DATA_DIR="$CLOUD_DIR/raw_data"
SCRIPTS_DIR="$CLOUD_DIR/scripts"

echo "=========================================="
echo "Creating Directories inside PVC..."
echo "=========================================="
kubectl exec -n $NAMESPACE $POD_NAME -- /bin/bash -c "mkdir -p $RAW_DATA_DIR $SCRIPTS_DIR $CLOUD_DIR/llm_training_ready_packed && chmod -R 777 $CLOUD_DIR"

echo "=========================================="
echo "Uploading Packing Python Scripts..."
echo "=========================================="
# Pack Scripts
kubectl cp /workspace/kubeflow-training-snippets/CLOUD_PACK_DATASET/cloud_pack_sft_qwen3_5_dataset.py $NAMESPACE/$POD_NAME:$SCRIPTS_DIR/
kubectl cp /workspace/kubeflow-training-snippets/CLOUD_PACK_DATASET/cloud_pack_sft_hcx_seed_omni_dataset.py $NAMESPACE/$POD_NAME:$SCRIPTS_DIR/
kubectl cp /workspace/kubeflow-training-snippets/CLOUD_PACK_DATASET/cloud_pack_sft_hcx_seed_think_dataset.py $NAMESPACE/$POD_NAME:$SCRIPTS_DIR/

# MLXP Upload Script & Env Variables
kubectl cp /workspace/kubeflow-training-snippets/DATAMANAGER_UPLOAD/upload_to_mlxp.py $NAMESPACE/$POD_NAME:$SCRIPTS_DIR/
kubectl cp /workspace/kubeflow-training-snippets/DATAMANAGER_UPLOAD/download_from_mlxp.py $NAMESPACE/$POD_NAME:$SCRIPTS_DIR/
if [ -f "/workspace/kubeflow-training-snippets/DATAMANAGER_UPLOAD/.env" ]; then
    kubectl cp /workspace/kubeflow-training-snippets/DATAMANAGER_UPLOAD/.env $NAMESPACE/$POD_NAME:$SCRIPTS_DIR/.env
else
    echo "⚠️ Warning: .env file not found in DATAMANAGER_UPLOAD ! API Upload might fail."
fi

# The Cloud Execution Orchestrator Script itself
kubectl cp /workspace/kubeflow-training-snippets/CLOUD_PACK_DATASET/run_cloud_pack.sh $NAMESPACE/$POD_NAME:$SCRIPTS_DIR/

echo "=========================================="
echo "Uploading Raw Datasets to MLXP DataManager..."
echo "=========================================="
LOCAL_INPUT_DIR="/workspace/2026_llm_data_generation/llm_training_ready"
WORKSPACE_NAME="YOUR_WORKSPACE"
UPLOAD_SCRIPT="/workspace/kubeflow-training-snippets/DATAMANAGER_UPLOAD/upload_large_to_mlxp.py"

# ---------------------------------------------------------
# Select which dataset(s) you want to process here:
# (Uncomment the ones you want to run)
# ---------------------------------------------------------
DATASETS=(
    # "data_13_korean_character_hf"
    # "data_korean_heritage_vqa_hf"
    # "data_public_executive_ocr_hf"
    "FineVision_converted_hf"
)

# UPLOAD_PIDS=()
# for ds in "${DATASETS[@]}"; do
#     echo "➡️  Uploading to MLXP -> $ds (화면에 진행률, 속도, 남은 시간이 표시됩니다)"
#     python3 $UPLOAD_SCRIPT \
#         --workspace "$WORKSPACE_NAME" \
#         --dataset-name "$ds" \
#         --local-dir "$LOCAL_INPUT_DIR/$ds"
        
#     if [ $? -ne 0 ]; then
#         echo "🚨 MLXP Upload failed for $ds"
#         exit 1
#     fi
#     echo "✅ $ds 업로드 완료!"
# done

echo "=========================================="
echo "Upload to PVC completed! 🎉"
echo "You can now run:"
echo "  kubectl apply -f 2_pack_job.yaml"
echo "=========================================="

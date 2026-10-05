#!/bin/bash
# =========================================================
# CLOUD CONVERT AND UPLOAD SCRIPT (Ran inside Kubeflow Job)
# =========================================================
PVC_MOUNT="/data"
CLOUD_DIR="$PVC_MOUNT/CLOUD_CONVERT_DATASET"
RAW_DATA_DIR="$CLOUD_DIR/raw_data"
SCRIPTS_DIR="$CLOUD_DIR/scripts"
OUTPUT_DIR="$CLOUD_DIR/converted_data"
WORKSPACE_NAME="YOUR_WORKSPACE"

DATASET_NAME="FineVision"
HF_CONVERTED_NAME="FineVision_hf_converted"

# Set working directory to load .env when executing download/upload
cd $SCRIPTS_DIR

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo " 1. Installing Requirements"
echo "=========================================="
pip install -U "ncloud-mlx[data-manager]" huggingface_hub python-dotenv pyarrow datasets Pillow transformers "qwen-vl-utils" ijson hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

export HF_HOME="$PVC_MOUNT/hf_cache"

echo "=========================================="
echo " 2. Downloading FineVision Raw from MLXP"
echo "=========================================="
RAW_TARGET_PATH="$RAW_DATA_DIR/$DATASET_NAME"

echo "➡️  Downloading from MLXP DataManager -> $WORKSPACE_NAME/$DATASET_NAME"
python3 $SCRIPTS_DIR/download_from_mlxp.py \
    --workspace "$WORKSPACE_NAME" \
    --dataset-name "$DATASET_NAME" \
    --local-dir "$RAW_TARGET_PATH" \
    --raw-download
    
if [ $? -ne 0 ]; then
    echo "🚨 Download failed!"
    exit 1
fi

echo "=========================================="
echo " 3. Running convert_datasets.py"
echo "=========================================="
CONVERTED_OUTPUT_PATH="$OUTPUT_DIR/$HF_CONVERTED_NAME"

python3 $SCRIPTS_DIR/convert_datasets.py \
    --input_path "$RAW_TARGET_PATH" \
    --output_path "$CONVERTED_OUTPUT_PATH"

if [ $? -ne 0 ]; then
    echo "🚨 Conversion failed!"
    exit 1
fi
echo "✅ Conversion complete."

echo "=========================================="
echo " 4. Uploading Result to MLXP"
echo "=========================================="
echo "➡️  Uploading to MLXP DataManager -> $WORKSPACE_NAME/$HF_CONVERTED_NAME"

python3 $SCRIPTS_DIR/upload_large_to_mlxp.py \
    --workspace "$WORKSPACE_NAME" \
    --dataset-name "$HF_CONVERTED_NAME" \
    --local-dir "$CONVERTED_OUTPUT_PATH"

if [ $? -ne 0 ]; then
    echo "🚨 Upload failed!"
    exit 1
fi

echo "=========================================="
echo "🎉 ALL PIPELINES FINISHED SUCCESSFULLY!"
echo "=========================================="

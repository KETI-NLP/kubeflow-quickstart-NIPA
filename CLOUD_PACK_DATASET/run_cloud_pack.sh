#!/bin/bash
# =========================================================
# CLOUD PACK AND UPLOAD SCRIPT (Ran inside Kubeflow Job)
# =========================================================
PVC_MOUNT="/data"
CLOUD_DIR="$PVC_MOUNT/CLOUD_PACK_DATASET"
RAW_DATA_DIR="$CLOUD_DIR/raw_data"
SCRIPTS_DIR="$CLOUD_DIR/scripts"
OUTPUT_DIR="$CLOUD_DIR/llm_training_ready_packed"
WORKSPACE_NAME="YOUR_WORKSPACE"

# Set working directory to load .env when executing upload
cd $SCRIPTS_DIR

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo " 1. Installing uv and Base Requirements"
echo "=========================================="
# Prevents uv venv from throwing error if environment already exists
export UV_VENV_CLEAR=1
# We install uv globally to manage our multi-environments cleanly
pip install -U uv

# Create MLXP Environment explicitly for safe data transfer without conflicts
echo "Setting up MLXP isolated environment..."
uv venv /opt/env_mlxp
uv pip install -p /opt/env_mlxp python-dotenv "ncloud-mlx[data-manager]"

# Create the HCX Environment (verified on 4.52.4)
echo "Setting up HCX isolated environment..."
uv venv /opt/env_hcx
uv pip install -p /opt/env_hcx -U huggingface_hub python-dotenv pyarrow "datasets>=3.0.0" Pillow transformers==4.52.4 "qwen-vl-utils" torch torchvision accelerate

# Create the QWen Environment (requires >=4.49.0 + custom wrapper)
echo "Setting up QWEN isolated environment..."
uv venv /opt/env_qwen
uv pip install -p /opt/env_qwen -U huggingface_hub python-dotenv pyarrow "datasets>=3.0.0" Pillow transformers>=4.49.0 "qwen-vl-utils" torch torchvision accelerate

# ---------------------------------------------------------
# Select which dataset(s) you want to process here:
# (Uncomment the ones you want to run)
# ---------------------------------------------------------
DATASETS=(
    "data_13_korean_character_hf"
    # "data_korean_heritage_vqa_hf"
    # "data_public_executive_ocr_hf"
    "FineVision_hf_converted"
)

export HF_HOME="$PVC_MOUNT/hf_cache"

RUN_IN_PARALLEL=true
# 1024코어, 1TB 메모리의 자원을 충분히 활용하기 위해 9개의 모델 패킹을 전부 동시에 실행합니다 (실제 파이썬 내에서 각 64 멀티프로세싱).
MAX_CONCURRENT_JOBS=16
PIDS=()
CMDS=()
FAILED_CMDS=()

launch_task() {
    local cmd_str="$1"
    
    if [ "$RUN_IN_PARALLEL" = true ]; then
        eval "$cmd_str" &
        PIDS+=($!)
        CMDS+=("$cmd_str")
        if [ ${#PIDS[@]} -ge "$MAX_CONCURRENT_JOBS" ]; then
            for i in "${!PIDS[@]}"; do
                wait "${PIDS[$i]}"
                if [ $? -ne 0 ]; then
                    FAILED_CMDS+=("${CMDS[$i]}")
                fi
            done
            PIDS=()
            CMDS=()
        fi
    else
        eval "$cmd_str"
        if [ $? -ne 0 ]; then
            FAILED_CMDS+=("$cmd_str")
        fi
    fi
}

echo "=========================================="
echo " 1.5 Downloading Raw Datasets from MLXP DataManager"
echo "=========================================="

for ds_name in "${DATASETS[@]}"; do
    if [ "$ds_name" == "data_13_korean_character_hf" ]; then
        echo "➡️  [Skipping Download] $ds_name is already downloaded locally in the PVC."
        continue
    fi
    
    input_path="$RAW_DATA_DIR/$ds_name"
    
    echo "➡️  [Downloading] Dataset: $ds_name from MLXP"
    /opt/env_mlxp/bin/python3 $SCRIPTS_DIR/download_from_mlxp.py \
        --workspace "$WORKSPACE_NAME" \
        --dataset-name "$ds_name" \
        --local-dir "$input_path" \
        --raw-download
    
    if [ $? -ne 0 ]; then
        echo "🚨 Failed to download $ds_name from MLXP. Exiting."
        exit 1
    fi
done





echo "=========================================="
echo " 2. Starting Packing Tasks"
echo "=========================================="

for ds_name in "${DATASETS[@]}"; do
    input_path="$RAW_DATA_DIR/$ds_name"
    base_no_hf="${ds_name%_hf}"
    
    if [ ! -d "$input_path" ]; then
        echo "🚨 Warning: Dataset $input_path does not exist on PVC. Skipping."
        continue
    fi

    # 1. Qwen3.5 버전 (Runs in /opt/env_qwen using the custom processor wrapper script)
    qwen_output_name="${base_no_hf}_qwen3_5_packed_sft_hf"
    cmd_qwen="/opt/env_qwen/bin/python3 $SCRIPTS_DIR/cloud_pack_sft_qwen3_5_dataset.py --data_path \"$input_path\" --output_path \"$OUTPUT_DIR/$qwen_output_name\""
    launch_task "$cmd_qwen"
    
    # 2. HCX Omni 버전 (Runs natively in /opt/env_hcx)
    hcx_output_name="${base_no_hf}_hcx_seed_omni_packed_sft_hf"
    cmd_hcx="/opt/env_hcx/bin/python3 $SCRIPTS_DIR/cloud_pack_sft_hcx_seed_omni_dataset.py --data_path \"$input_path\" --output_path \"$OUTPUT_DIR/$hcx_output_name\""
    launch_task "$cmd_hcx"

    # 3. HCX Think 버전 (Runs natively in /opt/env_hcx)
    hcx_think_output_name="${base_no_hf}_hcx_seed_think_packed_sft_hf"
    cmd_hcx_think="/opt/env_hcx/bin/python3 $SCRIPTS_DIR/cloud_pack_sft_hcx_seed_think_dataset.py --data_path \"$input_path\" --output_path \"$OUTPUT_DIR/$hcx_think_output_name\""
    launch_task "$cmd_hcx_think"
done

# 잔여 프로세스 대기
if [ ${#PIDS[@]} -gt 0 ]; then
    for i in "${!PIDS[@]}"; do
        wait "${PIDS[$i]}"
        if [ $? -ne 0 ]; then
            FAILED_CMDS+=("${CMDS[$i]}")
        fi
    done
fi

if [ ${#FAILED_CMDS[@]} -gt 0 ]; then
    echo "🚨 THE FOLLOWING PACKING COMMANDS FAILED:"
    for failed_cmd in "${FAILED_CMDS[@]}"; do
        echo " - $failed_cmd"
    done
    exit 1
fi

echo "✅ All datasets have been packed successfully!"

echo "=========================================="
echo " 3. Starting Direct MLXP Upload Tasks"
echo "=========================================="

declare -A MLXP_NAMES
# FineVision mappings
MLXP_NAMES["FineVision_hf_converted_hcx_seed_omni_packed_sft_hf"]="finevision_hcx_omni_sft_hf"
MLXP_NAMES["FineVision_hf_converted_hcx_seed_think_packed_sft_hf"]="finevision_hcx_think_sft_hf"
MLXP_NAMES["FineVision_hf_converted_qwen3_5_packed_sft_hf"]="finevision_qwen3_5_sft_hf"

# Existing mappings
MLXP_NAMES["data_13_korean_character_hcx_seed_omni_packed_sft_hf"]="data_13_kr_char_hcx_omni_sft_hf"
MLXP_NAMES["data_13_korean_character_hcx_seed_think_packed_sft_hf"]="data_13_kr_char_hcx_think_sft_hf"
MLXP_NAMES["data_13_korean_character_qwen3_5_packed_sft_hf"]="data_13_kr_char_qwen3_5_sft_hf"
MLXP_NAMES["data_korean_heritage_vqa_hcx_seed_omni_packed_sft_hf"]="data_kr_heri_vqa_hcx_omni_sft_hf"
MLXP_NAMES["data_korean_heritage_vqa_hcx_seed_think_packed_sft_hf"]="data_kr_heri_vqa_hcx_think_sft_hf"
MLXP_NAMES["data_korean_heritage_vqa_qwen3_5_packed_sft_hf"]="data_kr_heri_vqa_qwen3_5_sft_hf"
MLXP_NAMES["data_public_executive_ocr_hcx_seed_omni_packed_sft_hf"]="data_pub_exec_ocr_hcx_omni_sft_hf"
MLXP_NAMES["data_public_executive_ocr_hcx_seed_think_packed_sft_hf"]="data_pub_exec_ocr_hcx_think_sft_hf"
MLXP_NAMES["data_public_executive_ocr_qwen3_5_packed_sft_hf"]="data_pub_exec_ocr_qwen3_5_sft_hf"

UPLOAD_PIDS=()
for ds_name in "${DATASETS[@]}"; do
    base_no_hf="${ds_name%_hf}"
    
    for suffix in "qwen3_5_packed_sft_hf" "hcx_seed_omni_packed_sft_hf" "hcx_seed_think_packed_sft_hf"; do
        local_dirname="${base_no_hf}_${suffix}"
        mlxp_dataset_name="${MLXP_NAMES[$local_dirname]}"
        
        full_local_path="$OUTPUT_DIR/$local_dirname"
        
        if [ -d "$full_local_path" ]; then
            echo "➡️  [Uploading] Dataset: $mlxp_dataset_name"
            # Background upload
            /opt/env_mlxp/bin/python3 upload_to_mlxp.py \
                --workspace "$WORKSPACE_NAME" \
                --dataset-name "$mlxp_dataset_name" \
                --local-dir "$full_local_path" > "/tmp/upload_${mlxp_dataset_name}.log" 2>&1 &
                
            UPLOAD_PIDS+=($!)
            sleep 2
        fi
    done
done

# Wait for multiple uploads
echo "Waiting for all MLXP uploads to complete..."
for pid in "${UPLOAD_PIDS[@]}"; do
    wait $pid
    if [ $? -ne 0 ]; then
        echo "🚨 Upload failed for PID $pid"
    fi
done

echo "✅ All pipelines finished successfully!"

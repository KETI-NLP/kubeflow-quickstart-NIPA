#!/bin/bash

set -euo pipefail

# ==============================================================================
# MLXP DataManager Large Upload Script for DPO datasets
# - Uses upload_large_to_mlxp.py / huggingface_hub.upload_large_folder
# - Defaults to the currently relevant Korean heritage VQA DPO dataset
# - Can also target the other DPO datasets from run_upload_dpo.sh
# ==============================================================================

echo "Installing mlxp data manager dependencies..."
pip install "ncloud-mlx[data-manager]" huggingface_hub python-dotenv

WORKSPACE_NAME="YOUR_WORKSPACE"

DATASET_NAMES=(
    "data_kr_heri_text_dpo_hf"
    "data_kr_heri_vqa_dpo_hf"
    "data_kr_obj_img_4_hallu_txt_dpo_hf"
    "data_kr_obj_img_5_hallu_wr_txt_dpo_hf"
)

LOCAL_DIRS=(
    "/workspace/2026_llm_data_generation/llm_training_ready/data_korean_heritage_text_hf"
    "/workspace/2026_llm_data_generation/llm_training_ready/data_korean_heritage_vqa_hf"
    "/workspace/2026_llm_data_generation/llm_training_ready/data_korean_object_image_4_hallucination_text_only_generation_hf"
    "/workspace/2026_llm_data_generation/llm_training_ready/data_korean_object_image_5_hallucination_normal_image_wrong_text_generation_hf"
)

if [ "${#DATASET_NAMES[@]}" -ne "${#LOCAL_DIRS[@]}" ]; then
    echo "❌ 오류: DATASET_NAMES 의 개수와 LOCAL_DIRS 의 개수가 일치하지 않습니다!"
    exit 1
fi

# 기본값은 현재 사용자가 언급한 VQA DPO 업로드
TARGET_INDEX="${TARGET_INDEX:-1}"

if ! [[ "$TARGET_INDEX" =~ ^[0-9]+$ ]]; then
    echo "❌ 오류: TARGET_INDEX 는 0 이상의 정수여야 합니다."
    exit 1
fi

if [ "$TARGET_INDEX" -lt 0 ] || [ "$TARGET_INDEX" -ge "${#DATASET_NAMES[@]}" ]; then
    echo "❌ 오류: TARGET_INDEX 범위가 잘못되었습니다. 사용 가능 범위: 0 ~ $((${#DATASET_NAMES[@]} - 1))"
    exit 1
fi

D_NAME="${DATASET_NAMES[$TARGET_INDEX]}"
L_DIR="${LOCAL_DIRS[$TARGET_INDEX]}"
LOG_FILE="upload_large_${D_NAME}.log"

echo
echo "=========================================="
echo "🚀 Large uploader로 데이터셋 업로드를 시작합니다."
echo "workspace     : $WORKSPACE_NAME"
echo "dataset name  : $D_NAME"
echo "local dir     : $L_DIR"
echo "log file      : $LOG_FILE"
echo "target index  : $TARGET_INDEX"
echo "=========================================="
echo

python3 upload_large_to_mlxp.py \
    --workspace "$WORKSPACE_NAME" \
    --dataset-name "$D_NAME" \
    --local-dir "$L_DIR" | tee "$LOG_FILE"

echo
echo "🎉 Large uploader 실행이 종료되었습니다."
echo "로그 파일: $LOG_FILE"

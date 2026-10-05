#!/bin/bash

# ==============================================================================
# MLXP DataManager Parallel Upload Script
# ==============================================================================

# 1. Dependencies Installation
echo "Installing mlxp data manager dependencies..."
pip install "ncloud-mlx[data-manager]" huggingface_hub python-dotenv

# 2. Configure Target details
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

# 배열 길이 확인 (두 리스트의 짝이 맞는지 검증)
if [ "${#DATASET_NAMES[@]}" -ne "${#LOCAL_DIRS[@]}" ]; then
    echo "❌ 오류: DATASET_NAMES 의 개수와 LOCAL_DIRS 의 개수가 일치하지 않습니다!"
    exit 1
fi

echo -e "\n=========================================="
echo "🚀 총 ${#DATASET_NAMES[@]}개의 데이터셋을 병렬로 업로드 시작합니다..."
echo "진행 상황은 각각의 로그 파일(upload_<dataset_name>.log)에 저장됩니다."
echo "=========================================="

# 3. Execute the Python Uploader in Parallel
for i in "${!DATASET_NAMES[@]}"; do
    D_NAME="${DATASET_NAMES[$i]}"
    L_DIR="${LOCAL_DIRS[$i]}"
    
    LOG_FILE="upload_${D_NAME}.log"
    
    echo "➡️  [작업 시작] 데이터셋: $D_NAME | 로컬 경로: $L_DIR | 진행상황 보기: (tail -f $LOG_FILE)"
    
    # 백그라운드(&)로 파이썬 스크립트를 독립 실행하고, 출력을 각자의 로그 파일에 씁니다.
    python3 upload_to_mlxp.py \
        --workspace "$WORKSPACE_NAME" \
        --dataset-name "$D_NAME" \
        --local-dir "$L_DIR" > "$LOG_FILE" 2>&1 &
        
    # Huggingface 토큰 인증 파일(login)에 여러 프로세스가 동시에 접근하여 깨지는(Race Condition) 에러를 방지하기 위해 2초씩 딜레이를 줍니다.
    sleep 2
done

echo -e "\n모든 업로드 프로세스가 백그라운드에 띄워졌습니다."
echo "완전 종료될 때까지 대기합니다... (이 터미널 창을 끄지 마세요!)"

# Wait for all background jobs to finish
wait

echo -e "\n🎉 모든 데이터셋의 병렬 업로드가 완료되었습니다!!"

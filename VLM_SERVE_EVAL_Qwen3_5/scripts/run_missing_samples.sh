#!/bin/bash
WORK_DIR="/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5"
cd "$WORK_DIR" || exit 1
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRUST_REMOTE_CODE=True
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export HF_HOME="/workspace/.cache/huggingface"
export HF_DATASETS_CACHE="/workspace/.cache/huggingface/datasets"
export LIGHTEVAL_CUSTOM_TEMPLATE="/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/no_think_chat_template.jinja"

#MODEL_REPO="Qwen/Qwen3.5-9B"
MODEL_REPO="/workspace/local_models/qwen3_5_9b_multimodal_sft_dpo_normalized/checkpoint-1253"
MODEL_ARGS="model_name=$MODEL_REPO,batch_size=4,max_length=8192"
VISION_ARGS=""
OUTPUT_DIR="./test_final_11_tasks/local_qwen3.5_9b"

TASKS=(
  "kmmlu_gen|0 custom_tasks/custom_kmmlu_task.py"
  "hae_rae_bench_gen|0 custom_tasks/custom_hae_rae_bench_task.py"
  "ifeval_ko_gen|0 custom_tasks/custom_ifeval_ko_task.py"
)

for task_info in "${TASKS[@]}"; do
  task=$(echo "$task_info" | awk '{print $1}')
  custom_task=$(echo "$task_info" | awk '{print $2}')
  task_dir="$OUTPUT_DIR/$(echo $task | cut -d':' -f1 | cut -d'|' -f1)"
  
  echo "Running missing task: $task"
  uv run --python .venv/bin/python run_lighteval_patched.py accelerate "$MODEL_ARGS" \
    "$task" \
    --custom-tasks "$custom_task" \
    --max-samples 100 \
    --save-details \
    --output-dir "$task_dir"
done

.venv/bin/python generate_report_v7.py
.venv/bin/python /workspace/telegram_bot/send_telegram_message.py text "✅ 누락되었던 3개 태스크(kmmlu, hae_rae_bench, ifeval_ko)의 100개 샘플 평가 및 리포트 업데이트가 성공적으로 완료되었습니다!"

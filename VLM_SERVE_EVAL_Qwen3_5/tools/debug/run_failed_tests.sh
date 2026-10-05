#!/bin/bash
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRUST_REMOTE_CODE=True
export CUDA_VISIBLE_DEVICES="4,5,6,7"
export HF_HOME="/workspace/.cache/huggingface"
export HF_DATASETS_CACHE="/workspace/.cache/huggingface/datasets"
MODEL_REPO="Qwen/Qwen3.5-9B"
MODEL_ARGS="model_name=$MODEL_REPO,batch_size=16,max_length=8192"

TASKS=(
  "mmmu_gen:Accounting|0 custom_tasks/custom_mmmu_task.py"
  "korean_heritage_reverse_qa|0 custom_tasks/custom_korean_heritage_reverse_qa_task.py"
  "korean_heritage_text_shortqa|0 custom_tasks/custom_korean_heritage_text_shortqa_task.py"
)

OUTPUT_DIR="./test_3_samples_run/local_qwen3.5_9b_vlm"
mkdir -p "$OUTPUT_DIR"

for task_info in "${TASKS[@]}"; do
  task=$(echo "$task_info" | awk '{print $1}')
  custom_task=$(echo "$task_info" | awk '{print $2}')
  
  task_dir="$OUTPUT_DIR/$(echo $task | cut -d':' -f1)"
  
  echo "=================================================="
  echo "Running $task with local model $MODEL_REPO (vLLM) on GPUs 4,5,6,7"
  echo "=================================================="
  
  VISION_ARGS="--vision-model"
  if [[ "$task" =~ "korean_heritage_text_shortqa" || "$task" =~ "korean_heritage_reverse_qa" || "$task" =~ "kmmlu_gen" || "$task" =~ "hae_rae_bench_gen" || "$task" =~ "ifeval_ko_gen" || "$task" =~ "hallusionbench_gen" ]]; then
      VISION_ARGS=""
  fi
  
  uv run --python .venv/bin/python -m lighteval accelerate "$MODEL_ARGS" \
    "$task" \
    $VISION_ARGS \
    --custom-tasks "$custom_task" \
    --max-samples 3 \
    --save-details \
    --output-dir "$task_dir"
    
  echo ""
done

echo "Done running remaining sample benchmarks."

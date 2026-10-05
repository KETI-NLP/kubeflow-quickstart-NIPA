#!/bin/bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# Source the virtual environment
source .venv/bin/activate

# Set environment variables
export VLLM_WORKER_MULTIPROCESS_METHOD=spawn
export ACCELERATE_LOG_LEVEL=info
export HF_HUB_ENABLE_HF_TRANSFER=1

# Model configuration
MODEL_REPO="Qwen/Qwen3.5-9B"
# WE ADDED generation_size=10
MODEL_ARGS="model_name=$MODEL_REPO,batch_size=1,max_length=4096,generation_size=10"

TASKS=(
  "chartqa_gen:default|0 custom_tasks/custom_chartqa_task.py"
)

for task_info in "${TASKS[@]}"; do
  TASK=$(echo "$task_info" | cut -d' ' -f1)
  CUSTOM_TASK_FILE=$(echo "$task_info" | cut -d' ' -f2)

  echo "Starting evaluation for $TASK..."

  accelerate launch --multi_gpu --num_processes=4 \
    -m lighteval endpoint accelerate \
    --model_args "$MODEL_ARGS" \
    --tasks "$TASK" \
    --custom_tasks "$CUSTOM_TASK_FILE" \
    --max_samples 1 \
    --output_dir ./test_chartqa \
    --save_details

  echo "Finished $TASK"
done

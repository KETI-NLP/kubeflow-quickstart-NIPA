#!/bin/bash
WORK_DIR="/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5"
cd "$WORK_DIR" || exit 1
set -euo pipefail

export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRUST_REMOTE_CODE=True

MODELS=("yaml_files/litellm_gpt5_4.yaml" "yaml_files/litellm_gemini_3_1_pro.yaml")
TASKS=(
  "mathvista_gen:default|0 custom_tasks/custom_mathvista_task.py"
  "hallusionbench_gen:default|0 custom_tasks/custom_hallusionbench_task.py"
)

MAX_SAMPLES=5
OUTPUT_BASE="./test_new_benchmarks_run"

for MODEL in "${MODELS[@]}"; do
  model_name=$(basename "$MODEL" .yaml)
  for TASK_INFO in "${TASKS[@]}"; do
    IFS=' ' read -r TASK_SPEC TASK_SCRIPT <<< "$TASK_INFO"
    task_name=$(echo "$TASK_SPEC" | cut -d':' -f1)
    
    echo "=================================================="
    echo "Running $task_name with $model_name"
    echo "=================================================="
    
    OUT_DIR="$OUTPUT_BASE/$model_name/$task_name"
    
    MODEL_YAML="$MODEL" MAX_SAMPLES="$MAX_SAMPLES" bash scripts/_run_lighteval_task.sh "$TASK_SPEC" "$TASK_SCRIPT" "$OUT_DIR" || true
  done
done

echo "Done running all benchmarks."

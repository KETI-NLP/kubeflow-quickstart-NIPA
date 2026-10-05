#!/bin/bash
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRUST_REMOTE_CODE=True
export CUDA_VISIBLE_DEVICES="4,5,6,7"

MODEL_REPO="Qwen/Qwen3.5-9B"
MODEL_ARGS="model_name=$MODEL_REPO,batch_size=1,max_length=4096"

echo "Running mathvista with $MODEL_REPO on GPUs 4,5,6,7"

uv run --python .venv/bin/python -m lighteval accelerate "$MODEL_ARGS" \
    "mathvista_gen:default|0" \
    --custom-tasks "custom_tasks/custom_mathvista_task.py" \
    --max-samples 2 \
    --output-dir "./test_new_benchmarks_run/local_qwen3.5_9b_smoke"

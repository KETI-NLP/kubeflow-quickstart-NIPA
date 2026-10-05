#!/bin/bash
export CUDA_VISIBLE_DEVICES="4"
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRUST_REMOTE_CODE=True

MODEL_REPO="Qwen/Qwen3.5-9B"
MODEL_ARGS="model_name=$MODEL_REPO,batch_size=1,max_new_tokens=128"

uv run --python .venv/bin/python -m lighteval accelerate "$MODEL_ARGS" \
    "chartqa_gen:default|0" \
    --vision-model \
    --custom-tasks "custom_tasks/custom_chartqa_task.py" \
    --max-samples 2 \
    --save-details \
    --output-dir "./debug_chartqa"

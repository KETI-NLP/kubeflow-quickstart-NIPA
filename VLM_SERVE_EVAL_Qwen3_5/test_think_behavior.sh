#!/bin/bash
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRUST_REMOTE_CODE=True
export CUDA_VISIBLE_DEVICES="0"

MODEL_REPO="/workspace/local_models/qwen3_5_9b_multimodal_sft_dpo_normalized/checkpoint-1253"
MODEL_ARGS="model_name=$MODEL_REPO,batch_size=1,max_length=8192"

echo "=== Running WITH --no-think ==="
export LIGHTEVAL_CUSTOM_TEMPLATE="/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/no_think_chat_template.jinja"
uv run --python .venv/bin/python run_lighteval_patched.py accelerate "$MODEL_ARGS" \
    "hae_rae_bench_gen|0" \
    --custom-tasks "custom_tasks/custom_hae_rae_bench_task.py" \
    --max-samples 1 \
    --output-dir "./test_think_debug/no_think"

echo "=== Running WITHOUT --no-think ==="
unset LIGHTEVAL_CUSTOM_TEMPLATE
uv run --python .venv/bin/python run_lighteval_patched.py accelerate "$MODEL_ARGS" \
    "hae_rae_bench_gen|0" \
    --custom-tasks "custom_tasks/custom_hae_rae_bench_task.py" \
    --max-samples 1 \
    --output-dir "./test_think_debug/with_think"

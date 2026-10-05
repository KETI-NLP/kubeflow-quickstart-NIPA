# scripts/env.example.sh
#
# Per-machine environment configuration template.
# Copy to scripts/env.local.sh and edit for each machine. env.local.sh is
# git-ignored so each environment keeps its own paths.
#
#   cp scripts/env.example.sh scripts/env.local.sh
#   $EDITOR scripts/env.local.sh
#
# Run scripts in this directory will source env.local.sh automatically when
# present. Variables already set in the calling shell take precedence, so
# `PYTHON_BIN=/other/python bash scripts/run_local_qwen.sh ...` still works.

# Path to the Python interpreter that has the project dependencies installed
# (torch with sm_100 for B200, transformers, lighteval, ...).
: "${PYTHON_BIN:=/path/to/your/conda_env_or_venv/bin/python}"

# Default GPU set used when CUDA_VISIBLE_DEVICES is not already set by the
# caller (e.g. when run_benchmark_workflow_parallel.sh splits 0-3 / 4-7).
: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"

# Number of data-parallel workers per `run_local_qwen.sh` invocation.
# Each worker loads the full model on one GPU, so set this to the number
# of GPUs you want one model run to use (typically equals the count in
# CUDA_VISIBLE_DEVICES). When run_benchmark_workflow_parallel.sh splits
# 8 GPUs into 0-3 / 4-7, NUM_PROCESSES=4 per worker is the right value.
: "${NUM_PROCESSES:=4}"

# Per-WORKER batch size for run_local_qwen.sh. Effective system batch
# = BATCH_SIZE * NUM_PROCESSES. Tune to GPU VRAM:
# B200 (180GB) with 9B VLM at max_length 8192: 4 is comfortable per worker
# (model 18GB + KV cache ~8GB = ~26GB/GPU). Bump to 8 if you have headroom.
: "${BATCH_SIZE:=4}"

# Telegram notification helper (set to empty to disable Telegram messages).
: "${TELEGRAM_PY:=/workspace/telegram_bot/send_telegram_message.py}"

# Hugging Face cache locations (override if your machine has a different
# shared cache mount).
: "${HF_HOME:=/workspace/.cache/huggingface}"
: "${HF_DATASETS_CACHE:=/workspace/.cache/huggingface/datasets}"

export PYTHON_BIN CUDA_VISIBLE_DEVICES NUM_PROCESSES BATCH_SIZE TELEGRAM_PY HF_HOME HF_DATASETS_CACHE

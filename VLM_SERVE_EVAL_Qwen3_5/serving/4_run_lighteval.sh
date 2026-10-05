#!/bin/bash
set -euo pipefail

OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
MODEL_NAME="${MODEL_NAME:-qwen3_5_9b_multimodal_sft}"
TASKS="${TASKS:-lighteval|gsm8k|0|0}"
OUTPUT_DIR="${OUTPUT_DIR:-./lighteval_results}"

mkdir -p "${OUTPUT_DIR}"

if ! command -v lighteval >/dev/null 2>&1; then
  echo "lighteval is not installed. Install it first: pip install lighteval litellm"
  exit 1
fi

LITELLM_MODEL_ARGS="provider=openai,model_name=${MODEL_NAME},base_url=${OPENAI_BASE_URL},api_key=${OPENAI_API_KEY}"
EXTRA_ARGS=()

if lighteval endpoint litellm --help 2>&1 | grep -q -- "--use-chat-template"; then
  EXTRA_ARGS+=("--use-chat-template")
fi

echo "Running LightEval against ${OPENAI_BASE_URL} with model=${MODEL_NAME}"
echo "Tasks: ${TASKS}"
echo "Output dir: ${OUTPUT_DIR}"

lighteval endpoint litellm \
  "${LITELLM_MODEL_ARGS}" \
  "${TASKS}" \
  --output-dir "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}"

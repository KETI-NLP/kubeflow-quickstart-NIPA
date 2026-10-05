#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_DIR="${OUTPUT_DIR:-./test_korean_heritage_name_vqa_gpt4o_run}"
MAX_SAMPLES="${MAX_SAMPLES:-50}"
export MAX_SAMPLES

exec bash scripts/_run_lighteval_task.sh \
  "korean_heritage_name_vqa|0" \
  "custom_tasks/custom_korean_heritage_name_vqa_task.py" \
  "$OUTPUT_DIR"

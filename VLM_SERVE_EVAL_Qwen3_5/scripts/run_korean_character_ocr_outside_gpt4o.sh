#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_DIR="${OUTPUT_DIR:-./test_korean_character_ocr_outside_gpt4o_run}"

exec bash scripts/_run_lighteval_task.sh \
  "korean_character_ocr_outside|0" \
  "custom_tasks/custom_korean_character_ocr_task.py" \
  "$OUTPUT_DIR"

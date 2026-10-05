#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <task_spec> <custom_task_path> <output_dir> [postprocess_mode]"
  exit 1
fi

TASK_SPEC="$1"
CUSTOM_TASK_PATH="$2"
OUTPUT_DIR="$3"
POSTPROCESS_MODE="${4:-none}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_YAML="${MODEL_YAML:-yaml_files/litellm_gpt4o.yaml}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"

cd "$ROOT_DIR"

echo "[run] root_dir=$ROOT_DIR"
echo "[run] model_yaml=$MODEL_YAML"
echo "[run] task=$TASK_SPEC"
echo "[run] custom_task=$CUSTOM_TASK_PATH"
echo "[run] output_dir=$OUTPUT_DIR"
echo "[run] max_samples=$MAX_SAMPLES"

uv run --python .venv/bin/python -m lighteval endpoint litellm \
  "$MODEL_YAML" \
  "$TASK_SPEC" \
  --custom-tasks "$CUSTOM_TASK_PATH" \
  --max-samples "$MAX_SAMPLES" \
  --save-details \
  --output-dir "$OUTPUT_DIR"

LATEST_RESULTS="$(
  uv run --python .venv/bin/python - "$OUTPUT_DIR" <<'PY'
from pathlib import Path
import sys
out = Path(sys.argv[1])
matches = sorted((out / "results").glob("**/results_*.json"))
print(matches[-1] if matches else "")
PY
)"

LATEST_DETAILS="$(
  uv run --python .venv/bin/python - "$OUTPUT_DIR" <<'PY'
from pathlib import Path
import sys
out = Path(sys.argv[1])
matches = sorted((out / "details").glob("**/details_*.parquet"))
print(matches[-1] if matches else "")
PY
)"

echo
echo "[done] latest results: $LATEST_RESULTS"
if [[ -n "$LATEST_DETAILS" ]]; then
  echo "[done] latest details: $LATEST_DETAILS"
fi

echo
echo "[metrics]"
uv run --python .venv/bin/python - "$LATEST_RESULTS" <<'PY'
import json
from pathlib import Path
import sys

results_path = Path(sys.argv[1])
with results_path.open("r", encoding="utf-8") as f:
    payload = json.load(f)

task_key = next(k for k in payload["results"].keys() if k != "all")
metrics = payload["results"][task_key]
for key in sorted(metrics):
    if key.endswith("_stderr"):
        continue
    print(f"{key}: {metrics[key]}")
PY
if [[ "$POSTPROCESS_MODE" == "text_shortqa" ]]; then
  echo
  echo "[postprocess] per-answer-type summary"
  uv run --python .venv/bin/python scripts/summarize_korean_heritage_text_shortqa_results.py \
    --eval-dir "$OUTPUT_DIR"
  echo "[done] per-answer-type json: $OUTPUT_DIR/analysis/per_answer_type_scores.json"
  echo "[done] per-answer-type md:   $OUTPUT_DIR/analysis/per_answer_type_scores.md"
fi

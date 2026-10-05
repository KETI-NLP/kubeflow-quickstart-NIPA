#!/bin/bash
# ==============================================================================
# VLM benchmark — endpoint (serving) variant of run_local_qwen.sh
#
# Instead of loading the model locally via `accelerate`, this drives an
# OpenAI-compatible serving endpoint (see serving/1_openai_compatible_api.py,
# served on k8s by serving/2_serve_vlm_qwen3_5.yaml + serving/3_port_forward.sh).
# Evaluation is fully decoupled from serving, so the same running endpoint can
# be benchmarked repeatedly without paying model-load cost each time.
#
# Usage:
#   bash scripts/run_endpoint_eval.sh <served_model_id> [--base-url URL]
#
#   <served_model_id>   Model id exposed by the endpoint's /v1/models
#                       (e.g. qwen3_5_9b_multimodal_sft_dpo_normalized/checkpoint-600)
#   --base-url URL      Endpoint base URL (default: $OPENAI_BASE_URL or
#                       http://127.0.0.1:8000/v1)
#
# Env:
#   ONLY_TASKS   space-separated task short-names to run a subset
#   MAX_SAMPLES  samples per task (default 100)
# ==============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$WORK_DIR" || exit 1
export PATH="$HOME/.local/bin:$PATH"

# Per-machine env (PYTHON_BIN, TELEGRAM_PY, HF_HOME, ...)
if [ -f "$SCRIPT_DIR/env.local.sh" ]; then
    . "$SCRIPT_DIR/env.local.sh"
elif [ -f "$SCRIPT_DIR/env.example.sh" ]; then
    . "$SCRIPT_DIR/env.example.sh"
fi

if [ -z "$PYTHON_BIN" ] || [ ! -x "$PYTHON_BIN" ]; then
    echo "❌ PYTHON_BIN is not set or not executable: '$PYTHON_BIN'"
    echo "   Copy scripts/env.example.sh to scripts/env.local.sh and set PYTHON_BIN."
    exit 1
fi

export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRUST_REMOTE_CODE=True

usage() {
    echo "Usage: $0 <served_model_id> [--base-url URL]"
    exit 1
}

MODEL_NAME=""
BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
while [ $# -gt 0 ]; do
    case "$1" in
        --base-url) BASE_URL="$2"; shift 2 ;;
        -h|--help)  usage ;;
        *)
            if [ -z "$MODEL_NAME" ]; then MODEL_NAME="$1"; shift
            else echo "Unknown argument: $1"; usage; fi
            ;;
    esac
done
[ -z "$MODEL_NAME" ] && { echo "Error: served model id is required."; usage; }

API_KEY="${OPENAI_API_KEY:-dummy}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"

# Endpoint health check — fail fast if the serve is not reachable.
echo "Checking endpoint $BASE_URL ..."
if ! curl -s -m 10 "$BASE_URL/models" | grep -q "\"$MODEL_NAME\""; then
    echo "❌ Endpoint not reachable, or model '$MODEL_NAME' not served at $BASE_URL/models"
    echo "   Available models:"
    curl -s -m 10 "$BASE_URL/models" 2>/dev/null | "$PYTHON_BIN" -c \
        "import sys,json; [print('   -', m['id']) for m in json.load(sys.stdin).get('data',[])]" 2>/dev/null \
        || echo "   (could not list models)"
    exit 1
fi
echo "✅ Endpoint OK, model '$MODEL_NAME' is served."

# lighteval's litellm backend passes config.model_name straight to
# litellm.completion(model=...), ignoring config.provider. litellm parses a
# leading "<provider>/" off model and falls back to "Provider NOT provided"
# when there is no slash — so a bare id (e.g. "qwen3_5_9b_simple_name_sft")
# fails the same way a slash-containing id (e.g. ".../checkpoint-600") does
# unless we prepend "openai/" ourselves.
MODEL_ARGS="provider=openai,model_name=openai/$MODEL_NAME,base_url=$BASE_URL,api_key=$API_KEY"

# Directory-safe tag from the served model id.
MODEL_TAG="$(echo "$MODEL_NAME" | sed 's:[/ ]:_:g')"
OUTPUT_DIR="./test_final_11_tasks_endpoint/$MODEL_TAG"
mkdir -p "$OUTPUT_DIR"

TASKS=(
  "mmbench_gen:en|0 custom_tasks/custom_mmbench_task.py"
  "scienceqa_gen:default|0 custom_tasks/custom_scienceqa_task.py"
  "mathvista_gen:default|0 custom_tasks/custom_mathvista_task.py"
  "chartqa_gen:default|0 custom_tasks/custom_chartqa_task.py"
  "ai2d_gen:default|0 custom_tasks/custom_ai2d_task.py"
  "hallusionbench_gen:default|0 custom_tasks/custom_hallusionbench_task.py"
  "korean_heritage_name_vqa|0 custom_tasks/custom_korean_heritage_name_vqa_task.py"
  "korean_heritage_knowledge|0 custom_tasks/custom_korean_heritage_knowledge_task.py"
  "korean_heritage_knowledge_mc|0 custom_tasks/custom_korean_heritage_knowledge_mc_task.py"
  "korean_heritage_knowledge_mc_noise|0 custom_tasks/custom_korean_heritage_knowledge_mc_task.py"
  "korean_heritage_knowledge_mc_bright|0 custom_tasks/custom_korean_heritage_knowledge_mc_task.py"
  "korean_heritage_knowledge_mc_rotate|0 custom_tasks/custom_korean_heritage_knowledge_mc_task.py"
  "korean_heritage_knowledge_mc_shift|0 custom_tasks/custom_korean_heritage_knowledge_mc_task.py"
  "korean_heritage_knowledge_mc_allperturb|0 custom_tasks/custom_korean_heritage_knowledge_mc_task.py"
  # OCR tracks are kept separate because their targets and metrics differ:
  # outdoor Hangul words/signs, boxed public-document text, and single characters.
  "korean_character_ocr|0 custom_tasks/custom_korean_character_ocr_task.py"
  "korean_character_ocr_public_executive|0 custom_tasks/custom_korean_character_ocr_task.py"
  "korean_character_ocr_data13|0 custom_tasks/custom_korean_character_ocr_task.py"
  "kmmlu_gen|0 custom_tasks/custom_kmmlu_task.py"
  "hae_rae_bench_gen|0 custom_tasks/custom_hae_rae_bench_task.py"
  "korean_heritage_reverse_qa|0 custom_tasks/custom_korean_heritage_reverse_qa_task.py"
  "korean_heritage_text_shortqa|0 custom_tasks/custom_korean_heritage_text_shortqa_task.py"
  "korean_heritage_reverse_mt_mc|0 custom_tasks/custom_korean_heritage_reverse_mt_mc_task.py"
  "korean_heritage_reverse_mt_free|0 custom_tasks/custom_korean_heritage_reverse_mt_free_task.py"
)
#"ifeval_ko_gen|0 custom_tasks/custom_ifeval_ko_task.py"

for task_info in "${TASKS[@]}"; do
  task=$(echo "$task_info" | awk '{print $1}')
  custom_task=$(echo "$task_info" | awk '{print $2}')
  task_short=$(echo "$task" | cut -d':' -f1 | cut -d'|' -f1)
  task_dir="$OUTPUT_DIR/$task_short"

  # Subset filter (substring-exact match on the task short name).
  if [ -n "$ONLY_TASKS" ]; then
      hit=0
      for keep in $ONLY_TASKS; do
          [ "$task_short" = "$keep" ] && { hit=1; break; }
      done
      if [ "$hit" -eq 0 ]; then
          echo "Skipping '$task' (not in ONLY_TASKS)"
          continue
      fi
  fi

  echo "=================================================="
  echo "Running $task against endpoint ($MODEL_NAME)"
  echo "=================================================="

  mkdir -p "$task_dir"
  RUN_LOG="$task_dir/lighteval_run.log"

  set -o pipefail
  "$PYTHON_BIN" run_lighteval_patched.py endpoint litellm "$MODEL_ARGS" \
      "$task" \
      --custom-tasks "$custom_task" \
      --save-details \
      --output-dir "$task_dir" \
      --max-samples "$MAX_SAMPLES" 2>&1 | tee "$RUN_LOG"
  RC=${PIPESTATUS[0]}
  set +o pipefail

  if [ "$RC" -ne 0 ]; then
      echo "⚠️ Task '$task' lighteval exited with code $RC — skipping and continuing." >&2
  fi
  echo ""
done

echo "Done running endpoint benchmark for model: $MODEL_NAME"
if [ -n "$TELEGRAM_PY" ] && [ -f "$TELEGRAM_PY" ]; then
    "$PYTHON_BIN" "$TELEGRAM_PY" text "run_endpoint_eval.sh ($MODEL_TAG) 모든 태스크 실행이 완료되었습니다."
fi

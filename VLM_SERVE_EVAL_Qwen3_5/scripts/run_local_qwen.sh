#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$WORK_DIR" || exit 1
export PATH="$HOME/.local/bin:$PATH"

# Per-machine env (PYTHON_BIN, HF_HOME, CUDA_VISIBLE_DEVICES, TELEGRAM_PY, ...)
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
    echo "Usage: $0 <model_path_or_repo> [--no-think]"
    echo ""
    echo "  <model_path_or_repo>  HuggingFace repo id (e.g. Qwen/Qwen3.5-9B)"
    echo "                        or local model path (e.g. /path/to/checkpoint)"
    echo "  --no-think            Disable <think> via custom chat template"
    exit 1
}

MODEL_REPO=""
DISABLE_THINK=false
for arg in "$@"; do
    case "$arg" in
        --no-think)
            DISABLE_THINK=true
            ;;
        -h|--help)
            usage
            ;;
        *)
            if [ -z "$MODEL_REPO" ]; then
                MODEL_REPO="$arg"
            else
                echo "Unknown argument: $arg"
                usage
            fi
            ;;
    esac
done

if [ -z "$MODEL_REPO" ]; then
    echo "Error: model path is required."
    usage
fi

MODEL_ARGS="model_name=$MODEL_REPO,batch_size=${BATCH_SIZE:-4},max_length=8192"

# Derive a directory-safe tag from the model path/repo for the output directory.
trimmed_repo="${MODEL_REPO%/}"
if [[ "$trimmed_repo" == */* ]]; then
    parent=$(basename "$(dirname "$trimmed_repo")")
    name=$(basename "$trimmed_repo")
    MODEL_TAG="${parent}_${name}"
else
    MODEL_TAG="$trimmed_repo"
fi

if [ "$DISABLE_THINK" = true ]; then
    echo "Disabling <think> using python wrapper injection."
    export LIGHTEVAL_CUSTOM_TEMPLATE="$WORK_DIR/no_think_chat_template.jinja"
fi

TASKS=(
  "mmbench_gen:en|0 custom_tasks/custom_mmbench_task.py"
  "scienceqa_gen:default|0 custom_tasks/custom_scienceqa_task.py"
  "mathvista_gen:default|0 custom_tasks/custom_mathvista_task.py"
  "chartqa_gen:default|0 custom_tasks/custom_chartqa_task.py"
  "ai2d_gen:default|0 custom_tasks/custom_ai2d_task.py"
  "hallusionbench_gen:default|0 custom_tasks/custom_hallusionbench_task.py"
  "korean_heritage_name_vqa|0 custom_tasks/custom_korean_heritage_name_vqa_task.py"
  "korean_heritage_knowledge|0 custom_tasks/custom_korean_heritage_knowledge_task.py"
  "korean_heritage_reverse_qa|0 custom_tasks/custom_korean_heritage_reverse_qa_task.py"
  "korean_heritage_text_shortqa|0 custom_tasks/custom_korean_heritage_text_shortqa_task.py"
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
)
#"ifeval_ko_gen|0 custom_tasks/custom_ifeval_ko_task.py"

OUTPUT_DIR="./test_final_11_tasks/$MODEL_TAG"

mkdir -p "$OUTPUT_DIR"

for task_info in "${TASKS[@]}"; do
  task=$(echo "$task_info" | awk '{print $1}')
  custom_task=$(echo "$task_info" | awk '{print $2}')

  task_dir="$OUTPUT_DIR/$(echo $task | cut -d':' -f1 | cut -d'|' -f1)"

  echo "=================================================="
  echo "Running $task with local model $MODEL_REPO on GPUs $CUDA_VISIBLE_DEVICES"
  echo "=================================================="

  VISION_ARGS="--vision-model"
  if [[ "$task" =~ "korean_heritage_text_shortqa" || "$task" =~ "korean_heritage_reverse_qa" || "$task" =~ "kmmlu_gen" || "$task" =~ "hae_rae_bench_gen" || "$task" =~ "ifeval_ko_gen" ]]; then
      VISION_ARGS=""
  fi

  # Allow callers to skip already-completed tasks by name (substring match).
  # Example:
  #   ONLY_TASKS="korean_heritage_name_vqa korean_character_ocr kmmlu_gen hae_rae_bench_gen" bash scripts/run_local_qwen.sh ...
  if [ -n "$ONLY_TASKS" ]; then
      task_short=$(echo "$task" | cut -d':' -f1 | cut -d'|' -f1)
      hit=0
      for keep in $ONLY_TASKS; do
          [ "$task_short" = "$keep" ] && { hit=1; break; }
      done
      if [ "$hit" -eq 0 ]; then
          echo "Skipping '$task' (not in ONLY_TASKS)"
          echo ""
          continue
      fi
  fi

  mkdir -p "$task_dir"
  RUN_LOG="$task_dir/lighteval_run.log"

  # Data-parallel launch via `accelerate launch`: each of NUM_PROCESSES
  # workers loads the full model on its assigned GPU (no naive layer-wise
  # MP), so all GPUs run inference in parallel. BATCH_SIZE in MODEL_ARGS
  # is per-worker; effective system batch is BATCH_SIZE * NUM_PROCESSES.
  ACCELERATE_BIN="$(dirname "$PYTHON_BIN")/accelerate"

  set -o pipefail
  "$ACCELERATE_BIN" launch \
      --num_processes="${NUM_PROCESSES:-4}" \
      --num_machines=1 \
      --main_process_port="${MASTER_PORT:-29500}" \
      --mixed_precision=bf16 \
      run_lighteval_patched.py accelerate "$MODEL_ARGS" \
        "$task" \
        $VISION_ARGS \
        --custom-tasks "$custom_task" \
        --save-details \
        --output-dir "$task_dir" \
        --max-samples 100 2>&1 | tee "$RUN_LOG"
  RC=${PIPESTATUS[0]}
  set +o pipefail

  # Hard stop only if the checkpoint did not fully load — those numbers
  # would be meaningless. Other lighteval failures (e.g. transient dataset
  # I/O) are reported but the loop continues so we still get partial
  # results for the other tasks.
  if grep -qE "Following weights were not initialized|not initialized from checkpoint|unexpected key" "$RUN_LOG"; then
      echo "❌ Detected weight-loading mismatch in '$task' log. Aborting model run." >&2
      echo "   (Benchmarking a partially loaded checkpoint would be meaningless.)" >&2
      exit 1
  fi
  if [ "$RC" -ne 0 ]; then
      echo "⚠️ Task '$task' lighteval exited with code $RC — skipping and continuing." >&2
  fi

  echo ""
done

echo "Done running configured benchmark tasks for model: $MODEL_REPO"
if [ -n "$TELEGRAM_PY" ] && [ -f "$TELEGRAM_PY" ]; then
    "$PYTHON_BIN" "$TELEGRAM_PY" text "run_local_qwen.sh ($MODEL_TAG) 모든 태스크 실행이 완료되었습니다."
fi

#!/bin/bash

# ==============================================================================
# KoMMERCE VLM Benchmark Workflow (Parallel, 2 models)
#
# 두 모델을 GPU 0-3 / 4-7 로 나눠서 동시에 평가하고,
# 둘 다 끝난 뒤 모델별 리포트 + 비교 리포트 + 텔레그램 알림을 보낸다.
#
# Usage:
#   bash scripts/run_benchmark_workflow_parallel.sh <model1> <model2> [--no-think]
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$WORK_DIR" || exit 1

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

usage() {
    echo "Usage: $0 <model1> <model2> [--no-think]"
    echo "  두 모델을 GPU 0-3 / 4-7 로 분할하여 병렬 실행한다."
    exit 1
}

MODELS=()
EXTRA_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --no-think)
            EXTRA_ARGS+=("$arg")
            ;;
        -h|--help)
            usage
            ;;
        *)
            MODELS+=("$arg")
            ;;
    esac
done

if [ ${#MODELS[@]} -ne 2 ]; then
    echo "❌ Error: 정확히 두 개의 모델을 지정해야 합니다 (현재: ${#MODELS[@]}개)."
    usage
fi

model_tag() {
    local repo="${1%/}"
    if [[ "$repo" == */* ]]; then
        local parent name
        parent=$(basename "$(dirname "$repo")")
        name=$(basename "$repo")
        echo "${parent}_${name}"
    else
        echo "$repo"
    fi
}

RESULT_BASE="test_final_11_tasks"
REPORT_DIR="$WORK_DIR/reports"
LOG_DIR="$WORK_DIR/parallel_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$REPORT_DIR" "$LOG_DIR"

MODEL1="${MODELS[0]}"
MODEL2="${MODELS[1]}"
TAG1=$(model_tag "$MODEL1")
TAG2=$(model_tag "$MODEL2")
LOG1="$LOG_DIR/run_${TAG1}.log"
LOG2="$LOG_DIR/run_${TAG2}.log"

echo "=============================================================================="
echo "🚀 두 모델 병렬 벤치마크 시작"
echo "  Model 1 : $MODEL1"
echo "    GPUs  : 0,1,2,3"
echo "    Log   : $LOG1"
echo "  Model 2 : $MODEL2"
echo "    GPUs  : 4,5,6,7"
echo "    Log   : $LOG2"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    echo "  옵션    : ${EXTRA_ARGS[*]}"
fi
echo "=============================================================================="

(
    export CUDA_VISIBLE_DEVICES="0,1,2,3"
    export MASTER_PORT=29501
    bash scripts/run_local_qwen.sh "$MODEL1" "${EXTRA_ARGS[@]}"
) > "$LOG1" 2>&1 &
PID1=$!

(
    export CUDA_VISIBLE_DEVICES="4,5,6,7"
    export MASTER_PORT=29502
    bash scripts/run_local_qwen.sh "$MODEL2" "${EXTRA_ARGS[@]}"
) > "$LOG2" 2>&1 &
PID2=$!

echo "  PID Model 1: $PID1"
echo "  PID Model 2: $PID2"
echo "  두 프로세스 종료 대기 중... (tail -f $LOG_DIR/*.log 로 진행 상황 확인 가능)"

wait "$PID1"; RC1=$?
wait "$PID2"; RC2=$?

echo "  Model 1 exit code: $RC1"
echo "  Model 2 exit code: $RC2"

# ---------- 모델별 리포트 ----------
SUCCESS_MODELS=()
SUCCESS_DIRS=()
SUCCESS_REPORTS=()
FAILED_MODELS=()

for i in 0 1; do
    if [ $i -eq 0 ]; then
        MODEL="$MODEL1"; TAG="$TAG1"; RC="$RC1"
    else
        MODEL="$MODEL2"; TAG="$TAG2"; RC="$RC2"
    fi

    if [ "$RC" -ne 0 ]; then
        echo "❌ 벤치마크 실패: $MODEL (exit=$RC)"
        "$PYTHON_BIN" "$TELEGRAM_PY" text "❌ 벤치마크 실행 중 오류 발생 (모델: $MODEL)"
        FAILED_MODELS+=("$MODEL")
        continue
    fi

    MODEL_RESULT_DIR="$RESULT_BASE/$TAG"
    PER_MODEL_REPORT="$REPORT_DIR/sample_report_${TAG}.md"

    echo "=============================================================================="
    echo "🚀 모델별 리포트 생성: $TAG"
    echo "=============================================================================="
    "$PYTHON_BIN" generate_report_v7.py \
        --model-dir "$MODEL_RESULT_DIR" \
        --output "$PER_MODEL_REPORT"
    if [ $? -ne 0 ]; then
        echo "❌ 모델별 리포트 생성 실패: $MODEL"
        "$PYTHON_BIN" "$TELEGRAM_PY" text "❌ 모델별 리포트 생성 실패 (모델: $MODEL)"
        FAILED_MODELS+=("$MODEL")
        continue
    fi

    SUCCESS_MODELS+=("$MODEL")
    SUCCESS_DIRS+=("$MODEL_RESULT_DIR")
    SUCCESS_REPORTS+=("$PER_MODEL_REPORT")
done

if [ ${#SUCCESS_MODELS[@]} -eq 0 ]; then
    echo "❌ 모든 모델 실패"
    "$PYTHON_BIN" "$TELEGRAM_PY" text "❌ 모든 모델 벤치마크가 실패했습니다."
    exit 1
fi

# ---------- 비교 리포트 ----------
COMPARISON_REPORT="$REPORT_DIR/comparison_report.md"
echo "=============================================================================="
echo "🚀 비교 리포트 생성 (${#SUCCESS_MODELS[@]}개 모델)"
echo "=============================================================================="
COMPARE_ARGS=()
for d in "${SUCCESS_DIRS[@]}"; do
    COMPARE_ARGS+=("--model-dir" "$d")
done
"$PYTHON_BIN" generate_comparison_report.py \
    "${COMPARE_ARGS[@]}" \
    --output "$COMPARISON_REPORT"
if [ $? -ne 0 ]; then
    echo "❌ 비교 리포트 생성 중 오류"
    "$PYTHON_BIN" "$TELEGRAM_PY" text "❌ 비교 리포트 생성 중 오류가 발생했습니다."
    exit 1
fi
echo "✅ 비교 리포트: $COMPARISON_REPORT"

# ---------- 텔레그램 알림 ----------
SUCCESS_LIST=$(printf '  - %s\n' "${SUCCESS_MODELS[@]}")
REPORT_LIST=$(printf '  - %s\n' "${SUCCESS_REPORTS[@]}")
if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    FAILED_LIST=$(printf '  - %s\n' "${FAILED_MODELS[@]}")
    MSG="✅ 병렬 벤치마크 평가 완료
성공 (${#SUCCESS_MODELS[@]}):
${SUCCESS_LIST}
실패 (${#FAILED_MODELS[@]}):
${FAILED_LIST}
모델별 리포트:
${REPORT_LIST}
비교 리포트: ${COMPARISON_REPORT}"
else
    MSG="✅ 병렬 벤치마크 평가 완료 (${#SUCCESS_MODELS[@]}개 모델)
성공 모델:
${SUCCESS_LIST}
모델별 리포트:
${REPORT_LIST}
비교 리포트: ${COMPARISON_REPORT}"
fi

"$PYTHON_BIN" "$TELEGRAM_PY" text "$MSG"
echo "🎉 모든 워크플로우가 완료되었습니다!"

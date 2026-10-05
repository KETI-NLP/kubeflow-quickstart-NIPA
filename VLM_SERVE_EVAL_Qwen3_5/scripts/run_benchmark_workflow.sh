#!/bin/bash

# ==============================================================================
# KoMMERCE VLM Benchmark Workflow
# (no-think 적용, 샘플 평가 진행, 모델별 리포트 + 비교 리포트 생성, 텔레그램 알림)
#
# Usage:
#   bash scripts/run_benchmark_workflow.sh <model_path> [<model_path> ...] [--no-think]
#
# 동작 순서:
#   1. 인자로 받은 모델들에 대해 순차적으로 run_local_qwen.sh 실행
#   2. 성공한 모델마다 generate_report_v7.py로 모델별 상세 리포트 생성
#   3. 성공 모델이 1개 이상이면 generate_comparison_report.py로 비교 리포트 생성
#   4. 텔레그램 알림 발송
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$WORK_DIR" || exit 1

usage() {
    echo "Usage: $0 <model_path_or_repo> [<model_path_or_repo> ...] [--no-think]"
    echo ""
    echo "  <model_path_or_repo>  HuggingFace repo id 또는 로컬 모델 경로 (1개 이상 지정 가능)"
    echo "  --no-think            run_local_qwen.sh 에 --no-think 옵션 전달"
    exit 1
}

# 모델 경로 -> 디렉터리 안전 태그 (run_local_qwen.sh 와 동일 규칙)
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

if [ ${#MODELS[@]} -eq 0 ]; then
    echo "❌ Error: 적어도 하나의 모델 경로를 지정해야 합니다."
    usage
fi

TELEGRAM_PY="$WORK_DIR/../telegram_bot/send_telegram_message.py"
PYTHON_BIN="$WORK_DIR/.venv/bin/python"
RESULT_BASE="test_final_11_tasks"
REPORT_DIR="$WORK_DIR/reports"
mkdir -p "$REPORT_DIR"

SUCCESS_MODELS=()
SUCCESS_TAGS=()
SUCCESS_DIRS=()
SUCCESS_REPORTS=()
FAILED_MODELS=()

# ---------- Step 1: 모델별 벤치마크 + 모델별 리포트 ----------
for MODEL in "${MODELS[@]}"; do
    TAG=$(model_tag "$MODEL")
    MODEL_RESULT_DIR="$RESULT_BASE/$TAG"
    PER_MODEL_REPORT="$REPORT_DIR/sample_report_${TAG}.md"

    echo "=============================================================================="
    echo "🚀 [Step 1] 벤치마크 평가 시작: $MODEL"
    echo "    태그        : $TAG"
    echo "    결과 디렉터리: $MODEL_RESULT_DIR"
    if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
        echo "    옵션        : ${EXTRA_ARGS[*]}"
    fi
    echo "=============================================================================="

    bash scripts/run_local_qwen.sh "$MODEL" "${EXTRA_ARGS[@]}"
    rc=$?

    if [ $rc -ne 0 ]; then
        echo "❌ 벤치마크 실행 중 오류 발생 (모델: $MODEL, exit=$rc)"
        "$PYTHON_BIN" "$TELEGRAM_PY" text "❌ 벤치마크 평가 실행 중 오류가 발생했습니다 (모델: $MODEL)"
        FAILED_MODELS+=("$MODEL")
        continue
    fi

    echo "✅ [Step 1] 벤치마크 평가 완료: $MODEL"

    echo "=============================================================================="
    echo "🚀 [Step 2] 모델별 상세 리포트 생성: $TAG"
    echo "=============================================================================="
    "$PYTHON_BIN" generate_report_v7.py \
        --model-dir "$MODEL_RESULT_DIR" \
        --output "$PER_MODEL_REPORT"
    rc=$?

    if [ $rc -ne 0 ]; then
        echo "❌ 모델별 리포트 생성 실패: $MODEL"
        "$PYTHON_BIN" "$TELEGRAM_PY" text "❌ 모델별 리포트 생성 실패 (모델: $MODEL)"
        FAILED_MODELS+=("$MODEL")
        continue
    fi

    SUCCESS_MODELS+=("$MODEL")
    SUCCESS_TAGS+=("$TAG")
    SUCCESS_DIRS+=("$MODEL_RESULT_DIR")
    SUCCESS_REPORTS+=("$PER_MODEL_REPORT")
    echo ""
done

if [ ${#SUCCESS_MODELS[@]} -eq 0 ]; then
    echo "❌ 모든 모델에 대해 벤치마크/리포트가 실패했습니다."
    "$PYTHON_BIN" "$TELEGRAM_PY" text "❌ 모든 모델 벤치마크가 실패했습니다."
    exit 1
fi

# ---------- Step 3: 비교 리포트 ----------
COMPARISON_REPORT="$REPORT_DIR/comparison_report.md"
echo "=============================================================================="
echo "🚀 [Step 3] 모델 간 비교 리포트 생성 (${#SUCCESS_MODELS[@]}개 모델)"
echo "=============================================================================="
COMPARE_ARGS=()
for d in "${SUCCESS_DIRS[@]}"; do
    COMPARE_ARGS+=("--model-dir" "$d")
done
"$PYTHON_BIN" generate_comparison_report.py \
    "${COMPARE_ARGS[@]}" \
    --output "$COMPARISON_REPORT"

if [ $? -ne 0 ]; then
    echo "❌ 비교 리포트 생성 중 오류가 발생했습니다."
    "$PYTHON_BIN" "$TELEGRAM_PY" text "❌ 비교 리포트 생성 중 오류가 발생했습니다."
    exit 1
fi

echo "✅ [Step 3] 비교 리포트 생성 완료: $COMPARISON_REPORT"
echo ""

# ---------- Step 4: 텔레그램 알림 ----------
echo "=============================================================================="
echo "🚀 [Step 4] 텔레그램 완료 메시지 발송"
echo "=============================================================================="

SUCCESS_LIST=$(printf '  - %s\n' "${SUCCESS_MODELS[@]}")
REPORT_LIST=$(printf '  - %s\n' "${SUCCESS_REPORTS[@]}")
if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    FAILED_LIST=$(printf '  - %s\n' "${FAILED_MODELS[@]}")
    MSG="✅ 벤치마크 평가 및 리포트 생성이 완료되었습니다.
성공 모델 (${#SUCCESS_MODELS[@]}):
${SUCCESS_LIST}
실패 모델 (${#FAILED_MODELS[@]}):
${FAILED_LIST}
모델별 리포트:
${REPORT_LIST}
비교 리포트: ${COMPARISON_REPORT}"
else
    MSG="✅ 벤치마크 평가 및 리포트 생성이 성공적으로 완료되었습니다 (${#SUCCESS_MODELS[@]}개 모델).
성공 모델:
${SUCCESS_LIST}
모델별 리포트:
${REPORT_LIST}
비교 리포트: ${COMPARISON_REPORT}"
fi

"$PYTHON_BIN" "$TELEGRAM_PY" text "$MSG"

echo "🎉 모든 워크플로우가 완료되었습니다!"

#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
Usage:
  ./run.sh <command> [--profile local|k8s] [--config path] [-- extra args]

Commands:
  serve                       Run the OpenAI-compatible API server
  deploy                      Apply the Kubernetes serving manifest
  port-forward                Port-forward the serving Service
  lighteval                   Run lighteval against the configured endpoint
  eval-checkpoints            Run local checkpoint sweep evaluation
  collect-checkpoint-summary  Merge checkpoint eval outputs
  prompt-grid                 Run local prompt-grid eval for one checkpoint
  collect-prompt-grid         Merge prompt-grid CSV summaries
  launch-checkpoint-eval      Launch Kubernetes jobs for parallel checkpoint eval
  launch-prompt-grid          Launch Kubernetes jobs for parallel prompt-grid eval

Examples:
  ./run.sh serve --profile local
  ./run.sh serve --profile k8s
  ./run.sh eval-checkpoints --profile local
  ./run.sh launch-checkpoint-eval --profile k8s
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

append_if_set() {
    local flag="$1"
    local value="${2:-}"
    if [[ -n "${value}" ]]; then
        CMD_ARGS+=("${flag}" "${value}")
    fi
}

require_profile_capability() {
    local capability="$1"
    if [[ "${PROFILE}" != "k8s" ]]; then
        die "${capability} is only available with --profile k8s"
    fi
}

load_config() {
    local default_config=""
    case "${PROFILE}" in
        k8s)
            default_config="${SCRIPT_DIR}/configs/k8s.env"
            ;;
        local)
            default_config="${SCRIPT_DIR}/configs/local.env"
            if [[ ! -f "${default_config}" ]]; then
                die "Missing ${default_config}. Copy configs/local.env.example to configs/local.env and edit it first."
            fi
            ;;
        *)
            die "Unknown profile: ${PROFILE}"
            ;;
    esac

    CONFIG_PATH="${CONFIG_PATH:-${default_config}}"
    [[ -f "${CONFIG_PATH}" ]] || die "Config file not found: ${CONFIG_PATH}"

    # shellcheck disable=SC1090
    source "${CONFIG_PATH}"

    PYTHON_BIN="${PYTHON_BIN:-python}"
    MODEL_ID="${MODEL_ID:-qwen3_5_vlm_ft}"
    HOST="${HOST:-127.0.0.1}"
    PORT="${PORT:-8000}"
    MODELS_CONFIG="${MODELS_CONFIG:-}"
    TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
    ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
    OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:${PORT}/v1}"
    OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
    TASKS="${TASKS:-lighteval|gsm8k|0|0}"
    CHECKPOINT_SUMMARY_OUTPUT_PREFIX="${CHECKPOINT_SUMMARY_OUTPUT_PREFIX:-parallel_summary}"
}

COMMAND="${1:-help}"
if [[ $# -gt 0 ]]; then
    shift
fi

PROFILE="${PROFILE:-k8s}"
CONFIG_PATH="${CONFIG_PATH:-}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)
            [[ $# -ge 2 ]] || die "--profile requires a value"
            PROFILE="$2"
            shift 2
            ;;
        --config)
            [[ $# -ge 2 ]] || die "--config requires a value"
            CONFIG_PATH="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

case "${COMMAND}" in
    help|-h|--help)
        usage
        exit 0
        ;;
esac

load_config

cd "${SCRIPT_DIR}"

case "${COMMAND}" in
    serve)
        [[ -n "${MODEL_PATH:-}" ]] || die "MODEL_PATH is required"
        CMD_ARGS=(
            "${PYTHON_BIN}" "1_openai_compatible_api.py"
            "--model-path" "${MODEL_PATH}"
            "--model-id" "${MODEL_ID}"
            "--host" "${HOST}"
            "--port" "${PORT}"
            "--torch-dtype" "${TORCH_DTYPE}"
            "--attn-implementation" "${ATTN_IMPLEMENTATION}"
        )
        append_if_set "--base-model-name" "${BASE_MODEL_NAME:-}"
        append_if_set "--processor-path" "${PROCESSOR_PATH:-}"
        append_if_set "--checkpoints-root" "${CHECKPOINTS_ROOT:-}"
        append_if_set "--models-config" "${MODELS_CONFIG:-}"
        exec "${CMD_ARGS[@]}" "${EXTRA_ARGS[@]}"
        ;;

    deploy)
        require_profile_capability "deploy"
        require_command kubectl
        exec kubectl apply -f "${K8S_SERVE_YAML:-2_serve_vlm_qwen3_5.yaml}" "${EXTRA_ARGS[@]}"
        ;;

    port-forward)
        require_profile_capability "port-forward"
        require_command kubectl
        exec kubectl -n "${NAMESPACE:-gpu-workspace}" port-forward \
            "svc/${SERVICE_NAME:-vlm-qwen3-5-serve}" \
            "${LOCAL_PORT:-8000}:${REMOTE_PORT:-8000}" \
            "${EXTRA_ARGS[@]}"
        ;;

    lighteval)
        export OPENAI_BASE_URL
        export OPENAI_API_KEY
        export MODEL_NAME="${MODEL_NAME:-${MODEL_ID}}"
        export TASKS
        export OUTPUT_DIR="${LIGHEVAL_OUTPUT_DIR:-./lighteval_results}"
        exec "${SCRIPT_DIR}/4_run_lighteval.sh" "${EXTRA_ARGS[@]}"
        ;;

    eval-checkpoints)
        [[ -n "${CHECKPOINTS_DIR:-}" ]] || die "CHECKPOINTS_DIR is required"
        [[ -n "${CHECKPOINT_EVAL_OUTPUT_DIR:-}" ]] || die "CHECKPOINT_EVAL_OUTPUT_DIR is required"
        CMD_ARGS=(
            "${PYTHON_BIN}" "5_eval_checkpoints.py"
            "--checkpoints-dir" "${CHECKPOINTS_DIR}"
            "--output-dir" "${CHECKPOINT_EVAL_OUTPUT_DIR}"
            "--torch-dtype" "${TORCH_DTYPE}"
            "--attn-implementation" "${ATTN_IMPLEMENTATION}"
        )
        append_if_set "--base-model-name" "${BASE_MODEL_NAME:-}"
        append_if_set "--processor-path" "${PROCESSOR_PATH:-}"
        append_if_set "--questions-file" "${QUESTIONS_FILE:-}"
        append_if_set "--checkpoint-glob" "${CHECKPOINT_GLOB:-}"
        append_if_set "--limit" "${CHECKPOINT_LIMIT:-}"
        exec "${CMD_ARGS[@]}" "${EXTRA_ARGS[@]}"
        ;;

    collect-checkpoint-summary)
        [[ -n "${PARALLEL_CHECKPOINT_EVAL_OUTPUT_DIR:-}" ]] || die "PARALLEL_CHECKPOINT_EVAL_OUTPUT_DIR is required"
        exec "${PYTHON_BIN}" "9_collect_checkpoint_eval_summary.py" \
            "--input-dir" "${PARALLEL_CHECKPOINT_EVAL_OUTPUT_DIR}" \
            "--output-prefix" "${CHECKPOINT_SUMMARY_OUTPUT_PREFIX}" \
            "${EXTRA_ARGS[@]}"
        ;;

    prompt-grid)
        [[ -n "${PROMPT_GRID_CHECKPOINT_DIR:-}" ]] || die "PROMPT_GRID_CHECKPOINT_DIR is required"
        [[ -n "${PROMPT_GRID_OUTPUT_DIR:-}" ]] || die "PROMPT_GRID_OUTPUT_DIR is required"
        CMD_ARGS=(
            "${PYTHON_BIN}" "11_eval_prompt_grid.py"
            "--checkpoint-dir" "${PROMPT_GRID_CHECKPOINT_DIR}"
            "--output-dir" "${PROMPT_GRID_OUTPUT_DIR}"
            "--torch-dtype" "${TORCH_DTYPE}"
            "--attn-implementation" "${ATTN_IMPLEMENTATION}"
        )
        append_if_set "--base-model-name" "${BASE_MODEL_NAME:-}"
        append_if_set "--processor-path" "${PROCESSOR_PATH:-}"
        exec "${CMD_ARGS[@]}" "${EXTRA_ARGS[@]}"
        ;;

    collect-prompt-grid)
        [[ -n "${PROMPT_GRID_OUTPUT_DIR:-}" ]] || die "PROMPT_GRID_OUTPUT_DIR is required"
        CMD_ARGS=(
            "${PYTHON_BIN}" "14_collect_prompt_grid_summary.py"
            "--input-dir" "${PROMPT_GRID_OUTPUT_DIR}"
        )
        append_if_set "--output-file" "${PROMPT_GRID_RANKING_FILE:-}"
        exec "${CMD_ARGS[@]}" "${EXTRA_ARGS[@]}"
        ;;

    launch-checkpoint-eval)
        require_profile_capability "launch-checkpoint-eval"
        export NAMESPACE="${NAMESPACE:-gpu-workspace}"
        export CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-}"
        export OUTPUT_DIR="${PARALLEL_CHECKPOINT_EVAL_OUTPUT_DIR:-}"
        export TEMPLATE_FILE="${K8S_CHECKPOINT_EVAL_TEMPLATE:-7_eval_single_checkpoint_job.yaml}"
        export CHECKPOINTS="${CHECKPOINTS:-}"
        exec "${SCRIPT_DIR}/8_launch_parallel_checkpoint_eval.sh" "${EXTRA_ARGS[@]}"
        ;;

    launch-prompt-grid)
        require_profile_capability "launch-prompt-grid"
        export NAMESPACE="${NAMESPACE:-gpu-workspace}"
        export CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-}"
        export OUTPUT_DIR="${PROMPT_GRID_OUTPUT_DIR:-}"
        export TEMPLATE_FILE="${K8S_PROMPT_GRID_TEMPLATE:-12_eval_single_checkpoint_prompt_job.yaml}"
        export CHECKPOINTS="${CHECKPOINTS:-}"
        exec "${SCRIPT_DIR}/13_launch_parallel_prompt_grid_eval.sh" "${EXTRA_ARGS[@]}"
        ;;

    *)
        usage
        die "Unknown command: ${COMMAND}"
        ;;
esac

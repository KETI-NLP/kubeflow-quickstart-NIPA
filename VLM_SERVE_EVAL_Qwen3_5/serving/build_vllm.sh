#!/bin/bash
set -euo pipefail
# Build & push the vLLM-based serving image.
#   VLLM_TAG must support Qwen3.5 (model_type "qwen3_5"); repo targets 0.19.x.
#
# Usage:
#   VLLM_TAG=v0.19.0 ./build_vllm.sh

IMAGE_NAME="${IMAGE_NAME:-registry.example.com/vlm-qwen3_5-serve-vllm}"
TAG="${TAG:-latest}"
VLLM_TAG="${VLLM_TAG:-latest}"

echo "================================================"
echo "Building vLLM serving image: ${IMAGE_NAME}:${TAG}"
echo "Base: vllm/vllm-openai:${VLLM_TAG}   Arch: linux/amd64"
echo "================================================"

if ! docker buildx inspect default > /dev/null 2>&1; then
    docker buildx create --use --name multi-arch-builder
else
    docker buildx use default
fi

docker buildx build \
    --platform linux/amd64 \
    --build-arg "VLLM_TAG=${VLLM_TAG}" \
    --tag "${IMAGE_NAME}:${TAG}" \
    -f Dockerfile.vllm \
    . \
    --push

echo "Build complete: ${IMAGE_NAME}:${TAG}"

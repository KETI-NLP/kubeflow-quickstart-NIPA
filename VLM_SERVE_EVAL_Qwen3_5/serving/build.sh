#!/bin/bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-registry.example.com/vlm-qwen3_5-serve}"
TAG="${TAG:-latest}"

echo "================================================"
echo "Building serving image: ${IMAGE_NAME}:${TAG}"
echo "Target Architecture: linux/amd64 (x86_64)"
echo "================================================"

if ! docker buildx inspect default > /dev/null 2>&1; then
    docker buildx create --use --name multi-arch-builder
else
    docker buildx use default
fi

docker buildx build \
    --platform linux/amd64 \
    --tag "${IMAGE_NAME}:${TAG}" \
    -f Dockerfile \
    . \
    --push

echo "Build complete: ${IMAGE_NAME}:${TAG}"

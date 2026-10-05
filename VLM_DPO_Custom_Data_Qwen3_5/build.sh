#!/bin/bash
set -e

IMAGE_NAME="${IMAGE_NAME:-registry.example.com/vlm-dpo-qwen3_5}"
TAG="${TAG:-ib}"

echo "================================================"
echo "Building Docker image: ${IMAGE_NAME}:${TAG}"
echo "Target Architecture: linux/amd64 (x86_64)"
echo "Profile: Infiniband / RDMA enabled userspace"
echo "================================================"

if ! docker buildx inspect default > /dev/null 2>&1; then
    docker buildx create --use --name multi-arch-builder
else
    docker buildx use default
fi

echo "Starting multi-arch build..."
docker buildx build \
    --platform linux/amd64 \
    --tag ${IMAGE_NAME}:${TAG} \
    -f Dockerfile \
    . \
    --push

echo "Build complete. If you are uploading to a remote registry, run:"
echo "docker push ${IMAGE_NAME}:${TAG}"

#!/bin/bash
set -e

# Your container registry URL (Change this to your actual registry if needed)
# Example: mlx-public.kr.ncr.ntruss.com/mlx/custom-llm-sft:1.0
IMAGE_NAME="registry.example.com/custom-vlm-dpo"
TAG="latest"

echo "================================================"
echo "Building Docker image: ${IMAGE_NAME}:${TAG}"
echo "Target Architecture: linux/amd64 (x86_64)"
echo "================================================"

# Check if buildx is available and create a builder if one doesn't exist
if ! docker buildx inspect default > /dev/null 2>&1; then
    docker buildx create --use --name multi-arch-builder
else
    docker buildx use default
fi

# Build for linux/amd64 and push if you have credentials
# We use --load here so it's available locally, but if pushing to a remote registry, change to --push
echo "Starting multi-arch build..."
docker buildx build \
    --platform linux/amd64 \
    --tag ${IMAGE_NAME}:${TAG} \
    -f Dockerfile \
    . \
    --push

echo "Build complete. If you are uploading to a remote registry, run:"
echo "docker push ${IMAGE_NAME}:${TAG}"

FROM nvcr.io/nvidia/pytorch:24.09-py3

# Set environment variables for non-interactive installs and HuggingFace cache
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/data/huggingface_cache
ENV PYTHONUNBUFFERED=1

# Install required system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install LLM required python packages
# Pin versions to ensure FSDP1 stability
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    "transformers>=4.50.0" \
    "accelerate>=0.34.2" \
    datasets \
    "peft" \
    evaluate \
    scipy \
    "trl==0.11.4" \
    "vllm" \
    "deepspeed==0.14.4" \
    ninja \
    qwen-vl-utils \
    torchvision \
    packaging

# Install flash-attention separately because it requires ninja to build
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# Set working directory
WORKDIR /app

# Copy the distributed training script
COPY 3_llm_rlhf_distributed.py .

# Default command
CMD ["python3"]

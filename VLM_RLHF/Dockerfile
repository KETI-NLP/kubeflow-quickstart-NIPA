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
# trl>=0.15.0 natively supports multimodal GRPOTrainer.
# vllm is required for trl imports even if not used for generation.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    "transformers>=4.57.0" \
    "accelerate>=1.2.0" \
    "datasets>=3.0.0" \
    "peft>=0.14.0" \
    evaluate \
    scipy \
    "trl==0.16.1" \
    "vllm>=0.7.1" \
    "deepspeed>=0.14.4" \
    "qwen-vl-utils[decord]==0.0.8" \
    "math_verify" \
    torchvision \
    ninja \
    packaging && \
    pip uninstall -y flash-attn

# Set working directory
WORKDIR /app

# Copy the distributed training script
COPY 3_vlm_rlhf_distributed.py .

# Default command
CMD ["python3"]

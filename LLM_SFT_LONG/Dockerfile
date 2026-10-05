# Use NVIDIA PyTorch base image (24.06 contains PyTorch 2.3 which has the IPv6 fallback without breaking FSDP1)
FROM nvcr.io/nvidia/pytorch:24.06-py3

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
# Upgrade pip to avoid dependency resolution backtracking delays
# Pin transformers and accelerate to known stable versions for FSDP1
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    "transformers>=4.48.0" \
    "accelerate>=1.0.0" \
    "numpy<2.0" \
    datasets \
    peft \
    evaluate \
    scipy \
    trl \
    "deepspeed>=0.14.4" \
    ninja \
    packaging \
    qwen-vl-utils \
    opencv-python-headless \
    pillow && \
    python3 -c 'import os, importlib.util; spec=importlib.util.find_spec("transformers"); p=os.path.join(os.path.dirname(spec.origin), "utils", "import_utils.py"); c=open(p).read().replace("is_available, torch_version = _is_package_available(\"torch\", return_version=True)", "is_available, torch_version = True, \"2.4.0\""); open(p, "w").write(c)' || true

# Install flash-attention separately because it requires ninja to build
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# Install ring-flash-attention
RUN pip install --no-cache-dir git+https://github.com/zhuzilin/ring-flash-attention.git

# Uninstall DeepSpeed because Trainer aggressively imports it but it breaks with PyTorch 2.4.0 custom_fwd
RUN pip uninstall -y deepspeed || true

# Set working directory
WORKDIR /app

# Copy the distributed training script
COPY 3_llm_sft_long_distributed.py .

# Default command
CMD ["python3"]

FROM nvcr.io/nvidia/pytorch:24.09-py3

# Set environment variables for non-interactive installs and HuggingFace cache
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/data/huggingface_cache
ENV PYTHONUNBUFFERED=1
ENV NCCL_IB_DISABLE=0
ENV NCCL_DEBUG=INFO
ENV NCCL_ASYNC_ERROR_HANDLING=1

# Install required system packages and RDMA/Infiniband userspace libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    ca-certificates \
    ninja-build \
    rdma-core \
    ibverbs-utils \
    ibverbs-providers \
    libibverbs1 \
    librdmacm1 \
    infiniband-diags \
    iproute2 \
    net-tools \
    pciutils \
    && rm -rf /var/lib/apt/lists/*

# Install LLM required python packages
# Upgrade pip to avoid dependency resolution backtracking delays
# Pin transformers and accelerate to known stable versions for FSDP1
# Install ncloud-mlx first using the base image's pip/setuptools to avoid 'dist_info' errors
RUN pip install --no-cache-dir "ncloud-mlx[data-manager]"

# Upgrade pip and install remaining packages
RUN pip install --no-cache-dir --upgrade pip wheel && \
    pip install --no-cache-dir \
    "transformers>=4.50.0" \
    "accelerate>=0.34.2" \
    datasets \
    "peft>=0.10.0" \
    evaluate \
    scipy \
    "trl>=0.15.0" \
    "deepspeed>=0.14.4" \
    "qwen-vl-utils" \
    torchvision \
    packaging

# Install flash-attention separately because it requires ninja to build
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# Set working directory
WORKDIR /app

# Copy the distributed training script
COPY 3_vlm_sft_distributed.py .

# Default command
CMD ["python3"]

FROM registry.example.com/vlm-sft-qwen3_5-packed:ib

ENV HF_HOME=/data/huggingface_cache
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir --upgrade pip wheel && \
    pip install --no-cache-dir \
    "fastapi>=0.115.0" \
    "uvicorn>=0.34.0" \
    "httpx>=0.28.0" \
    "pillow>=11.0.0" \
    "safetensors>=0.5.0"

WORKDIR /app

COPY 1_openai_compatible_api.py /app/1_openai_compatible_api.py
COPY 5_eval_checkpoints.py /app/5_eval_checkpoints.py
COPY 10_eval_universal_prompt_checkpoints.py /app/10_eval_universal_prompt_checkpoints.py
COPY 11_eval_prompt_grid.py /app/11_eval_prompt_grid.py
COPY 9_collect_checkpoint_eval_summary.py /app/9_collect_checkpoint_eval_summary.py
COPY chat_ui.html /app/chat_ui.html

CMD ["python", "/app/1_openai_compatible_api.py", "--help"]

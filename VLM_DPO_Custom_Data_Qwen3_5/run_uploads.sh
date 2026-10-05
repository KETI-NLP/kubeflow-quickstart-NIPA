#!/bin/bash
export MLX_APIKEY="${MLX_APIKEY:?Set MLX_APIKEY in your environment}"
export MLX_ENDPOINT_URL="https://kpb4r.mlxp.ncloud.com/"
PROJECT="YOUR_WORKSPACE-h200-128"

echo "=========================================================="
echo "Starting Upload 2: qwen3_5_9b_multimodal_sft_dpo_normalized (final)"
echo "=========================================================="
python /app/9_upload_model_to_mlxp.py \
    --project $PROJECT \
    --model-name "qwen3_5_9b_multimodal_sft_dpo_normalized" \
    --version "v1.0" \
    --local-path "/data/checkpoints/upload_tmp_dpo_final" > /tmp/upload_2.log 2>&1

echo "Upload 2 completed. Check /tmp/upload_2.log for details."

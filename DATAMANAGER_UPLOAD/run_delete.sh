TARGETS=(
    data_korean_object_image_5_hallucination_normal_image_wrong_text_generation-hcx-so-packed-sft-ko
    data_korean_object_image_5_hallucination_normal_image_wrong_text_generation-qwen3-packed-sft-ko
    data_korean_object_image_4_hallucination_text_only_generation-hcx-seed-omni-packed-sft-ko
    data_korean_object_image_4_hallucination_text_only_generation-qwen3-packed-sft-ko
    data_korean_object_image_3_generation-hcx-seed-omni-packed-sft-ko	
    data_korean_object_image_3_generation-qwen3-packed-sft-ko
    data_030_korean_character_outside-hcx-seed-omni-packed-sft-ko
    data_030_korean_character_outside-qwen3-packed-sft-ko
    data_13_korean_character-hcx-seed-omni-packed-sft-ko
    data_13_korean_character-qwen3-packed-sft-ko
    data_030_korean_character_outside-ko
    data_13_korean_character-ko
    data_korean_object_image_3_generation-ko
    data_korean_object_image_4_hallucination_text_only_generation-ko
    data_korean_object_image_5_hallucination_normal_image_wrong_text_generation-ko
)
WORKSPACE="YOUR_WORKSPACE"
for ds in "${TARGETS[@]}"; do
    python3 delete_mlxp_repo.py \
      --workspace "$WORKSPACE" \
      --dataset-names "${ds}" \
      --force
done
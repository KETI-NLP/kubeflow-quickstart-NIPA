## Paired Raw/Packed Analysis

Test data:

- Raw: `/home/user/mlxp/test_data/data_korean_object_image_3_generation_raw/train`
- Packed: `/home/user/mlxp/test_data/data_korean_object_image_3_generation_qwen3_5_packed_sft_hf`

Observed schemas:

- Raw sample: `messages`, `images`
- Packed sample: `input_ids`, `labels`, `images`

Key findings:

1. The packed dataset is not a simple concatenation of the first raw samples.
   - Raw train length: `6287`
   - Packed length: `169`
   - Packed sample `0` contains `20` images and `8192` tokens.
   - This matches the provided FFD packing code structure.

2. The packed image-token counts are consistent with the raw replay path.
   - Replaying raw samples with the packing-time path:
     - `apply_chat_template(...)`
     - `process_vision_info(qwen_messages)`
     - `processor(text=[text], images=image_inputs, ...)`
   - For `packed[0]`, the summed raw replay image-token count matched exactly:
     - packed image tokens: `4160`
     - raw replay sum: `4160`
   - For `packed[1]`, the summed raw replay image-token count also matched exactly:
     - packed image tokens: `4128`
     - raw replay sum: `4128`

3. The current training-style processor path is what drifts.
   - Current training-style check:
     - `processor(text=[""], images=[sample_images], ...)`
   - Results on the paired packed data:
     - `packed[0]`: `4160 -> 4160` (ok)
     - `packed[1]`: `4128 -> 4144` (mismatch)
     - `packed[2]`: `4176 -> 4192` (mismatch)
     - `packed[3]`: `4192 -> 4208` (mismatch)

4. A concrete mismatch example:
   - `packed[1]`
   - packed token count: `4128`
   - current training-style feature count: `4144`
   - raw replay sum: `4128`

Conclusion:

- The packed dataset itself is not the broken part.
- The mismatch comes from recomputing vision features at training time with a processor path that is not equivalent to the packing-time path.
- In other words:
  - packing-time multimodal expansion is correct
  - packed `input_ids` image-pad counts are correct
  - training-time `processor(text=[""], images=[...])` is not reproducing the same per-sample feature counts for many packed samples

Most likely root cause:

- The original packing path depends on the full raw message structure (`messages` -> `apply_chat_template` -> `process_vision_info`).
- After packing, only `input_ids`, `labels`, and flattened packed `images` remain.
- That means the current training collator does not have enough original structure to faithfully reconstruct the same vision processing path just from packed samples.

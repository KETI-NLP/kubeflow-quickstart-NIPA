# Processor Path Comparison

검증 목적:
- 같은 이미지 샘플을 넣었을 때
  - 패킹 스타일 호출
    - `process_vision_info(qwen_messages)`
    - `processor(text=[packing_text], images=clean_image_inputs, ...)`
  - 학습 스타일 호출
    - `processor(text=[""], images=[images], ...)`
- 두 경로의 `image_grid_thw`, `feature_tokens`, `image_tokens`가 같은지 확인

검증 환경:
- Processor: `Qwen/Qwen3.5-27B`
- `spatial_merge_size=2`
- 더미 이미지 2장
  - image 1: `512x384`
  - image 2: `341x512`
- 입력 텍스트:

```text
Compare these two images.
<image>
Then this one.
<image>
```

결과:

```json
{
  "processor": "Qwen/Qwen3.5-27B",
  "spatial_merge_size": 2,
  "single_sample_two_images": {
    "input_text": "Compare these two images.\n<image>\nThen this one.\n<image>",
    "image_sizes": [
      [512, 384],
      [341, 512]
    ],
    "packing_style": {
      "input_len": 385,
      "image_tokens": 352,
      "image_grid_thw": [
        [1, 24, 32],
        [1, 32, 20]
      ],
      "feature_tokens": 352
    },
    "training_style": {
      "input_len": 0,
      "image_tokens": 0,
      "image_grid_thw": [
        [1, 24, 32],
        [1, 32, 22]
      ],
      "feature_tokens": 368
    }
  }
}
```

핵심 해석:
- 같은 이미지 2장을 넣었는데도 두 경로의 `image_grid_thw`가 달라졌다.
- 첫 번째 이미지는 동일:
  - `[1, 24, 32]`
- 두 번째 이미지는 다름:
  - 패킹 스타일: `[1, 32, 20]`
  - 학습 스타일: `[1, 32, 22]`
- 따라서 최종 `feature_tokens`도 달라졌다.
  - 패킹 스타일: `352`
  - 학습 스타일: `368`

결론:
- `text=[""]` 경로는 multi-image 샘플에서 패킹 시 processor 호출 결과를 재현하지 못할 수 있다.
- 현재 packed dataset mismatch의 유력 원인 중 하나는 이 processor 호출 방식 차이다.

추가 분리 실험 결과:

```json
{
  "A_text_packing__images_flat": {
    "input_len": 385,
    "image_tokens": 352,
    "image_grid_thw": [[1, 24, 32], [1, 32, 20]],
    "feature_tokens": 352
  },
  "B_text_packing__images_nested": {
    "input_len": 401,
    "image_tokens": 368,
    "image_grid_thw": [[1, 24, 32], [1, 32, 22]],
    "feature_tokens": 368
  },
  "C_text_empty__images_flat": {
    "input_len": 0,
    "image_tokens": 0,
    "image_grid_thw": [[1, 24, 32], [1, 32, 20]],
    "feature_tokens": 352
  },
  "D_text_empty__images_nested": {
    "input_len": 0,
    "image_tokens": 0,
    "image_grid_thw": [[1, 24, 32], [1, 32, 22]],
    "feature_tokens": 368
  }
}
```

분리 실험 해석:
- `flat` 이미지 입력인 `A`와 `C`는 `image_grid_thw`와 `feature_tokens`가 같았다.
- `nested` 이미지 입력인 `B`와 `D`는 서로 같았고, `flat` 입력과 달랐다.
- 따라서 이번 dummy 실험 기준으로는 `text=[packing_text]` vs `text=[""]`보다 **`images`를 `flat`으로 주는지 `nested`로 주는지**가 feature 수를 바꾼 핵심 요인이었다.

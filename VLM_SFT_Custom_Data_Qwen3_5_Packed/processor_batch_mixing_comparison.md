# Processor Batch Mixing Comparison

검증 목적:
- 같은 샘플 A를
  - 단독으로 `processor(text=[""], images=[sample_a], ...)`에 넣었을 때와
  - 다른 샘플 B와 함께 `processor(text=["", ""], images=[sample_a, sample_b], ...)`에 넣었을 때
- 샘플 A의 `image_grid_thw`와 `feature_tokens`가 달라지는지 확인

검증 환경:
- Processor: `Qwen/Qwen3.5-27B`
- `spatial_merge_size=2`
- 샘플 A 이미지 2장
  - `[512x384, 341x512]`
- 샘플 B 이미지 1장
  - `[512x512]`

결과:

```json
{
  "sample_a_image_sizes": [
    [512, 384],
    [341, 512]
  ],
  "sample_b_image_sizes": [
    [512, 512]
  ],
  "alone": {
    "image_grid_thw": [
      [1, 24, 32],
      [1, 32, 22]
    ],
    "feature_tokens": 368
  },
  "batched": {
    "image_grid_thw_all": [
      [1, 24, 32],
      [1, 32, 22],
      [1, 32, 32]
    ],
    "sample_a_prefix_rows": [
      [1, 24, 32],
      [1, 32, 22]
    ],
    "sample_a_feature_tokens_from_prefix": 368
  }
}
```

핵심 해석:
- 샘플 A를 단독으로 넣었을 때와 배치로 넣었을 때, 샘플 A의 prefix row는 동일했다.
- `feature_tokens`도 동일했다.
  - 단독: `368`
  - 배치 내 샘플 A: `368`

결론:
- 이 dummy 실험 기준으로는 `배치 혼합` 자체는 주원인으로 보이지 않는다.
- 현재 더 강한 원인 후보는:
  - 패킹 스타일 호출과 학습 스타일 호출의 차이
  - 그중에서도 **`images`를 `flat`으로 주는지 `nested`로 주는지** 차이

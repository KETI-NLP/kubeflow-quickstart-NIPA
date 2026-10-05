# Korean Heritage Text ShortQA Build Summary

소스:
- `/workspace/2026_llm_data_generation/korean_heritage_multimodal_hallucination_dpo/dpo_dataset_generated_text.json`

생성 파일:
- [korean_heritage_text_shortqa_candidates.json](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/tmp_eval_datasets/korean_heritage_text_shortqa_candidates.json)
- [korean_heritage_text_shortqa_benchmark.json](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/tmp_eval_datasets/korean_heritage_text_shortqa_benchmark.json)
- 생성 스크립트: [build_korean_heritage_text_shortqa.py](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/build_korean_heritage_text_shortqa.py)

선정 원칙:
- `type0_text_normal`만 사용
- 장문 설명형이 아니라 `단일 short-answer fact`로 안정적으로 추출 가능한 경우만 채택
- 문화재별 후보 fact를 여러 개 만든 뒤, 가장 신뢰도 높은 fact 하나를 benchmark용으로 선택
- 지나치게 애매한 `owner/manager` 같은 추출은 첫 버전에서 제외

최종 결과:
- candidate pool: `22,965`
- benchmark samples: `11,366`
- benchmark 기준 문화재 수: `11,366`

benchmark answer type 분포:
- `designation_date`: `5,751`
- `creation_year`: `3,784`
- `quantity`: `1,068`
- `designation_type`: `690`
- `material`: `73`

예시:

```text
질문: 화성 만의사 지장시왕도의 문화재 지정일은 언제인가?
정답: 2023년 8월 22일
타입: designation_date
```

```text
질문: 양헌수 승전비의 제작 또는 조성 시기는 언제인가?
정답: 1873년
타입: creation_year
```

```text
질문: 보성 문익점 부조묘와 고문서의 지정 수량은 얼마인가?
정답: 25점
타입: quantity
```

```text
질문: 임실 진구사지의 지정 종별은 무엇인가?
정답: 사적
타입: designation_type
```

```text
질문: 화성 만의사 지장시왕도의 바탕 재질은 무엇인가?
정답: 비단
타입: material
```

메모:
- 첫 버전은 `짧고 채점하기 쉬운 fact`를 최우선으로 했기 때문에 coverage보다 precision을 우선했다.
- 이후 benchmark task를 붙일 때는 `strict EM` 위주로 시작하고, 필요하면 날짜/수량 전용 정규화만 얹는 것이 적절하다.
- `designation_name` 타입은 text-only QA에서 질문에 정답이 노출되는 문제가 커서 최종 benchmark에서 제외했다.
- `material` 타입은 전수 검토 후 수동 보정과 제외를 일부 반영했다.
  - 수동 보정: `21`개
  - 제외: `2`개
  - `material` 문항은 70개 전수 재검토 후, 각 샘플에 대해 `재질만` 묻는 benchmark 질문을 수동 정의했다.

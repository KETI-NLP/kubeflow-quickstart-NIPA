# GPT-4o Korean Heritage Text ShortQA Smoke Summary

- output dir: [test_korean_heritage_text_shortqa_gpt4o_smoke_v1](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/test_korean_heritage_text_shortqa_gpt4o_smoke_v1)
- results: [results_2026-04-13T03-30-50.944371.json](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/test_korean_heritage_text_shortqa_gpt4o_smoke_v1/results/openai/gpt-4o/results_2026-04-13T03-30-50.944371.json)
- details: [details_korean_heritage_text_shortqa|0_2026-04-13T03-30-50.944371.parquet](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/test_korean_heritage_text_shortqa_gpt4o_smoke_v1/details/openai/gpt-4o/2026-04-13T03-30-50.944371/details_korean_heritage_text_shortqa%7C0_2026-04-13T03-30-50.944371.parquet)
- benchmark json: [korean_heritage_text_shortqa_benchmark.json](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/tmp_eval_datasets/korean_heritage_text_shortqa_benchmark.json)
- benchmark hf: [korean_heritage_text_shortqa_benchmark_hf](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/tmp_eval_datasets/korean_heritage_text_shortqa_benchmark_hf)

실행 설정:
- model: `openai/gpt-4o`
- task: `korean_heritage_text_shortqa|0`
- max samples: `100`
- custom task: [custom_korean_heritage_text_shortqa_task.py](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/custom_tasks/custom_korean_heritage_text_shortqa_task.py)

전체 점수:
- `text_shortqa_format_valid = 1.0000`
- `text_shortqa_empty_response = 0.0000`
- `text_shortqa_strict_em = 0.0700`
- `text_shortqa_relaxed_em = 0.0700`
- `text_shortqa_ordered_recall = 0.5504`
- `text_shortqa_ordered_precision = 0.5249`
- `text_shortqa_ordered_f1 = 0.5326`

Smoke 샘플 내 타입 분포:
- `designation_date`: `57`
- `creation_year`: `25`
- `quantity`: `15`
- `designation_type`: `3`

타입별 성능:
- `designation_date`: `strict_em = 0.0000`, `ordered_f1 = 0.5832`
- `creation_year`: `strict_em = 0.0800`, `ordered_f1 = 0.4589`
- `quantity`: `strict_em = 0.2667`, `ordered_f1 = 0.5033`
- `designation_type`: `strict_em = 0.3333`, `ordered_f1 = 0.3333`

예시:

```text
질문: 공주 충청감영 측우기의 제작 또는 조성 시기는 언제인가?
정답: 1837년
예측: Answer: 1837년
strict_em: 1.0
```

```text
질문: 구성동유적의 문화재 지정일은 언제인가?
정답: 1998년 7월 21일
예측: Answer: 2001년 1월 29일
strict_em: 0.0
ordered_f1: 0.5000
```

```text
질문: 재조본 경률이상 권1의 지정 수량은 얼마인가?
정답: 1권
예측: Answer: 1책
strict_em: 0.0
ordered_f1: 0.5000
```

```text
질문: 기림사약사전의 지정 종별은 무엇인가?
정답: 사적
예측: Answer: 보물
strict_em: 0.0
ordered_f1: 0.0000
```

메모:
- 형식 준수는 안정적이었다.
- 다만 날짜/연도 계열은 한국 문화재 지식이 아니라도 그럴듯한 다른 연도를 답하는 경향이 보였다.
- 첫 `100`개 smoke에는 `material`과 `designation_name`이 포함되지 않았다. 이 타입까지 보려면 `max-samples`를 더 늘려야 한다.

# GPT-4o Korean Heritage Text ShortQA Answer Examples

- benchmark json: [korean_heritage_text_shortqa_benchmark.json](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/tmp_eval_datasets/korean_heritage_text_shortqa_benchmark.json)
- eval results: [results_2026-04-13T03-30-50.944371.json](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/test_korean_heritage_text_shortqa_gpt4o_smoke_v1/results/openai/gpt-4o/results_2026-04-13T03-30-50.944371.json)
- details: [details_korean_heritage_text_shortqa|0_2026-04-13T03-30-50.944371.parquet](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/test_korean_heritage_text_shortqa_gpt4o_smoke_v1/details/openai/gpt-4o/2026-04-13T03-30-50.944371/details_korean_heritage_text_shortqa%7C0_2026-04-13T03-30-50.944371.parquet)

전체 요약:
- `text_shortqa_format_valid = 1.0000`
- `text_shortqa_strict_em = 0.0700`
- `text_shortqa_ordered_f1 = 0.5326`

## Correct Examples

```text
질문: 공주 충청감영 측우기의 제작 또는 조성 시기는 언제인가?
정답: 1837년
예측: Answer: 1837년
strict_em: 1.0
ordered_f1: 1.0
```

```text
질문: 경주 보문사지 석조의 지정 종별은 무엇인가?
정답: 보물
예측: Answer: 보물
strict_em: 1.0
ordered_f1: 1.0
```

```text
질문: 송조표전총류 권7의 지정 수량은 얼마인가?
정답: 1책
예측: Answer: 1책
strict_em: 1.0
ordered_f1: 1.0
```

```text
질문: 완주 송광사 소조사천왕상의 제작 또는 조성 시기는 언제인가?
정답: 1636년
예측: Answer: 1636년
strict_em: 1.0
ordered_f1: 1.0
```

## Wrong But Close

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
질문: 충주 윤양계 고택의 제작 또는 조성 시기는 언제인가?
정답: 1873년
예측: Answer: 1917년
strict_em: 0.0
ordered_f1: 0.6000
```

```text
질문: 강진 까막섬 상록수림의 문화재 지정일은 언제인가?
정답: 1966년 1월 13일
예측: Answer: 1962년 12월 3일
strict_em: 0.0
ordered_f1: 0.8000
```

```text
질문: 의성 대곡사 지장보살도의 문화재 지정일은 언제인가?
정답: 2010년 11월 11일
예측: Answer: 2010년 8월 5일
strict_em: 0.0
ordered_f1: 0.7000
```

## Wrong Examples

```text
질문: 기림사약사전의 지정 종별은 무엇인가?
정답: 사적
예측: Answer: 보물
strict_em: 0.0
ordered_f1: 0.0
```

```text
질문: 용흥사부도군의 제작 또는 조성 시기는 언제인가?
정답: 1717년
예측: Answer: 고려 후기
strict_em: 0.0
ordered_f1: 0.0
```

```text
질문: 진천 산수리 마애여래좌상의 지정 수량은 얼마인가?
정답: 3구
예측: Answer: 1기
strict_em: 0.0
ordered_f1: 0.0
```

```text
질문: 사월의 지정 수량은 얼마인가?
정답: 2책
예측: Answer: 정보가 제공되지 않아 알 수 없음
strict_em: 0.0
ordered_f1: 0.0
```

```text
질문: 오봉사지부도의 제작 또는 조성 시기는 언제인가?
정답: 1677년
예측: Answer: 고려시대 후기
strict_em: 0.0
ordered_f1: 0.0
```

```text
질문: 문경 봉암사 봉황문의 지정 수량은 얼마인가?
정답: 1동
예측: Answer: 2점
strict_em: 0.0
ordered_f1: 0.0
```

## Quick Read

- 형식은 잘 지키지만, `지정일`과 `제작연도`에서는 실제 정답이 아니라 다른 그럴듯한 날짜를 말하는 경우가 많았습니다.
- `수량`은 상대적으로 맞는 경우가 더 있었지만, `1권/1책`, `1동/2점`처럼 단위가 바뀌는 실수가 보였습니다.
- `지정 종별`은 sample 수가 적긴 하지만, `사적`과 `보물`을 혼동하는 사례가 보였습니다.

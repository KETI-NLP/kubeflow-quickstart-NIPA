# GPT-4o Korean Heritage Text ShortQA Prompt Comparison

비교 대상:
- v1: [test_korean_heritage_text_shortqa_gpt4o_smoke_v1](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/test_korean_heritage_text_shortqa_gpt4o_smoke_v1)
- v2 prompt: [test_korean_heritage_text_shortqa_gpt4o_smoke_v2_prompt](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/test_korean_heritage_text_shortqa_gpt4o_smoke_v2_prompt)
- task: [custom_korean_heritage_text_shortqa_task.py](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/custom_tasks/custom_korean_heritage_text_shortqa_task.py)

전체 점수 변화:
- `strict_em`: `0.0700 -> 0.0900`
- `ordered_f1`: `0.5326 -> 0.5951`
- `format_valid`: `1.0000 -> 1.0000`

바뀐 점:
- `creation_year`는 `YYYY년`만 답하게 제약
- `designation_date`는 `YYYY년 M월 D일`만 답하게 제약
- `quantity`는 `숫자+단위`만 답하게 제약
- `designation_type`, `material`도 타입별 예시와 금지 예시를 추가

질문별 비교:

```text
질문: 오봉사지부도의 제작 또는 조성 시기는 언제인가?
정답: 1677년

v1 예측: Answer: 고려시대 후기
v1 strict_em: 0.0
v1 ordered_f1: 0.0

v2 예측: Answer: 1687년
v2 strict_em: 0.0
v2 ordered_f1: 0.8000
```

```text
질문: 재조본 경률이상 권1의 지정 수량은 얼마인가?
정답: 1권

v1 예측: Answer: 1책
v1 strict_em: 0.0
v1 ordered_f1: 0.5000

v2 예측: Answer: 1권
v2 strict_em: 1.0
v2 ordered_f1: 1.0000
```

```text
질문: 구성동유적의 문화재 지정일은 언제인가?
정답: 1998년 7월 21일

v1 예측: Answer: 2001년 1월 29일
v1 strict_em: 0.0
v1 ordered_f1: 0.5000

v2 예측: Answer: 2001년 1월 29일
v2 strict_em: 0.0
v2 ordered_f1: 0.5000
```

해석:
- `quantity`처럼 출력 형식과 단위 혼동이 문제였던 경우는 프롬프트 개선 효과가 있었다.
- `creation_year`도 `고려시대 후기` 같은 서술형 오답을 `1687년`처럼 채점 가능한 형식으로 바꾸는 효과가 있었다.
- 하지만 `designation_date`처럼 모델이 아예 다른 날짜를 알고 있다고 믿고 답하는 경우는 프롬프트만으로 잘 교정되지 않았다.

다음 추천:
- `designation_date`는 prompt 개선과 별도로 few-shot 예시를 더 넣거나,
- smoke 기준으로는 `quantity`와 `creation_year`를 우선 핵심 지표로 보고,
- 날짜류는 더 큰 샘플에서 별도 분석하는 편이 낫다.

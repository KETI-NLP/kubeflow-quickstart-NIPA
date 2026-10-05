# GPT-4o Korean Heritage Name VQA LCS Eval Summary

실험:
- task: `korean_heritage_name_vqa`
- model: `openai/gpt-4o`
- setting: smoke run 20 samples
- output dir: [test_korean_heritage_name_vqa_gpt4o_smoke_v3_lcs](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/test_korean_heritage_name_vqa_gpt4o_smoke_v3_lcs)
- results: [results_2026-04-13T01-57-51.917246.json](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/test_korean_heritage_name_vqa_gpt4o_smoke_v3_lcs/results/openai/gpt-4o/results_2026-04-13T01-57-51.917246.json)

추가한 metric:
- `name_vqa_ordered_recall`
- `name_vqa_ordered_precision`
- `name_vqa_ordered_f1`

정의:
- 정답과 예측을 정규화한 뒤 문자 단위 `LCS` 길이를 계산
- `ordered_recall = LCS / len(gold)`
- `ordered_precision = LCS / len(pred)`
- `ordered_f1 = 2PR / (P + R)`

결과:
- `name_vqa_format_valid = 1.0000`
- `name_vqa_empty_response = 0.0000`
- `name_vqa_strict_em = 0.0000`
- `name_vqa_relaxed_em = 0.0000`
- `name_vqa_ordered_recall = 0.2296`
- `name_vqa_ordered_precision = 0.2542`
- `name_vqa_ordered_f1 = 0.2187`

해석:
- 완전일치 기준에서는 여전히 0점이다.
- 하지만 ordered metric으로 보면, 평균적으로 정답 문자열의 약 23% 정도는 순서를 유지한 채 예측 안에 살아 있다고 볼 수 있다.
- 즉 완전한 문화재명 식별은 잘 안 되지만, 부분적으로 이름 조각을 맞추거나 비슷한 계열 명칭을 내는 경향은 일부 보인다.

메모:
- 이번 재실행은 같은 모델 출력에 대해 metric만 확장해서 다시 계산한 케이스다.
- `LiteLLM` 캐시가 응답을 재사용했기 때문에 실행이 매우 빨랐다.

예시:

1. near-miss에 가까운 케이스

```text
gold: 구례 화엄사 각황전 앞 석등
pred: 화엄사 각황전 앞 석등
ordered_recall: 0.8182
ordered_precision: 1.0000
ordered_f1: 0.9000
```

설명:
- 지역명 `구례`만 빠졌고 나머지 핵심 명칭은 순서를 유지한 채 거의 그대로 들어 있다.

2. 정답 전체는 보존됐지만 불필요한 접미가 붙은 케이스

```text
gold: 석남역사
pred: 석남역사소설
ordered_recall: 1.0000
ordered_precision: 0.6667
ordered_f1: 0.8000
```

설명:
- 정답 문자열 전체는 예측 안에 포함되어 있다.
- 다만 `소설`이 붙어서 precision이 깎인다.

3. 일부 문자열만 우연히 겹치는 케이스

```text
gold: 채화칠기
pred: 국보 제78호 나전칠기합
ordered_recall: 0.5000
ordered_precision: 0.1818
ordered_f1: 0.2667
```

설명:
- `칠기` 같은 일부 조각이 겹쳐 recall은 약간 올라가지만, 실제 정답이라고 보기엔 precision이 낮다.

4. 계열은 비슷하지만 실제 명칭은 다른 케이스

```text
gold: 영원사목불좌상및복장유물
pred: 금동여래좌상
ordered_recall: 0.1667
ordered_precision: 0.3333
ordered_f1: 0.2222
```

설명:
- 불상 계열이라는 점은 비슷하지만 이름 차이가 커서 score가 낮다.

5. 완전 오답 케이스

```text
gold: 초조본 아비담팔건도론 권24
pred: 무구정광대다라니경
ordered_recall: 0.0000
ordered_precision: 0.0000
ordered_f1: 0.0000
```

설명:
- 순서를 유지한 공통 문자열이 사실상 없어 0점으로 떨어진다.

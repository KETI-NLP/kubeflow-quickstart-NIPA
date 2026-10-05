# Korean Heritage Text ShortQA Methodology

- source dataset: `/workspace/2026_llm_data_generation/korean_heritage_multimodal_hallucination_dpo/dpo_dataset_generated_text.json`
- builder script: [build_korean_heritage_text_shortqa.py](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/build_korean_heritage_text_shortqa.py)
- outputs:
  - [korean_heritage_text_shortqa_candidates.json](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/tmp_eval_datasets/korean_heritage_text_shortqa_candidates.json)
  - [korean_heritage_text_shortqa_benchmark.json](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/tmp_eval_datasets/korean_heritage_text_shortqa_benchmark.json)
- example inspection file: [korean_heritage_text_shortqa_source_vs_benchmark_examples.md](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/korean_heritage_text_shortqa_source_vs_benchmark_examples.md)

## Goal

목표는 원본 DPO-style text QA 데이터에서, 장문 설명형 QA를 그대로 평가에 쓰는 대신 `짧고 채점 가능한 fact QA`만 뽑아 benchmark용 JSON으로 재구성하는 것입니다.

핵심 아이디어:
- 원본 `source_query`와 `source_chosen`에서 특정 `fact slot`만 추출
- 답을 단답형으로 정규화
- 문화재별로 여러 후보가 생기면 가장 신뢰도 높은 것 하나만 benchmark용으로 선택

이 작업은 LLM을 사용하지 않았고, 전부 규칙 기반으로 수행했습니다.

## High-Level Pipeline

전체 흐름은 아래 순서입니다.

1. 대용량 JSON을 `ijson`으로 스트리밍 읽기
2. `qa_error_type == "type0_text_normal"`만 사용
3. 각 row에서 `heritage_name`, `user_query`, `chosen`, `rejected`를 정규화
4. `source_query + source_chosen`에 대해 여러 extractor를 순서대로 적용
5. 추출된 `(heritage_name, answer_type, answer)`를 candidate로 저장
6. 같은 의미 후보는 dedupe
7. 문화재별 후보 중 우선순위가 가장 높은 1개를 benchmark sample로 선택
8. JSON 저장

구현 위치:
- source path / output path: [build_korean_heritage_text_shortqa.py:25](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/build_korean_heritage_text_shortqa.py:25)
- 스트리밍 reader: [build_korean_heritage_text_shortqa.py:343](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/build_korean_heritage_text_shortqa.py:343)
- main loop: [build_korean_heritage_text_shortqa.py:364](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/build_korean_heritage_text_shortqa.py:364)

## Step 1. Row Filtering

원본 데이터 전체를 쓰지 않고, 먼저 아래 조건을 만족하는 row만 대상으로 삼았습니다.

- `qa_error_type == "type0_text_normal"`
- `heritage_name`, `user_query`, `chosen`이 비어 있지 않음

구현:
- [build_korean_heritage_text_shortqa.py:375](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/build_korean_heritage_text_shortqa.py:375)
- [build_korean_heritage_text_shortqa.py:386](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/build_korean_heritage_text_shortqa.py:386)

이유:
- `type1_text_false_premise`는 거짓 전제를 포함하므로 1차 factual short-answer benchmark에는 부적합
- 비어 있는 질문/정답은 추출 신뢰도가 낮음

## Step 2. Text Normalization

각 row에서 먼저 문자열을 정리했습니다.

정규화 대상:
- `heritage_name`
- `user_query`
- `chosen`
- `rejected`

주요 처리:
- 공백 축소
- 따옴표 정규화
- 문화재명 끝 괄호 제거

예:
- `영천 향사당 입규 현판 (永川 鄕射堂 立規 懸板)` -> `영천 향사당 입규 현판`

구현:
- 공백 축소: [build_korean_heritage_text_shortqa.py:133](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/build_korean_heritage_text_shortqa.py:133)
- 문화재명 정규화: [build_korean_heritage_text_shortqa.py:141](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/build_korean_heritage_text_shortqa.py:141)
- 괄호 제거: [build_korean_heritage_text_shortqa.py:150](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/build_korean_heritage_text_shortqa.py:150)
- 정답 문자열 정리: [build_korean_heritage_text_shortqa.py:181](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/build_korean_heritage_text_shortqa.py:181)

## Step 3. Fact Extraction from `source_query` + `source_chosen`

핵심은 `source_query`와 `source_chosen`을 같이 보고, 특정 short-answer fact만 뽑는 것입니다.

원칙:
- 질문에서 특정 사실을 묻는지 먼저 확인
- 답변(`chosen`)에서 정답 문자열을 정규식 또는 안전한 어휘 목록으로 추출
- 하나의 row에서 여러 타입이 추출될 수도 있음

현재 활성화된 extractor:
- `designation_date`
- `creation_year`
- `quantity`
- `holding_institution`
- `designation_type`
- `material`
- `mounting_format`

등록 위치:
- [build_korean_heritage_text_shortqa.py:329](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/build_korean_heritage_text_shortqa.py:329)

### 3.1 `designation_date`

질문이 `지정일`, `등록일`, `지정 날짜` 등을 묻는지 확인한 뒤, 답변에서 `YYYY년 M월 D일` 패턴을 뽑습니다.

구현:
- [build_korean_heritage_text_shortqa.py:219](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/build_korean_heritage_text_shortqa.py:219)

예:
- source_query: `이 고문서들이 유형문화재로 지정된 날짜를 확인해 주세요.`
- source_chosen: `... 1988년 12월 21일에 전라남도 유형문화재로 지정되었습니다.`
- answer: `1988년 12월 21일`

### 3.2 `creation_year`

질문이 `제작/조성/건립/창건/작성 연도`를 묻는지 보고, 답변에서 첫 번째 `YYYY년`을 뽑습니다.

구현:
- [build_korean_heritage_text_shortqa.py:228](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/build_korean_heritage_text_shortqa.py:228)

주의:
- 답변에 여러 연도가 있으면 첫 번째 연도를 선택하는 구조라, 문맥상 더 적합한 뒤 연도를 놓칠 수 있습니다.

예:
- `무장향교는 1420년에 창건되었고, 대성전은 1842년에 중건되었습니다.`
- 현재 추출 answer: `1420년`

이 케이스는 benchmark 질문과의 정합성이 애매할 수 있습니다.

### 3.3 `quantity`

질문에 `수량`, `몇 권`, `몇 점`, `몇 기` 같은 표현이 있으면 답변에서 `숫자+단위`를 추출합니다.

구현:
- [build_korean_heritage_text_shortqa.py:242](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/build_korean_heritage_text_shortqa.py:242)

허용 단위 예:
- `점`
- `권`
- `책`
- `구`
- `동`
- `기`
- `그루`

예:
- `총 3점의 수묵산수화가 수록되어 있습니다.` -> `3점`
- `총 2기의 무덤이 조성되어 있습니다.` -> `2기`

주의:
- 원문 답변이 `1권 1첩`처럼 복합 수량이면 단일 answer로 축약될 수 있습니다.

### 3.4 `holding_institution`

질문이 `소장 기관`을 묻고, 답변에 `...에서 소장` 형태가 있을 때 기관명을 뽑습니다.

구현:
- [build_korean_heritage_text_shortqa.py:251](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/build_korean_heritage_text_shortqa.py:251)

현재 benchmark 최종본에서는 거의 채택되지 않았고, 1차 버전에선 보수적으로 배제된 경우가 많습니다.

### 3.5 `designation_type`

질문이 `종별`, `분류`, `유형문화재`, `보물`, `국보`, `사적`, `명승` 등을 포함하면, 답변 안에서 `SAFE_LEGAL_TYPES` 목록에 있는 표현을 찾습니다.

구현:
- 안전한 종별 목록: [build_korean_heritage_text_shortqa.py:39](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/build_korean_heritage_text_shortqa.py:39)
- extractor: [build_korean_heritage_text_shortqa.py:302](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/build_korean_heritage_text_shortqa.py:302)

예:
- 답변에 `... 사적 제390호 ... 석조 자체는 보물 ...` 둘 다 있을 수 있음
- 이 경우 질문 문맥에 따라 의도와 다른 타입이 잡힐 수 있으므로 검수가 필요함

이 타입은 성능도 괜찮은 샘플이 있지만, 원문 질문과 benchmark 질문 간 간극이 생기기 쉬운 대표 유형입니다.

### 3.6 `material`

질문이 `바탕 재질`, `바탕 재료`, `재질은 무엇` 등을 묻는지 보고, 답변 안에서 제한된 재질 목록을 찾습니다.

구현:
- [build_korean_heritage_text_shortqa.py:311](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/build_korean_heritage_text_shortqa.py:311)

초기 자동 추출 목록:
- `비단`
- `종이`
- `한지`
- `목재`
- `나무`
- `석재`
- `가죽`

문제:
- 이 방식은 `비교 대상 재질`이나 `문장 속 일반명사`를 잘못 잡을 수 있습니다.
- 예: `명주 바탕은 종이보다 ...`에서 `종이`를 잘못 추출

그래서 `material`은 전수 검토 후 수동 보정을 추가했습니다.

수동 보정/제외:
- 수동 보정 dict: [build_korean_heritage_text_shortqa.py:81](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/build_korean_heritage_text_shortqa.py:81)
- 제외 set: [build_korean_heritage_text_shortqa.py:106](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/build_korean_heritage_text_shortqa.py:106)
- 적용 위치: [build_korean_heritage_text_shortqa.py:393](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/build_korean_heritage_text_shortqa.py:393)

추가 처리:
- `material`은 70개 전수 재검토 후, 각 샘플에 대해 `재질만` 묻는 benchmark 질문을 수동 정의
- 구현:
  - 질문 override dict: [build_korean_heritage_text_shortqa.py:112](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/build_korean_heritage_text_shortqa.py:112)
  - 적용 위치: [build_korean_heritage_text_shortqa.py:477](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/build_korean_heritage_text_shortqa.py:477)

이유:
- 원문 질문 그대로 유지하면 `영향`, `조성 시기 + 재질`, `크기 + 재질` 같은 복합 질문이 남아 benchmark 정답과 불일치할 수 있었음
- 그래서 현재는 `성안의 초상의 바탕 재질은 무엇인가?`, `나주 영천사 목조아미타여래좌상의 재질은 무엇인가?`처럼 정답과 정확히 맞는 질문으로 수동 정리함

### 3.7 `mounting_format`

질문이 `장황 형식`을 묻고 답변에 `...형 장황`이 있으면 추출합니다.

구현:
- [build_korean_heritage_text_shortqa.py:320](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/build_korean_heritage_text_shortqa.py:320)

현재 최종 benchmark에서는 거의 채택되지 않았습니다.

### Removed Type: `designation_name`

초기 버전에는 `designation_name` extractor가 있었지만, 최종 text-only benchmark에서는 제외했습니다.

이유:
- 질문 안에 문화재명이 이미 들어가면 정답이 노출되는 경우가 많았음
- 실제 확인 결과 `designation_name` 샘플 37개 중 상당수가 `source_query` 또는 `benchmark_question`에 정답이 직접 포함되어 있었음
- text-only 조건에서는 이 타입이 공정한 평가 문항이 되기 어렵다고 판단함

## Step 4. Candidate Construction

추출이 성공하면 candidate 하나를 만듭니다.

candidate 필드:
- `heritage_name`
- `heritage_name_full`
- `answer_type`
- `answer`
- `benchmark_question`
- `source_query`
- `source_chosen`
- `source_rejected`
- `qa_error_type`
- `hallucination_type`
- `paraphrase_count`
- `source_queries`

구현:
- dataclass: [build_korean_heritage_text_shortqa.py:187](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/build_korean_heritage_text_shortqa.py:187)
- 생성 위치: [build_korean_heritage_text_shortqa.py:409](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/build_korean_heritage_text_shortqa.py:409)

중요:
- benchmark 질문은 보통 `QUESTION_TEMPLATES`로 새로 작성
- 단, `material`은 원본 질문 그대로 유지

## Step 5. Dedupe and Paraphrase Aggregation

같은 문화재에서 같은 타입, 같은 답이 여러 번 나오면 하나의 candidate로 합칩니다.

dedupe key:
- `(heritage_name, answer_type, answer)`

구현:
- key 생성: [build_korean_heritage_text_shortqa.py:403](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/build_korean_heritage_text_shortqa.py:403)

동일 key가 다시 나오면:
- `paraphrase_count += 1`
- `source_queries`에 다른 질문을 최대 10개까지 누적

구현:
- [build_korean_heritage_text_shortqa.py:425](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/build_korean_heritage_text_shortqa.py:425)

의미:
- 같은 fact를 paraphrase 여러 개로 묻는 경우를 하나의 candidate fact로 합침

## Step 6. Choosing One Benchmark Sample per Heritage

한 문화재에서 여러 fact 후보가 생기면, benchmark에는 하나만 남깁니다.

선택 기준은 `TYPE_PRIORITY`, `paraphrase_count`, `answer length`입니다.

우선순위:
- `designation_date`: 100
- `creation_year`: 90
- `quantity`: 80
- `holding_institution`: 75
- `designation_name`: 70
- `designation_type`: 65
- `material`: 60
- `mounting_format`: 55

구현:
- priority table: [build_korean_heritage_text_shortqa.py:186](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/build_korean_heritage_text_shortqa.py:186)
- 선택 함수: [build_korean_heritage_text_shortqa.py:353](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/build_korean_heritage_text_shortqa.py:353)

정확한 sort key:
- 높은 `type priority`
- 높은 `paraphrase_count`
- 짧은 `answer`

즉, 문화재별로 후보가 여러 개면
- 더 안전한 타입을 우선
- 더 자주 반복되는 fact를 우선
- 답이 지나치게 긴 경우는 덜 선호

이 방식 때문에:
- `designation_date`나 `creation_year`가 자주 benchmark 대표 샘플이 됨
- `material`은 후보가 있어도 덜 자주 최종 채택됨

## Step 7. Final Output Structure

최종 benchmark JSON은 아래 구조를 가집니다.

```json
{
  "metadata": {
    "task_name": "korean_heritage_text_shortqa_benchmark",
    "selection_policy": "one best high-confidence short-answer fact per heritage",
    "benchmark_count": 11382,
    "benchmark_type_counts": {
      "designation_date": 5751,
      "creation_year": 3784,
      "quantity": 1068,
      "designation_type": 672,
      "designation_name": 37,
      "material": 70
    }
  },
  "samples": [
    {
      "heritage_name": "...",
      "answer_type": "...",
      "answer": "...",
      "benchmark_question": "...",
      "source_query": "...",
      "source_chosen": "..."
    }
  ]
}
```

저장 위치:
- candidate payload: [build_korean_heritage_text_shortqa.py:442](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/build_korean_heritage_text_shortqa.py:442)
- benchmark payload: [build_korean_heritage_text_shortqa.py:456](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/build_korean_heritage_text_shortqa.py:456)

## Why Some Benchmark Questions Look Different from Source

원본 `source_query`는 paraphrase가 섞여 있고, 여러 사실을 한 번에 묻는 경우도 많았습니다.

그래서 benchmark에선 보통 아래처럼 바꿨습니다.

예 1:
- source_query: `만의사 지장시왕도의 문화재 지정 명칭과 지정일을 알려주세요.`
- source_chosen: `지정 명칭은 ... 2023년 8월 22일에 ... 지정되었습니다.`
- benchmark_question: `화성 만의사 지장시왕도의 문화재 지정일은 언제인가?`
- answer: `2023년 8월 22일`

예 2:
- source_query: `공주 충청감영 측우기가 제작된 시기는 언제이며 어느 임금 때입니까?`
- source_chosen: `1837년(헌종 3)에 제작되었습니다.`
- benchmark_question: `공주 충청감영 측우기의 제작 또는 조성 시기는 언제인가?`
- answer: `1837년`

예 3:
- source_query: `성안의 초상의 바탕 재료가 종이가 아닌 명주인 점이 작품에 어떤 영향을 주나요?`
- source_chosen: `명주 바탕은 ...`
- benchmark_question: `성안의 초상의 바탕 재질은 무엇인가?`
- answer: `명주`

이렇게 변환한 이유:
- 한 문제에 한 정답만 남기기 위해
- 모델 간 비교가 쉬운 short-answer benchmark로 만들기 위해

## Known Failure Modes

이 방식은 빠르고 재현 가능하지만, 의미 해석을 완전히 하진 않기 때문에 아래 문제가 생길 수 있습니다.

### 1. Question-target mismatch

원문 질문은 A를 묻는데, benchmark 질문은 B로 일반화되거나 좁혀질 수 있습니다.

예:
- 원문: `무장향교 창건일과 대성전 중건 연도`
- benchmark: `무장향교대성전의 제작 또는 조성 시기`
- answer: `1420년`

### 2. Answer simplification loss

원문 답에 `1권 1첩`처럼 복합 정보가 있는데, benchmark answer는 `1권`만 남을 수 있습니다.

예:
- `재조본 경률이상 권1`

### 3. Keyword-triggered false extraction

질문이나 답변에 특정 키워드가 들어 있어 extractor가 잘못 반응할 수 있습니다.

예:
- `사적기`의 `사적`을 `designation_type=사적`으로 잘못 받아들일 가능성
- `종이보다`의 `종이`를 실제 재질로 잘못 추출하는 경우

### 4. Multiple facts in one answer

`source_chosen`에 여러 연도, 여러 수량, 여러 대상이 있으면 현재 규칙은 일부만 택합니다.

### 5. Heritage-level selection bias

문화재별로 하나만 남기기 때문에, 다른 좋은 fact가 있어도 benchmark 대표 샘플에서 빠질 수 있습니다.

## Why We Chose Rules Instead of LLM Rewriting

이번 버전은 LLM을 쓰지 않고 규칙 기반으로 만들었습니다.

장점:
- 빠름
- 비용 없음
- 재현 가능
- 대량 처리 가능

단점:
- 의미 해석이 얕음
- 문맥상 더 적절한 fact를 놓칠 수 있음
- 질문과 benchmark 타깃 간 불일치가 생길 수 있음

즉, 현재 데이터셋은 `precision-first v1`이라고 보는 것이 맞습니다.

## Practical Review Guidance

검수할 때는 아래 순서로 보는 것이 좋습니다.

1. `source_query`가 실제로 그 fact를 묻고 있는가
2. `source_chosen`에서 `answer`가 직접적으로 뽑히는가
3. `benchmark_question`이 `source_query`의 의미를 과도하게 바꾸지 않았는가
4. `answer`가 너무 과도하게 축약되지는 않았는가
5. 특히 `designation_type`, `quantity`, `material`은 문맥 오해가 없는지 확인

검수용 샘플:
- [korean_heritage_text_shortqa_source_vs_benchmark_examples.md](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/korean_heritage_text_shortqa_source_vs_benchmark_examples.md)

## Suggested Next Improvements

다음 버전에서 개선할 수 있는 방향:

- `designation_type`의 trigger를 더 보수적으로 바꾸기
- `creation_year`에서 여러 연도 중 어느 연도를 뽑을지 문맥 규칙 강화
- `quantity`의 복합 표기(`1권 1첩`) 처리 개선
- `material`과 `designation_type`에 대해 추가 수동 검수
- 검수 후 `clean_v2` benchmark를 별도 저장

현재 상태에서의 추천 해석:
- `designation_date`, `creation_year`, `quantity`는 비교적 쓸 만한 축
- `designation_type`과 일부 `material`은 검수 후 쓰는 편이 안전함

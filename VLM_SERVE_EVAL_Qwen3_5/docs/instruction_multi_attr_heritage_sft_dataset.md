# 한국 문화재 multi-attribute SFT 데이터셋 생성 지시서

이 문서는 **벤치마크 생성 에이전트**에 전달하기 위한 instruction입니다. 목표는 `Qwen/Qwen3.5-9B` (base) 의 한국 문화재 지식 능력을 끌어올리는 SFT 학습 데이터셋을 만드는 것입니다.

---

## 1. 목적

`Qwen/Qwen3.5-9B` (base) 가 보유한 한계를 보강할 SFT 데이터셋을 생성합니다. 평가 결과 분석은 [qwen3.5-9b_base_mc_insights_for_training_data.md](./qwen3.5-9b_base_mc_insights_for_training_data.md) 참고. 핵심 목표 3가지:

1. **format 도달률 ↑** — `Answer: <키워드>` 로 끝나는 짧은 답안 패턴 학습
2. **multi-task** — 한 가지 답안 패턴(예: 이름만)으로 collapse하지 않도록 5속성 + 설명을 함께 학습
3. **`year` / `location` 약점 보완** — 지정 연도·소재지에 대한 fact recall 학습

평가에서 base 모델의 약점:

| 카테고리 | strict_mc_match (MC, 100 샘플) |
|---|---:|
| category | 0.70 |
| era | 0.625 |
| location | 0.348 |
| year | 0.061 |
| **all** | **0.39** |

`year` / `location` 두 카테고리를 끌어올리는 데 데이터셋 구성이 집중되어야 합니다.

---

## 2. 원본 자원

빌드는 기존 패턴을 따릅니다. 다음 자산을 그대로 활용:

| 자산 | 경로 |
|---|---|
| 기존 빌드 스크립트 (참고) | [scripts/build_korean_heritage_name_vqa.py](../scripts/build_korean_heritage_name_vqa.py), [scripts/build_korean_heritage_name_vqa_hf.py](../scripts/build_korean_heritage_name_vqa_hf.py) |
| Heritage raw source | 기존 빌드 스크립트가 사용하는 동일 source (예: 문화재청 API / 캐시 JSON) |
| 이미지 캐시 | [tmp_eval_datasets/korean_heritage_name_vqa_image_cache/](../tmp_eval_datasets/korean_heritage_name_vqa_image_cache/) (그대로 재사용) |
| 평가 데이터 (참고용 record 구조) | [tmp_eval_datasets/korean_heritage_knowledge_hf/](../tmp_eval_datasets/korean_heritage_knowledge_hf/), [tmp_eval_datasets/korean_heritage_knowledge_mc_hf/](../tmp_eval_datasets/korean_heritage_knowledge_mc_hf/) |
| **평가 sample에 *포함된 heritage*는 학습 데이터에서 제외** | 위 두 평가셋의 `sample_id` / `heritage_name` 으로 필터링 |

⚠️ **train/eval leakage 금지** — 평가 데이터셋에 들어간 heritage 항목은 학습 셋에서 빼야 합니다. 자세히는 §9 "데이터 분리" 참조.

---

## 3. 생성 단위 — heritage 1개당 11 샘플

각 heritage 는 다음 정보를 가짐 (현 평가 데이터와 동일 schema):

```python
{
  "heritage_id": str,
  "heritage_name": str,         # 예: "보은 선병우 고가 (報恩 宣炳禹 古家)"
  "era": str,                   # 예: "조선"
  "designation_year": int,      # 예: 1986
  "location": str,              # 광역시·도, 예: "충북"
  "category": str,              # 상위 카테고리, 예: "등록문화유산"
  "image_path": str,            # 절대경로
  "description": str,           # 한 문단 설명 (위키/문화재청 캡션 등에서 추출)
}
```

이 한 heritage 로 다음 **11개 single-turn 학습 샘플** 생성:

| # | 형식 | 카테고리 | 예시 질문 |
|---|---|---|---|
| 1 | 주관식 | 명칭 | "사진 속 문화재의 명칭은?" |
| 2 | 주관식 | 시대 | "이 문화재가 만들어진 시대는?" |
| 3 | 주관식 | 연도 | "이 문화재의 지정 연도는?" |
| 4 | 주관식 | 위치 | "이 문화재의 소재지(광역시·도)는?" |
| 5 | 주관식 | 카테고리 | "이 문화재의 상위 카테고리는?" |
| 6 | 객관식 | 명칭 | "명칭은? A. ... B. ... C. ... D. ..." |
| 7 | 객관식 | 시대 | "시대는? A. ... B. ... C. ... D. ..." |
| 8 | 객관식 | 연도 | "지정 연도는? A. ... B. ... C. ... D. ..." |
| 9 | 객관식 | 위치 | "소재 광역시·도? A. ... B. ... C. ... D. ..." |
| 10 | 객관식 | 카테고리 | "상위 카테고리? A. ... B. ... C. ... D. ..." |
| 11 | 자연문 설명 | (collapse 방지) | "이 문화재를 한 문단으로 설명해 주세요." |

→ **heritage 1,000개 × 11 샘플 = 11,000 SFT 레코드** 가 초기 목표 규모. 가능하면 더 많아도 됩니다.

---

## 4. 샘플별 상세 스펙

### 4.1 공통 형식

모든 샘플은 **single-turn**:

```json
{
  "image": "<image_path>",
  "messages": [
    {"role": "user",      "content": "<prompt>"},
    {"role": "assistant", "content": "<short reasoning + Answer: X>"}
  ],
  "metadata": {
    "heritage_id": "...",
    "attr_type": "name|era|year|location|category|description",
    "qa_format": "free|mc|description",
    "answer_keyword": "보은 선병우 고가",
    "answer_letter": null
  }
}
```

prompt와 답안 모두 **한국어**가 기본 (모델이 영어로 새지 않게).

### 4.2 주관식 (샘플 1~5)

**prompt template** (속성별):

```
[이미지] 이 사진에 보이는 문화재에 대해 답해 주세요.

질문: {질문 본문}

반드시 마지막 줄을 다음 형식으로 끝내세요:
Answer: <짧은 정답 키워드>
```

`질문 본문` 예시:

- 명칭: "이 문화재의 명칭은 무엇입니까?"
- 시대: "이 문화재는 어느 시대에 만들어졌습니까? (예: 조선)"
- 연도: "이 문화재가 문화재로 지정된 연도는 몇 년입니까? (4자리 서기 연도)"
- 위치: "이 문화재의 소재지를 광역시·도 단위로 답해 주세요."
- 카테고리: "이 문화재의 상위 카테고리는 무엇입니까? (예: 유적건조물, 기록유산)"

**assistant 응답 template** (200~500 토큰 이내):

```
{브리프 시각 관찰 1~2문장}
{브리프 추론 1~3문장}

Answer: {정답 키워드}
```

예시 (명칭, 보은 선병우 고가):

```
사진은 충북 보은 지역의 전통 한옥 마을 풍경입니다. 산자락에 자리 잡은 대규모 양반가 한옥으로, 보은 선병우 고가의 전형적인 외관과 일치합니다.

Answer: 보은 선병우 고가
```

예시 (지정 연도):

```
이 사진은 보은 선병우 고가입니다. 등록문화유산으로 1986년 지정되었습니다.

Answer: 1986
```

**중요**: reasoning 첫 문장에 **heritage 식별**을 명시해야 함 (chain of thought 의 첫 step). 그 후 그 식별에 기반한 attribute 답.

### 4.3 객관식 (샘플 6~10)

**prompt template**:

```
[이미지] 이 사진에 보이는 문화재에 대해 답해 주세요.

{질문 본문}
A. {옵션 A}
B. {옵션 B}
C. {옵션 C}
D. {옵션 D}

다음을 읽고 정답을 고르시오. 응답의 마지막 줄은 반드시 'Answer: $LETTER' 형식이어야 합니다. $LETTER 는 A, B, C, D 중 하나입니다.
```

**옵션 배치 규칙**:
- 정답 letter 위치는 A·B·C·D 에 **고르게 분포**되도록 (한 카테고리 안에서 25%씩)
- 정답을 항상 같은 위치에 두면 모델이 위치 학습으로 cheat 가능 → 반드시 무작위
- 옵션 텍스트는 짧고 명확하게 (예: "A. 1993", "A. 조선", "A. 충북")

**assistant 응답 template**:

```
{브리프 시각 관찰 1~2문장}
{각 옵션 비교 1~3문장}

Answer: {정답 letter}
```

예시 (시대, 한벽루, gold=C 조선):

```
사진은 제천 청풍 한벽루입니다. 누각 양식과 단청 형태로 보아 조선 시대 건축물입니다.
A 삼국이나 B 발해 시대 건축은 양식이 다르고, D 현대도 명백히 아닙니다.

Answer: C
```

### 4.4 자연문 설명 (샘플 11)

**prompt**:

```
[이미지] 이 사진에 보이는 문화재를 한 문단(3~5문장)으로 설명해 주세요. 명칭, 시대, 위치, 분류, 지정 연도, 특징을 포함해 주세요.
```

**assistant 응답** (200~600 토큰):

자연문으로 명칭·시대·위치·카테고리·지정 연도·특징을 포함한 1문단. `Answer:` 라인 *없음* (이 샘플의 목적은 free-form 능력 유지).

예시:

```
보은 선병우 고가는 충북 보은에 위치한 조선 후기의 전통 한옥으로, 1986년에 등록문화유산으로 지정되었습니다. 산자락에 자리 잡은 대규모 양반가 건축으로, 안채·사랑채·행랑채가 잘 보존되어 있어 조선시대 상류층 주거 양식을 연구하는 데 중요한 자료입니다.
```

→ 이 샘플은 **collapse 방지용**. 모델이 `Answer: X` 패턴에만 fit 하지 않도록 free-form description을 함께 학습.

---

## 5. 객관식 distractor 생성 규칙

distractor(오답 옵션 3개) 의 품질이 객관식 학습의 핵심. 너무 쉬우면 학습 효과가 낮고, 너무 비슷하면 노이즈가 됨. 카테고리별 규칙:

### 5.1 명칭 distractor

- **같은 카테고리** 문화재 중에서 선택 (예: 정답이 한옥이면 다른 한옥들)
- 정답과 **이름 어휘가 일부 겹치는 hard negative** 1개 포함 (예: 정답 "보은 선병우 고가" → distractor "보은 우당고택")
- 가능하면 같은 시대·같은 광역시·도 출신 문화재로 채워서 시각적 유사 confusion 학습

### 5.2 시대 distractor

- 정답을 포함한 **시대 풀**: `삼국`, `통일신라`, `발해`, `고려`, `조선`, `대한제국`, `일제강점기`, `현대`
- 정답 외 무작위 3개 선택
- 정답이 `조선` 이면 인접 시대(`고려`, `대한제국`)를 한 개 이상 포함하면 hard

### 5.3 지정 연도 distractor

- 정답 연도 ± 5~20년 범위에서 3개 (전부 그럴듯한 다른 연도)
- 예: 정답 1986 → distractors `[1993, 1999, 2020]` 같은 식
- 너무 동떨어진 연도(예: 1500년)는 피함 — 시각 단서로도 구분 가능해 학습 효과 ↓
- year 옵션은 **항상 4자리 서기 연도**

### 5.4 위치(광역시·도) distractor

- 정답 외 무작위 3개를 한국 광역시·도 17개 중에서 선택:
  ```
  서울, 부산, 대구, 인천, 광주, 대전, 울산, 세종,
  경기, 강원, 충북, 충남, 전북, 전남, 경북, 경남, 제주
  ```
- 정답과 **인접 광역시·도** 1개 포함 (예: 정답 충북 → distractor 충남)

### 5.5 카테고리 distractor

- 카테고리 풀: `유적건조물`, `유물`, `기록유산`, `자연유산`, `등록문화유산`, `무형유산`
- 정답 외 무작위 3개

---

## 6. 출력 형식

JSONL 1줄 = 1 SFT 레코드:

```json
{"image": "<abs_path>", "messages": [...], "metadata": {...}}
```

파일 출력 경로 (제안):

```
tmp_eval_datasets/korean_heritage_multi_attr_sft.jsonl
```

추가로 HF Arrow 변환본도 함께 만들면 lighteval 등에서 재사용 편리:

```
tmp_eval_datasets/korean_heritage_multi_attr_sft_hf/
```

빌드 스크립트는 [scripts/build_korean_heritage_name_vqa.py](../scripts/build_korean_heritage_name_vqa.py) → `_hf.py` 패턴을 그대로 따라:

- `scripts/build_korean_heritage_multi_attr_sft.py` (raw → JSONL)
- `scripts/build_korean_heritage_multi_attr_sft_hf.py` (JSONL → HF Arrow)

---

## 7. 데이터 셔플 / collapse 방지

빌드 시 11샘플을 **heritage별로 묶지 말 것**. 최종 JSONL은 attribute·format·heritage 가 골고루 섞이게:

- heritage 단위로 11 샘플 생성 후 → **전체 풀에서 무작위 셔플** → 그 순서대로 JSONL 출력
- 학습 시 dataloader 가 다시 셔플하지만, 한 epoch 안의 순서를 미리 섞어두면 디버깅·분석 편리
- 같은 heritage의 11샘플이 연속해서 등장하면 *학습 step 단위로 multi-task 다양성*이 깨짐 (한 step에 같은 이미지만 들어옴)

---

## 8. 품질 체크 rubric

생성 후 자체 검증 통과해야 함:

### 8.1 reasoning 길이

- 주관식·객관식 reasoning **200~500 토큰** 사이가 90% 이상
- 1,000 토큰 초과는 5% 미만 (긴 thinking 학습 방지)

### 8.2 `Answer: X` 라인

- 주관식: 마지막 줄이 `Answer: <키워드>` 형식 — 100%
- 객관식: 마지막 줄이 `Answer: [A-D]` 정확히 한 글자 — 100%
- 자연문 설명: `Answer:` 라인 **없음** — 100%

### 8.3 반복 패턴 금지

생성된 응답에서 같은 6~30글자 substring 이 5회 이상 연속 반복되면 reject. 정규식:

```python
re.search(r"(.{6,30})\1{4,}", response)
```

### 8.4 정답 일관성 (sanity check)

- 명칭 답안의 `answer_keyword` == `heritage_name`
- 객관식 정답 letter 의 텍스트 == `answer_keyword`
- year 답안은 4자리 숫자 문자열만

### 8.5 옵션 letter 분포 (객관식)

같은 attribute 카테고리 안에서 정답 letter 의 A·B·C·D 분포가 20~30% 사이. (편향되면 모델이 letter 위치 학습)

### 8.6 distractor 품질

- 객관식 distractor 3개가 모두 정답과 **다르면서** **그럴듯**한지 — 자동 검사 어려우면 LLM judge 로 sample 100건 정도 spot check.

---

## 9. 데이터 분리 (train/eval leakage 방지)

다음 평가 데이터셋에 **포함된 `sample_id` 또는 `heritage_name`** 은 학습 데이터에서 **제외**:

| 평가 데이터셋 | 경로 | 제외 기준 |
|---|---|---|
| korean_heritage_name_vqa | [tmp_eval_datasets/korean_heritage_name_vqa_hf/](../tmp_eval_datasets/korean_heritage_name_vqa_hf/) | `heritage_name` 일치 시 제외 |
| korean_heritage_knowledge | [tmp_eval_datasets/korean_heritage_knowledge_hf/](../tmp_eval_datasets/korean_heritage_knowledge_hf/) | `heritage_name` 일치 시 제외 |
| korean_heritage_knowledge_mc | [tmp_eval_datasets/korean_heritage_knowledge_mc_hf/](../tmp_eval_datasets/korean_heritage_knowledge_mc_hf/) | `heritage_name` 일치 시 제외 |

빌드 스크립트 첫 단계에 이 필터를 명시적으로 호출:

```python
def load_eval_heritage_names() -> set[str]:
    names = set()
    for ds_path in [
        "tmp_eval_datasets/korean_heritage_name_vqa_hf",
        "tmp_eval_datasets/korean_heritage_knowledge_hf",
        "tmp_eval_datasets/korean_heritage_knowledge_mc_hf",
    ]:
        ds = datasets.load_from_disk(ds_path)
        names.update(r["heritage_name"] for r in ds)
    return names
```

---

## 10. 산출물 (제출 형식)

생성 에이전트는 다음을 함께 제출:

1. **데이터셋 파일**: `tmp_eval_datasets/korean_heritage_multi_attr_sft.jsonl` + `_hf/`
2. **빌드 스크립트**: `scripts/build_korean_heritage_multi_attr_sft.py` (+ `_hf.py`)
3. **데이터셋 README**: `tmp_eval_datasets/korean_heritage_multi_attr_sft_README.md` 에 다음 통계 보고:
   - 총 레코드 수 + heritage 수
   - attribute별 / format별 분포
   - 응답 길이 분포 (평균·median·95th percentile)
   - 객관식 정답 letter 분포 (per attribute)
   - 평가셋 heritage 누락 확인 결과
4. **샘플 미리보기**: 무작위 20건 (각 attribute·format 골고루) full text 를 README 안에 inline

---

## 11. 검증 (이 데이터로 학습 후 무엇을 측정?)

학습 1차 라운드 후 동일 100샘플 MC + 100샘플 주관식으로 측정해 다음 지표를 비교:

| metric | base 현재 | 목표 |
|---|---:|---:|
| `heritage_mc_format_valid` | 0.60 | ≥ 0.90 |
| `strict_mc_match` (MC) | 0.39 | ≥ 0.60 |
| `heritage_mc_year` | 0.06 | ≥ 0.25 |
| `heritage_mc_location` | 0.35 | ≥ 0.50 |
| `heritage_knowledge_em` (주관식) | 0.08 | ≥ 0.25 |
| `korean_heritage_name_vqa` (judge) | 10% | ≥ 30% |
| `mmbench_gen` (collapse 감시) | 0.59 | 유지 (>0.55) |
| `scienceqa_gen` (collapse 감시) | 0.77 | 유지 (>0.70) |

마지막 두 개 (mmbench, scienceqa) 가 떨어지면 = collapse 신호. 이 경우 sample 11(자연문 설명) 비중 ↑ 또는 일반 VLM 태스크 데이터 추가 mixing 필요.

---

## 부록: 참고 문서

- [qwen3.5-9b_base_mc_insights_for_training_data.md](./qwen3.5-9b_base_mc_insights_for_training_data.md) — 평가 데이터 분석 + 학습 데이터 디자인 정당화
- [qwen3.5-9b_base_heritage_knowledge_qa_samples.md](./qwen3.5-9b_base_heritage_knowledge_qa_samples.md) — base 모델의 실제 응답 패턴 10건 full

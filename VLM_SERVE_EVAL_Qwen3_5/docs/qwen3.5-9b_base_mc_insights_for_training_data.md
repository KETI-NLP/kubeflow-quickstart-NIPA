# 객관식 평가에서 얻은 학습 데이터 생성 인사이트

`Qwen/Qwen3.5-9B` (base) 의 `korean_heritage_knowledge_mc` 평가(`generation_size=8192`, 100샘플)와 동일 문화재의 주관식 평가를 비교 분석해, 다음 SFT/DPO 라운드에 어떤 데이터를 만들어야 점수를 끌어올릴 수 있는지 정리.

샘플 자료는 [qwen3.5-9b_base_heritage_knowledge_qa_samples.md](./qwen3.5-9b_base_heritage_knowledge_qa_samples.md) 참고.

---

## 1. 정량적 요약

### 1.1 메트릭 표

| metric | 주관식 (free-form) | 객관식 (MC, gen=8192) | 비고 |
|---|---:|---:|---|
| format_valid | 0.170 | 0.600 | 답안 라인 도달률 |
| 전체 정답률 | 0.080 | 0.390 | knowledge_em / strict_mc_match |
| 답 도달 시 정답률 | 47% (8/17) | **65% (39/60)** | conditional accuracy |
| 4지선다 무작위 | — | 0.250 | MC 정답률은 random+14 pp |

### 1.2 카테고리별 (MC, 100건)

| category | n | hits | format_valid (해당 카테고리) | hits rate | conditional acc | 의미 |
|---|---:|---:|---:|---:|---:|---|
| `category` | 20 | 14 | 14/20 = 70% | **70.0%** | **100%** | 답만 정하면 무조건 맞음 — 시각 단서 강함 |
| `era` | 24 | 15 | 18/24 = 75% | 62.5% | 83% | 한복·기와 등 시각 단서로 시대 추정 가능 |
| `location` | 23 | 8 | 14/23 = 61% | 34.8% | 57% | 광역 단위 visual cue 약함 |
| `year` | 33 | 2 | 14/33 = 42% | **6.1%** | **14%** | random(25%) 미만 — 순수 외부지식, 시각 무력 |

### 1.3 응답 분포 (raw)

| | 주관식 (gen=256) | 객관식 (gen=8192) |
|---|---:|---:|
| 평균 응답 길이 | 1,521 char | 9,955 char |
| 최대 응답 길이 | 2,064 char | 30,492 char |
| `Answer:` 라인 도달 | 17/100 | 54/100 |
| `<think>`/`</think>` 토큰 사용 | (불가, 토큰 한도) | 54/100 |
| 반복 루프 의심 | 4/100 | 2/100 |

### 1.4 year 카테고리 letter 매트릭스

| | pred=A | B | C | D | (none) |
|---|---:|---:|---:|---:|---:|
| gold=A (n=7) | **1** | 1 | 0 | 0 | 5 |
| gold=B (n=12) | 1 | **1** | 3 | 2 | 5 |
| gold=C (n=7) | 1 | 0 | **0** | 0 | 6 |
| gold=D (n=7) | 1 | 0 | 1 | **0** | 5 |

→ 21/33 (64%) 가 답 도달 실패. 답 도달한 12건 중 정답 단 **2건** (gold=A·B에서 각 1). letter 편향은 작지만, **knowledge 자체가 없어서 추측조차 제대로 못 함**. random(25%)에 못 미치는 이유는 *분모에 (none) = 자동 오답이 포함*되기 때문.

### 1.5 자주 등장한 hallucinated heritage 후보

오답 응답들에서 "이 (사진/문화재) 은 X" 식 패턴으로 잡힌 잘못된 식별 단어 빈도:

| 후보 단어 | 빈도 | 비고 |
|---|---:|---|
| **삼신도** | 416 | 한 응답 내에서 수십 회 반복되며 사실상 anchor 됨 |
| 목마라 | 144 | 동일 — 응답 안에서 자가 반복 |
| 한산대첩비 | 133 | 동일 |
| 삼존도 | 116 | 동일 |
| 국립중앙박물관 | 92 | 위치 추측의 default 답안 |
| 대장경일람 | 70 | 책류 heritage에 대한 default 추측 |

→ 모델은 *모르는 상태에서 친숙한 한국 문화재 이름을 luminous answer로 들이댄다*. 같은 단어가 한 응답 안에서 100+ 번 반복되는 패턴 = self-repetition/loop.

---

## 2. 핵심 진단

### 2.1 점수 계층 구조

```
strict_mc_match (0.39)
   ≤ format_valid_per_category 의 합 (0.60)
       ≤ 답 도달 시 정답률 × 형식 도달률
```

세 가지 병목이 곱셈으로 작용:
1. **답 도달률** (format_valid) — thinking이 너무 길어서 commit 못 함
2. **올바른 heritage 식별** — 잘못 식별하면 모든 attribute QA 오답
3. **fact 자체** (특히 year) — 시각으로 안 풀림, 외부 지식 필요

### 2.2 카테고리별 약점 분해

- **category, era**: 시각 단서로 풀리는 *visual taxonomy* 문제. 현재도 비교적 잘함 → 큰 향상 여지 작음.
- **location**: 광역 단위라 한정된 visual cue → **광역시/도 식별을 위한 추가 데이터** 필요.
- **year**: 거의 순수 *factual recall* → **(heritage, year) 외워야 하는** 문제.

---

## 3. 학습 데이터 생성 인사이트 (실행 권장)

> 각 항목: **무엇이** / **왜** / **어떻게 만들지**

### 인사이트 1: name VQA 단독 SFT는 역효과

- **무엇**: `simple_name_sft` 는 base 대비 `heritage_knowledge` 모든 메트릭에서 **하락** (knowledge_em 0.08→0.05, category 0.20→0.15, era 0.04→0.00 등).
- **왜**: name VQA 한 가지 패턴 ("이 유물의 명칭은 X입니다") 에 collapse → 다른 형식의 답을 할 능력 상실.
- **데이터 정책**: **multi-task SFT**. 한 heritage 당 최소 다섯 가지 질문 변형을 묶어 학습:
  1. 이름 식별: "이 사진의 문화재는?"
  2. 시대: "만들어진 시대는?"
  3. 위치: "소재 광역시/도?"
  4. 분류: "상위 카테고리?"
  5. 연도: "지정 연도는?"
  6. 일반 설명: "이 문화재를 설명해 주세요" (general VLM 능력 유지)
- 각 답은 명확한 형식(`Answer: <키워드>`)으로 끝나도록 통일.

### 인사이트 2: 답 도달률(format_valid)이 가장 큰 단기 이득

- **무엇**: 60건이 형식 도달, 그 중 65%가 정답. 도달률만 60→90%로 끌어도 strict_mc_match ≈ 0.39·(0.90/0.60) ≈ **0.59** 까지 자동 상승.
- **왜**: 모델이 thinking을 멈추고 commit 하는 패턴을 못 학습. 토큰 한도 안에서 답 라인 못 찍음.
- **데이터 정책**:
  - **짧은-thinking 데모**: reasoning 200~500 토큰 + `</think>` + `Answer: X` 패턴의 SFT 예시 대량.
  - **No-thinking 직답 데모**도 섞기: "Answer: X" 직답 + 그 뒤 한 문장 설명.
  - 길이 제어 학습: 같은 질문에 short/long 답 변형.

### 인사이트 3: year/location 은 text-only knowledge로

- **무엇**: 4개 옵션이 모두 그럴듯한 연도/광역 → 시각으로 풀 수 없음.
- **왜**: 모델은 (heritage_name → year/location) factoid 자체를 모름.
- **데이터 정책**:
  - **이미지 없는 텍스트 SFT** 도 추가: `"보은 선병우 고가의 지정 연도는?" → "1986"`.
  - 다양한 phrasing: "X는 몇 년에 지정?", "X 등록 연도", "X의 보물 지정일자" 등.
  - 광역시/도도 동일 패턴.
- 이렇게 하면 image-conditional path와 fact-recall path가 모두 학습됨.

### 인사이트 4: hard-negative heritage pair로 시각 disambiguation

- **무엇**: "한벽루" 사진을 "화랑대"로 식별, "건칠반"을 "울산박물관 칠기"로 식별. 비슷한 이름·외형이 confusion 유발.
- **왜**: 시각적 유사 heritage 사이를 구분하는 contrastive 학습이 부재.
- **데이터 정책**:
  - 같은 카테고리 내 비슷한 heritage 쌍 (정자·누각 류, 불상 류, 고서적 류) 으로 **(anchor image, name) + (hard negative name)** 트리플 생성.
  - SFT 예시: "이 사진은 ___입니다. (혼동되는 ___ 이/가 아닙니다.)" 형식.

### 인사이트 5: 자가 반복 루프 억제 — 다양성 있는 SFT 데이터

- **무엇**: "삼신도" 416회, "다마도리(玉造)가 아니라" 무한 반복. 응답이 같은 후보를 anchor 잡고 100+회 반복.
- **왜**: temperature=0 + 빈약한 SFT → 모델이 uncertainty 상태에서 같은 단어로 회귀.
- **데이터 정책**:
  - SFT 응답에 **반복 페턴 금지** rubric 적용.
  - "Uncertainty 표명 후 commit" 데모: "확실하지 않으나 ~로 추정합니다. Answer: X".
  - DPO 단계에서 *반복적 응답*을 reject pair로 쓰기.

### 인사이트 6: 한국어 컨텍스트인데 영문/한자로 추론

- **무엇**: 응답 다수가 영문 thinking ("Let's reconsider…", "Hwaseong Haenggung?") 으로 진행되다가 commit 못 함. 한자(`玉造`)로 빠지는 경우도.
- **왜**: base의 일반 multilingual reasoning bias.
- **데이터 정책**: 한국어 reasoning trace 데이터 가중. 한자 음역은 *마지막 답안*에만 허용 (예시 답에 `(漢字)` 부속).

### 인사이트 7: MC 옵션을 활용한 reasoning 패턴을 free-form에 이식

- **무엇**: MC 정답률 0.39 vs 주관식 0.08. 옵션 4개를 제거하기만 해도 큰 격차.
- **왜**: MC에서 모델은 "A는 ~여서 제외, B는 ~여서 제외" 식 plausibility ranking을 사용 → 비록 약하지만 commit.
- **데이터 정책**:
  - **MC reasoning을 free-form 답으로 변환한 페어**:
    - MC prompt: "A.1993 B.1999 **C.1986** D.2020 → Answer: C"
    - Free-form prompt: "지정 연도는?" + 답 "1986" + reasoning "(1993·1999·2020도 가능했지만 1986년 지정)".
  - 모델에게 "옵션이 없어도 plausibility ranking을 떠올리도록" SFT.

### 인사이트 8: heritage_name 식별이 모든 attribute QA의 prerequisite

- **무엇**: 객관식·주관식 양쪽에서 모델이 heritage를 잘못 식별하면 era/year/location 답도 따라 틀림.
- **왜**: attribute는 (heritage, attr) lookup. heritage가 틀리면 attribute도 틀림.
- **데이터 정책**:
  - **2-step CoT 데모**: "(1) 식별: 이 사진은 X 입니다. (2) X의 시대는 Y 입니다. Answer: Y."
  - 학습 시 *식별 단계가 틀리면* 전체 응답 페널티 (DPO 단계).

### 인사이트 9: random 미만 (year=6%) → 추측 단계도 못 함

- **무엇**: 4지선다 무작위 25%인데 6%. 답 도달 21/33 실패 + 도달한 12건 중 2건만 적중.
- **왜**: knowledge가 없을 때 모델이 "I don't know → don't answer" 로 가서 commit 못 함. 답 도달한 경우도 *jittered guess*.
- **데이터 정책**:
  - **불확실성 표명 + 강제 commit 데모**: "정확한 연도는 모르나 옵션 중 가장 가능성 높은 것을 고르면 ~". MC 가이드에는 "모를 때도 한 글자 골라라" 강제.
  - Temperature 다양화로 SFT 데이터 (지금은 모두 T=0 으로 같은 패턴).

### 인사이트 10: thinking 토큰 효율 — short reasoning 데이터

- **무엇**: MC 응답 평균 10K char, 최대 30K char. 길이 늘려서 점수 얻는 것은 토큰 비용 12배.
- **왜**: 모델이 "왜?" 를 무한히 자가 반박. commit 학습 부족.
- **데이터 정책**:
  - **gen=1024 안에 답이 끝나는 짧은 reasoning** 예시 비중 ↑.
  - "최대 N 토큰" 컨디셔닝: 학습 prompt에 `[max 500 tokens]` 지시 포함.

---

## 4. 우선순위 권장 (Effort vs Score Lift)

| 우선순위 | 데이터 | 예상 lift (strict_mc_match) | 노력 |
|---|---|---:|---|
| ★★★ | 형식 도달률을 60→90%로 — short answer SFT | +0.20 | 낮음 (생성 자동화 쉬움) |
| ★★★ | year/location text-only factoid | +0.10 (year 6→25% 가정) | 중 (DB에서 직접 추출) |
| ★★ | hard-negative heritage pair (contrastive) | +0.05~0.10 | 중 (유사도 기반 쌍 생성) |
| ★★ | multi-task SFT (5+1 형식 묶음) | +0.10 (분산 효과) | 중-높 (질문 다양화) |
| ★ | MC→Free-form reasoning 이식 | +0.05 | 낮음 (자동 변환) |
| ★ | 반복 루프 DPO 페어 | +0.03 | 낮음 (현재 응답에서 자동 마이닝) |

총합 가정: 점수 0.39 → **~0.65** 까지 무리 없이 가능. 단 simple_name_sft 처럼 한 가지로 치우치면 다시 collapse 위험 — 균형 유지가 핵심.

---

## 5. 검증 루프 (eval-driven iteration)

각 데이터 라운드 후 동일 100샘플 MC + 100샘플 주관식으로 측정. 다음 지표를 모니터링:

- `heritage_mc_format_valid` — 답 도달률
- `heritage_mc_year` — knowledge recall 능력
- `heritage_knowledge_em` (주관식) — open-ended 능력
- `korean_heritage_name_vqa` (judge 재채점) — name 식별이 살아있나
- `mmbench_gen` / `scienceqa_gen` 등 일반 VLM 태스크 — collapse 방지 모니터

> `simple_name_sft` 처럼 *원하는 한 가지가 올랐지만 나머지가 사라진* 패턴을 빠르게 감지.

---

## 6. 자료 출처

- 평가 raw: [test_final_11_tasks_endpoint/Qwen_Qwen3.5-9B/korean_heritage_knowledge_mc/](../test_final_11_tasks_endpoint/Qwen_Qwen3.5-9B/korean_heritage_knowledge_mc/)
- 동일 5쌍 응답 비교: [qwen3.5-9b_base_heritage_knowledge_qa_samples.md](./qwen3.5-9b_base_heritage_knowledge_qa_samples.md)
- 태스크 정의:
  - [custom_tasks/custom_korean_heritage_knowledge_mc_task.py](../custom_tasks/custom_korean_heritage_knowledge_mc_task.py)
  - [custom_tasks/custom_korean_heritage_knowledge_task.py](../custom_tasks/custom_korean_heritage_knowledge_task.py)
- 모델: `Qwen/Qwen3.5-9B` (HF) · vLLM 0.21.0 DP=8 · `generation_size=8192`

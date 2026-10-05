# 이미지 robustness 벤치마크 — perturbation severity 정의

`korean_heritage_knowledge_mc` robustness 변형(noise/brightness/rotation/shift/all)의 **severity 값이 실제로 얼마나 센지** 정리한 문서. 코드는 [custom_tasks/custom_korean_heritage_knowledge_mc_task.py](../custom_tasks/custom_korean_heritage_knowledge_mc_task.py) 의 `_perturb_image()`.

severity는 환경변수 `HERITAGE_MC_PERTURB_SEVERITY` 로 조절 (기본 2, 범위 1~5).

---

## 1. severity별 변형 강도 (공식)

각 변형 타입의 강도는 `severity` 에 선형 비례합니다.

| 변형 | 공식 | sev 1 | sev 2 | sev 3 | sev 4 | sev 5 |
|---|---|---:|---:|---:|---:|---:|
| **noise** (가우시안) | σ = severity × 12 (0~255 스케일) | σ=12 | σ=24 | σ=36 | σ=48 | σ=60 |
| **brightness** (밝기) | ±(severity × 12)% | ±12% | ±24% | ±36% | ±48% | ±60% |
| **rotation** (회전) | ±(severity × 4)° | ±4° | ±8° | ±12° | ±16° | ±20° |
| **shift** (평행이동) | ±(severity × 2.5)% (가로·세로 각각) | ±2.5% | ±5% | ±7.5% | ±10% | ±12.5% |
| **all** | 위 4개를 noise→brightness→rotation→shift 순서로 *모두* 적용 | — | — | — | — | — |

### 강도 감 잡기

- **noise σ=24 (sev2)**: 픽셀값이 평균적으로 ±24 (0~255 중 약 ±9%) 흔들림 → 화질 거친 사진 수준.
- **noise σ=48 (sev4)**: ±19% → 상당히 거친 노이즈. 미세 글자·문양이 뭉개짐.
- **brightness ±48% (sev4)**: 거의 2배 밝거나 절반 어두움.
- **rotation ±16° (sev4)**: 눈에 띄게 기운 사진. 빈 모서리는 회색(128,128,128)으로 채움.
- **shift ±10% (sev4)**: 이미지가 가로/세로로 10%씩 밀려 잘리고 회색 여백 생김.

> 주의: 실제 평가는 위 **± 범위 안에서 무작위 추출**한 값을 적용합니다 (rotation·shift는 uniform, brightness는 부호 ±1 무작위, noise는 가우시안). seed는 이미지 경로 해시로 고정 → 같은 이미지엔 항상 같은 변형 (재현 가능).

---

## 2. 시각 비교 (severity 0~5, 최대 강도 기준)

아래 그리드는 한 문화재 이미지(영덕 장륙사 건칠관음보살좌상)에 변형을 **최대 강도로** 적용한 모습. 행 = 변형 타입, 열 = severity (clean / 1 / 2 / 3 / 4 / 5).

![severity grid](perturbation_examples/severity_grid.jpg)

(실제 평가는 ± 범위 내 무작위라 sev N의 평균 강도는 위 최대치보다 약함. 위 그리드는 "sev N에서 *최악의 경우* 얼마나 망가지나"를 보여줌.)

### noise — clean / sev2 / sev4 확대 비교

| clean | sev 2 (σ=24) | sev 4 (σ=48) |
|---|---|---|
| ![](perturbation_examples/noise_sev0.jpg) | ![](perturbation_examples/noise_sev2.jpg) | ![](perturbation_examples/noise_sev4.jpg) |

### all (4종 동시) — clean / sev2 / sev4 확대 비교

| clean | sev 2 | sev 4 |
|---|---|---|
| ![](perturbation_examples/all_sev0.jpg) | ![](perturbation_examples/all_sev2.jpg) | ![](perturbation_examples/all_sev4.jpg) |

---

## 3. severity가 점수에 미친 실제 영향 (`qwen3_5_9b_multi_attr_sft`)

같은 모델·같은 100샘플에서 severity 2 vs 4 의 `strict_mc_match`:

| perturbation | clean | sev 2 | sev 4 | sev4 Δclean |
|---|---:|---:|---:|---:|
| clean (기준) | 0.860 | 0.860 | 0.860 | — |
| rotation | | 0.830 | 0.840 | −0.02 |
| shift | | 0.840 | 0.820 | −0.04 |
| brightness | | 0.810 | 0.790 | −0.07 |
| noise | | 0.790 | **0.670** | **−0.19** |
| **all** | | 0.780 | **0.580** | **−0.28** |

해석:
- **noise가 severity에 가장 민감** — σ 24→48 로 키우니 −0.07 → −0.19. 미세 시각 디테일을 가장 많이 파괴.
- **rotation·shift는 sev4에서도 거의 영향 없음** — 문화재의 전역 구조(형태/윤곽)는 기하 변형에 강건.
- **all(4종 동시)이 가장 가혹** — sev4에서 0.86 → 0.58 (−0.28). 주 기여는 noise.
- 무너지는 카테고리는 일관되게 **year / location** (fine-grained fact). era / category는 변형에 둔감.

---

## 4. 권장 severity 사용

| 목적 | severity | 비고 |
|---|---|---|
| 가벼운 robustness 확인 (기본) | 2 | 일상적 화질 저하 수준 |
| 스트레스 테스트 | 4 | 명확한 degradation, 모델 약점 부각 |
| 극단 (최대) | 5 | 사람도 식별 어려울 정도 — 하한 측정용 |

실행 예:

```bash
HERITAGE_MC_PERTURB_SEVERITY=4 \
ONLY_TASKS="korean_heritage_knowledge_mc_noise korean_heritage_knowledge_mc_bright korean_heritage_knowledge_mc_rotate korean_heritage_knowledge_mc_shift korean_heritage_knowledge_mc_allperturb" \
OPENAI_BASE_URL=http://127.0.0.1:8003/v1 \
bash scripts/run_endpoint_eval.sh <model_id>
```

> ⚠️ severity를 바꿔 재실행할 때는 **lighteval 캐시를 먼저 비워야** 합니다. 캐시 키가 텍스트 doc 해시 기준이라, 안 지우면 이전 severity의 생성 결과를 그대로 재사용해 이미지 변형이 무시됩니다:
> ```bash
> rm -rf ~/.cache/huggingface/lighteval/openai/<model>/*/korean_heritage_knowledge_mc_*
> ```
> 또 같은 task명이라 결과 디렉토리가 덮어쓰여지니, 비교가 필요하면 이전 결과를 먼저 백업하세요.

---

## 5. 생성 방법 (이 문서의 그리드)

`docs/perturbation_examples/severity_grid.jpg` 는 `_perturb_image()` 와 동일한 공식으로 최대 강도를 적용해 생성. 다른 이미지로 다시 만들려면 source 경로만 바꿔 재생성하면 됩니다 (생성 스니펫은 이 문서 작성 시 사용한 PIL 코드 참고).

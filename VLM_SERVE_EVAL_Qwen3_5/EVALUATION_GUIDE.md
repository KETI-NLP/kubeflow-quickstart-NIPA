# VLM 벤치마크 실행 가이드 (Qwen3.5 VLM)

이 문서는 학습한 Qwen3.5 VLM 체크포인트를 **평가 완료**까지 돌리는 두 가지 경로를 정리합니다.

| 경로 | 모델 로딩 방식 | 평가 러너 | 언제 쓰나 |
|------|----------------|-----------|-----------|
| **A. 로컬 직접 로딩** | lighteval이 `accelerate` 백엔드로 모델을 in-process 로드 | `scripts/run_local_qwen.sh` | GPU가 있는 머신에서 한 번만 평가할 때. 서버를 띄울 필요 없음 |
| **B. k8s 서빙 + 엔드포인트 평가** | k8s Deployment가 OpenAI 호환 API로 모델을 서빙 | `scripts/run_endpoint_eval.sh` | 모델을 한 번 띄워두고 여러 번/여러 체크포인트를 반복 평가할 때 |

두 경로 모두 `custom_tasks/`의 커스텀 태스크 정의를 사용하며, lighteval CLI는 `run_lighteval_patched.py` 래퍼를 통해 호출됩니다(멀티모달 이미지 전달, 다중 경로 데이터셋 등 패치 포함). 실제 실행 태스크 목록은 각 러너의 `TASKS` 배열이 기준이며, 엔드포인트 러너에는 문화재 reverse-QA/reverse-MT 등 추가 태스크도 포함됩니다.

한국어 OCR은 서로 성격이 다른 세 트랙을 별도 점수로 평가합니다.

- `korean_character_ocr`: outdoor 한글 단어·간판 OCR(기존 대표 태스크명)
- `korean_character_ocr_public_executive`: 표시 영역 안의 공공문서 텍스트 OCR
- `korean_character_ocr_data13`: 한글·영문·숫자·기호를 포함한 단일 문자/짧은 문자열 OCR

`korean_character_ocr_all_tests`도 정의돼 있지만, 서로 다른 문자 범위와 메트릭을 한 점수로 섞지 않기 위해 기본 러너에서는 위 세 트랙을 각각 실행합니다.

---

## 사전 준비 (공통)

### 의존성 설치

```bash
# 일반 환경
pip install -r requirements.txt
# B200 / Blackwell(sm_100) 환경
pip install -r requirements.b200.txt
```

`lighteval`, `litellm`, `transformers`, `torch`, `accelerate`, `qwen_vl_utils` 등이 포함됩니다.

### 머신별 환경 파일 설정 (경로 A·B 공통)

`scripts/` 의 실행 스크립트는 `scripts/env.local.sh` 를 자동으로 source 합니다. 머신마다 한 번 만들어 두면 됩니다.

```bash
cp scripts/env.example.sh scripts/env.local.sh
$EDITOR scripts/env.local.sh
```

반드시 채워야 하는 값:

- `PYTHON_BIN` — 프로젝트 의존성이 설치된 Python 인터프리터 절대경로
- `CUDA_VISIBLE_DEVICES`, `NUM_PROCESSES`, `BATCH_SIZE` — (경로 A에서 사용) GPU 수에 맞춰 설정
- `HF_HOME`, `HF_DATASETS_CACHE` — Hugging Face 캐시 경로
- `TELEGRAM_PY` — 완료 알림이 필요 없으면 빈 값으로 둠

> `env.local.sh` 는 git-ignore 되어 있어 머신별 경로가 충돌하지 않습니다.

---

## 경로 A — 로컬에서 모델 직접 로딩 후 평가

서버를 띄우지 않고, lighteval이 모델을 직접 GPU에 로드해 평가합니다. `accelerate launch` 로 데이터 병렬 실행하며, 각 워커가 모델 전체를 한 GPU에 올립니다.

### A-1. 실행

```bash
cd VLM_SERVE_EVAL_Qwen3_5

bash scripts/run_local_qwen.sh /path/to/qwen3_5_9b_multimodal_sft
```

- 인자: 로컬 체크포인트 경로 **또는** HF repo id (예: `Qwen/Qwen3.5-9B`)
- `--no-think` 옵션: `no_think_chat_template.jinja` 를 주입해 `<think>` 비활성화

### A-2. 옵션

```bash
# 일부 태스크만 실행 (태스크 short-name 공백 구분)
ONLY_TASKS="korean_heritage_name_vqa kmmlu_gen" \
  bash scripts/run_local_qwen.sh /path/to/checkpoint

# GPU / 배치 조정 (env.local.sh 값을 호출 시 덮어쓰기)
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PROCESSES=4 BATCH_SIZE=8 \
  bash scripts/run_local_qwen.sh /path/to/checkpoint
```

### A-3. 동작 방식

- 태스크별로 `accelerate launch ... run_lighteval_patched.py accelerate <MODEL_ARGS> <task> ...` 실행
- 텍스트 전용 태스크(`kmmlu_gen`, `hae_rae_bench_gen`, `ifeval_ko_gen`)는 `--vision-model` 플래그를 빼고 실행
- **가중치 로딩 실패 감지**: 로그에 `weights were not initialized` 류 메시지가 보이면 즉시 중단합니다 (부분 로드된 체크포인트의 점수는 무의미하므로). 그 외 일시적 실패는 경고만 남기고 다음 태스크로 진행

### A-4. 결과 위치

```
./test_final_11_tasks/<MODEL_TAG>/<task_name>/
  ├── lighteval_run.log        # 실행 로그
  ├── results/                 # lighteval 점수 JSON
  └── details/                 # --save-details 의 상세 parquet
```

`<MODEL_TAG>` 은 모델 경로에서 자동 생성됩니다 (예: `.../qwen3_5_9b_multimodal_sft` → `parentdir_qwen3_5_9b_multimodal_sft`).

---

## 경로 B — k8s로 서빙 후 엔드포인트 평가

모델을 OpenAI 호환 API로 k8s에 한 번 띄우고, 평가 클라이언트가 HTTP로 접근합니다. 서빙과 평가가 분리되어 있어 같은 서버를 여러 번 재평가할 수 있습니다.

### B-1. 서빙 이미지 빌드

```bash
cd VLM_SERVE_EVAL_Qwen3_5/serving
./build.sh
# 산출 이미지: registry.example.com/vlm-qwen3_5-serve:latest
```

서빙 코드(`1_openai_compatible_api.py`)만 바뀐 게 아니라면 매번 다시 빌드할 필요는 없습니다.

### B-2. 평가할 모델 확인 / 배포

`serving/2_serve_vlm_qwen3_5.yaml` 의 `args` 가 어떤 체크포인트를 띄울지 결정합니다. 기본값:

- `--model-path /data/checkpoints/qwen3_5_9b_multimodal_sft`
- `--base-model-name Qwen/Qwen3.5-9B` (체크포인트에 `config.json` 이 없을 때 아키텍처 로드용)
- `--model-id qwen3_5_9b_multimodal_sft` ← **평가 시 쓰는 모델 id**
- `--checkpoints-root /data/checkpoints` (여러 체크포인트 lazy-load 선택용)

다른 체크포인트를 평가하려면 YAML의 `--model-path` / `--model-id` 를 먼저 맞춥니다. 배포:

```bash
cd VLM_SERVE_EVAL_Qwen3_5/serving

# 방법 1: kubectl 직접
kubectl apply -f 2_serve_vlm_qwen3_5.yaml
kubectl -n gpu-workspace rollout status deploy/vlm-qwen3-5-serve

# 방법 2: 통합 진입점 (configs/k8s.env 사용)
./run.sh deploy --profile k8s
```

로그로 모델 로딩 완료 확인:

```bash
kubectl -n gpu-workspace logs -f deploy/vlm-qwen3-5-serve
```

### B-3. 포트포워딩

평가 클라이언트가 접근할 수 있도록 Service를 로컬 포트로 포워딩합니다. (별도 터미널에서 계속 떠 있어야 함)

```bash
cd VLM_SERVE_EVAL_Qwen3_5/serving
./3_port_forward.sh
# = ./run.sh port-forward --profile k8s
# svc/vlm-qwen3-5-serve 8000 → localhost:8000
```

헬스체크:

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/v1/models   # 평가할 model id가 보이는지 확인
```

### B-4. 엔드포인트 평가 실행

```bash
cd VLM_SERVE_EVAL_Qwen3_5

bash scripts/run_endpoint_eval.sh qwen3_5_9b_multimodal_sft
```

- 인자: 엔드포인트의 `/v1/models` 가 노출하는 **모델 id** (`--model-id` 와 동일, 또는 `checkpoints-root` 하위 체크포인트 id)
- 다른 엔드포인트 지정: `--base-url http://HOST:PORT/v1`

```bash
# 일부 태스크만 / 샘플 수 조정
ONLY_TASKS="korean_character_ocr kmmlu_gen" MAX_SAMPLES=50 \
  bash scripts/run_endpoint_eval.sh qwen3_5_9b_multimodal_sft
```

스크립트는 먼저 `/v1/models` 에 해당 모델이 떠 있는지 확인하고(없으면 즉시 종료), 태스크별로 `run_lighteval_patched.py endpoint litellm` 을 호출합니다. 모델 id에 `/` 가 포함돼도(`.../checkpoint-600`) `provider=openai` 를 명시해 litellm이 올바르게 처리합니다.

### B-5. 결과 위치

```
./test_final_11_tasks_endpoint/<MODEL_TAG>/<task_name>/
  ├── lighteval_run.log
  ├── results/
  └── details/
```

---

### (참고) k8s 없이 로컬에서 서빙만 하고 싶을 때

엔드포인트 방식(경로 B)을 k8s 없이 쓰려면 로컬에서 서버를 직접 띄울 수 있습니다.

```bash
cd VLM_SERVE_EVAL_Qwen3_5/serving
cp configs/local.env.example configs/local.env   # 최초 1회, 경로 수정
./run.sh serve --profile local                   # http://127.0.0.1:8000

# 이후 다른 터미널에서
cd VLM_SERVE_EVAL_Qwen3_5
bash scripts/run_endpoint_eval.sh <MODEL_ID>
```

---

## 평가 후 채점 보조 (OCR / name-VQA)

`korean_character_ocr` 와 `korean_heritage_name_vqa` 는 정답 매칭이 까다로워, lighteval 실행 후 LLM judge로 재채점합니다. judge 결과는 `*_llm_judge_cache.json` 에 캐시됩니다.

```bash
python scripts/ocr_llm_judge.py        # OCR 태스크 재채점
python scripts/name_vqa_llm_judge.py   # name-VQA 태스크 재채점
```

---

## 자주 겪는 문제

| 증상 | 원인 / 해결 |
|------|-------------|
| 모델/설정을 바꿨는데 점수가 동일 | lighteval이 이전 결과를 캐시함. `find ~/.cache/huggingface/lighteval -type d -name "*<task>*" -exec rm -rf {} +` 후 재실행 |
| 경로 A에서 `weight-loading mismatch` 로 중단 | 체크포인트가 베이스 모델 아키텍처와 안 맞음. `--base-model-name` / 체크포인트 경로 확인 |
| 경로 B에서 `Endpoint not reachable` | 포트포워딩이 떠 있는지, `/v1/models` 의 모델 id가 인자와 일치하는지 확인 |
| `PYTHON_BIN is not set` | `scripts/env.local.sh` 미생성. `cp scripts/env.example.sh scripts/env.local.sh` 후 `PYTHON_BIN` 설정 |
| 컨텍스트 길이 초과 오류 | `custom_tasks/` 의 `generation_size`, YAML의 `max_model_length` 확인 (루트 `README.md` 참고) |

---

## 관련 문서

- `README.md` — lighteval + LiteLLM 사용법, 커스텀 태스크 / YAML 작성, 디버깅 스크립트
- `serving/README.md` — 서빙 서버 상세, 체크포인트 sweep / prompt-grid 평가
- `docs/` — 한국지식·OCR 커스텀 태스크 제작 방법론

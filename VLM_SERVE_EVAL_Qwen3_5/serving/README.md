# Qwen3.5 VLM 서빙 + 웹 채팅 + Lighteval

이 디렉터리는 PVC에 저장된 학습 완료 Qwen3.5 VLM 체크포인트를 불러와 다음 두 가지를 한 번에 확인하기 위한 새 파이프라인입니다.

- OpenAI 호환 API로 모델 서빙
- 같은 API를 이용한 웹 채팅 테스트와 `lighteval` 벤치마크 실행

## 실행 프로파일

이제 실행 방식은 `local` 과 `k8s` 두 가지 프로파일로 통일할 수 있습니다.

- 공용 진입점: `./run.sh`
- 쿠버네티스 기본 설정: `configs/k8s.env`
- 로컬 설정 템플릿: `configs/local.env.example`

로컬에서 처음 쓸 때는 아래처럼 설정 파일을 한 번 준비하면 됩니다.

```bash
cp configs/local.env.example configs/local.env
vi configs/local.env
```

추천 명령 형태:

```bash
./run.sh serve --profile local
./run.sh serve --profile k8s
./run.sh deploy --profile k8s
./run.sh port-forward --profile k8s
./run.sh eval-checkpoints --profile local
./run.sh launch-checkpoint-eval --profile k8s
./run.sh prompt-grid --profile local
./run.sh launch-prompt-grid --profile k8s
```

의도는 다음과 같습니다.

- Python 코드는 환경을 몰라도 되게 유지
- 경로/포트/출력 위치는 프로파일 설정 파일로 분리
- 기존 쿠버네티스 실행 경로는 유지
- 사용자는 `무슨 기능인지` 와 `어느 프로파일인지` 만 기억하면 됨

로컬에서 여러 학습 모델을 UI 드롭다운으로 비교하려면 `configs/local.env`에 아래처럼 명시 모델 목록을 연결합니다.

```bash
MODELS_CONFIG=configs/local_compare_models.json
```

`configs/local_compare_models.json` 형식:

```json
{
  "models": [
    {"id": "dpo_final", "path": "/workspace/local_models/chat_compare/dpo_final"},
    {"id": "pre_dpo_checkpoint_300", "path": "/workspace/local_models/chat_compare/pre_dpo_checkpoint_300"}
  ]
}
```

서버가 뜨면 `http://127.0.0.1:8000/chat` 에서 모델을 선택할 수 있고, 선택한 모델은 첫 요청 시 lazy-load 됩니다.

기본 방향은 다음과 같습니다.

- 모델은 PVC 경로(`/data/checkpoints/...`)에서 직접 읽습니다.
- API는 `/v1/chat/completions`, `/v1/completions`, `/v1/models` 를 제공합니다.
- 브라우저에서는 `http://localhost:8000/chat` 으로 접속해 바로 테스트합니다.
- `lighteval` 은 LiteLLM backend를 통해 같은 OpenAI 호환 API를 호출합니다.

## 파일 구성

- `1_openai_compatible_api.py`
  - Qwen3.5 VLM 체크포인트를 로드하고 OpenAI 호환 API와 웹 채팅 페이지를 함께 제공합니다.
- `chat_ui.html`
  - 이미지 업로드가 가능한 간단한 브라우저 채팅 화면입니다.
- `2_serve_vlm_qwen3_5.yaml`
  - 쿠버네티스 Deployment + Service입니다. 기본값은 9B 체크포인트를 1 GPU로 띄웁니다.
- `3_port_forward.sh`
  - 로컬 브라우저/로컬 Lighteval에서 접근할 수 있도록 `svc/vlm-qwen3-5-serve` 를 포트포워딩합니다.
- `4_run_lighteval.sh`
  - 현재 떠 있는 OpenAI 호환 API 대상으로 `lighteval endpoint litellm` 을 실행합니다.
- `Dockerfile` / `build.sh`
  - 서빙용 이미지를 빌드합니다.
- `5_eval_checkpoints.py`
  - `checkpoint-*` 를 순회하며 간단한 수학 질의를 수행하고, checkpoint별 응답/정답 여부를 PVC에 JSON/CSV로 저장합니다.
- `6_eval_checkpoints_job.yaml`
  - 위 스크립트를 GPU 1장으로 실행하는 배치 Job 예시입니다.
- `7_eval_single_checkpoint_job.yaml`
  - checkpoint 1개만 평가하는 Job 템플릿입니다.
- `8_launch_parallel_checkpoint_eval.sh`
  - `checkpoint-*` 별 Job을 각각 생성해서 여러 노드에서 병렬 평가를 시작합니다.
- `9_collect_checkpoint_eval_summary.py`
  - 병렬로 생성된 `checkpoint-*.json` 을 다시 하나의 summary JSON/CSV로 합칩니다.
- `korean_common_knowledge_questions_10.json`
  - 한국 상식 10문항 예시 질문 세트입니다.

## 기본 가정

- 기본 체크포인트 경로: `/data/checkpoints/qwen3_5_9b_multimodal_sft`
- 기본 베이스 모델: `Qwen/Qwen3.5-9B`
- 기본 네임스페이스: `gpu-workspace`
- 기본 PVC: `training-pvc`

중요한 점:

- 현재 학습 결과물 예시를 보면 `model.safetensors` 는 있지만 `config.json` 이 없을 수 있습니다.
- 그래서 기본 서버는 `--base-model-name Qwen/Qwen3.5-9B` 로 아키텍처를 불러오고, 그 위에 로컬 `model.safetensors` 를 덮어씌우도록 구성했습니다.
- 만약 체크포인트 폴더에 `config.json` 까지 완비돼 있으면 `--base-model-name` 없이도 바로 로드할 수 있습니다.

## Step 1. 서빙 이미지 빌드

```bash
cd /home/user/mlxp/VLM_SERVE_EVAL_Qwen3_5
./build.sh
```

기본 이미지 이름:

```bash
registry.example.com/vlm-qwen3_5-serve:latest
```

## Step 2. API 서버 배포

```bash
kubectl apply -f 2_serve_vlm_qwen3_5.yaml
kubectl -n gpu-workspace rollout status deploy/vlm-qwen3-5-serve
```

로그 확인:

```bash
kubectl -n gpu-workspace logs -f deploy/vlm-qwen3-5-serve
```

다른 체크포인트를 띄우고 싶으면 YAML 안의 아래 인자만 먼저 맞추면 됩니다.

- `--model-path`
- `--base-model-name`
- `--model-id`
- GPU / 메모리 리소스

27B 체크포인트라면 1 GPU 기본값으로는 부족할 수 있으니 리소스와 로딩 방식을 다시 잡아야 합니다.

## Step 3. 웹 채팅 테스트

포트포워딩:

```bash
./3_port_forward.sh
```

브라우저 접속:

```text
http://localhost:8000/chat
```

빠른 헬스체크:

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/v1/models
```

텍스트 채팅 테스트 예시:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3_5_9b_multimodal_sft",
    "messages": [
      {"role": "system", "content": [{"type": "text", "text": "Answer in Korean."}]},
      {"role": "user", "content": [{"type": "text", "text": "간단히 자기소개해줘."}]}
    ],
    "max_tokens": 256,
    "temperature": 0.0
  }'
```

이미지 포함 테스트는 웹 UI를 쓰는 편이 가장 편합니다.

## Step 4. Lighteval 실행

공식 문서상 `lighteval` 은 다양한 백엔드를 지원하고, OpenAI 호환 API는 `endpoint litellm` 경로로 연결할 수 있습니다.

- Lighteval docs: https://huggingface.co/docs/lighteval/en/index
- LiteLLM backend docs: https://huggingface.co/docs/lighteval/en/use-litellm-as-backend

로컬 머신에서 실행 예시:

```bash
pip install lighteval litellm
OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
OPENAI_API_KEY=dummy \
MODEL_NAME=qwen3_5_9b_multimodal_sft \
TASKS='lighteval|gsm8k|0|0' \
./4_run_lighteval.sh
```

다른 태스크 예시:

```bash
TASKS='lighteval|mmlu:abstract_algebra|0|0'
```

주의:

- 이 서버는 우선 `chat/completions` 중심의 OpenAI 호환 API를 제공합니다.
- `lighteval` 버전에 따라 `--use-chat-template` 플래그 요구 여부가 달라질 수 있어서, 스크립트가 help 출력 기준으로 자동 감지하도록 해두었습니다.
- 현재 `lighteval` 은 주로 텍스트 벤치마크에 먼저 연결하는 것이 자연스럽습니다. 멀티모달 태스크까지 확장하려면 별도 task 설정을 추가로 맞춰야 합니다.

## Step 5. Checkpoint별 간단 질의 평가

API를 띄우지 않고, PVC 안의 `checkpoint-*` 를 직접 순회하면서 간단한 수학 문제를 던져 응답을 저장할 수 있습니다.

기본 Job 적용:

```bash
kubectl apply -f 6_eval_checkpoints_job.yaml
kubectl -n gpu-workspace logs -f job/qwen3-5-checkpoint-math-eval
```

기본 입력 경로:

```text
/data/checkpoints/qwen3_5_9b_multimodal_sft_lr_1e-5
```

기본 결과 저장 경로:

```text
/data/eval_results/qwen3_5_9b_multimodal_sft_lr_1e-5_math_eval
```

저장 산출물:

- `run_config.json`
- `summary.json`
- `summary.csv`
- `responses.jsonl`
- `responses.csv`
- `checkpoint-100.json` 같은 checkpoint별 상세 결과 파일

`summary.csv` 에는 checkpoint별 정답 개수와 정확도가 들어가고, `responses.csv` 에는 각 질문의 원문 응답과 추출된 답이 함께 남습니다.

다른 경로를 쓰고 싶으면 YAML 안의 아래 인자만 바꾸면 됩니다.

- `--checkpoints-dir`
- `--processor-path`
- `--output-dir`
- `--questions-file`
- `--limit`

직접 실행 예시:

```bash
python 5_eval_checkpoints.py \
  --checkpoints-dir /data/checkpoints/qwen3_5_9b_multimodal_sft_lr_1e-5 \
  --processor-path /data/checkpoints/qwen3_5_9b_multimodal_sft_lr_1e-5 \
  --output-dir /data/eval_results/qwen3_5_9b_multimodal_sft_lr_1e-5_math_eval
```

질문 세트를 바꾸고 싶으면 아래 형식의 JSON 파일을 만들어 `--questions-file` 로 넘기면 됩니다.

```json
[
  {"id": "q1", "question": "27 + 15 = ?", "answer": "42"},
  {"id": "q2", "question": "81 / 9 = ?", "answer": "9"}
]
```

## 병렬 평가

checkpoint마다 별도 Job을 띄워 여러 GPU 노드에서 병렬로 끝내고 싶다면 아래 스크립트를 쓰면 됩니다.

```bash
cd /home/user/mlxp/VLM_SERVE_EVAL_Qwen3_5
./8_launch_parallel_checkpoint_eval.sh
```

기본 출력 경로:

```text
/data/eval_results/qwen3_5_9b_multimodal_sft_lr_1e-5_math_eval_parallel
```

병렬 평가가 모두 끝난 뒤 요약본 합치기:

```bash
python 9_collect_checkpoint_eval_summary.py \
  --input-dir /data/eval_results/qwen3_5_9b_multimodal_sft_lr_1e-5_math_eval_parallel
```

특정 checkpoint만 병렬 대상으로 제한하고 싶으면:

```bash
CHECKPOINTS="checkpoint-300 checkpoint-400 checkpoint-500" \
./8_launch_parallel_checkpoint_eval.sh
```

## 로컬 직접 실행 예시

PVC 대신 로컬 다운로드 모델을 바로 띄우고 싶으면:

```bash
python 1_openai_compatible_api.py \
  --model-path /home/user/mlxp/VLM_SFT_Custom_Data_Qwen3_5_Packed/trained_model_output/qwen3_5_9b_multimodal_sft \
  --base-model-name Qwen/Qwen3.5-9B \
  --model-id qwen3_5_9b_multimodal_sft
```

## 확장 포인트

- 인증 헤더가 필요한 내부 API 게이트웨이 추가
- streaming 응답 구현
- 멀티턴 대화 로그 저장
- Lighteval 전용 쿠버네티스 Job 추가
- 27B용 다중 GPU 추론 배포 YAML 추가

## vLLM 기반 서빙 (대안)

기본 서버(`1_openai_compatible_api.py`)는 transformers `generate()`를 FastAPI로 감싼
구현이라 **pod당 생성을 1건씩 직렬 처리**합니다 (continuous batching 없음). 처리량이
중요하면 vLLM 기반 서빙을 별도 Deployment로 띄울 수 있습니다.

관련 파일:

- `vllm_qwen35_plugin/`
  - vLLM 플러그인 패키지. SFT/DPO 체크포인트는 비전 타워가
    `model.language_model.visual.*` 에 중첩돼 있어 vLLM 기본 weight-mapper로는
    비전 텐서 333개가 로드되지 않습니다. 이 플러그인이 `vllm.general_plugins`
    entry point로 등록돼 **모든 vLLM 프로세스**(API 서버, EngineCore, TP worker)에서
    weight-mapper를 패치합니다.
- `Dockerfile.vllm` / `build_vllm.sh`
  - `vllm/vllm-openai` 베이스 이미지에 위 플러그인을 설치합니다.
  - `VLLM_TAG`는 **Qwen3.5 아키텍처(`model_type: qwen3_5`, 하이브리드 linear+full
    attention)를 지원하는 vLLM 버전**이어야 합니다. 리포는 0.19.x 라인을 targeting
    합니다(`run_lighteval_patched.py` 주석 참고).
- `2b_serve_vllm.yaml`
  - 기본 서버와 **별도로** 뜨는 Deployment + Service(`vlm-qwen3-5-serve-vllm`).
    기본 서버는 fallback으로 그대로 둡니다.

빌드 및 배포:

```bash
cd serving
VLLM_TAG=v0.19.0 ./build_vllm.sh          # docker buildx 필요
kubectl apply -f 2b_serve_vllm.yaml
kubectl -n gpu-workspace rollout status deploy/vlm-qwen3-5-serve-vllm
kubectl -n gpu-workspace port-forward svc/vlm-qwen3-5-serve-vllm 8000:8000
```

평가 실행 (litellm 모델명에 `openai/` 접두사 필요 — 모델 id에 `/`가 있으면 필수):

```bash
bash scripts/run_endpoint_eval.sh qwen3_5_9b_multimodal_sft_dpo_normalized/checkpoint-600
```

주의사항:

- vLLM은 **서버당 모델 1개**입니다. 기본 서버처럼 `--checkpoints-root` 로 여러
  체크포인트를 lazy-load 하지 못합니다. 다른 체크포인트를 평가하려면
  `2b_serve_vllm.yaml` 의 모델 경로와 `--served-model-name` 을 바꿔 재배포합니다.
- continuous batching 덕분에 단일 GPU에서도 동시 요청 처리량이 크게 향상됩니다.
  멀티 GPU가 필요하면 `--tensor-parallel-size` 와 pod의 `nvidia.com/gpu` 를 함께
  올립니다.
- Qwen3.5 하이브리드 linear-attention VLM의 vLLM 서빙은 vLLM 버전에 민감합니다.
  비전 텐서 미초기화/아키텍처 오류가 나면 `VLLM_TAG` 를 조정하세요.

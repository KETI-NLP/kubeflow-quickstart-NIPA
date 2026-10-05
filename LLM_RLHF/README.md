# LLM 분산 학습 파이프라인 가이드 (RLHF / PPO + FSDP)

이 디렉토리는 Kubernetes 환경에서 언어 모델(LLM)을 다중 노드/다중 GPU(FSDP) 위에서 **RLHF(Reinforcement Learning from Human Feedback)** 방식으로 얼라인먼트 튜닝하고, 최종 학습 결과를 로컬로 안전하게 추출하기 위한 자동화 파이프라인입니다. 

코드 베이스는 TRL 라이브러리의 `PPOTrainer`를 기반으로 구성되었으며, 보상 모델(Reward Model)은 현재 뼈대 구조로 자리잡혀 있습니다.

---

## 🚀 전체 학습 파이프라인 (5-Step)

### Step 0. 커스텀 훈련 환경 구축 (최초 1회)
PPO 분산 훈련에 필수적인 의존성(`trl`, `accelerate`, `deepspeed`)들이 FSDP 환경(PyTorch 2.3)에서 크래시 없이 동작하도록 핀(Pin) 고정된 Docker 이미지를 빌드합니다.
```bash
./build.sh
```
> 완료 시 `registry.example.com/custom-llm-sft:latest` 이미지가 로컬 레지스트리에 등록됩니다.

### Step 1. 로컬에 RLHF 데이터셋 준비
PPO 훈련 시 정책(Policy)이 텍스트를 생성하도록 질문(Prompt)만 제공하는 선호도 데이터셋(`Anthropic/hh-rlhf`)을 파싱 후 저장합니다.
```bash
python 1_download_to_local.py
```
> 완료 시 현재 경로에 `./rlhf_dataset` 폴더가 생성됩니다. (스키마: `query`)

### Step 2. 데이터셋을 클라우드(PVC)로 업로드
분산 워커들이 접근하여 프롬프트들을 읽을 수 있도록 클러스터의 영구 스토리지(PVC)로 데이터를 복사합니다.
```bash
./2_upload_dataset.sh
```

### Step 3. RLHF (PPO) 분산 학습 시작
Kubernetes에 PyTorchJob을 배포하여 PPO 알고리즘 기반 생성 및 최적화를 시작합니다. 
```bash
kubectl apply -f 4_llm_pytorchjob.yaml
```
> **모니터링:** `kubectl logs -f llm-rlhf-dist-worker-0 pytorch` 명령어로 보상(Reward) 평균값 및 생성 지표를 실시간 모니터링할 수 있습니다.

### Step 4. 학습 완료된 모델 로컬로 다운로드
모든 PPO 에포크가 종료되면 워커 0번이 저장한 모델 파라미터(`*.safetensors`)를 로컬로 다운로드합니다.
```bash
./5_download_model.sh
```
> 다운로드 후 `./trained_model_output/llm-rlhf` 에 파일들이 보존되며 더미 파드는 시스템에서 정리됩니다.

---

## 📂 저장소 파일 요약

*   **`Dockerfile` / `build.sh`**
    *   분산 훈련과 PPO 최적화 과정에서 발생하는 TRL 및 딥스피드 충돌을 원천 방어하도록 설계된 커스텀 컨테이너 환경 파일입니다.
*   **`1_download_to_local.py`**
    *   `Anthropic/hh-rlhf` 등에서 프롬프트(`query`)만을 추출하여 데이터셋을 조형하는 역할을 합니다.
*   **`2_upload_to_pvc_direct.yaml`** / **`2_upload_dataset.sh`**
    *   더미 컨테이너를 스피닝(Spinning)하여 로컬 프롬프트 데이터셋을 PVC 안으로 일괄 복사해 주는 단방향 스크립트 셋입니다.
*   **`3_llm_rlhf_distributed.py`**
    *   **가장 중요한 메인 알고리즘 스크립트**입니다. TRL `PPOTrainer`를 사용하며, 토큰 생성 루프, Dummy Reward Model 연산, PPO Policy 최적화를 FSDP 규격 위에서 실행되도록 구현되어 있습니다. (실 사용 시 보상 함수를 교체하면 됩니다.)
*   **`4_llm_pytorchjob.yaml`**
    *   위 PPO 알고리즘 파이썬 파일을 사용할 컴퓨팅 리소스(GPU 분산) 명세와 함께 클러스터로 넘겨주는 스케줄링 파일입니다.
*   **`5_download_model.sh`**
    *   학습된 최종 결과 모델을 클러스터로부터 내 로컬 PC로 꺼내옵니다.

---

## ⚠️ 사전 준비 사항 (Prerequisites)

이 파이프라인을 온전히 실행하기 위해 클러스터 내에 미리 확보되어야 하는 설정들입니다.

1.  **네임스페이스 및 노드 라벨 (`Namespace & NodeSelector`)**
    *   모든 배포 파일(`.yaml` 및 `.sh`)은 `gpu-workspace` 네임스페이스를 바라보도록 작성되어 있습니다.
    *   파드가 뜰 GPU 노드는 `example.com/gpu-pool: YOUR_GPU_NODE_POOL` 라벨을 가져야 합니다.
2.  **공유 스토리지 (`PVC`)**
    *   `training-pvc` 이라는 이름의 PVC가 해당 네임스페이스 안에 사전에 생성(`Bound` 상태)되어 있어야 합니다. 이 공간에 데이터셋과 모델의 결과물이 저장됩니다.
3.  **프라이빗 레지스트리 인증 (`ImagePullSecrets`)**
    *   직접 빌드한 커스텀 이미지(`ketiair.com:...`)를 워커 노드들이 다운로드할 수 있도록, 쿠버네티스 내에 `byunggill` 이라는 이름의 도커 레지스트리 Secret이 생성되어 있어야 합니다. (적용처: `4_llm_pytorchjob.yaml`의 `imagePullSecrets`)
4.  **HuggingFace 접근 권한 (`HF_TOKEN` 환경변수 - 선택적)**
    *   추후 Llama 등 권한이 필요한 모델을 가져올 때 토큰이 요구될 수 있습니다.

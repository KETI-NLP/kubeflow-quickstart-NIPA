# LLM 분산 학습 파이프라인 가이드 (Full FSDP)

이 디렉토리는 Kubernetes 환경에서 언어 모델(LLM)을 다중 노드/다중 GPU(FSDP)로 파인튜닝(SFT)하고, 최종 학습 결과를 로컬로 안전하게 추출하기 위한 자동화 파이프라인입니다.

---

## 🚀 전체 학습 파이프라인 (5-Step)

### Step 0. 커스텀 훈련 환경 구축 (최초 1회)
학습에 필요한 패키지들이 사전 세팅된 Docker 이미지를 빌드합니다.
```bash
./build.sh
```
> 완료 시 `registry.example.com/custom-llm-sft:latest` 이미지가 로컬 레지스트리에 등록됩니다.

### Step 1. 로컬에 데이터셋 준비
학습에 사용할 데이터를 HuggingFace에서 로컬 머신으로 다운로드합니다.
```bash
python 1_download_to_local.py
```
> 완료 시 현재 경로에 `./alpaca_dataset` 폴더가 생성됩니다.

### Step 2. 데이터셋을 클라우드(PVC)로 업로드
학습 파드들이 읽을 수 있도록 로컬 데이터를 쿠버네티스 스토리지(PVC)로 복사합니다.
```bash
./2_upload_dataset.sh
```

### Step 3. 분산 파인튜닝 시작 (FSDP)
Kubernetes에 PyTorchJob을 배포하여 다중 GPU 분산 학습을 시작합니다.
```bash
kubectl apply -f 4_llm_pytorchjob.yaml
```
> **모니터링:** `kubectl logs -f llm-sft-dist-worker-0 pytorch` 명령어로 학습 로그 및 Loss 단계를 확인할 수 있습니다.

### Step 4. 학습 완료된 모델(Checkpoint) 로컬로 다운로드
학습이 모두 종료되면 PVC에 저장된 최종 모델 파라미터(`*.safetensors`)를 내 PC로 복사해옵니다.
```bash
./5_download_model.sh
```
> 완료 시 `./trained_model_output/llm-sft` 폴더에 학습 결과물이 저장되며, 다운로드용으로 띄웠던 임시 파드는 자동 삭제됩니다.

---

## 📂 저장소 파일 요약

*   **`Dockerfile` / `build.sh`**
    *   분산 학습에 필요한 라이브러리(`transformers`, `trl`, `deepspeed` 등)가 안정적으로 세팅된 환경을 빌드하는 스크립트입니다.
*   **`1_download_to_local.py`**
    *   HuggingFace에서 SFT용 훈련 데이터(Alpaca Cleaned)를 다운로드 받아 로컬 디스크에 저장하는 파이썬 스크립트입니다.
*   **`2_upload_to_pvc_direct.yaml`**
    *   로컬 호스트와 클러스터 PVC 간의 파일 복사(`kubectl cp`)를 중계하기 위해 가볍게 띄우는 더미(Dummy) 파드 명세서입니다.
*   **`2_upload_dataset.sh`**
    *   더미 파드를 생성하고 로컬 데이터셋을 PVC 안으로 일괄 복사해 주는 자동화 쉘 스크립트입니다.
*   **`3_llm_sft_distributed.py`**
    *   HuggingFace `Trainer` 인스턴스를 활용하여 대형 언어 모델 배포 및 FSDP 파인튜닝 로직이 구현된 메인 훈련 코드입니다.
*   **`4_llm_pytorchjob.yaml`**
    *   `3_llm_sft_distributed.py`를 다수의 GPU 노드 워커들에 분산 배치 및 실행시키는 Kubeflow PyTorchJob 명세 파일입니다.
*   **`5_download_model.sh`**
    *   모델 훈련이 완료된 후 PVC에 저장된 체크포인트를 로컬 로컬 머신으로 일괄 다운로드하고 더미 파드를 자동 철거하는 쉘 스크립트입니다.

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
    *   만약 Llama-3 등 권한 승인이 필요한 Gated 모델/데이터를 훈련에 사용할 경우, 다운로드 스크립트 실행 전 로컬이나 파드의 환경변수에 `HF_TOKEN`을 등록해야 합니다.

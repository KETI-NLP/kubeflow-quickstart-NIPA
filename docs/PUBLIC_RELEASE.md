# 환경 설정과 실행 안내

GPU 지원 사업에서 제공받은 Kubeflow 환경을 활용하며 작성한 학습·데이터 패킹·서빙 예제입니다. 각 디렉터리는 독립적인 실험 파이프라인이며, 모든 예제가 모든 GPU나 라이브러리 버전에서 검증된 것은 아닙니다.

## 사용 범위

이 저장소는 독립적인 소스 배포본입니다. 원본 실험 저장소나 소스 내보내기 작업은 필요하지 않습니다. 모델 가중치와 데이터셋은 포함하지 않으며 실행 의존성 및 GPU 클러스터는 별도로 준비해야 합니다. `/workspace` 경로와 인프라 이름은 예시이므로 자신의 환경에 맞춰 설정하세요.

## 시작 전 환경 설정

GPU 제공자가 발급한 kubeconfig를 사용하고 다음을 확인합니다.

```bash
kubectl config current-context
kubectl get crd pytorchjobs.kubeflow.org
kubectl -n YOUR_NAMESPACE get pvc
```

YAML 및 셸 스크립트에서 네임스페이스, PVC, 이미지 레지스트리, imagePullSecrets, nodeSelector, 모델·데이터 경로를 수정하세요. 공개 사본의 기본 예시는 `gpu-workspace`, `training-pvc`, `registry.example.com`, `registry-credentials`입니다. 노드 라벨과 RDMA/InfiniBand 설정은 제공자마다 달라 그대로 적용할 수 없습니다.

`Worker.replicas`, `elasticPolicy.minReplicas/maxReplicas`, `nProcPerNode`, 컨테이너 GPU 요청·제한은 함께 맞춰야 합니다. 예를 들어 워커 2개와 노드당 GPU 4개는 총 8 GPU입니다. 작은 GPU 구성에서는 모델 크기, 정밀도, 배치 크기와 샤딩 설정도 다시 결정해야 합니다.

토큰은 소스에 입력하지 말고 Secret 또는 로컬 환경변수로 전달합니다. `VLM_DPO` 예제는 `huggingface-token` Secret의 `HF_TOKEN`을 참조합니다. 셸에서 토큰을 입력하여 Secret을 만드는 예시는 다음과 같습니다.

```bash
read -rsp 'HF token: ' HF_TOKEN; echo
printf '%s' "$HF_TOKEN" | kubectl -n YOUR_NAMESPACE create secret generic huggingface-token --from-file=HF_TOKEN=/dev/stdin
unset HF_TOKEN
```

평가 설정의 `REPLACE_WITH_API_KEY`는 자신의 키로 바꾼 로컬 사본에서 사용하세요. 이 사본은 `.env` 파일과 마찬가지로 Git 추적에서 제외해야 합니다. YAML의 `${VARIABLE}`은 소비 라이브러리에서 지원 여부를 확인하지 않고 사용하면 자동 치환되지 않습니다.

## 학습 흐름

1. 원하는 모델과 학습 방식에 해당하는 디렉터리를 선택합니다. Qwen3.5 멀티모달 SFT는 `VLM_SFT_Custom_Data_Qwen3_5`, 패킹 데이터용은 같은 이름에 `_Packed`가 붙은 디렉터리입니다.
2. Dockerfile과 build.sh의 이미지 이름 및 의존성을 확인합니다. build.sh는 레지스트리에 이미지를 푸시할 수 있습니다.
3. 학습 스크립트의 인자와 데이터 스키마를 확인하여 데이터를 준비하고 PVC에 업로드합니다. 모델 다운로드 Job이 있으면 먼저 실행하고 캐시 경로를 일치시키세요. 일부 학습 코드는 오프라인 캐시를 요구합니다.
4. ConfigMap에 스크립트를 마운트하는 YAML은 해당 ConfigMap을 먼저 만듭니다. 예를 들어 `LLM_SFT`에서는 `kubectl -n YOUR_NAMESPACE create configmap llm-sft-script --from-file=3_llm_sft_distributed.py`가 필요합니다.
5. YAML을 먼저 검증한 뒤 적용하고 로그와 체크포인트를 확인합니다.

```bash
kubectl apply --dry-run=server -f YOUR_TRAINING_JOB.yaml
kubectl apply -f YOUR_TRAINING_JOB.yaml
kubectl -n YOUR_NAMESPACE get pytorchjobs,pods
kubectl -n YOUR_NAMESPACE logs -f YOUR_WORKER_POD -c pytorch
```

기존 하위 README는 실험 당시 설명이므로 명령과 파일명이 현재 파일 구성과 다를 수 있습니다. 특히 `LLM_SFT` 코드의 실제 모델 클래스는 Qwen3-VL입니다. 일반 텍스트 모델을 인자로 바꾸는 것만으로 호환되는 것은 아닙니다.

## 패킹과 서빙

`CLOUD_PACK_DATASET`에는 Qwen3.5, HyperCLOVAX Seed Omni/Think용 별도 패킹 스크립트가 있습니다. 각 스크립트의 `--help`에서 입력·출력 인자를 확인하고, 학습용 processor와 같은 모델 계열로 패킹하세요. `run_cloud_pack.sh`는 MLXP 다운로드와 여러 작업을 실행하므로 데이터 목록, 환경 설정, 동시 작업 수를 먼저 확인해야 합니다. MLXP는 해당 플랫폼을 사용하는 경우에만 필요합니다.

서빙은 `VLM_SERVE_EVAL_Qwen3_5/serving`의 프로파일을 사용합니다.

```bash
cd VLM_SERVE_EVAL_Qwen3_5/serving
cp configs/local.env.example configs/local.env
# local.env에서 Python 경로와 모델 경로 수정
./run.sh serve --profile local
```

Kubernetes 서빙은 `configs/k8s.env`와 배포 YAML을 자신의 환경에 맞춘 뒤 `./run.sh deploy --profile k8s`로 실행합니다. 평가 태스크 중 문화유산·OCR 태스크는 별도 데이터가 필요하며, 소스 공개만으로 데이터가 제공되지는 않습니다.

## 공개 범위와 라이선스

이 저장소의 자체 코드에는 Apache-2.0을 적용합니다. 외부 코드에는 원본 라이선스가 우선하며 [외부 코드 안내](../THIRD_PARTY_NOTICES.md)를 확인하세요. 기관·사업의 공개 조건과 코드에 대한 권한은 게시 전에 확인해야 합니다. 모델 가중치와 데이터는 각각의 배포 조건을 따릅니다.

공개 사본 생성과 정적 검증만 수행했으며 새 환경에서 실제 GPU 학습·서빙은 검증하지 않았습니다. 공개 전에는 선택한 대표 파이프라인을 작은 구성에서 실행하고 실행 환경과 결과를 기록하세요.

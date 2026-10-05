# 2. 데이터 준비와 선택적 패킹

[이전: 환경 준비](../01-environment/README.md) · [전체 순서](../../../README.md) · [다음: 학습 실행](../03-training/README.md)

처음에는 패킹하지 않은 데이터로 시작하세요. [Qwen3.5 일반 SFT README](../../../VLM_SFT_Custom_Data_Qwen3_5/README.md)의 데이터 설명과 `3_vlm_sft_distributed.py`를 확인합니다. 이 코드는 `messages`와 `images`를 학습 중 processor로 처리합니다. 임의의 CSV·JSON을 경로에 넣는 것만으로 학습할 수는 없습니다.

Hugging Face `Dataset` 또는 `DatasetDict`로 데이터를 준비하고 `save_to_disk`로 저장하세요. 이미지 표현과 메시지의 이미지 참조는 해당 학습 코드에서 읽을 수 있는 형식이어야 합니다. 처음에는 작은 샘플로 컬럼과 이미지 로딩을 확인합니다.

```python
from datasets import load_from_disk

dataset = load_from_disk("./my_sft_dataset")
print(dataset)
```

PVC로 복사할 때는 다른 기본 예제의 [업로드 절차](../../../VLM_SFT/README.md)를 참고할 수 있습니다. 해당 예제의 `2_upload_to_pvc_direct.yaml`에서 네임스페이스·PVC·노드 설정을 바꾼 뒤 업로드용 Pod를 생성합니다. 저장소 루트에서 실행하는 예시는 다음과 같습니다.

```bash
kubectl apply -f VLM_SFT/2_upload_to_pvc_direct.yaml
kubectl -n YOUR_NAMESPACE get pods
kubectl -n YOUR_NAMESPACE wait --for=condition=Ready pod/YOUR_UPLOAD_POD --timeout=300s
kubectl cp ./my_sft_dataset YOUR_NAMESPACE/YOUR_UPLOAD_POD:/data/my_sft_dataset
```

Pod 이름은 해당 YAML의 `metadata.name`을 사용합니다. 복사에는 컨테이너의 `tar`가 필요합니다. 학습 YAML의 `--data_path`를 `/data/my_sft_dataset`으로 변경하세요. MLXP를 사용하는 경우에는 해당 서비스 계정·데이터셋·SDK가 필요하며, 로컬 PVC 데이터 경로로 실습할 때는 MLXP가 필요하지 않습니다.

## 패킹은 일반 SFT를 확인한 뒤 선택하세요

패킹은 여러 샘플의 토큰을 긴 시퀀스로 묶는 전처리입니다. 패킹 스크립트와 학습 processor의 모델 계열을 일치시켜야 합니다. Qwen3.5 예시는 해당 의존성이 설치된 환경에서 다음과 같이 실행합니다.

```bash
python CLOUD_PACK_DATASET/cloud_pack_sft_qwen3_5_dataset.py --help
python CLOUD_PACK_DATASET/cloud_pack_sft_qwen3_5_dataset.py \
  --data_path /data/my_sft_dataset \
  --output_path /data/my_sft_dataset_packed \
  --model_name Qwen/Qwen3.5-9B \
  --max_seq_len 8192 --limit 100
```

`/data` 경로는 PVC를 마운트한 컨테이너 내부 경로입니다. 로컬 실행이면 로컬 데이터 경로로 바꾸세요. `--limit 100`으로 소량 확인 후 전체 데이터로 실행합니다. 입력 스키마는 패킹 스크립트의 변환 로직도 확인해야 합니다.

패킹 결과는 [Qwen3.5 Packed SFT](../../../VLM_SFT_Custom_Data_Qwen3_5_Packed/README.md)에서 사용합니다. HyperCLOVAX용 패킹 코드는 `CLOUD_PACK_DATASET/cloud_pack_sft_hcx_seed_omni_dataset.py`와 `cloud_pack_sft_hcx_seed_think_dataset.py`입니다. `run_cloud_pack.sh`는 MLXP 전송과 다수의 작업을 수행하므로 데이터 목록·경로·동시 실행 수를 수정한 뒤 사용하세요.

다음 단계로 넘어가기 전에 학습 Pod에서 읽을 데이터 경로와 해당 학습 코드에 맞는 데이터 형식을 확정하세요.

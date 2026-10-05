# VLM DPO Qwen3.5 + MLXP

이 디렉터리는 `VLM_DPO` 구조를 유지하면서, Qwen3.5 모델과 Naver MLXP 데이터셋을 사용하도록 맞춘 Kubeflow DPO 학습 파이프라인입니다.

기본 데이터셋:

- `YOUR_WORKSPACE/data_kr_obj_img_4_hallu_txt_dpo_hf`
- `YOUR_WORKSPACE/data_kr_obj_img_5_hallu_wr_txt_dpo_hf`
- `YOUR_WORKSPACE/data_kr_heri_vqa_dpo_hf`

## 파일 구성

- `1_download_to_local.py`: MLXP 데이터셋을 로컬 디스크로 내려받아 `save_to_disk()` 형태로 저장합니다.
- `2_upload_dataset.sh`: 로컬에 저장한 데이터셋을 PVC로 복사합니다. MLXP에서 직접 읽을 경우 생략 가능합니다.
- `3_vlm_dpo_distributed.py`: Qwen3.5용 DPO 분산 학습 진입점입니다. 여러 DPO 데이터셋을 쉼표로 넘기면 자동으로 병합합니다.
- `4_llm_pytorchjob.yaml`: Kubeflow PyTorchJob 예시입니다. 기본값은 MLXP에서 직접 데이터를 읽습니다.
- `5_download_model.sh`: PVC에 저장된 체크포인트를 로컬로 복사합니다.
- `6_cache_datasets_to_pvc.yaml`: MLXP Data Manager dataset snapshot을 PVC로 내려받고, 로컬 경로로 다시 열리는지 검증합니다.

## 빠른 사용 순서

1. 이미지 빌드

```bash
./build.sh
```

2. 필요하면 로컬에 데이터셋 캐시

```bash
python 1_download_to_local.py
```

3. Kubeflow 실행

```bash
kubectl apply -f 4_llm_pytorchjob.yaml
```

현재 기본 시작 모델은 SFT가 끝난 체크포인트 `/data/checkpoints/qwen3_5_9b_multimodal_sft`입니다.

여러 DPO 데이터셋을 함께 학습하려면 `4_llm_pytorchjob.yaml`의 `--data_path`를 아래처럼 쉼표로 연결하면 됩니다.

```bash
--data_path YOUR_WORKSPACE/data_kr_obj_img_4_hallu_txt_dpo_hf,YOUR_WORKSPACE/data_kr_obj_img_5_hallu_wr_txt_dpo_hf,YOUR_WORKSPACE/data_kr_heri_vqa_dpo_hf
```

4. 학습 완료 후 결과 다운로드

```bash
./5_download_model.sh
```

## 메모

- `4_llm_pytorchjob.yaml`의 `MLXP_API_KEY`, `MLXP_ENDPOINT_URL`, `MLX_APIKEY`는 실제 값으로 채워야 합니다.
- `3_vlm_dpo_distributed.py`는 입력 데이터셋마다 `prompt/chosen/rejected` 컬럼을 검사한 뒤 병합합니다.
- `6_cache_datasets_to_pvc.yaml`은 `snapshot_download()`로 원본 dataset snapshot을 PVC에 받은 뒤 `mlx.sdk.data.load_dataset(local_path)`로 검증합니다.
- `3_vlm_dpo_distributed.py`는 이제 Hugging Face `load_from_disk()` 형식뿐 아니라 snapshot으로 받은 로컬 dataset 디렉터리도 fallback으로 읽을 수 있습니다.
- 현재 기본 모델은 `/data/checkpoints/qwen3_5_9b_multimodal_sft`로 설정해 두었고, 필요하면 YAML과 스크립트 인자를 바꿔 다른 시작 체크포인트로 바꿀 수 있습니다.

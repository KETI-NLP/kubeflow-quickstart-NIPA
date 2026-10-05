# VLM DPO HyperClovaX + MLXP

이 디렉터리는 `VLM_DPO` 구조를 유지하면서, HyperClovaX 모델과 Naver MLXP DPO 데이터셋을 사용하도록 맞춘 Kubeflow DPO 학습 파이프라인입니다.

기본 모델:

- `naver-hyperclovax/HyperCLOVAX-SEED-Think-32B`

호환 모델 예시:

- `naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B`

기본 데이터셋:

- `YOUR_WORKSPACE/data_kr_obj_img_4_hallu_txt_dpo_hf`

## 파일 구성

- `1_download_to_local.py`: MLXP 데이터셋을 로컬 디스크로 내려받아 `save_to_disk()` 형태로 저장합니다.
- `2_upload_dataset.sh`: 로컬에 저장한 데이터셋을 PVC로 복사합니다. MLXP에서 직접 읽을 경우 생략 가능합니다.
- `3_vlm_dpo_distributed.py`: HyperClovaX용 DPO 분산 학습 진입점입니다. 여러 DPO 데이터셋을 쉼표로 넘기면 자동으로 병합합니다.
- `4_llm_pytorchjob.yaml`: Think-32B 기준 Kubeflow PyTorchJob 예시입니다.
- `5_download_model.sh`: PVC에 저장된 체크포인트를 로컬로 복사합니다.

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

4. 학습 완료 후 결과 다운로드

```bash
./5_download_model.sh
```

## 여러 데이터셋 병합

다른 DPO 데이터셋 업로드가 끝나면 `4_llm_pytorchjob.yaml`의 `--data_path`에 쉼표로 추가하면 됩니다.

```bash
--data_path YOUR_WORKSPACE/data_kr_obj_img_4_hallu_txt_dpo_hf,YOUR_WORKSPACE/another_dpo_dataset,YOUR_WORKSPACE/yet_another_dpo_dataset
```

각 입력 데이터셋은 `prompt/chosen/rejected` 컬럼을 가져야 하며, 스크립트가 이를 검사한 뒤 병합합니다.

## 모델 메모

- 기본값은 `Think-32B`지만 `--model_name naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B`처럼 바꿔서 8B도 같은 코드로 사용할 수 있게 맞춰두었습니다.
- 모델 로딩은 `VLM_SFT_Custom_Data_HyperClovaX_Packed`의 HyperClovaX 호환 패치와 RoPE 우회 로직을 반영했습니다.

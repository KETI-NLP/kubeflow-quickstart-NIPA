# VLM SFT — Qwen3.5 (Non-Packed / On-the-fly Tokenization)

`VLM_SFT_Custom_Data_Qwen3_5_Packed` 의 **비(非)-Packed 버전**입니다.
Kubernetes(FSDP) 환경에서 Qwen3.5 VLM을 SFT 파인튜닝합니다.

## Packed 버전과의 차이

| | `_Packed` | **이 디렉토리 (non-packed)** |
|---|---|---|
| 입력 데이터셋 컬럼 | `input_ids` + `labels` + `images` (사전 토크나이즈됨) | **`messages` + `images`** (원본 SFT 포맷) |
| 토크나이즈 시점 | 별도 `pack_sft_qwen3_5_dataset.py` 선행 실행 필요 | **학습 중 data collator가 실시간 처리** |
| 별도 packing 단계 | 필요 | **불필요** |

즉 `convert_to_llm_training_ready/convert_datasets.py` 변환 결과물(`messages`+`images`)을
**packing 없이 그대로** 학습에 투입할 수 있습니다.

코드 차이는 `3_vlm_sft_distributed.py` 한 파일에만 있습니다:
- `vlm_data_collator` — 각 샘플의 `messages`+`images` 를 Qwen3.5 프로세서에 통과시켜
  `input_ids`/`labels`/`pixel_values`/`mm_token_type_ids` 를 실시간 생성. user 프롬프트 구간은
  `-100` 마스킹하여 assistant 응답에만 loss가 걸리도록 함.
- 데이터셋 검증 로직 — `input_ids`/`labels` 대신 `messages` 존재 여부를 검사.

그 외 Qwen3.5 모델 로딩·FSDP·vision merger/projector 몽키패치·콜백 등은 `_Packed` 와 동일합니다.

---

## 학습 파이프라인

### Step 0. 학습용 Docker 이미지 빌드 (최초 1회)
```bash
./build.sh
```
> 기본값으로 `registry.example.com/vlm-sft-qwen3_5:ib` 이미지가 레지스트리에 푸시됩니다.
> (`3_vlm_sft_distributed.py` 가 `/app/` 에 COPY되므로, 코드 수정 시 반드시 재빌드해야 합니다.)

### Step 1. 분산 파인튜닝 시작 (FSDP)
데이터셋은 MLXP DataManager에서 직접 로드하므로 별도 다운로드/업로드 단계가 없습니다.
```bash
kubectl apply -f 4_llm_pytorchjob_9b_simple_name_sft.yaml
```
> **모니터링:** `kubectl -n gpu-workspace logs -f vlm-sft-qwen3-5-9b-simple-name-worker-0`

### Step 2. 학습 완료 모델 로컬 다운로드
```bash
./5_download_model.sh
```

### Step 3. TensorBoard
```bash
kubectl apply -f 6_tensorboard_pod.yaml
kubectl -n gpu-workspace port-forward pod/vlm-sft-tensorboard 6006:6006
```

---

## 학습 잡 YAML

- **`4_llm_pytorchjob_9b_simple_name_sft.yaml`** — 문화재 이름 식별(simple name) SFT 잡.
  - 데이터셋: `YOUR_WORKSPACE/data_kr_heri_vqa_simple_name_sft_hf` (MLXP, 14,315건 / messages+images)
  - 모델: `/data/checkpoints/qwen3_5_vl_dpo` · 프로세서: `Qwen/Qwen3.5-9B`
  - 출력: `/data/checkpoints/qwen3_5_9b_simple_name_sft`
- 그 외 `4_llm_pytorchjob_9b*.yaml` / `27b*.yaml` 는 `_Packed` 에서 상속된 템플릿입니다.
  non-packed 코드 기준이므로 `--data_path` 에는 `messages`+`images` 컬럼을 가진 데이터셋을 지정해야 합니다.

> **노드 수:** `4_llm_pytorchjob_9b_simple_name_sft.yaml` 은 2노드(16 GPU)로 설정돼 있습니다
> (`nProcPerNode: 8` × `replicas: 2`). 노드 수를 바꾸려면 `minReplicas`/`maxReplicas`/`replicas` 를 함께 수정하세요.

`--data_path` 는 쉼표로 여러 MLXP 데이터셋을 동시에 지정할 수 있습니다.

---

## 사전 준비 사항 (Prerequisites)

1. **네임스페이스/노드 라벨** — `gpu-workspace` 네임스페이스, `example.com/gpu-pool: YOUR_GPU_NODE_POOL` 라벨.
2. **PVC** — `training-pvc` (Bound 상태). 체크포인트/로그 저장.
3. **레지스트리 Secret** — `byunggill` 도커 레지스트리 Secret (`imagePullSecrets`).
4. **MLXP 인증** — 잡 YAML의 `MLX_API_KEY`/`MLXP_ENDPOINT_URL` 환경변수로 MLXP 데이터셋 로드.
5. **Infiniband** — `hostNetwork`, `IPC_LOCK`, `NCCL_IB_DISABLE=0` 등 RDMA 경로 사용 설정 반영됨.

# 3. 이미지 빌드와 학습 실행

[이전: 데이터 준비](../02-data/README.md) · [전체 순서](../../../README.md) · [다음: 결과 확인](../04-checkpoints/README.md)

[Qwen3.5 일반 SFT README](../../../VLM_SFT_Custom_Data_Qwen3_5/README.md)를 읽고 해당 디렉터리에서 시작합니다. 패킹 데이터는 [Packed SFT README](../../../VLM_SFT_Custom_Data_Qwen3_5_Packed/README.md)를 사용하세요.

```bash
cd VLM_SFT_Custom_Data_Qwen3_5
bash build.sh
```

실행 전에 build.sh의 이미지 주소와 태그를 변경하고 레지스트리에 로그인하세요. 이 명령은 이미지를 빌드하고 푸시합니다. Dockerfile에 학습 코드가 복사되므로 코드 변경 시 재빌드가 필요합니다. 학습 YAML의 `image`를 같은 주소·태그로 맞춥니다.

첫 실행은 `4_llm_pytorchjob_9b.yaml`을 복사해 자신의 설정을 넣습니다.

```bash
cp 4_llm_pytorchjob_9b.yaml my_sft_job.yaml
```

`my_sft_job.yaml`에서 아래 항목을 수정합니다.

- 환경 단계에서 확인한 네임스페이스·PVC·GPU 수·이미지·노드 라벨.
- `--model_name`: 사용할 모델 또는 PVC에 준비한 모델 경로.
- `--processor_name`이 있으면 모델과 호환되는 processor 경로.
- `--data_path`: 앞 단계에서 준비한 데이터 경로.
- `--output_path`, `--logging_dir`: 결과와 로그 저장 경로.
- 처음에는 `--max_steps 10`, `--per_device_train_batch_size 1`로 데이터 로딩·학습·최종 저장을 확인. 모델 크기와 메모리 요구량은 줄어들지 않습니다.
- 로컬 PVC 데이터를 사용하면 MLXP 전용 Secret·환경변수 참조 제거. 필요한 Hugging Face 토큰은 별도 Secret으로 연결.

```bash
kubectl apply --dry-run=server -f my_sft_job.yaml
kubectl apply -f my_sft_job.yaml
kubectl -n YOUR_NAMESPACE get pytorchjobs,pods
kubectl -n YOUR_NAMESPACE logs -f YOUR_WORKER_POD -c pytorch
```

dry-run은 Kubernetes 명세를 확인하며 데이터 형식이나 GPU 메모리 적합성까지 검증하지는 않습니다. Pod 이름은 조회 결과를 사용하세요. 로그에서 모델·데이터 로딩과 학습 loss를 확인합니다. Pending이면 `kubectl -n YOUR_NAMESPACE describe pod YOUR_WORKER_POD`로 자원·PVC·스케줄링 상태를 확인합니다.

첫 작업이 끝나고 모델 저장까지 확인한 다음 `max_steps`와 데이터 규모를 늘리세요. 이전 작업이 GPU를 점유하고 있는지도 확인합니다.

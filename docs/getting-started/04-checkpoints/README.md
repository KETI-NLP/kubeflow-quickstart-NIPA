# 4. 체크포인트와 로그 확인

[이전: 학습 실행](../03-training/README.md) · [전체 순서](../../../README.md) · [다음: 서빙·평가](../05-serving/README.md)

[Qwen3.5 SFT README](../../../VLM_SFT_Custom_Data_Qwen3_5/README.md)의 모델 다운로드·TensorBoard 항목을 참고하세요. 로그의 학습 완료와 모델 저장 완료를 확인하고, 학습 YAML의 `--output_path`를 기록합니다. 학습 중 저장한 `checkpoint-*`와 최종 출력 디렉터리는 구분해야 합니다.

```bash
kubectl -n YOUR_NAMESPACE get pytorchjobs,pods
kubectl -n YOUR_NAMESPACE logs YOUR_WORKER_POD -c pytorch --tail=100
```

종료한 학습 Pod에 접속할 수 없다면 PVC를 마운트한 업로드·다운로드용 Pod에서 결과를 확인하세요. 추출 스크립트의 네임스페이스, PVC, 원격 결과 경로와 로컬 저장 경로를 먼저 수정합니다.

```bash
cd VLM_SFT_Custom_Data_Qwen3_5
bash 5_download_model.sh
```

모델 가중치만이 아니라 config, tokenizer, processor 등 로딩에 필요한 파일도 함께 확인합니다. 분산 체크포인트 형식에 따라 별도 변환이 필요할 수 있으므로 최종 모델 저장 로그와 파일 형식을 확인하세요.

TensorBoard를 사용할 경우 `6_tensorboard_pod.yaml`의 환경 설정과 로그 경로를 학습 설정에 맞춥니다.

```bash
kubectl apply -f 6_tensorboard_pod.yaml
kubectl -n YOUR_NAMESPACE get pods
kubectl -n YOUR_NAMESPACE port-forward pod/YOUR_TENSORBOARD_POD 6006:6006
```

브라우저에서 `http://127.0.0.1:6006`을 열어 loss와 학습 단계를 확인합니다. 다음 단계에는 서빙 프로세스가 접근할 수 있는 최종 모델 경로가 필요합니다. 로컬 서빙이면 다운로드한 경로, 클러스터 서빙이면 PVC 경로를 사용하세요.

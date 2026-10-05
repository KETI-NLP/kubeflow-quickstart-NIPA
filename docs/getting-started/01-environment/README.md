# 1. 환경 준비

[전체 순서](../../../README.md) · [다음: 데이터 준비](../02-data/README.md)

저장소를 clone한 컴퓨터는 명령을 실행하는 곳이며, 학습은 GPU 클러스터에서 실행합니다. 로컬에는 Git, Python, kubectl과 이미지 빌드용 Docker가 필요합니다. GPU 제공자가 발급한 kubeconfig와 네임스페이스를 준비하세요.

```bash
kubectl config current-context
kubectl get crd pytorchjobs.kubeflow.org
kubectl -n YOUR_NAMESPACE get pvc
kubectl -n YOUR_NAMESPACE get pods
```

`YOUR_NAMESPACE`는 할당받은 값으로 바꿉니다. PVC가 `Bound`이고 PyTorchJob CRD가 존재하는지 확인하세요. 조회 권한이 없으면 제공자에게 확인합니다.

[전체 환경 안내](../../PUBLIC_RELEASE.md)의 설정 항목을 읽고, 사용할 예제의 YAML·build.sh·업로드 및 다운로드 스크립트에서 다음 값을 맞추세요.

| 설정 | 자신의 환경에서 지정할 값 |
|---|---|
| namespace | 할당받은 네임스페이스 |
| persistentVolumeClaim.claimName | 공유 PVC 이름 |
| image / build.sh 이미지 이름 | 자신이 빌드·푸시할 이미지 주소 |
| imagePullSecrets | 해당 레지스트리 인증 Secret |
| nodeSelector | 제공자의 GPU 노드 라벨 |
| replicas / nProcPerNode / GPU 요청·제한 | 할당 GPU 수와 일치하는 구성 |
| /data 경로 | 데이터, 모델 캐시, 결과 저장 위치 |

첫 예제는 `VLM_SFT_Custom_Data_Qwen3_5`입니다. 일부 YAML은 많은 GPU·메모리와 InfiniBand를 요구합니다. `minReplicas`, `maxReplicas`, `Worker.replicas`를 함께 맞추고, 노드당 프로세스 수와 GPU 요청·제한도 일치시키세요. 모델이 들어갈 GPU 메모리와 학습 코드의 FSDP 설정까지 확인해야 합니다.

모델 접근 토큰과 레지스트리 인증정보는 환경변수 또는 Kubernetes Secret으로 전달하세요. 로컬 PVC 데이터를 사용할 때는 YAML의 MLXP 전용 환경변수와 Secret 참조를 제거하고, Hugging Face 토큰이 필요하면 별도의 HF_TOKEN Secret을 연결합니다. MLXP 키와 Hugging Face 토큰은 서로 다른 인증정보입니다.

다음 단계로 넘어가기 전에 클러스터 접근, PVC, GPU 구성과 이미지 주소를 확인하세요.

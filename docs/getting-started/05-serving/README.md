# 5. 모델 서빙과 평가

[이전: 결과 확인](../04-checkpoints/README.md) · [전체 순서](../../../README.md)

[서빙 README](../../../VLM_SERVE_EVAL_Qwen3_5/serving/README.md)를 읽고 앞 단계의 최종 모델을 연결합니다. 이 서빙 디렉터리에는 Transformers 기반 API·채팅 서버와 별도 vLLM 실행 구성이 있습니다. 기본 `run.sh serve`의 구현과 모델 지원 범위를 확인하고 사용하세요.

```bash
cd VLM_SERVE_EVAL_Qwen3_5/serving
cp configs/local.env.example configs/local.env
```

`local.env`에서 `PYTHON_BIN`, `MODEL_PATH`, `MODEL_ID`, `BASE_MODEL_NAME`, 필요시 `PROCESSOR_PATH`와 GPU 관련 설정을 수정합니다. Python 환경에는 선택한 서빙 구현의 의존성이 설치되어 있어야 하고, 로컬 서빙에는 모델을 로딩할 GPU가 필요합니다. Dockerfile 및 상세 README에서 설치 구성을 확인하세요.

```bash
bash run.sh serve --profile local
```

기본 포트는 8000입니다. 서버 로그에서 모델 로딩을 확인한 뒤 다른 터미널에서 API를 조회합니다.

```bash
curl http://127.0.0.1:8000/v1/models
```

기본 채팅 UI는 `http://127.0.0.1:8000/chat`입니다. 반환된 모델 ID와 설정한 `MODEL_ID`가 맞는지 확인하고 간단한 질의로 추론을 확인하세요.

클러스터 서빙은 `configs/k8s.env`와 해당 배포 YAML의 이미지·GPU·PVC·모델 경로를 수정한 뒤 실행합니다.

```bash
bash run.sh deploy --profile k8s
bash run.sh port-forward --profile k8s
```

평가는 [평가 README](../../../VLM_SERVE_EVAL_Qwen3_5/README.md)와 [평가 상세 가이드](../../../VLM_SERVE_EVAL_Qwen3_5/EVALUATION_GUIDE.md)를 확인하세요. LightEval·LiteLLM 설치 및 버전에 맞는 태스크 설정이 필요합니다. 실행 중인 API를 `OPENAI_BASE_URL`로 연결하고 모델 ID·태스크·출력 디렉터리를 설정합니다.

```bash
bash run.sh lighteval --profile local
```

문화유산·OCR 등 커스텀 태스크는 별도 평가 데이터가 필요합니다. 평가 결과와 사용한 모델, 데이터, 생성 설정을 함께 기록하세요. 일반 SFT와 추론을 확인한 뒤 [DPO](../../../VLM_DPO_Custom_Data_Qwen3_5/README.md), [RLHF](../../../VLM_RLHF/README.md), [패킹 학습](../../../VLM_SFT_Custom_Data_Qwen3_5_Packed/README.md)으로 확장할 수 있습니다.

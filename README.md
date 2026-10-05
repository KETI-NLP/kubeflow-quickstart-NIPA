# Kubeflow GPU 학습·패킹·서빙 예제

본 저장소는 **NIPA 정보통신산업진흥원의 「첨단 GPU 활용 지원 사업」의 결과물**입니다. 사업을 통해 제공받은 Kubeflow GPU 환경을 활용하며 작성한 LLM·VLM 데이터 준비, 패킹, 분산 학습, 체크포인트 추출, 서빙·평가 예제를 공유합니다.

## 과제와 결과물 모델

**과제명:** K-Agentic VLM: 한국형 멀티 모달 지식과 도구를 정확하게 이해하고 실행하는 한국형 VLM 에이전트 고도화

본 과제의 결과물 모델 2개는 Hugging Face에서 확인할 수 있습니다.

- [Qwen3.5-KETI-HAECHI-27B](https://huggingface.co/KETI-NLP/Qwen3.5-KETI-HAECHI-27B)
- [HyperCLOVA-X-KETI-HAECHI-32B](https://huggingface.co/KETI-NLP/HyperCLOVA-X-KETI-HAECHI-32B)

모델의 사용 방법과 배포 조건은 각 모델 페이지를 참고하세요.

## 처음 사용하는 분: 아래 순서로 진행하세요

이 저장소를 clone한 뒤 아래 문서를 1번부터 읽고 따라 하세요. 첫 실습은 Qwen3.5 일반 SFT를 기준으로 안내합니다. 패킹과 DPO·RLHF는 기본 학습 흐름을 확인한 뒤 선택하면 됩니다.

```bash
git clone https://github.com/byunggilljoe/kubeflow-quickstart-NIPA.git kubeflow-training-snippets
cd kubeflow-training-snippets
```

원본 실험 저장소 없이 이 저장소만으로 소스를 사용할 수 있습니다. GPU 클러스터, 모델·데이터와 실행 의존성은 별도로 준비해야 합니다.

| 순서 | 읽고 따라 할 문서 | 이 단계에서 할 일 |
|---|---|---|
| 1 | [환경 준비 README](docs/getting-started/01-environment/README.md) | 클러스터 접근, GPU·PVC·레지스트리 확인, 예제 설정 변경 |
| 2 | [데이터 준비·패킹 README](docs/getting-started/02-data/README.md) | 일반 SFT 데이터 준비 및 PVC 업로드, 필요하면 패킹 |
| 3 | [학습 실행 README](docs/getting-started/03-training/README.md) | 이미지 빌드, 작은 학습 작업 실행, 로그 확인 |
| 4 | [학습 결과 확인 README](docs/getting-started/04-checkpoints/README.md) | 체크포인트·TensorBoard 확인, 모델 추출 |
| 5 | [서빙·평가 README](docs/getting-started/05-serving/README.md) | 모델 서빙, API·채팅 확인, 평가 실행 |

전체 설정 항목은 [환경 설정과 실행 안내](docs/PUBLIC_RELEASE.md)에 정리되어 있습니다. 각 단계 README는 처음 실행할 때 필요한 순서와 수정할 설정을 안내하고, 기존 예제별 README는 상세 구현을 설명합니다. 실제 GPU 실행은 자신의 환경에서 확인해야 합니다.

## 예제 선택

| 목적 | 코드와 상세 안내 |
|---|---|
| 첫 VLM SFT 실습 | [Qwen3.5 일반 SFT](VLM_SFT_Custom_Data_Qwen3_5/README.md) |
| 패킹 데이터로 VLM SFT | [Qwen3.5 Packed SFT](VLM_SFT_Custom_Data_Qwen3_5_Packed/README.md) |
| 다른 VLM의 SFT | [Qwen3-VL SFT](VLM_SFT_Custom_Data_Qwen3_VL/README.md), [HyperCLOVAX SFT](VLM_SFT_Custom_Data_HyperClovaX/README.md) |
| 기본·장문 학습 | [LLM_SFT](LLM_SFT/README.md), [LLM_SFT_LONG](LLM_SFT_LONG/README.md) |
| 선호도 학습 | [LLM DPO](LLM_DPO/README.md), [VLM DPO](VLM_DPO/README.md), [Qwen3.5 커스텀 DPO](VLM_DPO_Custom_Data_Qwen3_5/README.md) |
| 강화학습 | [LLM RLHF](LLM_RLHF/README.md), [VLM RLHF](VLM_RLHF/README.md) |
| 데이터 변환·패킹·전송 | [데이터 단계 안내](docs/getting-started/02-data/README.md), `CLOUD_CONVERT_DATASET`, `CLOUD_PACK_DATASET`, `DATAMANAGER_UPLOAD` |
| 서빙·평가 상세 | [서빙 README](VLM_SERVE_EVAL_Qwen3_5/serving/README.md), [평가 README](VLM_SERVE_EVAL_Qwen3_5/README.md) |

학습 디렉터리는 대체로 Dockerfile, 이미지 빌드 스크립트, Python 학습 코드와 Kubeflow PyTorchJob YAML로 구성됩니다. 분산 방식과 의존성은 예제마다 다르며, 데이터셋이나 모델 가중치는 포함하지 않습니다.

## 라이선스

자체 코드는 [Apache-2.0](LICENSE)으로 배포합니다. 외부 코드에는 [해당 코드의 라이선스](THIRD_PARTY_NOTICES.md)가 적용됩니다. 모델과 데이터는 각각의 배포 조건을 확인하세요.

## 사업 지원

본 저장소는 **NIPA 정보통신산업진흥원의 「첨단 GPU 활용 지원 사업」의 결과물**로, 사업을 통해 활용한 Kubeflow GPU 환경에서의 학습·패킹·서빙 경험을 다른 사용자도 참고하고 활용할 수 있도록 공개합니다.

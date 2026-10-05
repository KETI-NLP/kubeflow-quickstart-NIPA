# LightEval LiteLLM 사용 가이드

이 디렉토리는 LightEval 프레임워크를 LiteLLM과 함께 사용하기 위한 커스텀 설정 및 도구들을 포함합니다.

---

## 개요

이 프로젝트는 LightEval을 사용하여 다양한 LLM 모델을 평가하기 위한 설정을 제공합니다. 주요 구성 요소는 다음과 같습니다:

- **litellm_model.py**: LiteLLM 모델 클라이언트 (수정된 버전)
- **yaml_files/**: 각 모델별 설정 파일
- **custom_tasks/**: 커스텀 테스크 정의 파일
- **debugging_files/**: 디버깅 및 출력 확인 스크립트

---

## litellm_model.py 수정 사항

기본 LightEval의 `litellm_model.py`를 수정하여 다음 기능을 추가했습니다:

### 수정 위치
원본 파일 위치: `miniconda3/envs/lighteval/lib/python3.10/site-packages/lighteval/models/endpoints/litellm_model.py`
수정된 파일 위치: `lighteval/litellm_model.py`

### 주요 수정 사항

#### 1. OpenAI 양식이 아닌 경우 충돌 방지
```python
# 212-213번째 줄
"logprobs": None,  # return_logits if self.provider == "openai" else None,
"stop": None,  # stop_sequence,
```
- OpenAI API가 아닌 다른 프로바이더를 사용할 때 `logprobs`와 `stop` 파라미터로 인한 충돌을 방지하기 위해 주석 처리

#### 2. 오류 발생 시 상세 정보 출력
```python
# 262-274번째 줄
except Exception as e:
    wait_time = min(
        64, self.API_RETRY_SLEEP * (self.API_RETRY_MULTIPLIER**attempt)
    )
    last_exception = e
    logger.warning(
        f"Error in API call: {e}, waiting {wait_time} seconds before retry {attempt + 1}/{self.API_MAX_RETRY}"
    )
    print(f"[LiteLLM ERROR] {repr(e)}")
    import traceback
    traceback.print_exc()  # 실제 예외 스택을 stderr에 강제로 출력
    time.sleep(wait_time)
```
- API 호출 실패 시 예외 정보와 스택 트레이스를 출력하여 디버깅 용이성 향상
- 최종 실패 시 `RuntimeError`로 상세한 오류 메시지와 함께 예외 발생

### 설치 방법

수정된 파일을 시스템 패키지에 복사하여 사용:

```bash
# 백업 생성
cp miniconda3/envs/lighteval/lib/python3.10/site-packages/lighteval/models/endpoints/litellm_model.py \
   miniconda3/envs/lighteval/lib/python3.10/site-packages/lighteval/models/endpoints/litellm_model.py.bak

# 수정된 파일로 교체
cp lighteval/litellm_model.py \
   miniconda3/envs/lighteval/lib/python3.10/site-packages/lighteval/models/endpoints/litellm_model.py
```

---

## YAML 설정 파일 작성

각 모델별로 YAML 설정 파일을 `yaml_files/` 디렉토리에 작성합니다.

### 파일 구조

```yaml
model_parameters:
  model_name: "openai/qwen3-32b"  # 모델 이름 (provider/모델명 형식)
  provider: "openai"              # 프로바이더 이름
  base_url: "http://127.0.0.1:4000/v1"  # API 엔드포인트 URL
  api_key: "sk-..."                # API 키
  concurrent_requests: 4          # 동시 요청 수
  verbose: true                    # 상세 로그 출력 여부
  api_max_retry: 8                 # 최대 재시도 횟수
  api_retry_sleep: 5.0             # 재시도 대기 시간 (초)
  api_retry_multiplier: 2.0        # 재시도 대기 시간 배수
  timeout: 600.0                   # 요청 타임아웃 (초)
  max_model_length: 32768          # 모델의 최대 컨텍스트 길이 (중요!)
  generation_parameters:
    temperature: 0.0               # 생성 온도
    top_p: 1.0                     # Top-p 샘플링
```

### 중요 설정 항목

#### max_model_length
- **반드시 정확히 설정해야 합니다**
- 모델의 실제 최대 컨텍스트 길이보다 작거나 같아야 합니다
- 서버/엔드포인트에서 제한하는 경우 그 값에 맞춰 설정
- 예시:
  - Qwen3-32b: `32768`
  - Solar-pro-22b: `4000`
  - Llama4-scout-17b: `8192` (서버 제한에 맞춤)

### 기존 YAML 파일 예시

- `litellm_qwen3.yaml`: Qwen3-32b 모델
- `litellm_qwen3_coder.yaml`: Qwen3-coder-30b 모델
- `litellm_solar.yaml`: Solar-pro-22b 모델
- `litellm_llama4.yaml`: Llama4-scout-17b 모델

---

## 커스텀 테스크 파일

`custom_tasks/` 디렉토리에 커스텀 테스크 정의 파일을 작성합니다.

### 기존 커스텀 테스크 파일

- `custom_kmmlu_task.py`: KMMLU 테스크
- `custom_mmlu_task.py`: MMLU 테스크
- `custom_gpqa_task.py`: GPQA 테스크
- `custom_gsm8k_task.py`: GSM8K 테스크
- `custom_ifeval_ko_task.py` : IFEval-ko 테스크
- `custom_hae_rae_bench_task.py` : HAERAE v1.1 테스크
- `custom_livecodebench_task.py` : LiveCodeBench 테스크
- `custom_mbpp_task.py` : MBPP 테스크

---

## 디버깅 파일 사용

`debugging_files/` 디렉토리의 디버깅 스크립트를 사용하여 모델 출력을 확인할 수 있습니다.

### 사용 방법

```bash
# KMMLU 예시
python debugging_files/debug_kmmlu_output.py \
  --subject Accounting \
  --num-samples 5 \
  --output-dir ./test_kmmlu_output \
  --yaml yaml_files/litellm_qwen3.yaml
```

### 주요 옵션

- `--subject`: 테스크 서브젝트 이름 (예: "Accounting", "Biology")
- `--num-samples`: 확인할 샘플 수 (기본값: 5)
- `--output-dir`: LightEval 출력 디렉토리 (캐시된 결과를 읽기 위해)
- `--yaml`: LiteLLM YAML 설정 파일 경로
- `--no-cache`: 캐시를 사용하지 않고 모델을 직접 호출

### 디버깅 스크립트 기능

1. **LightEval 캐시 로드**: 이미 실행한 결과가 있으면 캐시에서 로드
2. **샘플별 상세 분석**: 각 샘플의 질문, 선택지, 정답, 모델 출력 표시
3. **메트릭 계산**: 실제 메트릭 계산 결과 확인
4. **정답 매칭 분석**: 모델 출력에서 정답이 포함되는지 상세 분석

### 디버깅 파일 목록

- `debug_kmmlu_output.py`: KMMLU 디버깅
- `debug_mmlu_output.py`: MMLU 디버깅
- `debug_gpqa_output.py`: GPQA 디버깅
- `debug_gsm8k_output.py`: GSM8K 디버깅
- `debug_ifeval_ko_output.py` : IFEval-ko 디버깅
- `debug_hae_rae_output.py` : HAERAE v1.1 디버깅
- `debug_livecodebench_output.py` : LiveCodeBench 디버깅
- `debug_mbpp_output.py` : MBPP 디버깅

---

## LightEval 실행 방법

### 기본 명령어 형식

```bash
lighteval endpoint litellm <YAML_FILE> <TASK> [OPTIONS]
```

### 예시: 커스텀 테스크 실행

```bash
# KMMLU 커스텀 테스크 실행
lighteval endpoint litellm \
  "$HOME/KETI/lighteval/lighteval/yaml_files/litellm_qwen3.yaml" \
  "kmmlu_gen:Accounting|0" \
  --custom-tasks "$HOME/KETI/lighteval/lighteval/custom_tasks/custom_kmmlu_task.py" \
  --max-samples 100 \
  --save-details \
  --output-dir ./test_kmmlu_output
```

#### MMLU 예시
```bash
# MMLU 커스텀 테스크 실행
lighteval endpoint litellm \
  "$HOME/KETI/lighteval/lighteval/yaml_files/litellm_qwen3.yaml" \
  "mmlu_gen:abstract_algebra|0" \
  --custom-tasks "$HOME/KETI/lighteval/lighteval/custom_tasks/custom_mmlu_task.py" \
  --max-samples 100 \
  --save-details \
  --output-dir ./test_mmlu_output
```

#### LiveCodeBench 예시 (v5 subset = qwen3 32b 기준)
```bash
# LiveCodeBench v5 subset 실행 (880개 문제)
lighteval endpoint litellm \
  "$HOME/KETI/lighteval/lighteval/yaml_files/litellm_qwen3.yaml" \
  "lcb_gen_qwen_v5|0" \
  --custom-tasks "$HOME/KETI/lighteval/lighteval/custom_tasks/custom_livecodebench_task.py" \
  --max-samples 100 \
  --save-details \
  --output-dir ./test_livecodebench_v5
```

#### LiveCodeBench 예시 (날짜 필터링 = llama4 기준)
```bash
# LiveCodeBench 날짜 필터링 버전 실행 (210개 문제)
# 만약 다른 날짜 범위 수정 원하는 경우 custom_livecodebench_task.py의 날짜 수정
lighteval endpoint litellm \
  "$HOME/KETI/lighteval/lighteval/yaml_files/litellm_qwen3.yaml" \
  "lcb_gen_qwen_date|0" \
  --custom-tasks "$HOME/KETI/lighteval/lighteval/custom_tasks/custom_livecodebench_task.py" \
  --max-samples 100 \
  --save-details \
  --output-dir ./test_livecodebench_date
```

#### gsm8k 예시
```bash
# GPQA 메인 데이터셋 실행
lighteval endpoint litellm \
  "$HOME/KETI/lighteval/lighteval/yaml_files/litellm_qwen3.yaml" \
  "gsm8k_gen" \
  --custom-tasks "$HOME/KETI/lighteval/lighteval/custom_tasks/custom_gsm8k_task.py" \
  --max-samples 100 \
  --save-details \
  --output-dir ./test_gsm8k_output
```

#### GPQA 예시
```bash
# GPQA 메인 데이터셋 실행
lighteval endpoint litellm \
  "$HOME/KETI/lighteval/lighteval/yaml_files/litellm_qwen3.yaml" \
  "gpqa_gen:main|0" \
  --custom-tasks "$HOME/KETI/lighteval/lighteval/custom_tasks/custom_gpqa_task.py" \
  --max-samples 100 \
  --save-details \
  --output-dir ./test_gpqa_output
```

#### MBPP 예시
```bash
# MBPP 테스트 세트 실행
lighteval endpoint litellm \
  "$HOME/KETI/lighteval/lighteval/yaml_files/litellm_qwen3.yaml" \
  "mbpp_gen:test|0" \
  --custom-tasks "$HOME/KETI/lighteval/lighteval/custom_tasks/custom_mbpp_task.py" \
  --max-samples 100 \
  --save-details \
  --output-dir ./test_mbpp_output
```

### 주요 옵션

- `--custom-tasks`: 커스텀 테스크 파일 경로
- `--max-samples`: 최대 샘플 수 (0은 전체)
- `--save-details`: 상세 결과 저장 (parquet 파일)
- `--output-dir`: 출력 디렉토리

### 테스크 이름 형식

```
<task_name>:<subject>|<num_shots>|<version>
```

예시:
- `kmmlu_gen:Accounting|0`: KMMLU Accounting 서브젝트, 0-shot
- `mmlu_gen:abstract_algebra|0`: MMLU abstract_algebra 서브젝트, 0-shot
- `kmmlu_gen` : KMMLU 전체, 0-shot

---

## 주의사항

### 1. 새로운 테스크 실행 시 디버깅 필수

- 새로운 테스크를 실행하거나 성능이 이상할 때는 **반드시 디버깅 스크립트로 출력을 확인**하세요
- 모델이 올바른 형식으로 답변하는지, 메트릭이 제대로 계산되는지 확인 필요
- **주의**: 디버깅 스크립트의 결과와 실제 lighteval 실행 결과/순서가 다를 수 있습니다. 샘플 확인용으로 사용하세요

### 2. 사이즈 관련 문제

사이즈 관련 문제 발생 시 다음을 확인하세요:

1. **custom task 파일의 `generation_size`**
   - LightevalTaskConfig의 `generation_size`가 `max_model_length`보다 충분히 작아야 합니다
   - generation_size를 따로 설정하지 않아도 됩니다(현재)

2. **YAML 파일의 `max_model_length`**
   - 모델의 실제 최대 컨텍스트 길이로 설정해야 합니다
   - 서버 제한이 있는 경우 그 값에 맞춰 설정
   - 현재 yaml 파일에 알맞게 설정되어 있습니다

---

## 문제 해결

### 문제 1: 다른 모델/설정으로 변경하였으나 같은 점수

**원인**: 기본적으로 이전에 돌린 결과가 있다면 자동으로 캐시 사용

**해결 방법**:
1. 캐시 삭제
```
find ~/.cache/huggingface/lighteval -type d -name "{task_name(예: *lcb*)}" -exec rm -rf {} + 2>/dev/null 
```
2. 재시도

---

## 파일 구조

```
lighteval/
├── README.md                    
├── litellm_model.py             # 수정된 LiteLLM 모델 클라이언트
├── ifeval_ko/                   # ifeval_ko metric 관련 정의
├── yaml_files/                  # 모델별 YAML 설정 파일
│   ├── litellm_qwen3.yaml
│   ├── litellm_qwen3_coder.yaml
│   ├── litellm_solar.yaml
│   └── litellm_llama4.yaml
├── custom_tasks/                # 커스텀 테스크 정의
│   ├── custom_kmmlu_task.py
│   ├── custom_mmlu_task.py
│   └── ...
└── debugging_files/              # 디버깅 스크립트
    ├── debug_kmmlu_output.py
    ├── debug_mmlu_output.py
    └── ...
```


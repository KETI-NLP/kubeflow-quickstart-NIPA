# Korean Knowledge Custom Tasks

이 문서는 현재 프로젝트에서 사용 중인 한국지식 및 OCR 계열 custom task를 한 번에 정리한 문서입니다.

정리 대상:
- 한국어 OCR benchmark
- 한국 문화재 이름 식별 VQA benchmark
- 한국 문화재 text short-answer QA benchmark

## 개요

현재 한국지식 관련 custom task는 아래 3축으로 나뉩니다.

1. OCR
- 이미지 1장을 보고 한국어 문구를 읽어내는 task
- 파일: [custom_korean_character_ocr_task.py](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/custom_tasks/custom_korean_character_ocr_task.py)

2. Name VQA
- 이미지 1장을 보고 문화재 이름을 맞히는 task
- 파일: [custom_korean_heritage_name_vqa_task.py](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/custom_tasks/custom_korean_heritage_name_vqa_task.py)

3. Text ShortQA
- 이미지 없이 텍스트 질문만 보고 한국 문화재 관련 fact를 단답형으로 맞히는 task
- 파일: [custom_korean_heritage_text_shortqa_task.py](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/custom_tasks/custom_korean_heritage_text_shortqa_task.py)

## 1. Korean Character OCR

### 목적

한국어 간판, 표지, 현판, 공공 표기 등의 이미지를 보고 텍스트를 인식합니다.

모델 출력 형식:

```text
OCR: <인식한 텍스트>
```

### task 이름

- `korean_character_ocr`
- `korean_character_ocr_public_executive`
- `korean_character_ocr_outside`
- `korean_character_ocr_data13`
- `korean_character_ocr_all_tests`

### 데이터셋 경로

기본 경로는 task 파일 내부에서 환경변수로 덮어쓸 수 있게 정의되어 있습니다.

- `KOREAN_CHARACTER_OCR_PUBLIC_EXECUTIVE_DATASET_PATH`
- `KOREAN_CHARACTER_OCR_OUTSIDE_DATASET_PATH`
- `KOREAN_CHARACTER_OCR_DATA13_DATASET_PATH`
- `KOREAN_CHARACTER_OCR_MULTI_DATASET_PATH`

현재 기본값:
- `public_executive`: `/workspace/2026_llm_data_generation/llm_training_ready/data_public_executive_ocr_hf/test`
- `outside`: `/workspace/2026_llm_data_generation/llm_training_ready/data_030_korean_character_outside_hf/test`
- `data13`: `/workspace/2026_llm_data_generation/llm_training_ready/data_13_korean_character_hf/test`

### 주요 metric

- `ocr_strict_em`
- `ocr_relaxed_em`
- `ocr_korean_only_em`
- `ocr_cer`
- `ocr_format_valid`
- `ocr_empty_response`

설명:
- `strict_em`: `OCR: ...` 마지막 줄의 문자열이 gold와 완전히 같은지
- `relaxed_em`: 공백/기호/포장 문구를 일부 정리한 뒤 일치하는지
- `korean_only_em`: 한글 문자열만 추출했을 때 같은지
- `cer`: 문자 오류율

### 실행 스크립트

- [run_korean_character_ocr_public_executive_gpt4o.sh](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/run_korean_character_ocr_public_executive_gpt4o.sh)
- [run_korean_character_ocr_outside_gpt4o.sh](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/run_korean_character_ocr_outside_gpt4o.sh)
- [run_korean_character_ocr_data13_gpt4o.sh](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/run_korean_character_ocr_data13_gpt4o.sh)

### 간단 실행 예시

```bash
./scripts/run_korean_character_ocr_public_executive_gpt4o.sh
./scripts/run_korean_character_ocr_outside_gpt4o.sh
./scripts/run_korean_character_ocr_data13_gpt4o.sh
```

샘플 수를 줄여 smoke test:

```bash
MAX_SAMPLES=1 ./scripts/run_korean_character_ocr_outside_gpt4o.sh
```

다른 모델 YAML 사용:

```bash
MODEL_YAML=yaml_files/litellm_gemini_2_5_pro.yaml \
MAX_SAMPLES=10 \
./scripts/run_korean_character_ocr_data13_gpt4o.sh
```

### 메모

- `public_executive`는 `MAX_SAMPLES=1`이어도 초기 준비 비용이 커서 매우 오래 걸릴 수 있습니다.
- `outside`, `data13`는 상대적으로 빠르게 smoke test가 끝났습니다.

## 2. Korean Heritage Name VQA

### 목적

문화재 이미지를 보고 공식 한글 명칭을 맞히는 VQA task입니다.

모델 출력 형식:

```text
Answer: <문화재명>
```

### task 이름

- `korean_heritage_name_vqa`

### 관련 파일

- task: [custom_korean_heritage_name_vqa_task.py](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/custom_tasks/custom_korean_heritage_name_vqa_task.py)
- JSON 생성: [build_korean_heritage_name_vqa.py](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/build_korean_heritage_name_vqa.py)
- HF 변환: [build_korean_heritage_name_vqa_hf.py](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/build_korean_heritage_name_vqa_hf.py)
- 기본 HF dataset 경로: `/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/tmp_eval_datasets/korean_heritage_name_vqa_paraphrase_hf`

### 데이터 특징

- 원천 데이터에서 문화재명 식별형만 추출
- `heritage_name`에서 한자 병기는 제거하고 한글 대표명만 정답으로 사용
- 여러 질문 패러프레이즈를 포함
- 이미지 입력을 실제로 사용

### 주요 metric

- `name_vqa_strict_em`
- `name_vqa_relaxed_em`
- `name_vqa_ordered_recall`
- `name_vqa_ordered_precision`
- `name_vqa_ordered_f1`
- `name_vqa_format_valid`
- `name_vqa_empty_response`

설명:
- `strict_em`: 정확한 문화재명 완전일치
- `relaxed_em`: 공백/문장부호 정규화 후 일치
- `ordered_*`: LCS 기반 부분일치 보조 지표

### 실행 스크립트

- [run_korean_heritage_name_vqa_gpt4o.sh](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/run_korean_heritage_name_vqa_gpt4o.sh)

### 간단 실행 예시

```bash
./scripts/run_korean_heritage_name_vqa_gpt4o.sh
```

적은 샘플로 빠른 확인:

```bash
MAX_SAMPLES=20 ./scripts/run_korean_heritage_name_vqa_gpt4o.sh
```

출력 디렉토리 지정:

```bash
OUTPUT_DIR=./test_name_vqa_manual_run \
MAX_SAMPLES=50 \
./scripts/run_korean_heritage_name_vqa_gpt4o.sh
```

### 참고 문서

- [gpt4o_korean_heritage_name_vqa_error_examples.md](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/gpt4o_korean_heritage_name_vqa_error_examples.md)
- [gpt4o_korean_heritage_name_vqa_lcs_eval_summary.md](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/gpt4o_korean_heritage_name_vqa_lcs_eval_summary.md)

## 3. Korean Heritage Text ShortQA

### 목적

텍스트 질문만 보고 한국 문화재 관련 fact를 단답형으로 맞히는 benchmark입니다.

모델 출력 형식:

```text
Answer: <짧은 정답>
```

### task 이름

- `korean_heritage_text_shortqa`

### 관련 파일

- task: [custom_korean_heritage_text_shortqa_task.py](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/custom_tasks/custom_korean_heritage_text_shortqa_task.py)
- JSON 생성: [build_korean_heritage_text_shortqa.py](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/build_korean_heritage_text_shortqa.py)
- HF 변환: [build_korean_heritage_text_shortqa_hf.py](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/build_korean_heritage_text_shortqa_hf.py)
- 결과 후처리: [summarize_korean_heritage_text_shortqa_results.py](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/summarize_korean_heritage_text_shortqa_results.py)

기본 HF dataset 경로:
- `/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/tmp_eval_datasets/korean_heritage_text_shortqa_benchmark_hf`

### 데이터 특징

- 원천 `text` DPO 데이터에서 단답형 fact만 추출
- 최종 포함 타입:
  - `designation_date`
  - `creation_year`
  - `quantity`
  - `designation_type`
  - `material`
- `designation_name`은 질문에 답이 포함되는 사례가 많아서 최종 benchmark에서 제거
- `material`은 전수 재검토 후 질문을 모두 `재질만 묻는 질문`으로 재작성

### 주요 metric

- `text_shortqa_strict_em`
- `text_shortqa_relaxed_em`
- `text_shortqa_ordered_recall`
- `text_shortqa_ordered_precision`
- `text_shortqa_ordered_f1`
- `text_shortqa_format_valid`
- `text_shortqa_empty_response`

### answer type별 제약

프롬프트에 answer type별 제약을 넣어두었습니다.

예:
- `creation_year`: 반드시 `YYYY년`
- `designation_date`: 반드시 `YYYY년 M월 D일`
- `quantity`: 반드시 `숫자+단위`

즉 모델이 `고려시대 후기`처럼 설명형으로 답하는 것을 줄이도록 설계했습니다.

### 실행 스크립트

- [run_korean_heritage_text_shortqa_gpt4o.sh](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/run_korean_heritage_text_shortqa_gpt4o.sh)

### 간단 실행 예시

```bash
./scripts/run_korean_heritage_text_shortqa_gpt4o.sh
```

작은 샘플 smoke:

```bash
MAX_SAMPLES=20 ./scripts/run_korean_heritage_text_shortqa_gpt4o.sh
```

다른 결과 디렉토리로 저장:

```bash
OUTPUT_DIR=./test_text_shortqa_manual_run \
MAX_SAMPLES=100 \
./scripts/run_korean_heritage_text_shortqa_gpt4o.sh
```

이 task는 실행 후 자동으로 타입별 요약도 생성합니다.

생성 파일:
- `<OUTPUT_DIR>/analysis/per_answer_type_scores.json`
- `<OUTPUT_DIR>/analysis/per_answer_type_scores.md`

### 참고 문서

- [korean_heritage_text_shortqa_build_summary.md](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/korean_heritage_text_shortqa_build_summary.md)
- [korean_heritage_text_shortqa_methodology.md](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/korean_heritage_text_shortqa_methodology.md)
- [korean_heritage_text_shortqa_source_vs_benchmark_examples.md](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/korean_heritage_text_shortqa_source_vs_benchmark_examples.md)
- [gpt4o_korean_heritage_text_shortqa_smoke_summary.md](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/gpt4o_korean_heritage_text_shortqa_smoke_summary.md)
- [gpt4o_korean_heritage_text_shortqa_prompt_comparison.md](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/gpt4o_korean_heritage_text_shortqa_prompt_comparison.md)

## 공통 실행 방식

모든 간단 실행 스크립트는 내부적으로 아래 helper를 사용합니다.

- [scripts/_run_lighteval_task.sh](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/scripts/_run_lighteval_task.sh)

공통 특징:
- `uv run --python .venv/bin/python -m lighteval ...` 사용
- 기본 모델 YAML: `yaml_files/litellm_gpt4o.yaml`
- 기본 `MAX_SAMPLES=100`
- `--save-details` 활성화

공통 환경변수:
- `MODEL_YAML`
- `MAX_SAMPLES`
- `OUTPUT_DIR`

예:

```bash
MODEL_YAML=yaml_files/litellm_gpt4o.yaml \
MAX_SAMPLES=10 \
OUTPUT_DIR=./tmp_run \
./scripts/run_korean_heritage_name_vqa_gpt4o.sh
```

## 결과 확인 방법

실행이 끝나면 보통 아래가 생성됩니다.

- `results/openai/gpt-4o/results_*.json`
- `details/openai/gpt-4o/.../details_*.parquet`

빠르게 보고 싶으면:

```bash
find <OUTPUT_DIR>/results -name 'results_*.json' | sort | tail -n 1
find <OUTPUT_DIR>/details -name 'details_*.parquet' | sort | tail -n 1
```

## 추천 시작점

처음 확인할 때는 아래 순서를 추천합니다.

1. OCR smoke

```bash
MAX_SAMPLES=1 ./scripts/run_korean_character_ocr_outside_gpt4o.sh
```

2. Name VQA smoke

```bash
MAX_SAMPLES=20 ./scripts/run_korean_heritage_name_vqa_gpt4o.sh
```

3. Text ShortQA smoke

```bash
MAX_SAMPLES=20 ./scripts/run_korean_heritage_text_shortqa_gpt4o.sh
```

이 순서로 보면
- 이미지 OCR 파이프라인
- 이미지 기반 한국지식 VQA
- 텍스트 기반 한국지식 QA

를 단계적으로 확인할 수 있습니다.

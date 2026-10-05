import custom_tasks.force_no_think_patch
from PIL import Image
"""
LiveCodeBench 커스텀 테스크 (Qwen용)
- v5 subset 사용 (release_v5, 총 880문제)
- 또는 날짜 필터링 사용 (llama용)

사용 방법:
    # v5 subset 사용 (qwen 공식)
    lighteval endpoint litellm "$HOME/KETI/lighteval/litellm_qwen3.yaml" \
      "lcb_gen_qwen_v5|0" \
      --custom-tasks "$HOME/KETI/lighteval/custom_livecodebench_filtered_task_qwen.py" \
      --num-samples 1 \
      --save-details \
      --output-dir ./test_livecodebench_qwen_v5
    
    # 날짜 필터링 사용 (llama 공식)
    lighteval endpoint litellm "$HOME/KETI/lighteval/litellm_llama4.yaml" \
      "lcb_gen_qwen_date|0" \
      --custom-tasks "$HOME/KETI/lighteval/custom_livecodebench_filtered_task_qwen.py" \
      --num-samples 1 \
      --save-details \
      --output-dir ./test_livecodebench_qwen_date
"""

import json
from datetime import datetime
from typing import Dict, Any, Callable, Optional

import numpy as np

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks.lcb.codegen_metrics import (
    codegen_metrics,
    translate_private_test_cases,
)
from lighteval.models.model_output import ModelResponse
from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetric, SamplingMethod


START_DATE = datetime(2024, 10, 1)  # 2024-10-01
END_DATE = datetime(2025, 2, 1)     # 2025-02-01


def parse_date(date_str: str) -> Optional[datetime]:
    if not date_str:
        return None
    try:
        if 'T' in date_str:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        for fmt in ['%Y-%m-%d', '%Y/%m/%d']:
            try:
                return datetime.strptime(date_str, fmt)
            except:
                continue
    except:
        pass
    return None


def filter_by_date(record: Dict[str, Any], start_date: datetime, end_date: datetime) -> bool:
    """
    레코드를 날짜로 필터링        
    Returns:
        bool: 날짜 범위 내에 있으면 True
    """
    contest_date_str = record.get("contest_date")
    if not contest_date_str:
        return False
    
    contest_date = parse_date(contest_date_str)
    if contest_date is None:
        return False
    
    return start_date <= contest_date < end_date


def record_to_sample(record: Dict[str, Any], use_date_filter: bool = False) -> Optional[Dict[str, Any]]:
    if use_date_filter:
        if not filter_by_date(record, START_DATE, END_DATE):
            return None
    return record


def extract_code_qwen(model_output: str) -> str:
    """
    Qwen용 코드 추출기
    
    1. 모든 ```python 또는 ``` 코드 블록 찾기
    2. stdin/stdout을 사용하는 블록 우선 선택
    3. 없으면 마지막 블록 선택
    4. 코드 블록 앞의 import 문도 함께 추출
    """
    if not model_output:
        return ""
    
    outputlines = model_output.split("\n")
    
    blocks = []
    block_start = None
    in_block = False
    
    for i, line in enumerate(outputlines):
        if "```python" in line.lower() or (line.strip() == "```" and not in_block):
            if block_start is not None and in_block:
                blocks.append((block_start, i))
            block_start = i
            in_block = True
        elif in_block and "```" in line:
            if i > block_start:
                blocks.append((block_start, i))
            in_block = False
            block_start = None
    
    if block_start is not None and in_block:
        blocks.append((block_start, len(outputlines)))
    
    # 코드 블록이 하나도 없으면, stdin/stdout 패턴 기반으로 추출 시도
    if not blocks:
        code_lines = []
        in_code = False
        for i, line in enumerate(outputlines):
            if any(keyword in line for keyword in ["input()", "sys.stdin", "readline", "print("]):
                in_code = True
            if in_code:
                code_lines.append(line)
                if i < len(outputlines) - 1:
                    next_line = outputlines[i + 1]
                    if next_line.strip().startswith("def ") or next_line.strip().startswith("class "):
                        break
        
        if code_lines:
            imports = []
            code_start_idx = next((i for i, line in enumerate(outputlines) if line in code_lines[:5]), 0)
            for i in range(max(0, code_start_idx - 30), code_start_idx):
                line = outputlines[i].strip()
                if line.startswith("import ") or line.startswith("from "):
                    imports.append(outputlines[i])
            
            for line in code_lines:
                if line.strip().startswith("import ") or line.strip().startswith("from "):
                    imports.append(line)
            
            seen = set()
            unique_imports = []
            for imp in imports:
                if imp not in seen:
                    seen.add(imp)
                    unique_imports.append(imp)
            
            if unique_imports:
                return "\n".join(unique_imports) + "\n\n" + "\n".join(code_lines)
            return "\n".join(code_lines)
        
        return ""
    
    # 코드 블록이 여러 개일 때, stdin/stdout/임포트 여부 기준으로 가장 적절한 블록 선택
    candidate_blocks = []
    for start_idx, end_idx in blocks:
        code_lines = outputlines[start_idx + 1:end_idx]
        block_content = "\n".join(code_lines).strip()
        
        if not block_content:
            continue
        
        has_stdin_stdout = any(keyword in block_content for keyword in [
            "input()", "sys.stdin", "readline", "readlines", "print("
        ])
        
        has_imports = any(
            line.strip().startswith("import ") or line.strip().startswith("from ") 
            for line in code_lines
        )
        
        candidate_blocks.append({
            "lines": code_lines,
            "content": block_content,
            "has_stdin_stdout": has_stdin_stdout,
            "has_imports": has_imports,
            "start_idx": start_idx,
            "end_idx": end_idx,
        })
    
    if not candidate_blocks:
        return ""
    
    # stdin/stdout 사용하는 블록 우선, 없으면 import 있는 블록, 그것도 없으면 마지막 블록
    stdin_blocks = [b for b in candidate_blocks if b["has_stdin_stdout"]]
    if stdin_blocks:
        selected_block = stdin_blocks[-1]
    else:
        import_blocks = [b for b in candidate_blocks if b["has_imports"]]
        if import_blocks:
            selected_block = import_blocks[-1]
        else:
            selected_block = candidate_blocks[-1]
    
    code_lines = selected_block["lines"]
    
    # "# YOUR CODE HERE" 같은 템플릿 라인은 제거
    cleaned_lines = []
    for line in code_lines:
        stripped = line.strip()     
        if stripped == "# YOUR CODE HERE" or stripped.startswith("# YOUR CODE"):
            continue
        cleaned_lines.append(line)
    
    if not cleaned_lines:
        return ""
    
    code_has_imports = any(
        line.strip().startswith("import ") or line.strip().startswith("from ")
        for line in cleaned_lines
    )
    
    imports = []
    if not code_has_imports:
        start_idx = selected_block["start_idx"]
        for i in range(max(0, start_idx - 30), start_idx):
            line = outputlines[i].strip()
            if line.startswith("import ") or line.startswith("from "):
                imports.append(outputlines[i])
    
    final_code = "\n".join(cleaned_lines)
    
    if imports and not code_has_imports:
        seen = set()
        unique_imports = []
        for imp in imports:
            if imp not in seen:
                seen.add(imp)
                unique_imports.append(imp)
        if unique_imports:
            return "\n".join(unique_imports) + "\n\n" + final_code
    
    return final_code  


def lcb_codegeneration_filtered_prompt_fn(line, task_name: str = "lcb_gen_filtered", use_date_filter: bool = False) -> Doc:
    """
    LiveCodeBench 프롬프트 함수
    GPQA 스타일로 instruction과 query를 분리하여 Bedrock API 호환성 확보
    """
    if use_date_filter:
        if not filter_by_date(line, START_DATE, END_DATE):
            return None  
    
    if not line.get("question_content"):
        return None  
    
    try:
        instruction = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."
        
        question_content = line.get("question_content", "").strip()
        if not question_content:
            return None  
        
        query = f"{instruction}\n\nQuestion: {question_content}\n\n"
        if starter_code := line.get("starter_code", None):
            starter_code = starter_code.strip() if starter_code else None
            if starter_code:
                query += "You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n"
                query += f"```python\n{starter_code}\n```\n\n"
            else:
                query += "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
                query += "```python\n# YOUR CODE HERE\n```\n\n"
        else:
            query += "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
            query += "```python\n# YOUR CODE HERE\n```\n\n"
        
        try:
            public_test_cases_str = line.get("public_test_cases", "[]")
            if isinstance(public_test_cases_str, str):
                public_test_cases = json.loads(public_test_cases_str)
            else:
                public_test_cases = public_test_cases_str
        except (json.JSONDecodeError, TypeError) as e:
            public_test_cases = []
        
        try:
            private_test_cases = translate_private_test_cases(line.get("private_test_cases", ""))
        except Exception:
            private_test_cases = []
        
        inputs = []
        outputs = []
        for test in public_test_cases + private_test_cases:
            if isinstance(test, dict) and "input" in test and "output" in test:
                inputs.append(test["input"])
                outputs.append(test["output"])
        
        fn_name = None
        try:
            metadata_str = line.get("metadata", "{}")
            if isinstance(metadata_str, str):
                metadata = json.loads(metadata_str)
            else:
                metadata = metadata_str
            fn_name = metadata.get("func_name", None) if metadata else None
        except (json.JSONDecodeError, TypeError, AttributeError):
            fn_name = None
        
        if not inputs or not outputs:
            return None  
        
        return Doc(
            task_name=task_name,
            query=query,
            choices=[""],
            gold_index=0,
            instruction=instruction,  
            specific={
                "inputs": inputs,
                "outputs": outputs,
                "fn_name": fn_name,
            },
        )
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            f"Error processing sample (question_id: {line.get('question_id', 'unknown')}): {e}. "
            f"Skipping this sample."
        )
        return None  


class CodegenMetricFiltered(SampleLevelComputation):
    """
    LiveCodeBench용 코드 생성 메트릭 (lighteval 기본 구현과 동일)
    """
    
    def compute(self, model_response: ModelResponse, doc: Doc, **kwargs) -> dict:
        """Estimates the Pass@1 metric for the code generation task."""
        assert doc.specific is not None, "Doc specific field is required for codegen_metric"

        predictions = model_response.final_text
        generated_code_snippets = [[extract_code_qwen(pred) for pred in predictions]]
        evaluation_sample = {
            "inputs": doc.specific["inputs"],
            "outputs": doc.specific["outputs"],
            "fn_name": doc.specific["fn_name"],
        }
        evaluation_sample = [{"input_output": json.dumps(evaluation_sample)}]

        metrics, _ = codegen_metrics(
            evaluation_sample,
            generated_code_snippets,
            k_list=[1],  
            num_process_evaluate=1,
            timeout=10,  
        )
        return metrics["pass@1"]


lcb_codegen_filtered_metric = SampleLevelMetric(
    metric_name="codegen_pass@1:16",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=CodegenMetricFiltered(),
    corpus_level_fn=np.mean,
    batched_compute=False,
)


def create_custom_livecodebench_qwen_task(
    subset: str = "release_v5",
    use_date_filter: bool = False,
    prompt_function: Optional[Callable] = None
) -> LightevalTaskConfig:
    """
    LiveCodeBench 커스텀 테스크 생성 (Qwen용)    
    Returns:
        LightevalTaskConfig: 커스텀 테스크 설정
    """
    if prompt_function is None:
        def prompt_fn(line, task_name: str = "lcb_gen_qwen"):
            return lcb_codegeneration_filtered_prompt_fn(line, task_name, use_date_filter)
        prompt_function = prompt_fn
    
    def sample_fn(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return record_to_sample(record, use_date_filter)
    
    task_name = "lcb_gen_qwen_v5" if not use_date_filter else "lcb_gen_qwen_date"
    
    # release_v5는 880개, 전체 데이터셋은 1055개
    actual_subset = None if use_date_filter else subset
    
    return LightevalTaskConfig(
        name=task_name,
        prompt_function=prompt_function,
        hf_repo="lighteval/code_generation_lite",
        hf_subset=actual_subset,  
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        metrics=[lcb_codegen_filtered_metric],  
        stop_sequence=[],  
        sample_fields=sample_fn,  
    )


CUSTOM_LIVECODEBENCH_FILTERED_TASKS = {
    "lcb_gen_qwen_v5": create_custom_livecodebench_qwen_task(subset="release_v5", use_date_filter=False),  # 880개
    "lcb_gen_qwen_date": create_custom_livecodebench_qwen_task(subset="release_v5", use_date_filter=True),  # 전체 데이터셋에서 날짜 필터링 (210개)
}

TASKS_TABLE = list(CUSTOM_LIVECODEBENCH_FILTERED_TASKS.values())


if __name__ == "__main__":
    print("=" * 80)
    print("LiveCodeBench Qwen 커스텀 테스크 설정 확인")
    print("=" * 80)
    
    print("\n[1] release_v5 subset 사용 (날짜 필터링 없음, 880개)")
    print("-" * 80)
    test_task_v5 = create_custom_livecodebench_qwen_task(subset="release_v5", use_date_filter=False)
    print(f"Task name: {test_task_v5.name}")
    print(f"Subset: {test_task_v5.hf_subset}")
    print(f"Date filter: 사용 안 함")
    print(f"Metrics: {test_task_v5.metrics}")
    print(f"Metric name: {test_task_v5.metrics[0].metric_name}")
    print(f"Category: {test_task_v5.metrics[0].category}")
    print(f"Generation size: {test_task_v5.generation_size}")
    print(f"Evaluation splits: {test_task_v5.evaluation_splits}")
    
    print("\n[2] 날짜 필터링 사용 (전체 데이터셋에서 필터링)")
    print("-" * 80)
    print(f"Date filter: {START_DATE.date()} ~ {END_DATE.date()}")
    test_task_date = create_custom_livecodebench_qwen_task(subset="release_v5", use_date_filter=True)
    print(f"Task name: {test_task_date.name}")
    print(f"Subset: {test_task_date.hf_subset} (None = 전체 데이터셋)")
    print(f"Date filter: {START_DATE.date()} ~ {END_DATE.date()}")
    print(f"Metrics: {test_task_date.metrics}")
    print(f"Metric name: {test_task_date.metrics[0].metric_name}")
    print(f"Category: {test_task_date.metrics[0].category}")
    print(f"Generation size: {test_task_date.generation_size}")
    print(f"Evaluation splits: {test_task_date.evaluation_splits}")
    
    print("\n" + "=" * 80)
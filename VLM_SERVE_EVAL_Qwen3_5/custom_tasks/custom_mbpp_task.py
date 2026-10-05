import custom_tasks.force_no_think_patch
from PIL import Image
"""
MBPP 커스텀 테스크 생성

사용 방법:
    lighteval endpoint litellm "$HOME/KETI/lighteval/litellm_solar.yaml" \
      "mbpp_gen:test|0" \
      --custom-tasks "$HOME/KETI/lighteval/custom_mbpp_task.py" \
      --num-samples 1 \
      --save-details \
      --output-dir ./test_mbpp_custom
"""

from typing import Dict, Any, Callable, Optional

import numpy as np

from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetric, SamplingMethod
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.models.model_output import ModelResponse
import subprocess
import tempfile
from pathlib import Path


def _extract_problem_description_only(text: str, test_list: list) -> str:
    """
    텍스트에서 문제 설명만 추출 (테스트 케이스 제거)
    """
    patterns_to_remove = [
        "Your code should satisfy these tests:",
        "Your code should pass these tests:",
        "Your code should satisfy these test cases:",
    ]
    
    result = text
    for pattern in patterns_to_remove:
        if pattern in result:
            idx = result.find(pattern)
            result = result[:idx].strip()
            break
    
    if test_list:
        for test in test_list:
            if test.strip() in result:
                result = result.replace(test.strip(), "").strip()
            if "assert" in test and len(test.split("assert")) > 1:
                assert_part = test.split("assert")[1].strip()
                if assert_part in result:
                    result = result.replace(assert_part, "").strip()
    
    return result.strip()

# lighteval 공식 extract_code 함수 <- 로는 완벽히 처리 X (특히 qwen)
# def extract_code(model_output: str) -> str:
#     outputlines = model_output.split("\n")
#     indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
#     if len(indexlines) < 2:
#         return ""
#     return "\n".join(outputlines[indexlines[-2] + 1 : indexlines[-1]])


def extract_code(model_output: str, test_list: list = None) -> str:
    """
    MBPP용 코드 추출: 코드 블록들 중 'def '를 포함하는 블록의 전체 내용 추출 (마커 제외)
    """
    outputlines = model_output.split("\n")
    
    target_function_name = None
    if test_list:
        for test in test_list:
            if "assert" in test and "(" in test:
                func_part = test.split("assert")[1].split("(")[0].strip()
                if func_part:
                    target_function_name = func_part
                    break
       
    blocks = []
    block_start = None
    
    for i, line in enumerate(outputlines):
        if "```python" in line or "```" in line:
            if block_start is None:
                block_start = i
            else:
                if i > block_start:
                    blocks.append((block_start, i))
                block_start = None
    
    # 각 블록을 확인하여 'def '를 포함하는 블록 찾기
    candidate_blocks = []
    for start_idx, end_idx in blocks:
        code_lines = outputlines[start_idx + 1:end_idx]
        block_content = "\n".join(code_lines)
        if "def " in block_content:
            candidate_blocks.append(code_lines)
    
    if target_function_name and candidate_blocks:
        for code_lines in candidate_blocks:
            code_text = "\n".join(code_lines)
            if f"def {target_function_name}" in code_text:
                full_code = _extract_with_imports(outputlines, code_lines)
                return full_code
    
    if candidate_blocks:
        code_lines = candidate_blocks[-1]
        full_code = _extract_with_imports(outputlines, code_lines)
        return full_code
    
    for i, line in enumerate(outputlines):
        if line.strip().startswith("def "):
            code_lines = [line]
            for j in range(i + 1, len(outputlines)):
                next_line = outputlines[j]
                if next_line.strip().startswith("def ") and j > i:
                    break
                code_lines.append(next_line)
            full_code = _extract_with_imports(outputlines, code_lines)
            return full_code
    
    return ""


def _extract_with_imports(outputlines: list, code_lines: list) -> str:
    """
    코드와 함께 필요한 import 문 추출
    """
    imports = []
    def_start_idx = None
    
    for i, line in enumerate(outputlines):
        if any(code_lines[0] in outputlines[j] for j in range(max(0, i-10), min(len(outputlines), i+10))):
            def_start_idx = i
            break
    
    if def_start_idx:
        for i in range(max(0, def_start_idx - 20), def_start_idx):
            line = outputlines[i].strip()
            if line.startswith("import ") or line.startswith("from "):
                imports.append(outputlines[i])
    
    seen = set()
    unique_imports = []
    for imp in imports:
        if imp not in seen:
            seen.add(imp)
            unique_imports.append(imp)
    
    if unique_imports:
        return "\n".join(unique_imports) + "\n\n" + "\n".join(code_lines)
    return "\n".join(code_lines)

class MBPPCodegenMetric(SampleLevelComputation):
    """
    MBPP용 코드 생성 메트릭 (LiveCodeBench의 CodegenMetric 구조 참고)
    test_list (assert 문 리스트)를 실행하여 평가
    """
    
    def compute(self, model_response: ModelResponse, doc: Doc, **kwargs) -> dict:
        assert doc.specific is not None, "Doc specific field is required for mbpp_codegen_metric"

        predictions = model_response.final_text
        test_list = doc.specific["test_list"]
        generated_code_snippets = [extract_code(pred, test_list) for pred in predictions]
        test_setup_code = doc.specific.get("test_setup_code", "")
        
        if not generated_code_snippets or not generated_code_snippets[0]:
            return 0.0
        
        passed = False
        for generated_code in generated_code_snippets:
            if not generated_code:
                continue
            
            full_code = f"""{test_setup_code}

{generated_code}

# Run tests - assert 실패 시 즉시 종료 (returncode != 0)
"""
            for assert_stmt in test_list:
                full_code += f"{assert_stmt}\n"
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(full_code)
                temp_file = f.name
            
            try:
                result = subprocess.run(
                    ['python', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=6
                )
                # subprocess 실행 결과로 판단
                # assert가 실패하면 AssertionError가 발생하여 프로그램이 종료되므로 returncode != 0
                # returncode == 0이면 모든 assert가 통과한 것
                if result.returncode == 0:
                    passed = True
                    break
            except subprocess.TimeoutExpired:
                pass
            except Exception:
                pass
            finally:
                try:
                    Path(temp_file).unlink()
                except:
                    pass
        
        return 1.0 if passed else 0.0


mbpp_pass_at_1_metric = SampleLevelMetric(
    metric_name="mbpp_pass@1",
    sample_level_fn=MBPPCodegenMetric(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)


def record_to_sample(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    MBPP 데이터셋 레코드를 샘플로 변환
    """
    return {
        "task_id": record.get("task_id", ""),
        "text": record.get("text") or record.get("prompt", ""),  
        "code": record.get("code", ""),
        "test_list": record.get("test_list", []),
        "test_setup_code": record.get("test_setup_code") or record.get("test_imports", ""),
        "challenge_test_list": record.get("challenge_test_list", []),
    }


def mbpp_prompt(line: Dict[str, Any], task_name: str = None) -> Doc:
    """
    MBPP 프롬프트 함수
    """
    text = line.get("text") or line.get("prompt", "")
    
    code = line.get("code", "")
    
    test_list = line.get("test_list", [])
    test_setup_code = line.get("test_setup_code") or line.get("test_imports", "")
    
    if code:
        problem_desc = _extract_problem_description_only(text, test_list)
        test_list_str = "\n".join(test_list) if test_list else ""
        if test_list_str:
            query = f"""{problem_desc} Your code should satisfy these tests:
    
{test_list_str}"""
        else:
            query = _extract_problem_description_only(text, test_list)
        choices = [code]
    elif test_list:
        test_list_str = "\n".join(test_list)
        query = f"""{text} Your code should satisfy these tests:
    
{test_list_str}"""
        choices = [""]  
    else:
        query = text
        choices = [""]
    
    doc = Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=0,
        instruction="",
        specific={
            "test_list": test_list,
            "test_setup_code": test_setup_code,
        },
    )
    
    return doc


def create_custom_mbpp_task(
    subset: str = "full",
    split: str = "test",
    prompt_function: Optional[Callable] = None
) -> LightevalTaskConfig:
    """
    MBPP 커스텀 테스크 생성
        
    Returns:
        LightevalTaskConfig: 커스텀 테스크 설정
    """
    if prompt_function is None:
        prompt_function = mbpp_prompt
    
    return LightevalTaskConfig(
        name=f"mbpp_gen:{subset}",
        prompt_function=prompt_function,
        hf_repo="google-research-datasets/mbpp",
        hf_subset=subset if subset != "full" else None,  
        hf_avail_splits=["train", "test", "validation", "prompt"],
        evaluation_splits=[split],
        few_shots_split="prompt",  
        few_shots_select="random_sampling",  
        num_fewshots=3,  
        metrics=[mbpp_pass_at_1_metric],  
        stop_sequence=["\n\n\n"],  
        version=1,
    
        sample_fields=record_to_sample,
    )


CUSTOM_MBPP_TASKS = {
    "mbpp_gen:test": create_custom_mbpp_task(subset="full", split="test"),
    "mbpp_gen:train": create_custom_mbpp_task(subset="full", split="train"),
    "mbpp_gen:validation": create_custom_mbpp_task(subset="full", split="validation"),
    "mbpp_gen_sanitized:test": create_custom_mbpp_task(subset="sanitized", split="test"),
}

TASKS_TABLE = list(CUSTOM_MBPP_TASKS.values())


if __name__ == "__main__":
    test_task = create_custom_mbpp_task()
    print(f"Task name: {test_task.name}")
    print(f"Metrics: {test_task.metrics}")
    print(f"Metric name: {test_task.metrics[0].metric_name}")
    print(f"Category: {test_task.metrics[0].category}")
    print(f"Generation size: {test_task.generation_size}")
    print(f"Evaluation splits: {test_task.evaluation_splits}")


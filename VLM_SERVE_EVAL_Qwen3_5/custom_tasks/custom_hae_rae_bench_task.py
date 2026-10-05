import custom_tasks.force_no_think_patch
from PIL import Image
"""
HAE_RAE_BENCH_1.1 커스텀 테스크 생성

사용 방법:
    lighteval endpoint litellm "$HOME/KETI/lighteval/litellm_qwen3.yaml" \
      "hae_rae_bench_gen:correct_definition_matching|0" \
      --custom-tasks "$HOME/KETI/lighteval/custom_hae_rae_bench_task.py" \
      --max-samples 5 \
      --save-details \
      --output-dir ./test_hae_rae_bench_custom
"""

import ast
import re
from typing import Callable, Dict, Any

import numpy as np

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetric, SamplingMethod
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.models.model_output import ModelResponse
from custom_tasks.custom_mc_metric import strict_mc_metric


def record_to_sample(record):
    return {
        "query": record["query"].strip(),
        "options": record["options"] if isinstance(record["options"], list) else [record["options"]],
        "answer": record["answer"].strip(),
    }


def is_multiple_choice(line: Dict[str, Any]) -> bool:
    """
    Returns:
        True: 객관식 (options가 리스트이고 answer가 (A), (B) 형식)
        False: 주관식 (options가 nan이거나 answer가 자유 텍스트)
    """
    answer_str = line.get("answer", "").strip()
    options = line.get("options", None)
    
    if answer_str.startswith("(") and answer_str.endswith(")"):
        answer_letter = answer_str[1:-1].strip()
        if answer_letter.upper() in LETTER_INDICES:
            return True
    
    # options가 리스트이고 길이가 2 이상이면 객관식
    if isinstance(options, list) and len(options) >= 2:
        return True
    
    # options가 문자열이고 파싱 가능한 리스트면 객관식
    if isinstance(options, str) and options.lower() != "nan":
        try:
            parsed = ast.literal_eval(options)
            if isinstance(parsed, list) and len(parsed) >= 2:
                return True
        except:
            pass
    
    # 그 외는 주관식
    return False


def hae_rae_bench_prompt_multiple_choice(line, task_name: str = None):
    """
    HAE_RAE_BENCH 객관식 문제 프롬프트 함수
    """
    answer_str = line["answer"].strip()
    # (A) -> A, (B) -> B 등으로 변환
    if answer_str.startswith("(") and answer_str.endswith(")"):
        answer_letter = answer_str[1:-1].strip()
    else:
        answer_letter = answer_str.strip()
    
    if isinstance(line["options"], list):
        choices = [str(opt).strip() for opt in line["options"]]
    elif isinstance(line["options"], str):
        try:
            choices = ast.literal_eval(line["options"])
            choices = [str(opt).strip() for opt in choices]
        except:
            choices = [line["options"].strip()]
    else:
        choices = [str(line["options"]).strip()]
    
    num_choices = len(choices)
    choice_letters = LETTER_INDICES[:num_choices]
    
    if answer_letter.upper() in choice_letters:
        gold_index = choice_letters.index(answer_letter.upper())
    else:
        gold_index = 0
    
    if gold_index >= len(choices):
        gold_index = 0
    
    query = f"""다음 객관식 질문에 대해 답변하시오. 응답의 마지막 줄은 다음 형식으로 작성하시오: '정답: $LETTER' (인용 없이) 이때 LETTER는 {''.join(choice_letters)} 중 하나입니다. 질문을 잘 읽고 답변을 작성하시오.

### 질문: {line['query']}
"""
    
    return Doc(
        task_name=task_name,
        query=query,
        choices=[f" {letter}" for letter in choice_letters],
        gold_index=gold_index,
        instruction="",
        specific={"answer": answer_str, "is_multiple_choice": True},
    )


def hae_rae_bench_prompt_subjective(line, task_name: str = None):
    """
    HAE_RAE_BENCH 주관식 문제 프롬프트 함수
    """
    answer_str = line["answer"].strip()
    
    query = f"""다음 질문에 대해 답변하시오. 질문을 잘 읽고 풀이과정 없이 답만 작성하시오.

### 질문: {line['query']}
"""
    
    return Doc(
        task_name=task_name,
        query=query,
        choices=[answer_str],  
        gold_index=0,
        instruction="",
        specific={"answer": answer_str, "is_multiple_choice": False},
    )


def hae_rae_bench_prompt(line, task_name: str = None):
    """
    문제 유형에 따라 객관식/주관식 프롬프트 선택
    """
    if is_multiple_choice(line):
        return hae_rae_bench_prompt_multiple_choice(line, task_name)
    else:
        return hae_rae_bench_prompt_subjective(line, task_name)


class SubjectiveAnswerMetric(SampleLevelComputation):
    
    def __init__(self, normalize: bool = True):
        """
            normalize: True면 공백, 대소문자 정규화 후 비교
        """
        self.normalize = normalize
    
    def _normalize_text(self, text: str) -> str:
        if not text:
            return ""
        normalized = re.sub(r'\s+', ' ', text.strip()).lower()
        return normalized
    
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        """
        주관식 답변 평가
        """
        if not doc.specific or "answer" not in doc.specific:
            return 0.0
        
        gold_answer = doc.specific["answer"]
        predictions = model_response.final_text
        
        if not predictions:
            return 0.0
        
        for pred in predictions:
            if self.normalize:
                gold_norm = self._normalize_text(gold_answer)
                pred_norm = self._normalize_text(pred)
                # 정답 전체가 출력에 포함되어 있으면 정답
                if gold_norm in pred_norm:
                    return 1.0
            else:
                gold_stripped = gold_answer.strip()
                if gold_stripped in pred:
                    return 1.0
        
        return 0.0


subjective_metric = SampleLevelMetric(
    metric_name="exact_match",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=SubjectiveAnswerMetric(normalize=True),
    corpus_level_fn=np.mean,
    batched_compute=False,
)


def create_custom_hae_rae_bench_task(
    subject: str,
    prompt_function: Callable = None
) -> LightevalTaskConfig:
    """
    HAE_RAE_BENCH 서브젝트별 커스텀 테스크 생성
    문제 유형에 따라 적절한 metric 사용:
    - 객관식: gpqa_instruct_metric (Answer: LETTER 형식)
    - 주관식: exact_match (자유 텍스트 답변)
        
    Returns:
        LightevalTaskConfig: 커스텀 테스크 설정
    """
    if prompt_function is None:
        prompt_function = hae_rae_bench_prompt
    
    from datasets import load_dataset
    try:
        dataset = load_dataset("HAERAE-HUB/HAE_RAE_BENCH_1.1", subject, split="test")
        if len(dataset) > 0:
            sample = dataset[0]
            is_mc = is_multiple_choice(sample)
        else:
            is_mc = True  
    except:
        is_mc = True  
    
    # 문제 유형에 따라 metric 선택
    if is_mc:
        metrics = [strict_mc_metric]
    else:
        metrics = [subjective_metric]
    
    return LightevalTaskConfig(
        name=f"hae_rae_bench_gen:{subject}",
        prompt_function=prompt_function,
        hf_repo="HAERAE-HUB/HAE_RAE_BENCH_1.1",
        hf_subset=subject,
        hf_avail_splits=["train", "test"],
        evaluation_splits=["test"],
        few_shots_split=None,  
        few_shots_select=None,
        metrics=metrics,
        stop_sequence=[],
        generation_size=512,
        version=1,
      
        sample_fields=record_to_sample,
    )


HAE_RAE_BENCH_SUBJECTS = [
    "correct_definition_matching",
    "csat_geo",
    "csat_law",
    "csat_socio",
    "date_understanding",
    "general_knowledge",
    "history",
    "loan_words",
    "lyrics_denoising",
    "proverbs_denoising",
    "rare_words",
    "standard_nomenclature",
    "reading_comprehension",
]


CUSTOM_HAE_RAE_BENCH_TASKS = {
    f"hae_rae_bench_gen:{subject}": create_custom_hae_rae_bench_task(subject)
    for subject in HAE_RAE_BENCH_SUBJECTS
}


TASKS_TABLE = list(CUSTOM_HAE_RAE_BENCH_TASKS.values())


if __name__ == "__main__":
    test_task = create_custom_hae_rae_bench_task("correct_definition_matching")
    print(f"Task name: {test_task.name}")
    print(f"Metrics: {test_task.metrics}")
    print(f"Metric name: {test_task.metrics[0].metric_name}")
    print(f"Category: {test_task.metrics[0].category}")
    print(f"Generation size: {test_task.generation_size}")
    print(f"Evaluation splits: {test_task.evaluation_splits}")


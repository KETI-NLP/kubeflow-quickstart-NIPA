import custom_tasks.force_no_think_patch
from PIL import Image
"""
MMLU 커스텀 테스크 생성 예제

사용 방법:
    lighteval endpoint litellm "$HOME/KETI/lighteval/litellm_qwen3.yaml" \
      "mmlu_gen:abstract_algebra|0" \
      --custom-tasks "$HOME/KETI/lighteval/custom_mmlu_task.py" \
      --max-samples 5 \
      --save-details \
      --output-dir ./test_mmlu_custom
"""

import re
from typing import Callable

import numpy as np
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetric, SamplingMethod
from lighteval.tasks.default_prompts import mmlu_helm, LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.models.model_output import ModelResponse
from lighteval.utils.utils import as_list


def custom_mmlu_prompt(line, task_name: str = None):
    """
    커스텀 MMLU 프롬프트 함수
    """
    subject = line["subject"]
    
    query = f"""Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.
    
Question: {line['question']}\n"""
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, line["choices"])])
    query += "Answer:"
    
    
    gold_ix = LETTER_INDICES.index(line["answer"]) if isinstance(line["answer"], str) else line["answer"]
    
    return Doc(
        task_name=task_name,
        query=query,
        choices=[" A", " B", " C", " D"],
        gold_index=gold_ix,
        fewshot_sorting_class=line["choices"][gold_ix],
        instruction="",  
    )


def create_custom_mmlu_task(
    subject: str, 
    prompt_function: Callable = None
) -> LightevalTaskConfig:
    """
    MMLU 서브젝트별 커스텀 테스크 생성    
    Returns:
        LightevalTaskConfig: 커스텀 테스크 설정
    """
    if prompt_function is None:
        prompt_function = custom_mmlu_prompt
    
    return LightevalTaskConfig(
        name=f"mmlu_gen:{subject}",  
        prompt_function=prompt_function,
        hf_repo="lighteval/mmlu",
        hf_subset=subject,
        hf_avail_splits=["auxiliary_train", "test", "validation", "dev"],
        evaluation_splits=["test"],
        few_shots_split="dev",
        few_shots_select=None,
        metrics=[Metrics.gpqa_instruct_metric], 
        stop_sequence=[],  
        version=1,
      
    )


# MMLU 전체 서브젝트 리스트
MMLU_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


CUSTOM_MMLU_TASKS = {
    f"mmlu_gen:{subject}": create_custom_mmlu_task(subject)
    for subject in MMLU_SUBJECTS
}


TASKS_TABLE = list(CUSTOM_MMLU_TASKS.values())


if __name__ == "__main__":
    test_task = create_custom_mmlu_task("abstract_algebra")
    print(f"Task name: {test_task.name}")
    print(f"Metrics: {test_task.metrics}")
    print(f"Metric name: {test_task.metrics[0].metric_name}")
    print(f"Category: {test_task.metrics[0].category}")


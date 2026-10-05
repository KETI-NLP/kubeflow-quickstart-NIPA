import custom_tasks.force_no_think_patch
"""
KMMLU 커스텀 테스크 생성
- lighteval의 기본 extractive_match 메트릭 사용

사용 방법:
    lighteval endpoint litellm "$HOME/KETI/lighteval/litellm_qwen3.yaml" \
      "kmmlu_gen:Accounting|0" \
      --custom-tasks "$HOME/KETI/lighteval/custom_kmmlu_task.py" \
      --max-samples 5 \
      --save-details \
      --output-dir ./test_kmmlu_custom
"""

from typing import Callable

from PIL import Image

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from custom_tasks.custom_mc_metric import strict_mc_metric


def record_to_sample(record):
    return {
        "question": record["question"].strip(),
        "A": record["A"].strip(),
        "B": record["B"].strip(),
        "C": record["C"].strip(),
        "D": record["D"].strip(),
        "answer": record["answer"],
        "Category": record.get("Category", ""),
    }


def kmmlu_prompt(line, task_name: str = None):
    
    if isinstance(line["answer"], int):
        gold_index = line["answer"]
        if gold_index > 0:
            gold_index = gold_index - 1
        if gold_index < 0 or gold_index >= 4:
            gold_index = 0
    elif isinstance(line["answer"], str):
        gold_index = LETTER_INDICES.index(line["answer"].upper()) if line["answer"].upper() in LETTER_INDICES else 0
    else:
        gold_index = 0
    
    choices = [line["A"], line["B"], line["C"], line["D"]]
    
    query = f"""다음 객관식 질문에 대해 답변하시오. 응답의 마지막 줄은 다음 형식으로 작성하시오: '정답: $LETTER' (인용 없이) 이때 LETTER는 ABCD 중 하나입니다. 질문을 잘 읽고 답변을 작성하시오.

{line['question']}
"""
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, choices)])
    query += "정답: 차근 차근 생각해봅시다."
    
    return Doc(
        task_name=task_name,
        query=query,
        choices=[" A", " B", " C", " D"],
        gold_index=gold_index,
        instruction="",  
    )


def create_custom_kmmlu_task(
    subject: str,
    prompt_function: Callable = None
) -> LightevalTaskConfig:
    """
    KMMLU 서브젝트별 커스텀 테스크 생성
    """
    if prompt_function is None:
        prompt_function = kmmlu_prompt
    
    return LightevalTaskConfig(
        name=f"kmmlu_gen:{subject}",
        prompt_function=prompt_function,
        hf_repo="HAERAE-HUB/KMMLU",
        hf_subset=subject,
        hf_avail_splits=["train", "dev", "test"],
        evaluation_splits=["test"],
        few_shots_split=None,  
        few_shots_select=None,
        metrics=[strict_mc_metric],
        stop_sequence=[],
        generation_size=512,
        version=1,
      
        sample_fields=record_to_sample,
    )


KMMLU_SUBJECTS = [
    "Accounting",
    "Agricultural-Sciences",
    "Aviation-Engineering-and-Maintenance",
    "Biology",
    "Chemical-Engineering",
    "Chemistry",
    "Civil-Engineering",
    "Computer-Science",
    "Construction",
    "Criminal-Law",
    "Ecology",
    "Economics",
    "Education",
    "Electrical-Engineering",
    "Electronics-Engineering",
    "Energy-Management",
    "Environmental-Science",
    "Fashion",
    "Food-Processing",
    "Gas-Technology-and-Engineering",
    "Geomatics",
    "Health",
    "Industrial-Engineer",
    "Information-Technology",
    "Interior-Architecture-and-Design",
    "Law",
    "Machine-Design-and-Manufacturing",
    "Management",
    "Maritime-Engineering",
    "Marketing",
    "Materials-Engineering",
    "Mechanical-Engineering",
    "Nondestructive-Testing",
    "Patent",
    "Political-Science-and-Sociology",
    "Psychology",
    "Public-Safety",
    "Railway-and-Automotive-Engineering",
    "Real-Estate",
    "Refrigerating-Machinery",
    "Social-Welfare",
    "Taxation",
    "Telecommunications-and-Wireless-Technology",
    "Korean-History",
    "Math",
]


CUSTOM_KMMLU_TASKS = {
    f"kmmlu_gen:{subject}": create_custom_kmmlu_task(subject)
    for subject in KMMLU_SUBJECTS
}


TASKS_TABLE = list(CUSTOM_KMMLU_TASKS.values())


if __name__ == "__main__":
    test_task = create_custom_kmmlu_task("Accounting")
    print(f"Task name: {test_task.name}")
    print(f"Metrics: {test_task.metrics}")
    print(f"Metric name: {test_task.metrics[0].metric_name}")
    print(f"Category: {test_task.metrics[0].category}")
    print(f"Generation size: {test_task.generation_size}")
    print(f"Evaluation splits: {test_task.evaluation_splits}")


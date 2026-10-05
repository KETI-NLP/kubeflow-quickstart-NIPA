import custom_tasks.force_no_think_patch
from PIL import Image
"""
GPQA 커스텀 테스크 생성

사용 방법:
    lighteval endpoint litellm "$HOME/KETI/lighteval/litellm_qwen3.yaml" \
      "gpqa_gen|0" \
      --custom-tasks "$HOME/KETI/lighteval/custom_gpqa_task.py" \
      --max-samples 5 \
      --save-details \
      --output-dir ./test_gpqa_custom
"""

from typing import Callable

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.default_prompts import LETTER_INDICES, gpqa_instruct
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


# lighteval의 기본 extractive_match 메트릭 사용 (gpqa_instruct_metric)


def record_to_sample(record):
    return {
        "Question": record["Question"].strip(),
        "Incorrect Answer 1": record["Incorrect Answer 1"],
        "Incorrect Answer 2": record["Incorrect Answer 2"],
        "Incorrect Answer 3": record["Incorrect Answer 3"],
        "Correct Answer": record["Correct Answer"],
    }


GPQA_SHUFFLE_SEED = 42

def _hash_string_to_int(s: str) -> int:
    return hash(s) % (2**31)

def gpqa_instruct_fixed(line, task_name: str = None):
    """
    같은 질문에 대해서는 항상 같은 순서가 나오도록 질문 내용을 시드로 사용
    """
    question = line["Question"].strip()
    question_seed = _hash_string_to_int(question) + GPQA_SHUFFLE_SEED
    
    correct_answer = line["Correct Answer"]
    incorrect_answers = [
        line["Incorrect Answer 1"],
        line["Incorrect Answer 2"],
        line["Incorrect Answer 3"]
    ]
    
    import random
    random_state = random.Random(question_seed)
    
    all_choices = incorrect_answers + [correct_answer]
    random_state.shuffle(all_choices)
    
    gold_index = all_choices.index(correct_answer)
    
    choices = all_choices
    
    instruction = "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering."
    query_template = "{Instruction}\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
    query = query_template.format(
        A=choices[0].strip(),
        B=choices[1].strip(),
        C=choices[2].strip(),
        D=choices[3].strip(),
        Question=line["Question"].strip(),
        Instruction=instruction,
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=LETTER_INDICES[: len(choices)],
        gold_index=gold_index,
        instruction=instruction,
    )


def create_custom_gpqa_task(
    subset: str,
    prompt_function: Callable = None,
    task_name_suffix: str = None
) -> LightevalTaskConfig:
    """
    GPQA 서브셋별 커스텀 테스크 생성
    Returns:
        LightevalTaskConfig: 커스텀 테스크 설정
    """
    if prompt_function is None:
        prompt_function = gpqa_instruct_fixed
    
    if task_name_suffix:
        task_name = f"gpqa_gen:{task_name_suffix}"
    else:
        task_name = f"gpqa_gen:{subset.replace('gpqa_', '')}"

    generation_size = 8192 if prompt_function in (gpqa_instruct, gpqa_instruct_fixed) else 50
    
    return LightevalTaskConfig(
        name=task_name,
        prompt_function=prompt_function,
        hf_repo="Idavidrein/gpqa",
        hf_subset=subset,
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split=None,
        few_shots_select=None,
        metrics=[Metrics.gpqa_instruct_metric],  # lighteval의 기본 extractive_match 메트릭 사용
        stop_sequence=[], 
        version=1,
      
        sample_fields=record_to_sample,  # 원래 GPQA와 동일
    )


# 모든 GPQA 커스텀 테스크 생성 (원래 GPQA의 모든 테스크를 generative로)

CUSTOM_GPQA_TASKS = {
    "gpqa_gen:main": create_custom_gpqa_task(
        subset="gpqa_main",
        prompt_function=gpqa_instruct_fixed, 
        task_name_suffix="main"
    ),
    "gpqa_gen:extended": create_custom_gpqa_task(
        subset="gpqa_extended",
        prompt_function=gpqa_instruct_fixed, 
        task_name_suffix="extended"
    ),
    "gpqa_gen:diamond": create_custom_gpqa_task(
        subset="gpqa_diamond",
        prompt_function=gpqa_instruct_fixed, 
        task_name_suffix="diamond"
    ),
}


TASKS_TABLE = list(CUSTOM_GPQA_TASKS.values())


if __name__ == "__main__":
    test_task = create_custom_gpqa_task("gpqa_main")
    print(f"Task name: {test_task.name}")
    print(f"Metrics: {test_task.metrics}")
    print(f"Metric name: {test_task.metrics[0].metric_name}")
    print(f"Category: {test_task.metrics[0].category}")


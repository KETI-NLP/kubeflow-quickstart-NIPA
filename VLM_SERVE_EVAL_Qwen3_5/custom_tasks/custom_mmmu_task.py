import custom_tasks.force_no_think_patch
from PIL import Image
"""
MMMU custom task for LightEval.

Usage:
    lighteval endpoint litellm "yaml_files/litellm_qwen3_vl.yaml" \
      "mmmu_gen:Accounting|0" \
      --custom-tasks "custom_tasks/custom_mmmu_task.py" \
      --max-samples 5 \
      --save-details \
      --output-dir ./test_mmmu_output
"""

from typing import Callable, Any
import ast
import json

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def record_to_sample(record):
    return {
        "id": record["id"],
        "question": record["question"],
        "options": record["options"],
        "answer": record["answer"],
        "image_1": record.get("image_1", None),
        "image_2": record.get("image_2", None),
        "image_3": record.get("image_3", None),
        "image_4": record.get("image_4", None),
        "image_5": record.get("image_5", None),
        "image_6": record.get("image_6", None),
        "image_7": record.get("image_7", None),
        "question_type": record["question_type"],
    }


def mmmu_prompt(line: dict[str, Any], task_name: str = None) -> Doc:
    question = line["question"]
    options = line["options"]
    answer = line["answer"]
    question_type = line["question_type"]

    images = []
    for i in range(1, 8):
        img_key = f"image_{i}"
        if line.get(img_key) is not None:
            images.append(line[img_key])

    # Multiple-choice
    if question_type == "multiple-choice":
        parsed_options = []
        if isinstance(options, str):
            try:
                parsed_options = ast.literal_eval(options)
            except:
                try:
                    parsed_options = json.loads(options)
                except:
                    parsed_options = []
        elif isinstance(options, list):
            parsed_options = options
            
        choices_text = ""
        choices = []
        if parsed_options:
            for idx, opt in enumerate(parsed_options):
                letter = LETTER_INDICES[idx]
                choices_text += f"{letter}. {opt}\n"
                choices.append(f" {letter}")
        
        query = (
            f"다음을 읽고 정답을 고르시오. 응답의 마지막 줄은 반드시 'Answer: $LETTER' (인용 없이) 형식이어야 합니다. $LETTER는 선택지 중 하나입니다.\n\n"
            f"{question}\n{choices_text}\n"
            f"CRITICAL INSTRUCTION: DO NOT output any thinking process, reasoning, or explanations. DO NOT output 'Thinking Process:'. Output ONLY the final answer.\n정답:"
        )
        
        gold_index = 0
        if answer in LETTER_INDICES:
            gold_index = LETTER_INDICES.index(answer)
            
        return Doc(
            task_name=task_name,
            query=query,
            choices=choices if choices else [" A", " B", " C", " D"],
            gold_index=gold_index,
            images=images,
            instruction="",
        )
    
    # Open-ended
    else:
        query = (
            f"다음을 읽고 질문에 답하시오. 풀이 과정 없이 정답만 짧게 적으시오.\n\n"
            f"{question}\n"
            f"CRITICAL INSTRUCTION: DO NOT output any thinking process, reasoning, or explanations. DO NOT output 'Thinking Process:'. Output ONLY the final answer.\n정답:"
        )
        return Doc(
            task_name=task_name,
            query=query,
            choices=[answer],
            gold_index=0,
            images=images,
            instruction="",
            specific={"answer": answer}
        )


def create_custom_mmmu_task(subject: str) -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name=f"mmmu_gen:{subject}",
        prompt_function=mmmu_prompt,
        hf_repo="MMMU/MMMU",
        hf_subset=subject,
        hf_avail_splits=["validation", "test"],
        evaluation_splits=["validation"],  # test split does not have answers
        few_shots_split=None,
        few_shots_select=None,
        metrics=[Metrics.gpqa_instruct_metric],  # for multiple-choice formatting
        stop_sequence=[],
        version=1,
    generation_size=2048,
    
        sample_fields=record_to_sample,
    )


# TODO: Add all other MMMU subjects here
MMMU_SUBJECTS = [
    "Accounting",
    "Math",
]

CUSTOM_MMMU_TASKS = {
    f"mmmu_gen:{subject}": create_custom_mmmu_task(subject)
    for subject in MMMU_SUBJECTS
}

TASKS_TABLE = list(CUSTOM_MMMU_TASKS.values())

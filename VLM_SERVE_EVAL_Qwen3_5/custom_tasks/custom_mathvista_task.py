import custom_tasks.force_no_think_patch
from PIL import Image
"""
MathVista custom task for LightEval.
Dataset: AI4Math/MathVista (testmini split, 1000 samples)
Task: Visual mathematical reasoning - diagrams, geometry, charts, etc.

Usage:
    lighteval endpoint litellm "yaml_files/litellm_gpt4o.yaml" \\
      "mathvista_gen:default|0" \\
      --custom-tasks "custom_tasks/custom_mathvista_task.py" \\
      --max-samples 5 --save-details --output-dir ./test_mathvista
"""

import re
from typing import Any

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from custom_tasks.custom_mc_metric import strict_mc_metric


from datasets import load_dataset

MATHVISTA_DS_MAP = None

def get_mathvista_image(pid):
    global MATHVISTA_DS_MAP
    if MATHVISTA_DS_MAP is None:
        ds = load_dataset("AI4Math/MathVista", "default", split="testmini")
        MATHVISTA_DS_MAP = {str(row["pid"]): row["decoded_image"] for row in ds}
    return MATHVISTA_DS_MAP.get(str(pid))


def record_to_sample(record):
    return {
        "pid": record["pid"],
        "question": record["question"],
        "answer": record["answer"],
        "image": record.get("decoded_image", None) or record.get("image", None),
        "choices": record.get("choices", None),  # present for multiple-choice
        "question_type": record.get("question_type", "free_form"),  # 'multi_choice' or 'free_form'
    }


def mathvista_prompt(line: dict[str, Any], task_name: str = None) -> Doc:
    question = line["question"]
    answer = str(line["answer"])
    question_type = line.get("question_type", "free_form")
    choices = line.get("choices", None)

    images = []
    if "pid" in line:
        img = get_mathvista_image(line["pid"])
        if img is not None:
            images.append(img)
    elif line.get("image") is not None:
        img = line["image"]
        if isinstance(img, str):
            img = Image.open(img)
        images.append(img)

    if question_type == "multi_choice" and choices:
        choices_text = ""
        choice_labels = []
        for idx, choice in enumerate(choices):
            letter = LETTER_INDICES[idx]
            choices_text += f"{letter}. {choice}\n"
            choice_labels.append(f" {letter}")

        query = (
            f"Look at the image carefully and answer the following question.\n\n"
            f"{question}\n{choices_text}\n"
            f"Respond with only the letter of the correct answer (e.g. '정답: A').\n정답:"
        )

        gold_index = 0
        # answer may be a letter or the full text
        if answer in LETTER_INDICES:
            gold_index = LETTER_INDICES.index(answer)
        else:
            # try to match choice text
            for idx, c in enumerate(choices):
                if c.strip() == answer.strip():
                    gold_index = idx
                    break

        return Doc(
            task_name=task_name,
            query=query,
            choices=choice_labels,
            gold_index=gold_index,
            images=images,
            instruction="",
        )
    else:
        # Free-form
        query = (
            f"Look at the image carefully and answer the following question concisely.\n\n"
            f"{question}\n정답:"
        )
        return Doc(
            task_name=task_name,
            query=query,
            choices=[answer],
            gold_index=0,
            images=images,
            instruction="",
            specific={"answer": answer},
        )


MATHVISTA_TASK = LightevalTaskConfig(
    name="mathvista_gen:default",
    prompt_function=mathvista_prompt,
    hf_repo="AI4Math/MathVista",
    hf_subset="default",
    hf_avail_splits=["testmini", "test"],
    evaluation_splits=["testmini"],
    few_shots_split=None,
    few_shots_select=None,
    metrics=[strict_mc_metric],
    stop_sequence=[],
    version=1,
    generation_size=2048,
    
    sample_fields=record_to_sample,
)

TASKS_TABLE = [MATHVISTA_TASK]

import custom_tasks.force_no_think_patch
from PIL import Image
"""
ScienceQA custom task for LightEval.
Dataset: derek-thomas/ScienceQA
Task: Science VQA with images - multiple choice science questions with optional image/hint.
Only samples WITH images are evaluated (image split).

Usage:
    lighteval endpoint litellm "yaml_files/litellm_gpt4o.yaml" \\
      "scienceqa_gen:default|0" \\
      --custom-tasks "custom_tasks/custom_scienceqa_task.py" \\
      --max-samples 5 --save-details --output-dir ./test_scienceqa
"""

from typing import Any

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from custom_tasks.custom_mc_metric import strict_mc_metric


def record_to_sample(record):
    return {
        "question": record["question"],
        "choices": record["choices"],       # list of strings
        "answer": record["answer"],         # int index of correct choice
        "image": record.get("image", None),
        "hint": record.get("hint", ""),
    }

def filter_images(line: dict[str, Any]) -> bool:
    return line.get("image") is not None

def scienceqa_prompt(line: dict[str, Any], task_name: str = None) -> Doc:
    question = line["question"]
    choices = line["choices"]
    answer_idx = line["answer"]
    hint = line.get("hint", "")

    images = []
    if line.get("image") is not None:
        img = line["image"]
        if isinstance(img, str):
            img = Image.open(img)
        images.append(img)
    else:
        # Dummy 1x1 image for text-only samples so VLM pipeline doesn't crash
        images.append(Image.new('RGB', (1, 1), color='white'))

    choices_text = ""
    choice_labels = []
    for idx, choice in enumerate(choices):
        letter = LETTER_INDICES[idx]
        choices_text += f"{letter}. {choice}\n"
        choice_labels.append(f" {letter}")

    query = ""
    if hint and hint.strip():
        query += f"Hint: {hint}\n\n"
    query += (
        f"Look at the image carefully and answer the science question.\n\n"
        f"{question}\n{choices_text}\n"
        f"Respond with only the letter of the correct answer (e.g. '정답: A').\n정답:"
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=choice_labels,
        gold_index=int(answer_idx),
        images=images,
        instruction="",
    )


SCIENCEQA_TASK = LightevalTaskConfig(
    name="scienceqa_gen:default",
    prompt_function=scienceqa_prompt,
    hf_repo="derek-thomas/ScienceQA",
    hf_subset="default",
    hf_avail_splits=["test", "validation", "train"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    metrics=[strict_mc_metric],
    stop_sequence=[],
    version=1,
    generation_size=2048,
    sample_fields=record_to_sample,
)

TASKS_TABLE = [SCIENCEQA_TASK]

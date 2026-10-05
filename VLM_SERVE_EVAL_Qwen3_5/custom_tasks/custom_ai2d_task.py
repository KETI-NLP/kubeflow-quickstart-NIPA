import custom_tasks.force_no_think_patch
from PIL import Image
"""
AI2D (AI2 Diagrams) custom task for LightEval.
Dataset: lmms-lab/ai2d
Task: Science diagram understanding (multiple-choice)

Usage:
    lighteval endpoint litellm "yaml_files/litellm_gpt4o.yaml" \\
      "ai2d_gen:default|0" \\
      --custom-tasks "custom_tasks/custom_ai2d_task.py" \\
      --max-samples 5 --save-details --output-dir ./test_ai2d
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
        "options": record["options"],   # list of strings
        "answer": record["answer"],     # index (int) or letter
        "image": record.get("image", None),
    }


def ai2d_prompt(line: dict[str, Any], task_name: str = None) -> Doc:
    question = line["question"]
    options = line["options"]  # list
    answer = line["answer"]

    images = []
    if line.get("image") is not None:
        img = line["image"]
        if isinstance(img, str):
            img = Image.open(img)
        images.append(img)

    choices_text = ""
    choice_labels = []
    for idx, opt in enumerate(options):
        letter = LETTER_INDICES[idx]
        choices_text += f"{letter}. {opt}\n"
        choice_labels.append(f" {letter}")

    query = (
        f"Look at the diagram carefully and answer the question.\n\n"
        f"{question}\n{choices_text}\n"
        f"Respond with only the letter of the correct answer (e.g. '정답: A').\n정답:"
    )

    # Determine gold_index
    gold_index = 0
    if isinstance(answer, int):
        gold_index = answer
    elif isinstance(answer, str):
        if answer in LETTER_INDICES:
            gold_index = LETTER_INDICES.index(answer)
        else:
            for idx, opt in enumerate(options):
                if opt.strip() == answer.strip():
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


AI2D_TASK = LightevalTaskConfig(
    name="ai2d_gen:default",
    prompt_function=ai2d_prompt,
    hf_repo="lmms-lab/ai2d",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    metrics=[strict_mc_metric],
    stop_sequence=[],
    version=1,
    generation_size=2048,
    
    sample_fields=record_to_sample,
)

TASKS_TABLE = [AI2D_TASK]

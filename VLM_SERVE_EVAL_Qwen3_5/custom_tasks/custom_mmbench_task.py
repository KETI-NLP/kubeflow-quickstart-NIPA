import custom_tasks.force_no_think_patch
from PIL import Image
"""
MMBench custom task for LightEval.

Usage:
    lighteval endpoint litellm "yaml_files/litellm_qwen3_vl.yaml" \
      "mmbench_gen:en|0" \
      --custom-tasks "custom_tasks/custom_mmbench_task.py" \
      --max-samples 5 \
      --save-details \
      --output-dir ./test_mmbench_output
"""

from typing import Callable, Any

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from custom_tasks.custom_mc_metric import strict_mc_metric


def record_to_sample(record):
    return {
        "question": record["question"],
        "A": record.get("A", ""),
        "B": record.get("B", ""),
        "C": record.get("C", ""),
        "D": record.get("D", ""),
        "answer": record.get("answer", ""),
        "image": record.get("image", None),
        "hint": record.get("hint", ""),
    }


def mmbench_prompt(line: dict[str, Any], task_name: str = None) -> Doc:
    question = line["question"]
    hint = line["hint"]
    answer = line["answer"]

    images = []
    if line.get("image") is not None:
        img = line["image"]
        if isinstance(img, str):
            img = Image.open(img)
        images.append(img)
        
    choices_text = ""
    choices = []
    for letter in ['A', 'B', 'C', 'D']:
        if line.get(letter):
            choices_text += f"{letter}. {line[letter]}\n"
            choices.append(f" {letter}")
    
    query = ""
    if hint and hint.strip():
        query += f"Hint: {hint}\n"
        
    query += f"""다음을 읽고 정답을 고르시오. 응답의 마지막 줄은 반드시 '정답: $LETTER' (인용 없이) 형식이어야 합니다. $LETTER는 선택지 중 하나입니다.\n\n{question}\n{choices_text}\n정답:"""
    
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


def create_custom_mmbench_task(subset: str) -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name=f"mmbench_gen:{subset}",
        prompt_function=mmbench_prompt,
        hf_repo="lmms-lab/MMBench_EN",  # Or other variants
        hf_subset="default",
        hf_avail_splits=["test", "validation", "dev"],
        evaluation_splits=["dev"], # dev split usually has answers
        few_shots_split=None,
        few_shots_select=None,
        metrics=[strict_mc_metric],
        stop_sequence=[],
        version=1,
    generation_size=2048,
    
        sample_fields=record_to_sample,
    )

# TODO: Add other MMBench variants if needed (e.g., MMBench_CN)
MMBENCH_SUBSETS = [
    "en",
]

CUSTOM_MMBENCH_TASKS = {
    f"mmbench_gen:{subset}": create_custom_mmbench_task(subset)
    for subset in MMBENCH_SUBSETS
}

TASKS_TABLE = list(CUSTOM_MMBENCH_TASKS.values())

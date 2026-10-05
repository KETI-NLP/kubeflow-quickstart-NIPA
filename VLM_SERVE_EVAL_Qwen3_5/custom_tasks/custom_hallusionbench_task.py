import custom_tasks.force_no_think_patch
from PIL import Image
"""
HallusionBench custom task for LightEval.
Dataset: lmms-lab/HallusionBench
Task: Visual hallucination detection - yes/no questions about image content.
Goal: Test whether the model can accurately perceive what IS and IS NOT in the image.

Usage:
    lighteval endpoint litellm "yaml_files/litellm_gpt4o.yaml" \\
      "hallusionbench_gen:default|0" \\
      --custom-tasks "custom_tasks/custom_hallusionbench_task.py" \\
      --max-samples 5 --save-details --output-dir ./test_hallusionbench
"""

from typing import Any
import re
import numpy as np

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod


def record_to_sample(record):
    return {
        "question": record["question"],
        "answer": record["gt_answer"],  # 0 or 1 (0=No, 1=Yes)
        "image": record.get("image", None),
        "category": record.get("category", ""),
    }

class HallusionFuzzyMatch(SampleLevelComputation):
    def compute(self, doc: Doc, model_response, **kwargs) -> float:
        pred = ""
        # Extract string response from different possible formats (dict in parquet vs ModelResponse in evaluation)
        if isinstance(model_response, list) and len(model_response) > 0:
            item = model_response[0]
        else:
            item = model_response
            
        if isinstance(item, dict):
            pred = item.get('text', '')
        elif hasattr(item, 'final_text'):
            if isinstance(item.final_text, list):
                pred = "".join(item.final_text)
            else:
                pred = item.final_text
        elif isinstance(item, str):
            pred = item
        else:
            pred = str(item)
            
        pred = str(pred).lower()
        gold = doc.choices[doc.gold_index].strip().lower()
        
        # Extract "yes" or "no" from prediction using regex
        matches = re.findall(r'\b(yes|no)\b', pred)
        if matches:
            extracted = matches[-1]
            return 1.0 if extracted == gold else 0.0
        
        # Fallback: check for Korean translations
        kr_matches = re.findall(r'(예|아니요|아니오|네)', pred)
        if kr_matches:
            extracted = 'yes' if kr_matches[-1] in ['예', '네'] else 'no'
            return 1.0 if extracted == gold else 0.0

        return 0.0

hallusion_regex_match = SampleLevelMetric(
    metric_name="fuzzy_match",
    sample_level_fn=HallusionFuzzyMatch(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)


def hallusionbench_prompt(line: dict[str, Any], task_name: str = None) -> Doc:
    question = line["question"]
    answer_raw = line["gt_answer"]

    images = []
    if line.get("image") is not None:
        img = line["image"]
        if isinstance(img, str):
            img = Image.open(img)
        images.append(img)

    # answer is 0=No, 1=Yes
    if isinstance(answer_raw, int):
        answer_str = "Yes" if answer_raw == 1 else "No"
        gold_index = answer_raw  # 0 -> No (idx 0), 1 -> Yes (idx 1)
    else:
        answer_str = str(answer_raw)
        gold_index = 1 if answer_str.lower() == "yes" else 0

    query = (
        f"Look at the image carefully and answer with only 'Yes' or 'No'. Do not provide any additional explanation.\n\n"
        f"{question}\nAnswer:"
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=[" No", " Yes"],
        gold_index=gold_index,
        images=images,
        instruction="",
    )


HALLUSIONBENCH_TASK = LightevalTaskConfig(
    name="hallusionbench_gen:default",
    prompt_function=hallusionbench_prompt,
    hf_repo="lmms-lab/HallusionBench",
    hf_subset="default",
    hf_avail_splits=["image", "non_image"],
    evaluation_splits=["image"],  # image split requires actual vision
    few_shots_split=None,
    few_shots_select=None,
    metrics=[hallusion_regex_match],
    stop_sequence=[],
    version=1,
    generation_size=2048,
    sample_fields=record_to_sample,
)

TASKS_TABLE = [HALLUSIONBENCH_TASK]

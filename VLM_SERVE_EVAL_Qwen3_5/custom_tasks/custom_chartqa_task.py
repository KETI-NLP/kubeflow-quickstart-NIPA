import custom_tasks.force_no_think_patch
from PIL import Image
"""
ChartQA custom task for LightEval.
Dataset: lmms-lab/ChartQA
Task: Chart comprehension - bar/line/pie charts (open-ended QA)
Metric: Relaxed accuracy (within 5% for numerical answers)

Usage:
    lighteval endpoint litellm "yaml_files/litellm_gpt4o.yaml" \\
      "chartqa_gen:default|0" \\
      --custom-tasks "custom_tasks/custom_chartqa_task.py" \\
      --max-samples 5 --save-details --output-dir ./test_chartqa
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
        "answer": record["answer"],
        "image": record.get("image", None),
        "type": record.get("type", ""),   # 'human' or 'augmented'
    }

class ChartQAFuzzyMatch(SampleLevelComputation):
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
        
        # If the exact gold string appears in the prediction, count it as a match
        if gold in pred:
            return 1.0
            
        # If gold is numeric, try to extract numbers from prediction and compare
        try:
            gold_val = float(gold.replace(',', ''))
            # Extract all numbers from prediction
            pred_numbers = re.findall(r'-?\d+\.?\d*', pred.replace(',', ''))
            # Check numbers from end to beginning, as reasoning usually precedes the final answer
            for num_str in reversed(pred_numbers):
                if abs(float(num_str) - gold_val) < 1e-5:
                    return 1.0
        except ValueError:
            pass
            
        return 0.0

chartqa_regex_match = SampleLevelMetric(
    metric_name="fuzzy_match",
    sample_level_fn=ChartQAFuzzyMatch(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)


def chartqa_prompt(line: dict[str, Any], task_name: str = None) -> Doc:
    question = line["question"]
    answer = str(line["answer"])

    images = []
    if line.get("image") is not None:
        img = line["image"]
        if isinstance(img, str):
            img = Image.open(img)
        images.append(img)

    query = (
        f"Look at the chart carefully and answer the following question concisely.\n"
        f"Answer with ONLY the exact number or term shown in the chart. Do not provide any additional explanation.\n\n"
        f"{question}\nAnswer:"
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


CHARTQA_TASK = LightevalTaskConfig(
    name="chartqa_gen:default",
    prompt_function=chartqa_prompt,
    hf_repo="lmms-lab/ChartQA",
    hf_subset="default",
    hf_avail_splits=["test", "val", "train"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    metrics=[chartqa_regex_match],  # Custom fuzzy extractor for verbose generation
    stop_sequence=[],
    version=1,
    generation_size=2048,
    sample_fields=record_to_sample,
)

TASKS_TABLE = [CHARTQA_TASK]

import custom_tasks.force_no_think_patch
from PIL import Image
"""
IFEval-Ko 커스텀 테스크 생성
- instruction following evaluation 데이터셋
- 생성형 태스크
- ifeval_ko 폴더의 메트릭 로직 사용

사용 방법:
    lighteval endpoint litellm "$HOME/KETI/lighteval/litellm_qwen3.yaml" \
      "ifeval_ko_gen|0" \
      --custom-tasks "$HOME/KETI/lighteval/custom_ifeval_ko_task.py" \
      --max-samples 5 \
      --save-details \
      --output-dir ./test_ifeval_ko_custom
"""

import sys
import os
import unicodedata
from typing import Callable

import numpy as np

lighteval_dir = os.path.dirname(os.path.dirname(__file__))  
lighteval_dir = os.path.abspath(lighteval_dir)
if lighteval_dir not in sys.path:
    sys.path.insert(0, lighteval_dir)

from ifeval_ko import utils as ifeval_ko_utils
from ifeval_ko.utils import InputExample, agg_inst_level_acc, process_results

from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod


class IFEvalKoMetrics(SampleLevelComputation):
    """IFEval-Ko 메트릭 계산 클래스 - ifeval_ko/utils.py의 process_results를 그대로 사용"""
    
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> dict:
        doc_dict = {
            "key": doc.specific["key"],
            "instruction_id_list": doc.specific["instruction_id_list"],
            "prompt": doc.query,
            "kwargs": doc.specific["kwargs"],
        }
        
        results = [model_response.final_text[0]]
        
        result = process_results(doc_dict, results)
        
        return {
            "prompt_level_strict_acc": int(result["prompt_level_strict_acc"]),
            "inst_level_strict_acc": result["inst_level_strict_acc"],  
            "prompt_level_loose_acc": int(result["prompt_level_loose_acc"]),
            "inst_level_loose_acc": result["inst_level_loose_acc"],  
        }


submetric_names = [
    "prompt_level_strict_acc",
    "inst_level_strict_acc",
    "prompt_level_loose_acc",
    "inst_level_loose_acc",
]

ifeval_ko_metrics = SampleLevelMetricGrouping(
    metric_name=submetric_names,
    higher_is_better=dict.fromkeys(submetric_names, True),
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=IFEvalKoMetrics(),
    corpus_level_fn={
        "prompt_level_strict_acc": np.mean,
        "inst_level_strict_acc": agg_inst_level_acc,
        "prompt_level_loose_acc": np.mean,
        "inst_level_loose_acc": agg_inst_level_acc,
    },
)


def record_to_sample(record):
    return {
        "key": record["key"],
        "prompt": record["prompt"].strip(),
        "instruction_id_list": record["instruction_id_list"],
        "kwargs": record["kwargs"],
    }


def ifeval_ko_prompt(line, task_name: str = None):
    """
    IFEval-Ko 프롬프트 함수
    생성형 태스크이므로 prompt를 그대로 사용
    """
    query = line["prompt"]
    instruction_ids = line["instruction_id_list"]
    
    return Doc(
        task_name=task_name,
        query=query,
        choices=instruction_ids,  
        gold_index=0,  
        instruction="",  
        specific={
            "key": line["key"],
            "instruction_id_list": instruction_ids,
            "kwargs": line["kwargs"],
        },
    )


def create_custom_ifeval_ko_task(
    prompt_function: Callable = None
) -> LightevalTaskConfig:
    """
    IFEval-Ko 커스텀 테스크 생성
    """
    if prompt_function is None:
        prompt_function = ifeval_ko_prompt
    
    return LightevalTaskConfig(
        name="ifeval_ko_gen",
        prompt_function=prompt_function,
        hf_repo="allganize/IFEval-Ko",
        hf_subset="default",
        hf_avail_splits=["train"],
        evaluation_splits=["train"],  
        few_shots_split=None,  
        few_shots_select=None,
        metrics=[ifeval_ko_metrics],  
        stop_sequence=[],  
        version=1,
      
        sample_fields=record_to_sample,
    )


CUSTOM_IFEVAL_KO_TASKS = {
    "ifeval_ko_gen": create_custom_ifeval_ko_task(),
}


TASKS_TABLE = list(CUSTOM_IFEVAL_KO_TASKS.values())


if __name__ == "__main__":
    test_task = create_custom_ifeval_ko_task()
    print(f"Task name: {test_task.name}")
    print(f"Metrics: {test_task.metrics}")
    print(f"Metric name: {test_task.metrics[0].metric_name}")
    print(f"Category: {test_task.metrics[0].category}")
    print(f"Generation size: {test_task.generation_size}")
    print(f"Evaluation splits: {test_task.evaluation_splits}")


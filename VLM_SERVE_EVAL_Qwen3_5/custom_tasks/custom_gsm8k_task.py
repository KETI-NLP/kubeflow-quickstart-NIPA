import custom_tasks.force_no_think_patch
from PIL import Image
"""
GSM8K 커스텀 테스크 생성

사용 방법:
    lighteval endpoint litellm "$HOME/KETI/lighteval/litellm_qwen3.yaml" \
      "gsm8k_gen|0" \
      --custom-tasks "$HOME/KETI/lighteval/custom_gsm8k_task.py" \
      --max-samples 5 \
      --save-details \
      --output-dir ./test_gsm8k_custom
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.default_prompts import gsm8k
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def custom_gsm8k_prompt(line, task_name: str = None):
    doc = gsm8k(line, task_name)
    doc.query = "The last line of your response should be of the following format: 'Answer: <short-answer>'(without quotes)\n" + doc.query
    return doc

def record_to_sample(record):
    return {
        "question": record["question"].strip(),
        "answer": record["answer"].strip(),
    }


def create_custom_gsm8k_task() -> LightevalTaskConfig:
    """
    GSM8K 커스텀 테스크 생성
    원래 GSM8K와 동일한 프롬프트를 사용하되, 커스텀 설정 가능

    Returns:
        LightevalTaskConfig: 커스텀 테스크 설정
    """
    return LightevalTaskConfig(
        name="gsm8k_gen",
        prompt_function=custom_gsm8k_prompt, 
        hf_repo="openai/gsm8k",
        hf_subset="main",
        hf_avail_splits=["train", "test"],
        evaluation_splits=["test"],  
        few_shots_split=None,
        few_shots_select=None,
        metrics=[Metrics.expr_gold_metric],  
        stop_sequence=["Question:"],  
        version=1,
      
        sample_fields=record_to_sample,  
    )


CUSTOM_GSM8K_TASKS = {
    "gsm8k_gen": create_custom_gsm8k_task(),
}


TASKS_TABLE = list(CUSTOM_GSM8K_TASKS.values())


if __name__ == "__main__":
    test_task = create_custom_gsm8k_task()
    print(f"Task name: {test_task.name}")
    print(f"Metrics: {test_task.metrics}")
    print(f"Metric name: {test_task.metrics[0].metric_name}")
    print(f"Category: {test_task.metrics[0].category}")
    print(f"Generation size: {test_task.generation_size}")
    print(f"Evaluation splits: {test_task.evaluation_splits}")


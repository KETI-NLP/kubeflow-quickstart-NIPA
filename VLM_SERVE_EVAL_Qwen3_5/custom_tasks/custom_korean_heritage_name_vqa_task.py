"""
Korean heritage name VQA custom task for LightEval.

This task reads a local Hugging Face `save_to_disk` dataset whose rows contain:
- question
- answer
- image_path

The model receives one image and must answer in the format:
`Answer: <문화재명>`
"""

from __future__ import annotations
import custom_tasks.force_no_think_patch

import hashlib
import os
import re
from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image

from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod


DEFAULT_DATASET_PATH = (
    "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/tmp_eval_datasets/"
    "korean_heritage_name_vqa_paraphrase_hf"
)
DATASET_PATH = os.environ.get("KOREAN_HERITAGE_NAME_VQA_DATASET_PATH", DEFAULT_DATASET_PATH)
DEFAULT_IMAGE_CACHE_DIR = (
    "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/tmp_eval_datasets/"
    "korean_heritage_name_vqa_image_cache"
)
IMAGE_CACHE_DIR = os.environ.get("KOREAN_HERITAGE_NAME_VQA_IMAGE_CACHE_DIR", DEFAULT_IMAGE_CACHE_DIR)


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_quotes(text: str) -> str:
    return text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")


def _normalize_name(text: str) -> str:
    text = _normalize_quotes(text)
    text = _collapse_ws(text)
    text = text.strip(" \t\n\r\"'")
    text = re.sub(r"\s+", "", text)
    return text


def _extract_answer_text(text: str) -> str:
    text = _normalize_quotes((text or "").strip())
    if not text:
        return ""

    # Match "Answer:" even if it's not at the start of a line
    matches = re.findall(r"Answer\s*:\s*(.+?)(?:\n|$)", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].strip()

    non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not non_empty_lines:
        return ""
    return non_empty_lines[-1]


def _lcs_length(a: str, b: str) -> int:
    if not a or not b:
        return 0

    prev = [0] * (len(b) + 1)
    for ca in a:
        curr = [0]
        for j, cb in enumerate(b, start=1):
            if ca == cb:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def _ordered_overlap_scores(gold: str, pred: str) -> tuple[float, float, float]:
    normalized_gold = _normalize_name(gold)
    normalized_pred = _normalize_name(pred)
    if not normalized_gold or not normalized_pred:
        return 0.0, 0.0, 0.0

    lcs = _lcs_length(normalized_gold, normalized_pred)
    recall = lcs / len(normalized_gold)
    precision = lcs / len(normalized_pred)
    if recall + precision == 0:
        f1 = 0.0
    else:
        f1 = 2 * recall * precision / (recall + precision)
    return recall, precision, f1


@lru_cache(maxsize=4096)
def _prepare_image_path(image_path: str) -> str:
    os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)
    digest = hashlib.sha1(image_path.encode("utf-8")).hexdigest()
    cached_path = os.path.join(IMAGE_CACHE_DIR, f"{digest}.jpg")
    if os.path.exists(cached_path):
        return cached_path

    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        rgb.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        rgb.save(cached_path, format="JPEG", quality=85, optimize=True)
    return cached_path


@lru_cache(maxsize=4096)
def _load_image(image_path: str) -> Image.Image:
    prepared_path = _prepare_image_path(image_path)
    image = Image.open(prepared_path)
    image.load()
    return image


def _extract_predictions(model_response) -> list[str]:
    preds = model_response.final_text or []
    if preds and isinstance(preds, list):
        if all(isinstance(x, str) and len(x) == 1 for x in preds):
            return ["".join(preds)]
    elif isinstance(preds, str):
        return [preds]
    return preds


class NameVQAStrictEMMetric(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        gold = doc.get_golds()[0] if doc.get_golds() else ""
        predictions = _extract_predictions(model_response)
        if not gold or not predictions:
            return 0.0

        for pred in predictions:
            answer = _extract_answer_text(pred)
            if answer == gold:
                return 1.0
        return 0.0


class NameVQARelaxedEMMetric(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        gold = doc.get_golds()[0] if doc.get_golds() else ""
        predictions = _extract_predictions(model_response)
        if not gold or not predictions:
            return 0.0

        normalized_gold = _normalize_name(gold)
        for pred in predictions:
            answer = _extract_answer_text(pred)
            if answer and _normalize_name(answer) == normalized_gold:
                return 1.0
        return 0.0


class NameVQAFormatValidMetric(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        predictions = _extract_predictions(model_response)
        if not predictions:
            return 0.0

        for pred in predictions:
            if re.search(r"Answer\s*:\s*(.+?)(?:\n|$)", pred or "", flags=re.IGNORECASE):
                return 1.0
        return 0.0


class NameVQAEmptyResponseMetric(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        predictions = _extract_predictions(model_response)
        if not predictions:
            return 1.0

        for pred in predictions:
            if pred and pred.strip():
                return 0.0
        return 1.0


class NameVQAOrderedRecallMetric(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        gold = doc.get_golds()[0] if doc.get_golds() else ""
        predictions = _extract_predictions(model_response)
        if not gold or not predictions:
            return 0.0

        best = 0.0
        for pred in predictions:
            answer = _extract_answer_text(pred)
            recall, _, _ = _ordered_overlap_scores(gold, answer)
            best = max(best, recall)
        return best


class NameVQAOrderedPrecisionMetric(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        gold = doc.get_golds()[0] if doc.get_golds() else ""
        predictions = _extract_predictions(model_response)
        if not gold or not predictions:
            return 0.0

        best = 0.0
        for pred in predictions:
            answer = _extract_answer_text(pred)
            _, precision, _ = _ordered_overlap_scores(gold, answer)
            best = max(best, precision)
        return best


class NameVQAOrderedF1Metric(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        gold = doc.get_golds()[0] if doc.get_golds() else ""
        predictions = _extract_predictions(model_response)
        if not gold or not predictions:
            return 0.0

        best = 0.0
        for pred in predictions:
            answer = _extract_answer_text(pred)
            _, _, f1 = _ordered_overlap_scores(gold, answer)
            best = max(best, f1)
        return best


name_vqa_strict_em_metric = SampleLevelMetric(
    metric_name="name_vqa_strict_em",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=NameVQAStrictEMMetric(),
    corpus_level_fn=np.mean,
    batched_compute=False,
)

name_vqa_relaxed_em_metric = SampleLevelMetric(
    metric_name="name_vqa_relaxed_em",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=NameVQARelaxedEMMetric(),
    corpus_level_fn=np.mean,
    batched_compute=False,
)

name_vqa_format_valid_metric = SampleLevelMetric(
    metric_name="name_vqa_format_valid",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=NameVQAFormatValidMetric(),
    corpus_level_fn=np.mean,
    batched_compute=False,
)

name_vqa_empty_response_metric = SampleLevelMetric(
    metric_name="name_vqa_empty_response",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=False,
    sample_level_fn=NameVQAEmptyResponseMetric(),
    corpus_level_fn=np.mean,
    batched_compute=False,
)

name_vqa_ordered_recall_metric = SampleLevelMetric(
    metric_name="name_vqa_ordered_recall",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=NameVQAOrderedRecallMetric(),
    corpus_level_fn=np.mean,
    batched_compute=False,
)

name_vqa_ordered_precision_metric = SampleLevelMetric(
    metric_name="name_vqa_ordered_precision",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=NameVQAOrderedPrecisionMetric(),
    corpus_level_fn=np.mean,
    batched_compute=False,
)

name_vqa_ordered_f1_metric = SampleLevelMetric(
    metric_name="name_vqa_ordered_f1",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=NameVQAOrderedF1Metric(),
    corpus_level_fn=np.mean,
    batched_compute=False,
)


def record_to_sample(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": record["sample_id"],
        "base_sample_id": record["base_sample_id"],
        "template_id": record["template_id"],
        "question": record["question"],
        "answer": record["answer"],
        "heritage_name": record["heritage_name"],
        "heritage_name_full": record.get("heritage_name_full", ""),
        "image_path": record["image_path"],
        "image_url": record.get("image_url", ""),
        "target_image_info": record.get("target_image_info", ""),
        "source_qa_error_type": record.get("source_qa_error_type", ""),
        "source_hallucination_type": record.get("source_hallucination_type", ""),
        "source_row_count": record.get("source_row_count", 1),
    }


def korean_heritage_name_vqa_prompt(line: dict[str, Any], task_name: str | None = None) -> Doc:
    answer = line["answer"]
    query = (
        f"{line['question']}\n\n"
        "반드시 마지막 줄에 다음 형식으로만 답하세요:\n"
        "Answer: <문화재명>\n\n"
        "좋은 답변 예시:\n"
        "Answer: 숭례문\n"
        "Answer: 경복궁 근정전\n\n"
        "틀린 답변 예시:\n"
        "이 이미지는 숭례문입니다.\n"
        "\"경복궁 근정전\"\n"
        "문화재명: 숭례문"
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=[answer],
        gold_index=0,
        images=[_load_image(line["image_path"])],
        instruction="",
        specific={
            "sample_id": line["sample_id"],
            "base_sample_id": line["base_sample_id"],
            "template_id": line["template_id"],
            "answer": answer,
            "image_path": line["image_path"],
            "image_url": line.get("image_url", ""),
            "source_row_count": line.get("source_row_count", 1),
        },
    )


CUSTOM_KOREAN_HERITAGE_NAME_VQA_TASK = LightevalTaskConfig(
    name="korean_heritage_name_vqa",
    prompt_function=korean_heritage_name_vqa_prompt,
    hf_repo=DATASET_PATH,
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    metrics=[
        name_vqa_format_valid_metric,
        name_vqa_empty_response_metric,
        name_vqa_strict_em_metric,
        name_vqa_relaxed_em_metric,
        name_vqa_ordered_recall_metric,
        name_vqa_ordered_precision_metric,
        name_vqa_ordered_f1_metric,
    ],
    stop_sequence=[],
    version=3,
    sample_fields=record_to_sample,
    generation_size=2048,
)


TASKS_TABLE = [CUSTOM_KOREAN_HERITAGE_NAME_VQA_TASK]


if __name__ == "__main__":
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Task name: {CUSTOM_KOREAN_HERITAGE_NAME_VQA_TASK.name}")
    print(f"Metrics: {CUSTOM_KOREAN_HERITAGE_NAME_VQA_TASK.metrics}")

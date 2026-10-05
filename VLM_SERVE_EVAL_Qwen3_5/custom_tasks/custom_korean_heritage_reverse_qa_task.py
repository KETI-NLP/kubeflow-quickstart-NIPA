"""
Korean heritage reverse QA custom task for LightEval.

기존 ShortQA  : 문화재명 → 속성(지정일, 재질 등) 답변
이번 ReverseQA: 속성/설명 → 문화재명 답변

Dataset rows contain:
  - sample_id     : str
  - heritage_id   : str  (e.g. "HF010461")
  - heritage_name : str  (정답 = 문화재 명칭)
  - question      : str  (마스킹된 단서들 + 지시문)
  - answer        : str  (= heritage_name)
  - num_clues     : int

The model must answer in the format:
  `Answer: <문화재 명칭>`
"""

from __future__ import annotations
import custom_tasks.force_no_think_patch

from PIL import Image

import os
import re
from typing import Any

import numpy as np

from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod


DEFAULT_DATASET_PATH = (
    "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/tmp_eval_datasets/"
    "korean_heritage_reverse_qa_benchmark_hf"
)
DATASET_PATH = os.environ.get(
    "KOREAN_HERITAGE_REVERSE_QA_DATASET_PATH", DEFAULT_DATASET_PATH
)


# ---------------------------------------------------------------------------
# 텍스트 유틸리티
# ---------------------------------------------------------------------------

def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_quotes(text: str) -> str:
    return (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )


def _extract_answer_text(text: str) -> str:
    """모델 출력에서 'Answer: <...>' 형식의 정답 부분을 추출한다."""
    text = _normalize_quotes((text or "").strip())
    if not text:
        return ""

    matches = re.findall(
        r"^\s*Answer\s*:\s*(.+?)\s*$", text, flags=re.IGNORECASE | re.MULTILINE
    )
    if matches:
        return matches[-1].strip()

    # 형식이 없으면 마지막 비어있지 않은 줄을 반환
    non_empty = [line.strip() for line in text.splitlines() if line.strip()]
    return non_empty[-1] if non_empty else ""


def _normalize_heritage_name(text: str) -> str:
    """문화재 명칭 정규화: 공백 제거, 따옴표·괄호 strip."""
    text = _normalize_quotes(text or "")
    text = text.strip(" \t\n\r\"'.,;:!?()")
    text = re.sub(r"\s+", "", text)   # 모든 공백 제거 (한국어 명칭은 붙여씀)
    return text


# ---------------------------------------------------------------------------
# LCS 기반 소프트 매칭
# ---------------------------------------------------------------------------

def _lcs_length(a: str, b: str) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for ca in a:
        curr = [0]
        for j, cb in enumerate(b, start=1):
            curr.append(prev[j - 1] + 1 if ca == cb else max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def _ordered_overlap_scores(gold: str, pred: str) -> tuple[float, float, float]:
    norm_gold = _normalize_heritage_name(gold)
    norm_pred = _normalize_heritage_name(pred)
    if not norm_gold or not norm_pred:
        return 0.0, 0.0, 0.0

    lcs = _lcs_length(norm_gold, norm_pred)
    recall = lcs / len(norm_gold)
    precision = lcs / len(norm_pred)
    f1 = (
        2 * recall * precision / (recall + precision)
        if (recall + precision) > 0
        else 0.0
    )
    return recall, precision, f1


# ---------------------------------------------------------------------------
# 메트릭 구현
# ---------------------------------------------------------------------------

def _extract_predictions(model_response) -> list[str]:
    preds = model_response.final_text or []
    if preds and isinstance(preds, list):
        if all(isinstance(x, str) and len(x) == 1 for x in preds):
            return ["".join(preds)]
    elif isinstance(preds, str):
        return [preds]
    return preds


class ReverseQAStrictEMMetric(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        gold = doc.get_golds()[0] if doc.get_golds() else ""
        predictions = _extract_predictions(model_response)
        if not gold or not predictions:
            return 0.0
        for pred in predictions:
            if _extract_answer_text(pred) == gold:
                return 1.0
        return 0.0


class ReverseQARelaxedEMMetric(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        gold = doc.get_golds()[0] if doc.get_golds() else ""
        predictions = _extract_predictions(model_response)
        if not gold or not predictions:
            return 0.0
        norm_gold = _normalize_heritage_name(gold)
        for pred in predictions:
            answer = _extract_answer_text(pred)
            if answer and _normalize_heritage_name(answer) == norm_gold:
                return 1.0
        return 0.0


class ReverseQAFormatValidMetric(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        predictions = _extract_predictions(model_response)
        if not predictions:
            return 0.0
        for pred in predictions:
            if re.search(
                r"^\s*Answer\s*:\s*.+$", pred or "", flags=re.IGNORECASE | re.MULTILINE
            ):
                return 1.0
        return 0.0


class ReverseQAEmptyResponseMetric(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        predictions = _extract_predictions(model_response)
        if not predictions:
            return 1.0
        for pred in predictions:
            if pred and pred.strip():
                return 0.0
        return 1.0


class ReverseQAOrderedRecallMetric(SampleLevelComputation):
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


class ReverseQAOrderedPrecisionMetric(SampleLevelComputation):
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


class ReverseQAOrderedF1Metric(SampleLevelComputation):
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


# ---------------------------------------------------------------------------
# SampleLevelMetric 인스턴스
# ---------------------------------------------------------------------------

reverse_qa_strict_em_metric = SampleLevelMetric(
    metric_name="reverse_qa_strict_em",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=ReverseQAStrictEMMetric(),
    corpus_level_fn=np.mean,
    batched_compute=False,
)

reverse_qa_relaxed_em_metric = SampleLevelMetric(
    metric_name="reverse_qa_relaxed_em",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=ReverseQARelaxedEMMetric(),
    corpus_level_fn=np.mean,
    batched_compute=False,
)

reverse_qa_format_valid_metric = SampleLevelMetric(
    metric_name="reverse_qa_format_valid",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=ReverseQAFormatValidMetric(),
    corpus_level_fn=np.mean,
    batched_compute=False,
)

reverse_qa_empty_response_metric = SampleLevelMetric(
    metric_name="reverse_qa_empty_response",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=False,
    sample_level_fn=ReverseQAEmptyResponseMetric(),
    corpus_level_fn=np.mean,
    batched_compute=False,
)

reverse_qa_ordered_recall_metric = SampleLevelMetric(
    metric_name="reverse_qa_ordered_recall",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=ReverseQAOrderedRecallMetric(),
    corpus_level_fn=np.mean,
    batched_compute=False,
)

reverse_qa_ordered_precision_metric = SampleLevelMetric(
    metric_name="reverse_qa_ordered_precision",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=ReverseQAOrderedPrecisionMetric(),
    corpus_level_fn=np.mean,
    batched_compute=False,
)

reverse_qa_ordered_f1_metric = SampleLevelMetric(
    metric_name="reverse_qa_ordered_f1",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=ReverseQAOrderedF1Metric(),
    corpus_level_fn=np.mean,
    batched_compute=False,
)


# ---------------------------------------------------------------------------
# 프롬프트 함수 및 LightEval Task 설정
# ---------------------------------------------------------------------------

def record_to_sample(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": record["sample_id"],
        "heritage_id": record["heritage_id"],
        "heritage_name": record["heritage_name"],
        "question": record["question"],
        "answer": record["answer"],
        "num_clues": int(record["num_clues"]),
    }


def korean_heritage_reverse_qa_prompt(
    line: dict[str, Any], task_name: str | None = None
) -> Doc:
    query = (
        f"{line['question']}\n\n"
        "CRITICAL INSTRUCTION: DO NOT output any thinking process, reasoning, or explanations. DO NOT output 'Thinking Process:'. Output ONLY the final answer."
    )
    return Doc(
        task_name=task_name,
        query=query,
        choices=[line["answer"]],
        gold_index=0,
        instruction="",
        specific={
            "sample_id": line["sample_id"],
            "heritage_id": line["heritage_id"],
            "heritage_name": line["heritage_name"],
            "num_clues": int(line["num_clues"]),
        },
    )


CUSTOM_KOREAN_HERITAGE_REVERSE_QA_TASK = LightevalTaskConfig(
    name="korean_heritage_reverse_qa",
    prompt_function=korean_heritage_reverse_qa_prompt,
    hf_repo=DATASET_PATH,
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    metrics=[
        reverse_qa_format_valid_metric,
        reverse_qa_empty_response_metric,
        reverse_qa_strict_em_metric,
        reverse_qa_relaxed_em_metric,
        reverse_qa_ordered_recall_metric,
        reverse_qa_ordered_precision_metric,
        reverse_qa_ordered_f1_metric,
    ],
    stop_sequence=[],
    version=1,
    
    sample_fields=record_to_sample,
    generation_size=2048,
)


TASKS_TABLE = [CUSTOM_KOREAN_HERITAGE_REVERSE_QA_TASK]


if __name__ == "__main__":
    print(f"Dataset path : {DATASET_PATH}")
    print(f"Task name    : {CUSTOM_KOREAN_HERITAGE_REVERSE_QA_TASK.name}")
    print(f"Metrics      : {[m.metric_name for m in CUSTOM_KOREAN_HERITAGE_REVERSE_QA_TASK.metrics]}")

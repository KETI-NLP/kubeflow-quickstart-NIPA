"""
Korean heritage text short-answer QA custom task for LightEval.

This task reads a local Hugging Face `save_to_disk` dataset whose rows contain:
- question
- answer
- answer_type

The model must answer in the format:
`Answer: <짧은 정답>`
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
    "korean_heritage_text_shortqa_benchmark_hf"
)
DATASET_PATH = os.environ.get("KOREAN_HERITAGE_TEXT_SHORTQA_DATASET_PATH", DEFAULT_DATASET_PATH)


TYPE_SPECIFIC_INSTRUCTIONS = {
    "designation_date": (
        "정답은 반드시 `YYYY년 M월 D일` 형식의 날짜 하나여야 합니다.\n"
        "다른 날짜를 추측하지 말고, 설명이나 시대명은 쓰지 마세요.\n"
        "좋은 답: `Answer: 1998년 7월 21일`\n"
        "나쁜 답: `Answer: 2001년 1월 29일`, `Answer: 조선 후기`"
    ),
    "creation_year": (
        "정답은 반드시 `YYYY년` 형식의 4자리 연도 하나여야 합니다.\n"
        "시대명, 왕대, 세기 표현은 쓰지 마세요.\n"
        "좋은 답: `Answer: 1677년`\n"
        "나쁜 답: `Answer: 고려시대 후기`, `Answer: 조선 선조 10년`"
    ),
    "quantity": (
        "정답은 반드시 `숫자+단위` 하나여야 합니다.\n"
        "단위는 질문의 대상에 맞는 원문 단위를 유지하세요. 예: `권`, `책`, `점`, `기`, `동`, `구`.\n"
        "좋은 답: `Answer: 1권`, `Answer: 3기`\n"
        "나쁜 답: `Answer: 1책`(정답이 1권일 때), `Answer: 약 3점`"
    ),
    "designation_type": (
        "정답은 반드시 문화재 지정 종별 하나여야 합니다.\n"
        "예: `보물`, `사적`, `국보`.\n"
        "설명은 쓰지 마세요."
    ),
    "designation_name": (
        "정답은 반드시 문화재 지정 명칭 하나여야 합니다.\n"
        "설명이나 부가 문장을 쓰지 마세요."
    ),
    "material": (
        "정답은 반드시 재질명 하나여야 합니다.\n"
        "예: `비단`, `명주`, `청동`, `목재 패널`.\n"
        "설명은 쓰지 마세요."
    ),
}


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_quotes(text: str) -> str:
    return text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")


def _extract_answer_text(text: str) -> str:
    text = _normalize_quotes((text or "").strip())
    if not text:
        return ""

    matches = re.findall(r"^\s*Answer\s*:\s*(.+?)\s*$", text, flags=re.IGNORECASE | re.MULTILINE)
    if matches:
        return matches[-1].strip()

    non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not non_empty_lines:
        return ""
    return non_empty_lines[-1]


def _normalize_date(text: str) -> str:
    text = _collapse_ws(text)
    text = re.sub(r"\s*년\s*", "년 ", text)
    text = re.sub(r"\s*월\s*", "월 ", text)
    text = re.sub(r"\s*일\s*", "일", text)
    return _collapse_ws(text)


def _normalize_quantity(text: str) -> str:
    text = _collapse_ws(text)
    text = re.sub(r"^총\s*", "", text)
    return text


def _normalize_short_answer(text: str, answer_type: str) -> str:
    text = _normalize_quotes(text or "")
    text = text.strip(" \t\n\r\"'.,:;!?")

    if answer_type == "designation_date":
        text = _normalize_date(text)
    elif answer_type == "quantity":
        text = _normalize_quantity(text)
    else:
        text = _collapse_ws(text)

    text = re.sub(r"\s+", "", text)
    return text


def _build_answer_instruction(answer_type: str) -> str:
    base = (
        "반드시 마지막 줄에 다음 형식으로만 답하세요:\n"
        "Answer: <짧은 정답>\n\n"
        "설명이나 추가 문장은 쓰지 마세요."
    )
    extra = TYPE_SPECIFIC_INSTRUCTIONS.get(answer_type, "")
    if not extra:
        return base
    return f"{base}\n\n{extra}"


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


def _ordered_overlap_scores(gold: str, pred: str, answer_type: str) -> tuple[float, float, float]:
    normalized_gold = _normalize_short_answer(gold, answer_type)
    normalized_pred = _normalize_short_answer(pred, answer_type)
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


def _extract_predictions(model_response) -> list[str]:
    preds = model_response.final_text or []
    if preds and isinstance(preds, list):
        if all(isinstance(x, str) and len(x) == 1 for x in preds):
            return ["".join(preds)]
    elif isinstance(preds, str):
        return [preds]
    return preds


class TextShortQAStrictEMMetric(SampleLevelComputation):
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


class TextShortQARelaxedEMMetric(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        gold = doc.get_golds()[0] if doc.get_golds() else ""
        answer_type = doc.specific.get("answer_type", "")
        predictions = _extract_predictions(model_response)
        if not gold or not predictions:
            return 0.0

        normalized_gold = _normalize_short_answer(gold, answer_type)
        for pred in predictions:
            answer = _extract_answer_text(pred)
            if answer and _normalize_short_answer(answer, answer_type) == normalized_gold:
                return 1.0
        return 0.0


class TextShortQAFormatValidMetric(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        predictions = _extract_predictions(model_response)
        if not predictions:
            return 0.0

        for pred in predictions:
            if re.search(r"^\s*Answer\s*:\s*.+$", pred or "", flags=re.IGNORECASE | re.MULTILINE):
                return 1.0
        return 0.0


class TextShortQAEmptyResponseMetric(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        predictions = _extract_predictions(model_response)
        if not predictions:
            return 1.0

        for pred in predictions:
            if pred and pred.strip():
                return 0.0
        return 1.0


class TextShortQAOrderedRecallMetric(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        gold = doc.get_golds()[0] if doc.get_golds() else ""
        answer_type = doc.specific.get("answer_type", "")
        predictions = _extract_predictions(model_response)
        if not gold or not predictions:
            return 0.0

        best = 0.0
        for pred in predictions:
            answer = _extract_answer_text(pred)
            recall, _, _ = _ordered_overlap_scores(gold, answer, answer_type)
            best = max(best, recall)
        return best


class TextShortQAOrderedPrecisionMetric(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        gold = doc.get_golds()[0] if doc.get_golds() else ""
        answer_type = doc.specific.get("answer_type", "")
        predictions = _extract_predictions(model_response)
        if not gold or not predictions:
            return 0.0

        best = 0.0
        for pred in predictions:
            answer = _extract_answer_text(pred)
            _, precision, _ = _ordered_overlap_scores(gold, answer, answer_type)
            best = max(best, precision)
        return best


class TextShortQAOrderedF1Metric(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        gold = doc.get_golds()[0] if doc.get_golds() else ""
        answer_type = doc.specific.get("answer_type", "")
        predictions = _extract_predictions(model_response)
        if not gold or not predictions:
            return 0.0

        best = 0.0
        for pred in predictions:
            answer = _extract_answer_text(pred)
            _, _, f1 = _ordered_overlap_scores(gold, answer, answer_type)
            best = max(best, f1)
        return best


text_shortqa_strict_em_metric = SampleLevelMetric(
    metric_name="text_shortqa_strict_em",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=TextShortQAStrictEMMetric(),
    corpus_level_fn=np.mean,
    batched_compute=False,
)

text_shortqa_relaxed_em_metric = SampleLevelMetric(
    metric_name="text_shortqa_relaxed_em",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=TextShortQARelaxedEMMetric(),
    corpus_level_fn=np.mean,
    batched_compute=False,
)

text_shortqa_format_valid_metric = SampleLevelMetric(
    metric_name="text_shortqa_format_valid",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=TextShortQAFormatValidMetric(),
    corpus_level_fn=np.mean,
    batched_compute=False,
)

text_shortqa_empty_response_metric = SampleLevelMetric(
    metric_name="text_shortqa_empty_response",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=False,
    sample_level_fn=TextShortQAEmptyResponseMetric(),
    corpus_level_fn=np.mean,
    batched_compute=False,
)

text_shortqa_ordered_recall_metric = SampleLevelMetric(
    metric_name="text_shortqa_ordered_recall",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=TextShortQAOrderedRecallMetric(),
    corpus_level_fn=np.mean,
    batched_compute=False,
)

text_shortqa_ordered_precision_metric = SampleLevelMetric(
    metric_name="text_shortqa_ordered_precision",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=TextShortQAOrderedPrecisionMetric(),
    corpus_level_fn=np.mean,
    batched_compute=False,
)

text_shortqa_ordered_f1_metric = SampleLevelMetric(
    metric_name="text_shortqa_ordered_f1",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=TextShortQAOrderedF1Metric(),
    corpus_level_fn=np.mean,
    batched_compute=False,
)


def record_to_sample(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": record["sample_id"],
        "heritage_name": record["heritage_name"],
        "heritage_name_full": record.get("heritage_name_full", ""),
        "question": record["question"],
        "answer": record["answer"],
        "answer_type": record["answer_type"],
        "source_query": record.get("source_query", ""),
        "source_chosen": record.get("source_chosen", ""),
        "source_rejected": record.get("source_rejected", ""),
        "qa_error_type": record.get("qa_error_type", ""),
        "hallucination_type": record.get("hallucination_type", ""),
        "paraphrase_count": int(record.get("paraphrase_count", 1)),
        "candidate_count_for_heritage": int(record.get("candidate_count_for_heritage", 1)),
    }


def korean_heritage_text_shortqa_prompt(line: dict[str, Any], task_name: str | None = None) -> Doc:
    answer = line["answer"]
    answer_type = line["answer_type"]
    query = (
        f"{line['question']}\n\n"
        f"{_build_answer_instruction(answer_type)}\n\n"
        "CRITICAL INSTRUCTION: DO NOT output any thinking process, reasoning, or explanations. DO NOT output 'Thinking Process:'. Output ONLY the final answer."
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=[answer],
        gold_index=0,
        instruction="",
        specific={
            "sample_id": line["sample_id"],
            "heritage_name": line["heritage_name"],
            "answer": answer,
            "answer_type": line["answer_type"],
            "source_query": line.get("source_query", ""),
            "paraphrase_count": int(line.get("paraphrase_count", 1)),
        },
    )


CUSTOM_KOREAN_HERITAGE_TEXT_SHORTQA_TASK = LightevalTaskConfig(
    name="korean_heritage_text_shortqa",
    prompt_function=korean_heritage_text_shortqa_prompt,
    hf_repo=DATASET_PATH,
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    metrics=[
        text_shortqa_format_valid_metric,
        text_shortqa_empty_response_metric,
        text_shortqa_strict_em_metric,
        text_shortqa_relaxed_em_metric,
        text_shortqa_ordered_recall_metric,
        text_shortqa_ordered_precision_metric,
        text_shortqa_ordered_f1_metric,
    ],
    stop_sequence=[],
    version=2,
    sample_fields=record_to_sample,
    generation_size=2048,
)


TASKS_TABLE = [CUSTOM_KOREAN_HERITAGE_TEXT_SHORTQA_TASK]


if __name__ == "__main__":
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Task name: {CUSTOM_KOREAN_HERITAGE_TEXT_SHORTQA_TASK.name}")
    print(f"Metrics: {CUSTOM_KOREAN_HERITAGE_TEXT_SHORTQA_TASK.metrics}")

"""
Korean character OCR custom task for LightEval.

This task uses the test split stored as a local Hugging Face `save_to_disk` dataset.
The model receives one image and must output the recognized Korean text in the
format `OCR: <text>` on the last line.
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
    "/workspace/2026_llm_data_generation/llm_training_ready/"
    "data_030_korean_character_outside_hf/test"
)
DATASET_PATH = os.environ.get("KOREAN_CHARACTER_OCR_DATASET_PATH", DEFAULT_DATASET_PATH)
DEFAULT_PUBLIC_EXECUTIVE_DATASET_PATH = (
    "/workspace/2026_llm_data_generation/llm_training_ready/"
    "data_public_executive_ocr_hf/test"
)
PUBLIC_EXECUTIVE_DATASET_PATH = os.environ.get(
    "KOREAN_CHARACTER_OCR_PUBLIC_EXECUTIVE_DATASET_PATH", DEFAULT_PUBLIC_EXECUTIVE_DATASET_PATH
)
DEFAULT_KOREAN_CHARACTER_OUTSIDE_DATASET_PATH = (
    "/workspace/2026_llm_data_generation/llm_training_ready/"
    "data_030_korean_character_outside_hf/test"
)
KOREAN_CHARACTER_OUTSIDE_DATASET_PATH = os.environ.get(
    "KOREAN_CHARACTER_OCR_OUTSIDE_DATASET_PATH", DEFAULT_KOREAN_CHARACTER_OUTSIDE_DATASET_PATH
)
DEFAULT_KOREAN_CHARACTER_13_DATASET_PATH = (
    "/workspace/2026_llm_data_generation/llm_training_ready/"
    "data_13_korean_character_hf/test"
)
KOREAN_CHARACTER_13_DATASET_PATH = os.environ.get(
    "KOREAN_CHARACTER_OCR_DATA13_DATASET_PATH", DEFAULT_KOREAN_CHARACTER_13_DATASET_PATH
)
DEFAULT_MULTI_DATASET_PATH = "::".join(
    [
        PUBLIC_EXECUTIVE_DATASET_PATH,
        KOREAN_CHARACTER_OUTSIDE_DATASET_PATH,
        KOREAN_CHARACTER_13_DATASET_PATH,
    ]
)
MULTI_DATASET_PATH = os.environ.get("KOREAN_CHARACTER_OCR_MULTI_DATASET_PATH", DEFAULT_MULTI_DATASET_PATH)


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _remove_spaces(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _extract_korean_text(text: str) -> str:
    chunks = re.findall(r"[가-힣]+", text)
    return "".join(chunks)


def _normalize_quotes(text: str) -> str:
    return text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")


def _normalize_ocr_punctuation(text: str) -> str:
    text = _normalize_quotes(text)
    text = text.replace("×", "x")
    text = re.sub(r"[∼〜～~]", "-", text)
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\s*:\s*", ":", text)
    text = re.sub(r"\(\s*", "(", text)
    text = re.sub(r"\s*\)", ")", text)
    text = re.sub(r"\s*-\s*", "-", text)
    text = text.replace("&", "")
    text = re.sub(r"^/", "", text)
    return _collapse_ws(text)


def _extract_ocr_line(text: str) -> str:
    text = _normalize_quotes(text.strip())
    # Match "OCR: <text>" anywhere, allowing thought process before it
    matches = re.findall(r"OCR\s*:\s*(.+?)(?:\n|$)", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return ""


def _strip_reference_wrappers(text: str) -> str:
    text = _normalize_ocr_punctuation(text)
    # Match both straight quotes; curved quotes were already normalized by
    # _normalize_ocr_punctuation above. Use a backreference so the opening
    # and closing quote are the same kind. Datasets quote the OCR text with
    # single quotes ('...'), so matching only '"' silently failed and the
    # fallback prefix-strip left fragments like "에 '..." in the gold.
    quoted = re.findall(r'(["\'])([^"\'\n]+)\1', text)
    if quoted:
        text = quoted[-1][1].strip()
    else:
        text = re.sub(r"^(결과|정답|답변)\s*:\s*", "", text)
        text = re.sub(r"^(문구|텍스트|속\s*텍스트)\s*:\s*", "", text)
        text = re.sub(r"^(텍스트는|보이는\s*글은)\s*", "", text)
        text = re.sub(r"^(확인된\s*문구는|읽히는\s*문구는|보이는\s*문구는)\s*", "", text)
        # `에` alone must be in the group — otherwise "이미지에" leaves "에 ".
        text = re.sub(r"^(사진|이미지)(에는|에서|에|속|의)?\s*", "", text)
        text = re.sub(
            r"\s*(이라고\s*(써\s*있습니다|적혀\s*있습니다|나옵니다|기록되어\s*있습니다|표기되어\s*있습니다)|"
            r"라고\s*(적혀\s*있습니다|기록되어\s*있습니다|표기되어\s*있습니다|확인됩니다|나옵니다)|"
            r"라는\s*문구가\s*보입니다|라는\s*문장이\s*보입니다|입니다|이\s*보입니다)\.?$",
            "",
            text,
        )

    text = text.strip(" \t\n\r\"'.,:;!?()[]{}")
    return _normalize_ocr_punctuation(text)


def _strip_prediction_wrappers(text: str) -> str:
    text = _extract_ocr_line(text)
    if not text:
        return ""
    return _strip_reference_wrappers(text)


def _generic_strip_prediction_wrappers(text: str) -> str:
    text = _extract_ocr_line(text)
    if not text:
        return ""
    return _normalize_ocr_punctuation(text.strip(" \t\n\r\"'"))


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = curr[j - 1] + 1
            delete_cost = prev[j] + 1
            replace_cost = prev[j - 1] + (ca != cb)
            curr.append(min(insert_cost, delete_cost, replace_cost))
        prev = curr
    return prev[-1]


def extract_gold_from_messages(messages: list[dict[str, str]]) -> str:
    assistant_messages = [m.get("content", "") for m in messages if m.get("role") == "assistant"]
    if not assistant_messages:
        return ""
    return _strip_reference_wrappers(assistant_messages[-1])


def _extract_predictions(model_response) -> list[str]:
    preds = model_response.final_text or []
    if preds and isinstance(preds, list):
        if all(isinstance(x, str) and len(x) == 1 for x in preds):
            return ["".join(preds)]
    elif isinstance(preds, str):
        return [preds]
    return preds


class OCRStrictEMMetric(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        gold = _strip_reference_wrappers(doc.get_golds()[0]) if doc.get_golds() else ""
        predictions = _extract_predictions(model_response)
        if not gold or not predictions:
            return 0.0

        for pred in predictions:
            ocr_text = _extract_ocr_line(pred).strip()
            if ocr_text == gold:
                return 1.0
        return 0.0


class OCRFormatValidMetric(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        predictions = _extract_predictions(model_response)
        if not predictions:
            return 0.0

        for pred in predictions:
            if _extract_ocr_line(pred):
                return 1.0
        return 0.0


class OCREmptyResponseMetric(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        predictions = _extract_predictions(model_response)
        if not predictions:
            return 1.0

        for pred in predictions:
            if pred and pred.strip():
                return 0.0
        return 1.0


class OCRRelaxedEMMetric(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        gold = _strip_reference_wrappers(doc.get_golds()[0]) if doc.get_golds() else ""
        predictions = _extract_predictions(model_response)
        if not gold or not predictions:
            return 0.0

        gold_korean = _extract_korean_text(gold)

        for pred in predictions:
            normalized_pred = _strip_prediction_wrappers(pred)
            if not normalized_pred:
                continue
            pred_korean = _extract_korean_text(normalized_pred)

            if _remove_spaces(normalized_pred) == _remove_spaces(gold):
                return 1.0
            if gold_korean and pred_korean == gold_korean:
                return 1.0
        return 0.0


class OCRKoreanOnlyEMMetric(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        gold = _extract_korean_text(_strip_reference_wrappers(doc.get_golds()[0])) if doc.get_golds() else ""
        predictions = _extract_predictions(model_response)
        if not gold or not predictions:
            return 0.0

        for pred in predictions:
            normalized_pred = _strip_prediction_wrappers(pred)
            if not normalized_pred:
                continue
            pred_korean = _extract_korean_text(normalized_pred)
            if pred_korean == gold:
                return 1.0
        return 0.0


class OCRCERMetric(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        gold = _extract_korean_text(_strip_reference_wrappers(doc.get_golds()[0])) if doc.get_golds() else ""
        predictions = _extract_predictions(model_response)
        if not gold or not predictions:
            return 1.0

        best_cer = 1.0
        for pred in predictions:
            normalized_pred = _extract_korean_text(_strip_prediction_wrappers(pred))
            if not normalized_pred:
                continue
            distance = _levenshtein(normalized_pred, gold)
            cer = distance / max(len(gold), 1)
            best_cer = min(best_cer, cer)
        return best_cer


class OCRGenericStrictEMMetric(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        gold = _strip_reference_wrappers(doc.get_golds()[0]) if doc.get_golds() else ""
        predictions = _extract_predictions(model_response)
        if not gold or not predictions:
            return 0.0

        for pred in predictions:
            normalized_pred = _generic_strip_prediction_wrappers(pred)
            if normalized_pred == gold:
                return 1.0
        return 0.0


class OCRGenericRelaxedEMMetric(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        gold = _strip_reference_wrappers(doc.get_golds()[0]) if doc.get_golds() else ""
        predictions = _extract_predictions(model_response)
        if not gold or not predictions:
            return 0.0

        gold_compact = _remove_spaces(gold)
        for pred in predictions:
            normalized_pred = _generic_strip_prediction_wrappers(pred)
            if not normalized_pred:
                continue
            if _remove_spaces(normalized_pred) == gold_compact:
                return 1.0
        return 0.0


class OCRGenericCERMetric(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        gold = _strip_reference_wrappers(doc.get_golds()[0]) if doc.get_golds() else ""
        predictions = _extract_predictions(model_response)
        if not gold or not predictions:
            return 1.0

        best_cer = 1.0
        for pred in predictions:
            normalized_pred = _generic_strip_prediction_wrappers(pred)
            if not normalized_pred:
                continue
            distance = _levenshtein(normalized_pred, gold)
            cer = distance / max(len(gold), 1)
            best_cer = min(best_cer, cer)
        return best_cer


ocr_strict_em_metric = SampleLevelMetric(
    metric_name="ocr_strict_em",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=OCRStrictEMMetric(),
    corpus_level_fn=np.mean,
    batched_compute=False,
)

ocr_format_valid_metric = SampleLevelMetric(
    metric_name="ocr_format_valid",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=OCRFormatValidMetric(),
    corpus_level_fn=np.mean,
    batched_compute=False,
)

ocr_empty_response_metric = SampleLevelMetric(
    metric_name="ocr_empty_response",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=False,
    sample_level_fn=OCREmptyResponseMetric(),
    corpus_level_fn=np.mean,
    batched_compute=False,
)

ocr_relaxed_em_metric = SampleLevelMetric(
    metric_name="ocr_relaxed_em",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=OCRRelaxedEMMetric(),
    corpus_level_fn=np.mean,
    batched_compute=False,
)

ocr_korean_only_em_metric = SampleLevelMetric(
    metric_name="ocr_korean_only_em",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=OCRKoreanOnlyEMMetric(),
    corpus_level_fn=np.mean,
    batched_compute=False,
)

ocr_cer_metric = SampleLevelMetric(
    metric_name="ocr_cer",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=False,
    sample_level_fn=OCRCERMetric(),
    corpus_level_fn=np.mean,
    batched_compute=False,
)

ocr_generic_strict_em_metric = SampleLevelMetric(
    metric_name="ocr_generic_strict_em",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=OCRGenericStrictEMMetric(),
    corpus_level_fn=np.mean,
    batched_compute=False,
)

ocr_generic_relaxed_em_metric = SampleLevelMetric(
    metric_name="ocr_generic_relaxed_em",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=OCRGenericRelaxedEMMetric(),
    corpus_level_fn=np.mean,
    batched_compute=False,
)

ocr_generic_cer_metric = SampleLevelMetric(
    metric_name="ocr_generic_cer",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=False,
    sample_level_fn=OCRGenericCERMetric(),
    corpus_level_fn=np.mean,
    batched_compute=False,
)


def record_to_sample(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": record["messages"],
        "num_images": len(record.get("images", [])),
        "gold_text": extract_gold_from_messages(record["messages"]),
        "dataset_source": record.get("__dataset_path", ""),
    }


def _extract_user_prompt(messages: list[dict[str, str]]) -> str:
    user_messages = [m.get("content", "") for m in messages if m.get("role") == "user"]
    if not user_messages:
        return ""
    text = user_messages[-1].replace("<image>\n", "").replace("<image>", "").strip()
    return _collapse_ws(text)


def korean_character_ocr_prompt(line: dict[str, Any], task_name: str | None = None) -> Doc:
    gold_text = extract_gold_from_messages(line["messages"])
    query = (
        "이미지에 보이는 한글 텍스트만 OCR하세요. "
        "설명, 번역, 따옴표, 괄호 설명, 접두 문구를 절대 추가하지 마세요. "
        "보이는 순서대로 그대로 적으세요. "
        "한 줄만 출력하세요. "
        "출력 형식은 오직 다음 하나만 허용됩니다: OCR: <한글 텍스트>\n\n"
        "좋은 답변 예시:\n"
        "OCR: 크리스마스\n"
        "OCR: 용한의원\n"
        "OCR: 삼겹살\n\n"
        "틀린 답변 예시:\n"
        "이 이미지에는 크리스마스라고 적혀 있습니다.\n"
        "\"용한의원\"\n"
        "OCR: 보건복지부 인증 치과전문의 서울모담치과"
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=[gold_text],
        gold_index=0,
        images=line.get("images", []),
        instruction="",
        specific={
            "gold_text": gold_text,
            "source_messages": line.get("messages", []),
            "dataset_source": line.get("__dataset_path", ""),
        },
    )


def public_executive_ocr_prompt(line: dict[str, Any], task_name: str | None = None) -> Doc:
    gold_text = extract_gold_from_messages(line["messages"])
    original_user_prompt = _extract_user_prompt(line.get("messages", []))
    query = (
        f"{original_user_prompt} "
        "표시된 영역 안의 텍스트만 OCR하세요. "
        "영역 밖 텍스트는 쓰지 마세요. "
        "설명, 번역, 따옴표, 접두 문구를 절대 추가하지 마세요. "
        "보이는 순서대로 그대로 적으세요. "
        "한 줄만 출력하세요. "
        "출력 형식은 오직 다음 하나만 허용됩니다: OCR: <텍스트>\n\n"
        "좋은 답변 예시:\n"
        "OCR: 환경보호과장\n"
        "OCR: 전화(0525)30-3363/전송(0525)30-3673/담당장병옥\n"
        "OCR: 61-1/전화(0525)30-1351/전송(0525)36-1978/담당배병갑\n\n"
        "틀린 답변 예시:\n"
        "초록색 사각형 안에는 환경보호과장이 쓰여 있습니다.\n"
        "OCR: 김해시 동상동 539.\n"
        "\"(055)322-2967\""
    ).strip()

    return Doc(
        task_name=task_name,
        query=query,
        choices=[gold_text],
        gold_index=0,
        images=line.get("images", []),
        instruction="",
        specific={
            "gold_text": gold_text,
            "source_messages": line.get("messages", []),
            "dataset_source": line.get("__dataset_path", ""),
            "original_user_prompt": original_user_prompt,
        },
    )


def data13_ocr_prompt(line: dict[str, Any], task_name: str | None = None) -> Doc:
    gold_text = extract_gold_from_messages(line["messages"])
    query = (
        "이미지에 보이는 텍스트를 문자 단위로 정확히 OCR하세요. "
        "한글, 영문, 숫자, 기호를 보이는 그대로 유지하세요. "
        "한 글자도 빼거나 더하지 마세요. "
        "설명, 번역, 따옴표, 접두 문구를 절대 추가하지 마세요. "
        "한 줄만 출력하세요. "
        "출력 형식은 오직 다음 하나만 허용됩니다: OCR: <텍스트>\n\n"
        "좋은 답변 예시:\n"
        "OCR: 결\n"
        "OCR: h\n"
        "OCR: OIL\n\n"
        "틀린 답변 예시:\n"
        "이미지에 결이라고 적혀 있습니다.\n"
        "\"h\"\n"
        "OCR: 보기"
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=[gold_text],
        gold_index=0,
        images=line.get("images", []),
        instruction="",
        specific={
            "gold_text": gold_text,
            "source_messages": line.get("messages", []),
            "dataset_source": line.get("__dataset_path", ""),
        },
    )


def build_ocr_task(
    name: str,
    dataset_path: str,
    version: int,
    prompt_function=korean_character_ocr_prompt,
    metrics=None,
) -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name=name,
        prompt_function=prompt_function,
        hf_repo=dataset_path,
        hf_subset="default",
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split=None,
        few_shots_select=None,
        metrics=metrics
        or [
            ocr_format_valid_metric,
            ocr_empty_response_metric,
            ocr_strict_em_metric,
            ocr_relaxed_em_metric,
            ocr_korean_only_em_metric,
            ocr_cer_metric,
        ],
        stop_sequence=[],
        version=version,
        sample_fields=record_to_sample,
        generation_size=2048,
    )


CUSTOM_KOREAN_CHARACTER_OCR_TASK = build_ocr_task(
    name="korean_character_ocr",
    dataset_path=DATASET_PATH,
    version=6,
)

CUSTOM_KOREAN_CHARACTER_OCR_PUBLIC_EXECUTIVE_TASK = build_ocr_task(
    name="korean_character_ocr_public_executive",
    dataset_path=PUBLIC_EXECUTIVE_DATASET_PATH,
    version=3,
    prompt_function=public_executive_ocr_prompt,
)

CUSTOM_KOREAN_CHARACTER_OCR_OUTSIDE_TASK = build_ocr_task(
    name="korean_character_ocr_outside",
    dataset_path=KOREAN_CHARACTER_OUTSIDE_DATASET_PATH,
    version=3,
)

CUSTOM_KOREAN_CHARACTER_OCR_DATA13_TASK = build_ocr_task(
    name="korean_character_ocr_data13",
    dataset_path=KOREAN_CHARACTER_13_DATASET_PATH,
    version=4,
    prompt_function=data13_ocr_prompt,
    metrics=[
        ocr_format_valid_metric,
        ocr_empty_response_metric,
        ocr_generic_strict_em_metric,
        ocr_generic_relaxed_em_metric,
        ocr_generic_cer_metric,
    ],
)

CUSTOM_KOREAN_CHARACTER_OCR_ALL_TESTS_TASK = build_ocr_task(
    name="korean_character_ocr_all_tests",
    dataset_path=MULTI_DATASET_PATH,
    version=3,
)

TASKS_TABLE = [
    CUSTOM_KOREAN_CHARACTER_OCR_TASK,
    CUSTOM_KOREAN_CHARACTER_OCR_PUBLIC_EXECUTIVE_TASK,
    CUSTOM_KOREAN_CHARACTER_OCR_OUTSIDE_TASK,
    CUSTOM_KOREAN_CHARACTER_OCR_DATA13_TASK,
    CUSTOM_KOREAN_CHARACTER_OCR_ALL_TESTS_TASK,
]


if __name__ == "__main__":
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Task name: {CUSTOM_KOREAN_CHARACTER_OCR_TASK.name}")
    print(f"Metrics: {CUSTOM_KOREAN_CHARACTER_OCR_TASK.metrics}")

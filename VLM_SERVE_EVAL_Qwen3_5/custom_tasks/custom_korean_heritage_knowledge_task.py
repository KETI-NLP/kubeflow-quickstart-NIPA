"""
Korean heritage knowledge VQA custom task for LightEval.

heritage_data.json 의 basic_info 4개 필드(시대/소재지/분류/지정연도)에 대한
이미지 기반 지식 평가. 각 샘플은:
- question
- answer_keywords (정답 키워드 리스트 — 하나라도 응답에 포함되면 정답)
- image_path
- category ('era' | 'location' | 'category' | 'year')

모델은 다음 형식으로 응답하도록 유도:
  Answer: <짧은 정답 키워드>

메트릭:
- heritage_knowledge_em: 전체 평균(모든 카테고리 합산)
- heritage_era_em / location_em / category_em / year_em: 카테고리별 평균
  (다른 카테고리 샘플엔 NaN 반환 -> nanmean 으로 무시)
"""

from __future__ import annotations
import custom_tasks.force_no_think_patch  # noqa: F401

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
    "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/"
    "tmp_eval_datasets/korean_heritage_knowledge_hf"
)
DATASET_PATH = os.environ.get("KOREAN_HERITAGE_KNOWLEDGE_DATASET_PATH", DEFAULT_DATASET_PATH)
DEFAULT_IMAGE_CACHE_DIR = (
    "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/"
    "tmp_eval_datasets/korean_heritage_knowledge_image_cache"
)
IMAGE_CACHE_DIR = os.environ.get("KOREAN_HERITAGE_KNOWLEDGE_IMAGE_CACHE_DIR", DEFAULT_IMAGE_CACHE_DIR)


# ==================== 이미지 로딩 (Name VQA 패턴 재사용) ====================

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


# ==================== 응답 파싱 / 채점 ====================

def _normalize_quotes(text: str) -> str:
    return text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")


def _extract_answer_text(text: str) -> str:
    """모델 출력에서 'Answer: ...' 라인을 우선 추출. 없으면 마지막 비어있지 않은 라인."""
    text = _normalize_quotes((text or "").strip())
    if not text:
        return ""
    matches = re.findall(r"Answer\s*:\s*(.+?)(?:\n|$)", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    non_empty = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return non_empty[-1] if non_empty else ""


def _extract_predictions(model_response: ModelResponse) -> list[str]:
    preds = model_response.final_text or []
    if preds and isinstance(preds, list):
        if all(isinstance(x, str) and len(x) == 1 for x in preds):
            return ["".join(preds)]
    elif isinstance(preds, str):
        return [preds]
    return preds


def _get_category(doc: Doc) -> str:
    spec = doc.specific or {}
    return spec.get("category", "")


def _get_answer_keywords(doc: Doc) -> list[str]:
    spec = doc.specific or {}
    return list(spec.get("answer_keywords") or [])


def _check_keyword_substring(prediction: str, keywords: list[str]) -> float:
    """응답에서 'Answer:' 추출 후, 어떤 정답 키워드라도 substring으로 포함되면 1.0."""
    if not keywords:
        return 0.0
    answer = _extract_answer_text(prediction)
    if not answer:
        return 0.0
    # 정답 키워드 substring 매칭 (조선/서울/유적건조물 등 단어 단위)
    for kw in keywords:
        if kw and kw in answer:
            return 1.0
    return 0.0


def _check_year_match(prediction: str, keywords: list[str]) -> float:
    """4자리 연도 정확 매칭. 응답에서 4자리 연도를 모두 찾고, 정답 연도가 있으면 1.0."""
    if not keywords:
        return 0.0
    answer = _extract_answer_text(prediction)
    if not answer:
        return 0.0
    years_in_pred = re.findall(r"(?<!\d)(\d{4})(?!\d)", answer)
    if not years_in_pred:
        # Fallback: 전체 응답에서도 찾아본다 (모델이 Answer: 라인을 못 만든 경우)
        years_in_pred = re.findall(r"(?<!\d)(\d{4})(?!\d)", prediction or "")
    return 1.0 if any(y in keywords for y in years_in_pred) else 0.0


# ==================== 메트릭 클래스 ====================

class HeritageOverallEM(SampleLevelComputation):
    """카테고리 무관 전체 정답률."""
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        preds = _extract_predictions(model_response)
        if not preds:
            return 0.0
        keywords = _get_answer_keywords(doc)
        cat = _get_category(doc)
        best = 0.0
        for p in preds:
            if cat == "year":
                s = _check_year_match(p, keywords)
            else:
                s = _check_keyword_substring(p, keywords)
            best = max(best, s)
        return best


class HeritagePerCategoryEM(SampleLevelComputation):
    """특정 카테고리에서만 0/1 점수, 다른 카테고리 샘플엔 NaN 반환."""
    def __init__(self, target_category: str):
        self.target_category = target_category

    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        cat = _get_category(doc)
        if cat != self.target_category:
            return float("nan")
        preds = _extract_predictions(model_response)
        if not preds:
            return 0.0
        keywords = _get_answer_keywords(doc)
        best = 0.0
        for p in preds:
            if cat == "year":
                s = _check_year_match(p, keywords)
            else:
                s = _check_keyword_substring(p, keywords)
            best = max(best, s)
        return best


class HeritageFormatValid(SampleLevelComputation):
    """응답에 'Answer:' 라인이 있으면 1.0."""
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        preds = _extract_predictions(model_response)
        for p in preds:
            if re.search(r"Answer\s*:\s*(.+?)(?:\n|$)", p or "", flags=re.IGNORECASE):
                return 1.0
        return 0.0


def _nanmean(values):
    """NaN 무시 평균. lighteval corpus_level_fn 으로 사용."""
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.nanmean(arr)) if not np.all(np.isnan(arr)) else 0.0


# ==================== 메트릭 등록 ====================

heritage_knowledge_em = SampleLevelMetric(
    metric_name="heritage_knowledge_em",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=HeritageOverallEM(),
    corpus_level_fn=np.mean,
    batched_compute=False,
)

heritage_era_em = SampleLevelMetric(
    metric_name="heritage_era_em",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=HeritagePerCategoryEM("era"),
    corpus_level_fn=_nanmean,
    batched_compute=False,
)

heritage_location_em = SampleLevelMetric(
    metric_name="heritage_location_em",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=HeritagePerCategoryEM("location"),
    corpus_level_fn=_nanmean,
    batched_compute=False,
)

heritage_category_em = SampleLevelMetric(
    metric_name="heritage_category_em",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=HeritagePerCategoryEM("category"),
    corpus_level_fn=_nanmean,
    batched_compute=False,
)

heritage_year_em = SampleLevelMetric(
    metric_name="heritage_year_em",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=HeritagePerCategoryEM("year"),
    corpus_level_fn=_nanmean,
    batched_compute=False,
)

heritage_format_valid = SampleLevelMetric(
    metric_name="heritage_format_valid",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=HeritageFormatValid(),
    corpus_level_fn=np.mean,
    batched_compute=False,
)


# ==================== Prompt / 데이터 매핑 ====================

def record_to_sample(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": record["sample_id"],
        "heritage_name": record["heritage_name"],
        "image_path": record["image_path"],
        "image_url": record.get("image_url", ""),
        "category": record["category"],
        "question": record["question"],
        "answer_keywords": list(record["answer_keywords"]),
        "answer_display": record.get("answer_display", ""),
    }


def korean_heritage_knowledge_prompt(line: dict[str, Any], task_name: str | None = None) -> Doc:
    answer_keywords = list(line["answer_keywords"])
    category = line["category"]
    question = line["question"]

    query = (
        f"{question}\n\n"
        "반드시 마지막 줄에 다음 형식으로만 답하세요:\n"
        "Answer: <짧은 정답 키워드>\n\n"
        "예시:\n"
        "Answer: 조선\n"
        "Answer: 서울\n"
        "Answer: 유적건조물\n"
        "Answer: 1962\n\n"
        "정답 키워드 하나만 짧게 적으세요. 부가 설명·문장은 다른 줄에 쓰고, 마지막 줄은 정확히 'Answer: 키워드' 형식만."
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=[answer_keywords[0] if answer_keywords else ""],
        gold_index=0,
        images=[_load_image(line["image_path"])],
        instruction="",
        specific={
            "sample_id": line["sample_id"],
            "heritage_name": line["heritage_name"],
            "image_path": line["image_path"],
            "category": category,
            "answer_keywords": answer_keywords,
            "answer_display": line.get("answer_display", ""),
        },
    )


CUSTOM_KOREAN_HERITAGE_KNOWLEDGE_TASK = LightevalTaskConfig(
    name="korean_heritage_knowledge",
    prompt_function=korean_heritage_knowledge_prompt,
    hf_repo=DATASET_PATH,
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    metrics=[
        heritage_format_valid,
        heritage_knowledge_em,
        heritage_era_em,
        heritage_location_em,
        heritage_category_em,
        heritage_year_em,
    ],
    stop_sequence=[],
    version=1,
    sample_fields=record_to_sample,
    generation_size=512,
)


TASKS_TABLE = [CUSTOM_KOREAN_HERITAGE_KNOWLEDGE_TASK]


if __name__ == "__main__":
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Task name: {CUSTOM_KOREAN_HERITAGE_KNOWLEDGE_TASK.name}")
    print(f"Metrics: {[m.metric_name for m in CUSTOM_KOREAN_HERITAGE_KNOWLEDGE_TASK.metrics]}")

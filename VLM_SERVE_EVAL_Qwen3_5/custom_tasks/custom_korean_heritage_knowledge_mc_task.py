"""
Korean Heritage Knowledge — **객관식(4지선다)** custom task for LightEval.

같은 500 heritage / 4 카테고리(시대/소재지/분류/지정연도)를 MMBench 스타일 MC로 묻는다.
정답 1개 + 오답 3개 → letter (A/B/C/D) 로 답.

채점:
- StrictMultipleChoiceMatch (custom_mc_metric) 의 letter 추출 로직을 재사용
- 카테고리별 메트릭은 자체 SampleLevelComputation 으로 NaN 활용 (nanmean)
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
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod
from custom_tasks.custom_mc_metric import StrictMultipleChoiceMatch, strict_mc_metric


DEFAULT_DATASET_PATH = (
    "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/"
    "tmp_eval_datasets/korean_heritage_knowledge_mc_hf"
)
DATASET_PATH = os.environ.get("KOREAN_HERITAGE_KNOWLEDGE_MC_DATASET_PATH", DEFAULT_DATASET_PATH)
DEFAULT_IMAGE_CACHE_DIR = (
    "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/"
    "tmp_eval_datasets/korean_heritage_knowledge_mc_image_cache"
)
IMAGE_CACHE_DIR = os.environ.get("KOREAN_HERITAGE_KNOWLEDGE_MC_IMAGE_CACHE_DIR", DEFAULT_IMAGE_CACHE_DIR)


# ==================== 이미지 로딩 + robustness perturbation ====================

import numpy as np
from PIL import ImageEnhance

# perturbation severity (1~5). 변형 강도. 환경변수로 조정 가능.
PERTURB_SEVERITY = int(os.environ.get("HERITAGE_MC_PERTURB_SEVERITY", "2"))


def _perturb_image(img: "Image.Image", perturb: str, severity: int, seed: int) -> "Image.Image":
    """이미지에 결정적(seed 고정) 변형을 가해 robustness 측정용 이미지를 만든다.
    perturb: none|noise|brightness|rotation|shift|all
    - noise:      가우시안 노이즈 (sigma = severity*12)
    - brightness: 밝기 ±(severity*12)%
    - rotation:   회전 ±(severity*4)도
    - shift:      평행이동 ±(severity*2.5)%
    - all:        위 4개 순차 적용
    """
    if perturb == "none" or severity <= 0:
        return img
    rng = np.random.RandomState(seed % (2 ** 31))
    types = ["noise", "brightness", "rotation", "shift"] if perturb == "all" else [perturb]
    out = img.convert("RGB")
    for t in types:
        if t == "noise":
            arr = np.asarray(out).astype(np.float32)
            arr = np.clip(arr + rng.normal(0, severity * 12.0, arr.shape), 0, 255).astype(np.uint8)
            out = Image.fromarray(arr)
        elif t == "brightness":
            factor = 1.0 + float(rng.choice([-1, 1])) * severity * 0.12
            out = ImageEnhance.Brightness(out).enhance(factor)
        elif t == "rotation":
            angle = float(rng.uniform(-1, 1)) * severity * 4.0
            out = out.rotate(angle, resample=Image.BILINEAR, fillcolor=(128, 128, 128))
        elif t == "shift":
            w, h = out.size
            mx = severity * 0.025
            dx = int(rng.uniform(-mx, mx) * w)
            dy = int(rng.uniform(-mx, mx) * h)
            out = out.transform((w, h), Image.AFFINE, (1, 0, -dx, 0, 1, -dy),
                                resample=Image.BILINEAR, fillcolor=(128, 128, 128))
    return out.convert("RGB")


@lru_cache(maxsize=8192)
def _prepare_image_path(image_path: str, perturb: str, severity: int) -> str:
    os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)
    # 캐시 키에 perturb 설정 포함 → clean/noise/rotation 캐시 충돌 없음
    key = f"{image_path}|{perturb}|{severity}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    cached = os.path.join(IMAGE_CACHE_DIR, f"{digest}.jpg")
    if os.path.exists(cached):
        return cached
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        rgb.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        if perturb != "none":
            # seed = image_path 해시 → 같은 이미지엔 항상 같은 변형 (재현성)
            seed = int(hashlib.sha1(image_path.encode("utf-8")).hexdigest()[:8], 16)
            rgb = _perturb_image(rgb, perturb, severity, seed)
        rgb.save(cached, format="JPEG", quality=85, optimize=True)
    return cached


@lru_cache(maxsize=8192)
def _load_image(image_path: str, perturb: str = "none", severity: int = 0) -> "Image.Image":
    img = Image.open(_prepare_image_path(image_path, perturb, severity))
    img.load()
    return img


# ==================== 카테고리별 메트릭 (StrictMultipleChoiceMatch 재사용) ====================

class HeritageMCPerCategory(SampleLevelComputation):
    """특정 카테고리에서만 0/1 점수, 다른 카테고리 샘플엔 NaN."""
    def __init__(self, target_category: str):
        self.target_category = target_category
        self._inner = StrictMultipleChoiceMatch()

    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        cat = (doc.specific or {}).get("category", "")
        if cat != self.target_category:
            return float("nan")
        return self._inner.compute(doc, model_response, **kwargs)


def _nanmean(values):
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.nanmean(arr)) if not np.all(np.isnan(arr)) else 0.0


class HeritageMCFormatValid(SampleLevelComputation):
    """응답에 'Answer:' 또는 '정답:' 라인 + A-D 글자가 있으면 1.0."""
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        text = model_response.final_text or []
        if isinstance(text, list):
            text = "".join(t if isinstance(t, str) else "" for t in text)
        if not isinstance(text, str):
            text = str(text)
        m = re.findall(r"(?i)(?:answer|정답|정답은)[\s]*:?[\s]*(.*)", text)
        if m and re.search(r"\b([A-D])\b", m[-1].upper()):
            return 1.0
        return 0.0


heritage_mc_era = SampleLevelMetric(
    metric_name="heritage_mc_era",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=HeritageMCPerCategory("era"),
    corpus_level_fn=_nanmean,
    batched_compute=False,
)
heritage_mc_location = SampleLevelMetric(
    metric_name="heritage_mc_location",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=HeritageMCPerCategory("location"),
    corpus_level_fn=_nanmean,
    batched_compute=False,
)
heritage_mc_category = SampleLevelMetric(
    metric_name="heritage_mc_category",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=HeritageMCPerCategory("category"),
    corpus_level_fn=_nanmean,
    batched_compute=False,
)
heritage_mc_year = SampleLevelMetric(
    metric_name="heritage_mc_year",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=HeritageMCPerCategory("year"),
    corpus_level_fn=_nanmean,
    batched_compute=False,
)
heritage_mc_format_valid = SampleLevelMetric(
    metric_name="heritage_mc_format_valid",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=HeritageMCFormatValid(),
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
        "A": record["A"], "B": record["B"], "C": record["C"], "D": record["D"],
        "answer": record["answer"],
        "correct_choice_text": record.get("correct_choice_text", ""),
    }


def _make_prompt_fn(perturb: str, severity: int):
    """perturbation 종류별로 묶인 prompt 함수 생성 (이미지 변형 적용)."""
    def _prompt(line: dict[str, Any], task_name: str | None = None) -> Doc:
        question = line["question"]
        choices_text = ""
        choices = []
        for letter in ("A", "B", "C", "D"):
            choices_text += f"{letter}. {line[letter]}\n"
            choices.append(f" {letter}")

        query = (
            f"{question}\n{choices_text}\n"
            "다음을 읽고 정답을 고르시오. 응답의 마지막 줄은 반드시 'Answer: $LETTER' "
            "(또는 '정답: $LETTER') 형식이어야 합니다. $LETTER 는 A, B, C, D 중 하나입니다.\n\n"
            "예시:\nAnswer: A\nAnswer: B"
        )

        answer = line["answer"]
        gold_index = LETTER_INDICES.index(answer) if answer in LETTER_INDICES else 0

        return Doc(
            task_name=task_name,
            query=query,
            choices=choices,
            gold_index=gold_index,
            images=[_load_image(line["image_path"], perturb, severity)],
            instruction="",
            specific={
                "sample_id": line["sample_id"],
                "heritage_name": line["heritage_name"],
                "image_path": line["image_path"],
                "category": line["category"],
                "correct_choice_text": line.get("correct_choice_text", ""),
                "perturb": perturb,
            },
        )
    return _prompt


_MC_METRICS = [
    heritage_mc_format_valid,
    strict_mc_metric,  # 전체 평균 (모든 카테고리 통합)
    heritage_mc_era,
    heritage_mc_location,
    heritage_mc_category,
    heritage_mc_year,
]


def _make_mc_task(name: str, perturb: str, severity: int) -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name=name,
        prompt_function=_make_prompt_fn(perturb, severity),
        hf_repo=DATASET_PATH,
        hf_subset="default",
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split=None,
        few_shots_select=None,
        metrics=_MC_METRICS,
        stop_sequence=[],
        version=1,
        sample_fields=record_to_sample,
        # The scored answer is a single A-D letter. Keep a small allowance for
        # incidental explanation while preventing non-EOS responses from
        # generating thousands of unnecessary tokens.
        generation_size=128,
    )


# clean(변형 없음) + 4종 단일 변형 + 전체 변형. 각각 별도 태스크라 결과 디렉토리 분리됨.
CUSTOM_KOREAN_HERITAGE_KNOWLEDGE_MC_TASK = _make_mc_task("korean_heritage_knowledge_mc", "none", 0)

TASKS_TABLE = [
    CUSTOM_KOREAN_HERITAGE_KNOWLEDGE_MC_TASK,
    _make_mc_task("korean_heritage_knowledge_mc_noise", "noise", PERTURB_SEVERITY),
    _make_mc_task("korean_heritage_knowledge_mc_bright", "brightness", PERTURB_SEVERITY),
    _make_mc_task("korean_heritage_knowledge_mc_rotate", "rotation", PERTURB_SEVERITY),
    _make_mc_task("korean_heritage_knowledge_mc_shift", "shift", PERTURB_SEVERITY),
    _make_mc_task("korean_heritage_knowledge_mc_allperturb", "all", PERTURB_SEVERITY),
]


if __name__ == "__main__":
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Task name: {CUSTOM_KOREAN_HERITAGE_KNOWLEDGE_MC_TASK.name}")
    print(f"Metrics: {[m.metric_name for m in CUSTOM_KOREAN_HERITAGE_KNOWLEDGE_MC_TASK.metrics]}")

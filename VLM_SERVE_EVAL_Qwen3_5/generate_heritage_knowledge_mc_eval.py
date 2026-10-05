"""
Korean Heritage Knowledge 평가 데이터셋 — **객관식(4지선다)** 버전.

generate_heritage_knowledge_eval.py 와 같은 500 heritage(heritage_eval_split.json 재사용)
+ 같은 4 카테고리(시대/소재지/분류/지정연도). 단, 각 질문에 정답 1개 + 오답 3개 = 4 보기를 제시하고
모델은 'Answer: X' 또는 '정답: X' (X ∈ A/B/C/D) 형식으로 letter를 골라야 한다.

채점은 기존 custom_mc_metric.strict_mc_metric 를 그대로 사용 가능 (letter 추출 패턴 동일).
오답(distractor) 풀:
- era: ERA_POOL 의 다른 시대들
- location: REGION_PREFIXES 의 다른 광역시/도들
- category: heritage_data 스캔으로 모은 상위 분류들
- year: 1962~2025 의 다른 연도들
"""

import argparse
import json
import os
import random
import re
import sys

HERITAGE_BASE_DIR = "/workspace/2026_llm_data_generation/korean_heritage_multimodal_hallucination_dpo"
sys.path.append(HERITAGE_BASE_DIR)
sys.path.append("/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5")
from image_cache_manager import ImageCacheManager  # noqa: E402
# 추출 헬퍼는 open-ended 버전과 공유 (시대/지역/분류/연도 정답 동일)
from generate_heritage_knowledge_eval import (  # noqa: E402
    clean_heritage_name, extract_era_keywords, extract_region,
    extract_top_category, extract_year, ERA_KEYWORDS, REGION_PREFIXES,
)

HERITAGE_DATA_PATH = os.path.join(HERITAGE_BASE_DIR, "heritage_data.json")
DEFAULT_DATASET_DIR = "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/tmp_eval_datasets/korean_heritage_knowledge_mc_hf"
DEFAULT_SPLIT_PATH = "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/heritage_eval_split.json"

CATEGORIES = ("era", "location", "category", "year")

QUESTION_TEMPLATES = {
    "era": [
        "이 문화재가 만들어진 시대는 언제인가요?",
        "이 문화재는 어느 시대의 것인가요?",
        "사진 속 문화재가 만들어진 시대를 알려주세요.",
    ],
    "location": [
        "이 문화재는 어디에 위치해 있나요? (광역시/도)",
        "이 문화재의 소재지(시/도)는 어디인가요?",
        "사진 속 문화재가 있는 광역시/도는 어디인가요?",
    ],
    "category": [
        "이 문화재의 최상위 분류는 무엇인가요?",
        "이 문화재가 속한 가장 큰 분류는 무엇인가요?",
        "사진 속 문화재의 상위 카테고리는 무엇인가요?",
    ],
    "year": [
        "이 문화재가 처음 문화재로 지정·등록된 연도는 언제인가요?",
        "이 문화재의 지정 연도는 몇 년인가요?",
        "사진 속 문화재가 문화재로 지정된 연도를 고르세요.",
    ],
}

ERA_POOL = sorted(set(ERA_KEYWORDS) | {
    "조선", "고려", "통일신라", "신라", "백제", "고구려", "발해", "가야", "삼국", "고조선",
    "대한제국", "일제강점기", "근대", "현대", "선사", "구석기", "신석기", "청동기", "철기",
}, key=lambda x: -len(x))

REGION_POOL = list(REGION_PREFIXES)


def collect_category_pool(heritage_data) -> list[str]:
    """heritage_data 전체에서 최상위 분류 키워드 수집 (빈도순)."""
    from collections import Counter
    cnt = Counter()
    for it in heritage_data:
        bi = it.get("basic_info") or {}
        top = extract_top_category(bi.get("분류", ""))
        if "무형유산" in bi.get("분류", ""):
            continue
        if top:
            cnt[top] += 1
    return [k for k, _ in cnt.most_common() if k]


def collect_year_pool(heritage_data) -> list[str]:
    """heritage_data 전체에서 등장하는 지정 연도들 (중복 제거)."""
    yrs = set()
    for it in heritage_data:
        bi = it.get("basic_info") or {}
        y = extract_year(bi.get("지정(등록)일", ""))
        if y:
            yrs.add(y)
    return sorted(yrs)


def make_mc_choices(correct: str, pool: list[str], rng: random.Random, num_distractors: int = 3):
    """정답 1개 + 오답 N개 -> 셔플 후 letter 배정. 풀이 부족하면 가용 만큼."""
    distractors = [c for c in pool if c != correct]
    rng.shuffle(distractors)
    distractors = distractors[:num_distractors]
    choices = [correct] + distractors
    rng.shuffle(choices)
    while len(choices) < 4:
        choices.append("(해당 없음)")
    correct_idx = choices.index(correct)
    return choices, ["A", "B", "C", "D"][correct_idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42,
                        help="MC 보기 셔플 시드 (heritage 선정은 open-ended 와 같은 split 재사용)")
    parser.add_argument("--dataset-out", type=str, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--split-in", type=str, default=DEFAULT_SPLIT_PATH,
                        help="heritage_eval_split.json 경로 (open-ended 와 동일 heritage 사용)")
    args = parser.parse_args()

    with open(HERITAGE_DATA_PATH, "r", encoding="utf-8") as f:
        heritage_data = json.load(f)

    with open(args.split_in, "r", encoding="utf-8") as f:
        split_info = json.load(f)
    eval_heritage_names = set(split_info["heritage_names"])
    print(f"평가 split 로드: {len(eval_heritage_names)} 문화재")

    cache_mgr = ImageCacheManager(HERITAGE_BASE_DIR)

    category_pool = collect_category_pool(heritage_data)
    year_pool = collect_year_pool(heritage_data)
    print(f"오답풀 — category: {len(category_pool)}개 (top 6: {category_pool[:6]})")
    print(f"오답풀 — year: {len(year_pool)}개 (범위 {year_pool[0]}~{year_pool[-1]})")
    print(f"오답풀 — era: {len(ERA_POOL)}개, location: {len(REGION_POOL)}개")

    # heritage_eval_split 의 순서 보존을 위해 heritage_data 인덱스로 매칭
    name_to_item = {clean_heritage_name(it.get("name", "")): it for it in heritage_data}

    records = []
    rng = random.Random(args.seed)
    sample_id = 0

    for name in split_info["heritage_names"]:
        item = name_to_item.get(name)
        if item is None:
            print(f"[경고] heritage_data 에서 못 찾음: {name}")
            continue
        bi = item.get("basic_info") or {}
        image_url = item.get("main_photo")
        if not image_url:
            continue
        image_path = cache_mgr.get_cache_path(name)
        if not os.path.exists(image_path):
            continue

        eras = extract_era_keywords(bi.get("시대", ""))
        region = extract_region(bi.get("소재지", ""))
        top_cat = extract_top_category(bi.get("분류", ""))
        year = extract_year(bi.get("지정(등록)일", ""))

        if not (eras and region and top_cat and year):
            continue

        answers_by_cat = {
            "era": (eras[0], ERA_POOL),
            "location": (region, REGION_POOL),
            "category": (top_cat, category_pool),
            "year": (year, year_pool),
        }

        for cat in CATEGORIES:
            correct, pool = answers_by_cat[cat]
            choices, letter = make_mc_choices(correct, pool, rng)
            q = rng.choice(QUESTION_TEMPLATES[cat])
            records.append({
                "sample_id": str(sample_id),
                "heritage_name": name,
                "image_path": image_path,
                "image_url": image_url,
                "category": cat,
                "question": q,
                "A": choices[0],
                "B": choices[1],
                "C": choices[2],
                "D": choices[3],
                "answer": letter,
                "correct_choice_text": correct,
            })
            sample_id += 1

    print(f"평가 레코드 (MC): {len(records)}건")

    from datasets import Dataset
    ds = Dataset.from_list(records)
    os.makedirs(args.dataset_out, exist_ok=True)
    ds.save_to_disk(args.dataset_out)
    print(f"\nHF Dataset 저장: {args.dataset_out}")
    print(f"  rows: {len(ds)}")
    print("\n샘플[0..3]:")
    for i in range(min(4, len(ds))):
        r = ds[i]
        print(f"  [{i}] {r['category']:8s} | Q: {r['question'][:46]}...")
        print(f"      A:{r['A'][:14]:<14} B:{r['B'][:14]:<14} C:{r['C'][:14]:<14} D:{r['D'][:14]:<14}")
        print(f"      정답: {r['answer']} ({r['correct_choice_text']})  heritage: {r['heritage_name'][:30]}")


if __name__ == "__main__":
    main()

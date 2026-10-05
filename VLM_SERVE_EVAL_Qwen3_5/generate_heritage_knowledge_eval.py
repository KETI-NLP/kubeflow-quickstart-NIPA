"""
Korean Heritage Knowledge 평가 데이터셋 생성기.

heritage_data.json 의 basic_info 4개 필드(시대/소재지/분류/지정연도)에서
결정적으로 정답 키워드를 추출해 (이미지, 질문, 정답 키워드) 형태의 평가 샘플을 만든다.
LLM 호출 없음. simple_name 패턴과 동일하게 ImageCacheManager 로 캐시 이미지만 사용.

출력:
- HF Dataset: tmp_eval_datasets/korean_heritage_knowledge_hf/
- 평가 전용 heritage 목록: heritage_eval_split.json (학습 데이터에서 차단용)
"""

import argparse
import json
import os
import random
import re
import sys

HERITAGE_BASE_DIR = "/workspace/2026_llm_data_generation/korean_heritage_multimodal_hallucination_dpo"
sys.path.append(HERITAGE_BASE_DIR)
from image_cache_manager import ImageCacheManager  # noqa: E402

HERITAGE_DATA_PATH = os.path.join(HERITAGE_BASE_DIR, "heritage_data.json")
DEFAULT_DATASET_DIR = "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/tmp_eval_datasets/korean_heritage_knowledge_hf"
DEFAULT_SPLIT_PATH = "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/heritage_eval_split.json"


# ==================== 정답 추출 헬퍼 ====================

# 한국사 시대 키워드 (긴 것 우선 매칭). 부분 매칭 충돌 방지 위해 sorted by -len.
_ERA_KEYWORDS_RAW = [
    "통일신라", "남북국", "고조선", "후삼국",
    "삼국", "고구려", "백제", "신라", "발해", "가야",
    "고려", "조선", "대한제국",
    "선사", "구석기", "신석기", "청동기", "철기",
    "일제강점기", "근대", "현대",
]
ERA_KEYWORDS = sorted(set(_ERA_KEYWORDS_RAW), key=lambda x: -len(x))

# 광역시/도 prefix (소재지 첫 토큰). 긴 것 먼저.
REGION_PREFIXES = sorted([
    "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
    "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
], key=lambda x: -len(x))


def clean_heritage_name(name: str) -> str:
    if not name:
        return ""
    name = re.sub(r"[^\S ]", "", name)
    name = re.sub(r" +", " ", name)
    return name.strip()


def extract_era_keywords(era_str: str) -> list[str]:
    """시대 문자열에서 표준 시대 키워드 추출.
    1) 명시적 시대 키워드 매칭 (긴 것 우선, 부분 매칭 충돌 회피)
    2) 동의어 처리 (여말선초, 일제시대 등)
    3) 연도 기반 추정 (1392~1897=조선, 1897~1910=대한제국, 1910~1945=일제강점기, 1945+=현대)
    """
    if not era_str:
        return []
    found = []

    # 1. 명시 키워드 (긴 것 먼저, 부분 매칭 제거)
    for kw in ERA_KEYWORDS:
        if kw in era_str and not any(kw in existing for existing in found):
            found.append(kw)

    # 2. 동의어
    if "여말선초" in era_str:
        for kw in ("고려", "조선"):
            if kw not in found:
                found.append(kw)
    if "일제시대" in era_str and "일제강점기" not in found:
        found.append("일제강점기")

    # 3. 연도 기반 추정 (4자리 연도가 명시된 경우에만)
    if not found:
        m = re.search(r"(\d{4})", era_str)
        if m:
            year = int(m.group(1))
            if 1392 <= year < 1897:
                found.append("조선")
            elif 1897 <= year < 1910:
                found.append("대한제국")
            elif 1910 <= year < 1945:
                found.append("일제강점기")
            elif year >= 1945:
                found.append("현대")

    return found


def extract_region(location_str: str) -> str:
    if not location_str:
        return ""
    s = location_str.strip()
    for pre in REGION_PREFIXES:
        if s.startswith(pre):
            return pre
    return ""


def extract_top_category(category_str: str) -> str:
    if not category_str:
        return ""
    parts = [p.strip() for p in category_str.split("/")]
    return parts[0] if parts and parts[0] else ""


def extract_year(date_str: str) -> str:
    if not date_str:
        return ""
    m = re.search(r"(\d{4})", date_str)
    return m.group(1) if m else ""


# ==================== 질문 템플릿 (카테고리당 3개) ====================

QUESTION_TEMPLATES = {
    "era": [
        "이 문화재가 만들어진 시대는 언제인가요?",
        "이 문화재는 어느 시대의 것인가요?",
        "사진 속 문화재가 만들어진 시대를 알려주세요.",
    ],
    "location": [
        "이 문화재는 어디에 위치해 있나요? 광역시/도 단위로 답해주세요.",
        "이 문화재의 소재지는 어디인가요? 시/도 이름으로 답해주세요.",
        "사진 속 문화재가 있는 광역시/도를 알려주세요.",
    ],
    "category": [
        "이 문화재의 분류(최상위 카테고리)는 무엇인가요?",
        "이 문화재가 속한 가장 큰 분류는 무엇인가요?",
        "사진 속 문화재의 상위 카테고리는 무엇인가요?",
    ],
    "year": [
        "이 문화재가 처음 문화재로 지정·등록된 연도(서기, 4자리)는 언제인가요?",
        "이 문화재의 지정 연도는 몇 년인가요? 4자리 서기 연도로 답해주세요.",
        "사진 속 문화재가 문화재로 지정된 연도(예: 1962)를 알려주세요.",
    ],
}

ANSWER_FORMAT_HINT = (
    "반드시 마지막 줄에 다음 형식으로만 답하세요:\n"
    "Answer: <짧은 정답 키워드>\n\n"
    "예시:\n"
    "Answer: 조선\n"
    "Answer: 서울\n"
    "Answer: 유적건조물\n"
    "Answer: 1962"
)

CATEGORIES = ("era", "location", "category", "year")


# ==================== 메인 ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-heritages", type=int, default=500,
                        help="평가셋에 포함할 문화재 개수")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-out", type=str, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--split-out", type=str, default=DEFAULT_SPLIT_PATH)
    args = parser.parse_args()

    with open(HERITAGE_DATA_PATH, "r", encoding="utf-8") as f:
        heritage_data = json.load(f)
    print(f"heritage_data.json loaded: {len(heritage_data)} entries")

    cache_mgr = ImageCacheManager(HERITAGE_BASE_DIR)

    # 1. 4개 카테고리 모두 결정적 추출 가능 + 이미지 캐시 있음 문화재만
    eligible = []
    stats = {"intangible": 0, "no_photo": 0, "no_image_cache": 0,
             "era_fail": 0, "location_fail": 0, "category_fail": 0, "year_fail": 0}

    for item in heritage_data:
        bi = item.get("basic_info") or {}
        if "무형유산" in bi.get("분류", ""):
            stats["intangible"] += 1
            continue
        name = clean_heritage_name(item.get("name", ""))
        image_url = item.get("main_photo")
        if not name or not image_url:
            stats["no_photo"] += 1
            continue
        image_path = cache_mgr.get_cache_path(name)
        if not os.path.exists(image_path):
            stats["no_image_cache"] += 1
            continue

        eras = extract_era_keywords(bi.get("시대", ""))
        region = extract_region(bi.get("소재지", ""))
        top_cat = extract_top_category(bi.get("분류", ""))
        year = extract_year(bi.get("지정(등록)일", ""))

        if not eras: stats["era_fail"] += 1; continue
        if not region: stats["location_fail"] += 1; continue
        if not top_cat: stats["category_fail"] += 1; continue
        if not year: stats["year_fail"] += 1; continue

        eligible.append({
            "name": name,
            "image_url": image_url,
            "image_path": image_path,
            "answers": {
                "era": eras,
                "location": [region],
                "category": [top_cat],
                "year": [year],
            },
        })

    print(f"제외 통계: {stats}")
    print(f"4-필드 모두 추출 가능한 캐시-보유 문화재: {len(eligible)}")

    if len(eligible) < args.num_heritages:
        print(f"[경고] 요청 {args.num_heritages} 보다 가용 {len(eligible)} 적음 -> 가용 전체 사용")
        args.num_heritages = len(eligible)

    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    selected = eligible[: args.num_heritages]
    print(f"평가셋 선정: {len(selected)} 문화재")

    # 2. 평가 split 기록 (학습 데이터 생성 시 차단용)
    os.makedirs(os.path.dirname(args.split_out), exist_ok=True)
    with open(args.split_out, "w", encoding="utf-8") as f:
        json.dump({
            "heritage_names": [h["name"] for h in selected],
            "seed": args.seed,
            "num_heritages": len(selected),
        }, f, ensure_ascii=False, indent=2)
    print(f"평가용 heritage 목록 저장: {args.split_out}")

    # 3. 평가 레코드 생성 (heritage × 4 카테고리)
    records = []
    sample_id = 0
    for h in selected:
        for cat in CATEGORIES:
            q = rng.choice(QUESTION_TEMPLATES[cat])
            ans_keywords = h["answers"][cat]
            records.append({
                "sample_id": str(sample_id),
                "heritage_name": h["name"],
                "image_path": h["image_path"],
                "image_url": h["image_url"],
                "category": cat,
                "question": q,
                "answer_keywords": ans_keywords,
                "answer_display": ans_keywords[0],
                "answer_format_hint": ANSWER_FORMAT_HINT,
            })
            sample_id += 1

    print(f"평가 레코드: {len(records)}건 ({len(selected)} 문화재 × {len(CATEGORIES)} 카테고리)")

    # 4. HF Dataset 저장
    from datasets import Dataset
    ds = Dataset.from_list(records)
    os.makedirs(args.dataset_out, exist_ok=True)
    ds.save_to_disk(args.dataset_out)
    print(f"\nHF Dataset 저장: {args.dataset_out}")
    print(f"  features: {ds.features}")
    print(f"  rows: {len(ds)}")
    print("\n샘플[0..3]:")
    for i in range(min(4, len(ds))):
        r = ds[i]
        print(f"  [{i}] {r['category']:8s} | Q: {r['question'][:50]}...")
        print(f"      A keywords: {r['answer_keywords']}  heritage: {r['heritage_name'][:30]}")


if __name__ == "__main__":
    main()

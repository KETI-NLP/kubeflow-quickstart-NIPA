"""
Korean Heritage Multi-Attribute SFT 데이터셋 빌드 (raw -> JSONL).

instruction: docs/instruction_multi_attr_heritage_sft_dataset.md

핵심 설계:
- heritage 5,321 (가용 전체) × 11 slot × N=10 paraphrase ≈ 585k SFT 레코드
- 11 slot = 주관식 5(name·era·year·location·category) + 객관식 5 + 자연문 설명 1
- reasoning 은 짧게: "이 사진은 {name}으로 보입니다."  (이미 학습된 name_vqa 가 image→name 받쳐줌)
- per heritage 마지막 1 paraphrase 는 test 블록, 나머지 9 는 train 블록
- 학습 풀의 질문 paraphrase 는 3개 eval HF 데이터셋의 byte-identical 문장과 disjoint
- 객관식 distractor: 카테고리별 hard negative + letter A/B/C/D 균등 분포

출력:
  tmp_eval_datasets/korean_heritage_multi_attr_sft.jsonl
"""

from __future__ import annotations
import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

HERITAGE_BASE_DIR = "/workspace/2026_llm_data_generation/korean_heritage_multimodal_hallucination_dpo"
sys.path.append(HERITAGE_BASE_DIR)
sys.path.append("/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5")

from image_cache_manager import ImageCacheManager  # noqa: E402
from generate_heritage_knowledge_eval import (  # noqa: E402
    clean_heritage_name, extract_era_keywords, extract_region,
    extract_top_category, extract_year, ERA_KEYWORDS, REGION_PREFIXES,
)

HERITAGE_DATA_PATH = os.path.join(HERITAGE_BASE_DIR, "heritage_data.json")
DEFAULT_OUT_JSONL = "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/tmp_eval_datasets/korean_heritage_multi_attr_sft.jsonl"

EVAL_DATASETS = [
    "tmp_eval_datasets/korean_heritage_knowledge_hf",
    "tmp_eval_datasets/korean_heritage_knowledge_mc_hf",
    "tmp_eval_datasets/korean_heritage_name_vqa_paraphrase_hf",
]

ATTRIBUTES = ("name", "era", "year", "location", "category")
FORMATS = ("free", "mc")  # 11번째 slot 은 description (별도)
CATEGORY_TOP_POOL = ["유적건조물", "유물", "기록유산", "자연유산", "등록문화유산", "무형유산"]


# ==================== 조사 처리 ====================

def josa(word: str, with_b: str, without_b: str) -> str:
    """받침 유무 (예: 은/는, 이/가)."""
    last = word[-1] if word else ""
    if not ("가" <= last <= "힣"):
        return without_b
    return with_b if (ord(last) - 0xAC00) % 28 != 0 else without_b


def josa_euro_ro(word: str) -> str:
    """으로/로 (받침 없거나 ㄹ받침이면 로)."""
    last = word[-1] if word else ""
    if not ("가" <= last <= "힣"):
        return "로"
    jong = (ord(last) - 0xAC00) % 28
    if jong == 0 or jong == 8:
        return "로"
    return "으로"


def display_name(name: str) -> str:
    """답변에 노출할 이름. 끝의 한자 괄호 (...) 제거."""
    s = re.sub(r"\s*\([^()]*\)\s*$", "", name).strip()
    return s if s else name


# ==================== Reasoning 앵커 (5개) ====================

REASONING_TEMPLATES = [
    "이 사진은 {name}{ero} 보입니다.",
    "사진 속 문화재는 {name}{ero} 보입니다.",
    "이 이미지는 {name}입니다.",
    "{name}의 사진으로 식별됩니다.",
    "사진을 보면 {name}임을 알 수 있습니다.",
]


def render_reasoning(name: str, tpl: str) -> str:
    return tpl.format(name=name, ero=josa_euro_ro(name))


# ==================== 학습용 질문 paraphrase 풀 (eval 의 20문장과 disjoint) ====================

TRAIN_QUESTIONS = {
    "name": [
        "이 사진에 나오는 한국 문화재의 정식 명칭을 알려주세요.",
        "사진에 보이는 문화재의 이름이 무엇인가요?",
        "사진 속 한국 문화재의 명칭을 한 줄로 적어주세요.",
        "이 이미지에 담긴 문화재는 무엇인가요? 이름만 답해주세요.",
        "위 사진의 문화재가 무엇인지 명칭을 말해주세요.",
        "사진 속 문화유산의 정확한 이름을 알려주실 수 있나요?",
        "이 한국 문화재의 명칭은 어떻게 됩니까?",
        "이 사진에서 보이는 문화재의 이름을 말씀해 주세요.",
        "사진에 등장하는 한국 문화유산은 무엇으로 불리나요?",
        "이미지 속 문화재가 무엇인지 그 명칭을 적어주세요.",
        "사진 속 문화재의 공식 명칭이 궁금합니다.",
        "이미지에 나타난 한국 문화재의 정식 이름을 알려주세요.",
        "사진 속 유산은 어떤 이름의 문화재인가요?",
    ],
    "era": [
        "사진 속 문화재는 어느 시기의 작품인가요?",
        "이 문화재가 제작된 시대가 언제인지 알려주세요.",
        "이 한국 문화재가 속하는 시대를 알려주실 수 있나요?",
        "이미지에 나온 문화재는 어떤 시대에 만들어졌나요?",
        "사진의 문화재가 만들어진 시기는 어느 시대에 해당합니까?",
        "이 문화재의 제작 시대를 한 단어로 표현해 주세요.",
        "이 유산이 만들어진 시대를 알려주십시오.",
        "이미지 속 문화재는 한국사 시대 구분상 어디에 속합니까?",
        "이 사진 속 문화재가 만들어진 시기는 어느 왕조/시대인가요?",
        "사진에 보이는 문화유산은 어느 시기 작품인지 답해주세요.",
        "이 문화재의 제작 시대명을 알려주세요.",
        "위 이미지의 문화재가 어느 시대에 제작되었는지 알려주세요.",
        "사진 속 문화재가 만들어진 시점의 한국사 시대는?",
    ],
    "year": [
        "사진 속 문화재가 공식 지정된 해(서기 4자리)를 알려주세요.",
        "이 문화재가 국가 문화재로 등재된 연도가 언제입니까?",
        "이미지의 문화재가 문화재로서 지정된 해는 몇 년인가요?",
        "이 한국 문화재가 처음으로 지정·등록된 해를 알려주세요.",
        "사진 속 유산의 문화재 지정 연도를 말씀해 주세요.",
        "이 문화재가 정식으로 문화재로 등재된 해는 언제입니까?",
        "이미지에 보이는 문화재가 지정된 연도(예: 1985)를 답해주세요.",
        "사진의 문화재가 등록된 연도를 4자리 서기로 답하세요.",
        "이 문화재가 처음 지정된 해(서기, 4자리)는 어떻게 됩니까?",
        "이미지 속 문화재의 정식 지정 연도를 알려주세요.",
        "사진의 문화재가 문화재로 인정된 해를 4자리 연도로 적어주세요.",
        "이 한국 문화재의 지정·등록 연도는 무엇인가요?",
        "위 이미지의 문화재가 지정 받은 해는 언제입니까?",
    ],
    "location": [
        "사진 속 문화재가 현재 위치한 광역시·도는 어디입니까?",
        "이 문화재의 소재지(시/도)를 알려주실 수 있나요?",
        "이미지에 나온 문화재는 어느 광역시·도에 있나요?",
        "사진의 문화재가 자리한 시도를 한 단어로 적어주세요.",
        "이 한국 문화재의 위치(광역시·도)는 어디입니까?",
        "사진 속 유산은 어느 시·도에 보존되어 있나요?",
        "이미지 속 문화재가 위치하는 광역시 또는 도를 알려주세요.",
        "사진의 문화재가 있는 행정구역(시/도)을 답해주세요.",
        "이 문화재가 보관·관리되는 광역시·도가 어디인가요?",
        "이미지 속 문화재의 행정 소재지(시/도)는 어디입니까?",
        "이 한국 문화재가 어느 시·도에 자리잡고 있는지 알려주세요.",
        "위 사진의 문화재 소재 시·도를 적어주세요.",
        "사진에 나온 문화재가 위치한 광역 행정구역을 말씀해 주세요.",
    ],
    "category": [
        "이 문화재가 속하는 가장 큰 분류는 무엇입니까?",
        "사진 속 문화재의 상위 분류(예: 유적건조물, 유물)는 무엇인가요?",
        "이 한국 문화재의 최상위 카테고리를 알려주세요.",
        "이미지의 문화재가 어떤 큰 분류에 속하는지 답해주세요.",
        "사진 속 유산의 가장 큰 분류명을 적어주세요.",
        "이 문화재의 분류 체계상 최상위 카테고리는 무엇입니까?",
        "이미지 속 문화재가 속한 1차 분류를 알려주세요.",
        "사진의 문화재는 어떤 카테고리(상위)에 들어갑니까?",
        "이 한국 문화유산의 최상위 분류명을 한 단어로 알려주세요.",
        "이미지에 보이는 문화재의 상위 분류 카테고리는?",
        "사진 속 문화재가 들어가는 가장 큰 분류는 어떤 것입니까?",
        "이 문화재가 분류상 어느 상위 카테고리에 해당하는지 알려주세요.",
        "위 이미지의 문화재 분류 체계상 최상위 항목은 무엇입니까?",
    ],
}


# ==================== 답변 키워드 핸들링 ====================

def first_answer(answers: list[str]) -> str:
    return answers[0] if answers else ""


def free_form_answer_text(attr: str, name: str, answers: list[str]) -> str:
    """주관식 정답을 자연스러운 형태로 반환 (Answer 라인 키워드는 항상 짧게)."""
    ans = first_answer(answers)
    return ans


# ==================== 객관식 distractor ====================

ERA_FOR_DISTRACTORS = [
    "삼국", "고구려", "백제", "신라", "통일신라", "발해", "고려",
    "조선", "대한제국", "일제강점기", "근대", "현대",
]

# 인접 시대 매핑 (hard negative)
ERA_NEIGHBORS = {
    "삼국": ["고구려", "백제", "신라"],
    "고구려": ["백제", "신라", "삼국"],
    "백제": ["고구려", "신라", "통일신라"],
    "신라": ["통일신라", "백제", "고구려"],
    "통일신라": ["신라", "발해", "고려"],
    "발해": ["통일신라", "고려"],
    "고려": ["조선", "통일신라"],
    "조선": ["고려", "대한제국"],
    "대한제국": ["조선", "일제강점기"],
    "일제강점기": ["대한제국", "근대", "현대"],
    "근대": ["일제강점기", "현대"],
    "현대": ["근대", "일제강점기"],
}

# 인접 광역시·도 (hard negative) — 같은/이웃 지역
LOCATION_NEIGHBORS = {
    "서울": ["경기", "인천"], "부산": ["경남", "울산"], "대구": ["경북", "경남"],
    "인천": ["경기", "서울"], "광주": ["전남", "전북"], "대전": ["충남", "충북", "세종"],
    "울산": ["경남", "부산"], "세종": ["충남", "충북", "대전"],
    "경기": ["서울", "강원", "충북", "인천"], "강원": ["경기", "경북", "충북"],
    "충북": ["충남", "경기", "강원", "전북", "경북", "대전", "세종"],
    "충남": ["충북", "경기", "전북", "대전", "세종"],
    "전북": ["전남", "충남", "충북", "경북", "경남"],
    "전남": ["전북", "경남", "광주"], "경북": ["충북", "강원", "전북", "경남", "대구"],
    "경남": ["경북", "전남", "전북", "부산", "대구", "울산"],
    "제주": [],  # 인접 없음 (섬)
}


def make_distractors(attr: str, correct: str, pool: list[str], rng: random.Random,
                     hard_set: list[str] | None = None, num: int = 3) -> list[str]:
    """오답 N개 추출. hard_set 있으면 1개 이상 포함."""
    available = [c for c in pool if c != correct]
    if not available:
        return []
    out = []
    if hard_set:
        hard_avail = [c for c in hard_set if c != correct and c in available]
        if hard_avail:
            pick = rng.choice(hard_avail)
            out.append(pick)
    while len(out) < num and len(available) > len(out):
        cand = rng.choice([c for c in available if c not in out])
        out.append(cand)
    return out[:num]


def make_year_distractors(correct_year: str, rng: random.Random, num: int = 3) -> list[str]:
    """정답 ±5~20년 범위에서 num개."""
    try:
        y = int(correct_year)
    except (TypeError, ValueError):
        return []
    out = set()
    attempts = 0
    while len(out) < num and attempts < 100:
        delta = rng.randint(5, 20)
        sign = rng.choice([-1, 1])
        cand = y + sign * delta
        if 1962 <= cand <= 2026 and cand != y:
            out.add(str(cand))
        attempts += 1
    while len(out) < num:
        # fallback: pick wider
        cand = rng.randint(1962, 2026)
        if cand != y:
            out.add(str(cand))
    return list(out)[:num]


def make_name_distractors(correct_name: str, by_cat_region: dict,
                          heritage_meta: dict, rng: random.Random, num: int = 3) -> list[str]:
    """같은 (category, region) 풀에서 어휘 겹침 hard negative 포함."""
    cat = heritage_meta["category"]
    region = heritage_meta["region"]
    pool_same = by_cat_region.get((cat, region), [])
    pool_same_cat = []
    for (c, r), names in by_cat_region.items():
        if c == cat and r != region:
            pool_same_cat.extend(names)

    # hard negative — 이름 어휘 겹침
    def overlap_score(other: str) -> int:
        a = set(re.findall(r"[가-힣]+", correct_name))
        b = set(re.findall(r"[가-힣]+", other))
        return len(a & b)

    candidates = [n for n in pool_same if n != correct_name]
    hard = sorted([(overlap_score(n), n) for n in candidates], key=lambda x: -x[0])
    out = []
    if hard and hard[0][0] >= 1:
        out.append(hard[0][1])

    # 나머지 — 같은 카테고리 + 다른 지역 또는 같은 (cat, region) 의 나머지
    while len(out) < num:
        merged = pool_same + pool_same_cat
        merged = [n for n in merged if n != correct_name and n not in out]
        if not merged:
            break
        out.append(rng.choice(merged))
    return out[:num]


# ==================== 자연문 설명 (sample #11) ====================

DESCRIPTION_META_TEMPLATES = [
    "{name}{eun} {location}에 위치한 {era} 시대의 {category}로, {year}년에 문화재로 지정되었습니다.",
    "{name}{eun} {era} 시대 작품으로 {location}에 있으며, {year}년에 {category}로 지정되었습니다.",
    "{location} 소재의 {name}{eun} {era}대의 {category}이며, 지정 연도는 {year}년입니다.",
    "{year}년에 {category}로 지정된 {name}{eun} {era} 시대의 작품이며 {location}에 위치합니다.",
    "이 문화재({name}){eun} {location}에 자리 잡은 {era} 시대 {category}입니다. {year}년에 지정되었습니다.",
    "{name}{eun} {category}로 분류되며, {era} 시대에 만들어져 {location}에 보존돼 있고, {year}년에 정식 지정되었습니다.",
    "{era}대의 작품인 {name}{eun} {location}에 있으며, {year}년 {category}로 지정되어 보호받고 있습니다.",
    "{name}{eun} {year}년에 지정된 {category}입니다. {era} 시대의 작품으로 {location}에 위치하고 있습니다.",
]
DESCRIPTION_WRAPPER_TEMPLATES = [
    "{snippet}",
    "이 문화재는 {snippet}",
    "특징적으로, {snippet}",
    "자세히 살펴보면, {snippet}",
    "역사적으로, {snippet}",
]
DESCRIPTION_PROMPT_TEMPLATES = [
    "이 사진에 보이는 문화재를 한 문단(3~5문장)으로 설명해 주세요. 명칭, 시대, 위치, 분류, 지정 연도, 특징을 포함해 주세요.",
    "사진 속 한국 문화재에 대해 명칭·시대·위치·분류·지정 연도·특징을 모두 포함하는 한 문단 설명을 작성해 주세요.",
    "이미지에 나온 문화재의 명칭, 만들어진 시대, 소재지, 분류, 지정 연도, 그리고 특징을 한 문단으로 알려주세요.",
    "사진의 문화재를 명칭·시대·소재지·상위 분류·지정 연도·주요 특징을 모두 포함해 3~5문장으로 설명해 주세요.",
    "이 한국 문화재의 명칭·시대·위치·분류·지정 연도·특징을 묶어서 한 문단으로 서술해 주세요.",
]


_DESC_PREFIX_NOISE = ("국가유산 설명", "유물 설명")


def clean_description_ko(text: str) -> str:
    """description_ko 앞쪽 boilerplate 제거 + 정리."""
    if not text:
        return ""
    # 반복 prefix 제거
    s = text
    for noise in _DESC_PREFIX_NOISE:
        s = re.sub(rf"(?:{noise})+", "", s)
    s = s.strip()
    return s


def description_snippet(desc_ko: str) -> str:
    """description_ko 의 첫 1~2 문장 추출."""
    s = clean_description_ko(desc_ko)
    if not s:
        return ""
    sents = re.split(r"(?<=[.!?다요])\s+", s)
    sents = [x for x in sents if x.strip()]
    return " ".join(sents[:2]) if sents else ""


def render_description(name: str, location: str, era: str, category: str,
                       year: str, desc_ko: str, meta_tpl: str, wrap_tpl: str) -> str:
    eun = josa(name, "은", "는")
    meta = meta_tpl.format(name=name, eun=eun, location=location, era=era,
                           category=category, year=year)
    snippet = description_snippet(desc_ko)
    wrapper = wrap_tpl.format(snippet=snippet).strip() if snippet else ""
    if wrapper:
        return f"{meta} {wrapper}"
    return meta


# ==================== Prompt 조립 ====================

ANSWER_FORMAT_HINT_FREE = (
    "반드시 마지막 줄을 다음 형식으로만 끝내세요:\n"
    "Answer: <짧은 정답 키워드>"
)
ANSWER_FORMAT_HINT_MC = (
    "다음을 읽고 정답을 고르시오. 응답의 마지막 줄은 반드시 'Answer: $LETTER' 형식이어야 합니다. "
    "$LETTER 는 A, B, C, D 중 하나입니다."
)


def build_free_user_prompt(question: str) -> str:
    return f"<image>\n{question}\n\n{ANSWER_FORMAT_HINT_FREE}"


def build_mc_user_prompt(question: str, choices: list[str]) -> str:
    body = "\n".join(f"{letter}. {text}" for letter, text in zip(["A","B","C","D"], choices))
    return f"<image>\n{question}\n{body}\n\n{ANSWER_FORMAT_HINT_MC}"


def build_description_prompt(prompt_tpl: str) -> str:
    return f"<image>\n{prompt_tpl}"


def build_free_assistant(reasoning: str, answer_kw: str) -> str:
    return f"{reasoning}\n\nAnswer: {answer_kw}"


def build_mc_assistant(reasoning: str, letter: str) -> str:
    return f"{reasoning}\n\nAnswer: {letter}"


def build_description_assistant(description: str) -> str:
    # Answer: 라인 없음 (collapse 방지용 free-form)
    return description


# ==================== Eval byte-overlap set ====================

def load_eval_overlap_keys(repo_root: str) -> set[tuple]:
    """3개 eval HF 셋에서 (heritage_name_canonical, question, answer_canonical) 튜플 수집."""
    from datasets import load_from_disk
    keys = set()
    for rel in EVAL_DATASETS:
        path = os.path.join(repo_root, rel)
        if not os.path.exists(path):
            print(f"[경고] eval 셋 없음 (스킵): {path}")
            continue
        ds = load_from_disk(path)
        cols = set(ds.features.keys())
        for r in ds:
            hn = r.get("heritage_name", "")
            hn_norm = display_name(hn)
            q = r.get("question", "")
            # 정답 추출 — 데이터셋마다 형식 다름
            if "answer_keywords" in cols:  # knowledge_hf
                ans = ",".join(sorted(r["answer_keywords"]))
            elif "correct_choice_text" in cols:  # knowledge_mc_hf
                ans = r["correct_choice_text"]
            elif "answer" in cols:  # name_vqa_paraphrase
                ans = r["answer"]
            else:
                ans = ""
            keys.add((hn_norm, q, ans))
    return keys


# ==================== Heritage 풀 ====================

def load_heritages(repo_root: str):
    with open(HERITAGE_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    cache_mgr = ImageCacheManager(HERITAGE_BASE_DIR)
    eligible = []
    stats = Counter()
    for item in data:
        bi = item.get("basic_info") or {}
        if "무형유산" in bi.get("분류", ""):
            stats["intangible"] += 1
            continue
        name = clean_heritage_name(item.get("name", ""))
        url = item.get("main_photo")
        if not name or not url:
            stats["no_photo"] += 1
            continue
        img_path = cache_mgr.get_cache_path(name)
        if not os.path.exists(img_path):
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
            "name_full": name,
            "name": display_name(name),
            "image_path": img_path,
            "image_url": url,
            "era": eras[0],  # 1개만 사용
            "region": region,
            "category": top_cat,
            "year": year,
            "description_ko": item.get("description_ko", ""),
        })
    return eligible, stats


# ==================== 샘플 빌드 (heritage 1개당 11 slot × N) ====================

def build_heritage_samples(heritage: dict, n: int, rng: random.Random,
                           by_cat_region: dict, all_names: list[str],
                           overlap_set: set[tuple],
                           letter_balance: defaultdict) -> tuple[list, list]:
    """heritage 1개에 대해 11 slot × N paraphrase = 11N 샘플 생성.
    반환: (train_records, test_records) - 마지막 1개를 test 로 분리.
    """
    samples_by_slot: dict[tuple, list[dict]] = defaultdict(list)
    hname = heritage["name"]
    hname_full = heritage["name_full"]
    img = heritage["image_path"]
    img_url = heritage["image_url"]

    # heritage 별 정답 (answer keyword)
    answers = {
        "name": hname,
        "era": heritage["era"],
        "year": heritage["year"],
        "location": heritage["region"],
        "category": heritage["category"],
    }
    # MC 풀 (정답 외 distractor 추출 풀)
    distractor_pool = {
        "name": None,  # 별도 함수
        "era": ERA_FOR_DISTRACTORS,
        "year": None,  # ±5~20
        "location": REGION_PREFIXES,
        "category": CATEGORY_TOP_POOL,
    }
    hard_neighbors = {
        "era": ERA_NEIGHBORS.get(heritage["era"], []),
        "location": LOCATION_NEIGHBORS.get(heritage["region"], []),
        "name": None, "year": None, "category": None,
    }

    # 5 attr × 2 format = 10 slot + 1 description = 11
    for attr in ATTRIBUTES:
        q_pool = TRAIN_QUESTIONS[attr]
        chosen_qs = rng.sample(q_pool, min(n, len(q_pool)))
        # 부족하면 중복 허용 (random.choices)
        while len(chosen_qs) < n:
            chosen_qs.append(rng.choice(q_pool))

        # === free-form (slot key: (attr, "free")) ===
        for i, q in enumerate(chosen_qs):
            ans_kw = answers[attr]
            tup = (hname, q, ans_kw)
            if tup in overlap_set:
                # 매우 드물지만 충돌 시 다른 q 로 교체
                for cand in q_pool:
                    if (hname, cand, ans_kw) not in overlap_set and cand not in chosen_qs:
                        q = cand
                        break
            reasoning = render_reasoning(hname, rng.choice(REASONING_TEMPLATES))
            user_prompt = build_free_user_prompt(q)
            assistant = build_free_assistant(reasoning, ans_kw)
            samples_by_slot[(attr, "free")].append({
                "image": img,
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant},
                ],
                "metadata": {
                    "heritage_id": hname,
                    "heritage_name_full": hname_full,
                    "attr_type": attr,
                    "qa_format": "free",
                    "answer_keyword": ans_kw,
                    "answer_letter": None,
                    "image_url": img_url,
                },
            })

        # === MC (slot key: (attr, "mc")) ===
        chosen_qs_mc = rng.sample(q_pool, min(n, len(q_pool)))
        while len(chosen_qs_mc) < n:
            chosen_qs_mc.append(rng.choice(q_pool))

        for i, q in enumerate(chosen_qs_mc):
            correct = answers[attr]
            # distractor 생성
            if attr == "name":
                distractors = make_name_distractors(correct, by_cat_region, {
                    "category": heritage["category"], "region": heritage["region"]
                }, rng)
            elif attr == "year":
                distractors = make_year_distractors(correct, rng)
            else:
                pool = distractor_pool[attr]
                hard = hard_neighbors.get(attr)
                distractors = make_distractors(attr, correct, pool, rng, hard_set=hard)

            # 4개 미만이면 fallback
            tries = 0
            while len(distractors) < 3 and tries < 200:
                tries += 1
                if attr == "year":
                    cand = str(rng.randint(1962, 2026))
                elif attr == "name":
                    cand = rng.choice(all_names)
                else:
                    pool = distractor_pool[attr] or []
                    pool_avail = [x for x in pool if x != correct]
                    if not pool_avail:
                        break
                    cand = rng.choice(pool_avail)
                if cand not in distractors and cand != correct:
                    distractors.append(cand)

            # letter 균등 분포: 가장 적게 쓰인 letter 위치에 정답 배치
            letter_counts = letter_balance[attr]
            target_letter = min("ABCD", key=lambda L: letter_counts[L])
            target_idx = "ABCD".index(target_letter)
            choices = [None] * 4
            choices[target_idx] = correct
            di = 0
            for j in range(4):
                if choices[j] is None:
                    choices[j] = distractors[di]
                    di += 1
            letter_counts[target_letter] += 1

            tup = (hname, q, correct)
            if tup in overlap_set:
                # mc 도 동일하게 처리
                for cand in q_pool:
                    if (hname, cand, correct) not in overlap_set and cand not in chosen_qs_mc:
                        q = cand
                        break

            reasoning = render_reasoning(hname, rng.choice(REASONING_TEMPLATES))
            user_prompt = build_mc_user_prompt(q, choices)
            assistant = build_mc_assistant(reasoning, target_letter)
            samples_by_slot[(attr, "mc")].append({
                "image": img,
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant},
                ],
                "metadata": {
                    "heritage_id": hname,
                    "heritage_name_full": hname_full,
                    "attr_type": attr,
                    "qa_format": "mc",
                    "answer_keyword": correct,
                    "answer_letter": target_letter,
                    "image_url": img_url,
                },
            })

    # description slot (11번째)
    desc_prompts = rng.sample(DESCRIPTION_PROMPT_TEMPLATES,
                              min(n, len(DESCRIPTION_PROMPT_TEMPLATES)))
    while len(desc_prompts) < n:
        desc_prompts.append(rng.choice(DESCRIPTION_PROMPT_TEMPLATES))

    meta_choices = rng.sample(DESCRIPTION_META_TEMPLATES,
                              min(n, len(DESCRIPTION_META_TEMPLATES)))
    while len(meta_choices) < n:
        meta_choices.append(rng.choice(DESCRIPTION_META_TEMPLATES))
    wrap_choices = [rng.choice(DESCRIPTION_WRAPPER_TEMPLATES) for _ in range(n)]

    for i in range(n):
        desc_text = render_description(
            hname, heritage["region"], heritage["era"],
            heritage["category"], heritage["year"], heritage["description_ko"],
            meta_choices[i], wrap_choices[i],
        )
        samples_by_slot[("description", "desc")].append({
            "image": img,
            "messages": [
                {"role": "user", "content": build_description_prompt(desc_prompts[i])},
                {"role": "assistant", "content": build_description_assistant(desc_text)},
            ],
            "metadata": {
                "heritage_id": hname,
                "heritage_name_full": hname_full,
                "attr_type": "description",
                "qa_format": "description",
                "answer_keyword": None,
                "answer_letter": None,
                "image_url": img_url,
            },
        })

    # train/test 분리: 각 slot 의 마지막 1개를 test, 나머지 N-1 을 train
    train, test = [], []
    for slot, records in samples_by_slot.items():
        train.extend(records[:-1])
        test.extend(records[-1:])
    return train, test


# ==================== 메인 ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-paraphrase", type=int, default=10,
                        help="slot 당 paraphrase 수 (마지막 1개는 test 블록)")
    parser.add_argument("--max-heritages", type=int, default=None,
                        help="테스트용 heritage 수 제한")
    parser.add_argument("--out-jsonl", type=str, default=DEFAULT_OUT_JSONL)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repo_root = "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5"

    # 1. eval byte-overlap set
    print("=== eval byte-overlap set 로드 ===")
    overlap_set = load_eval_overlap_keys(repo_root)
    print(f"  exclusion 튜플: {len(overlap_set)}건")

    # 2. heritage 로드
    print("=== heritage 풀 로드 ===")
    eligible, hstats = load_heritages(repo_root)
    print(f"  통계: {dict(hstats)}")
    print(f"  4-필드 추출 + 캐시 이미지 보유: {len(eligible)}")

    if args.max_heritages:
        random.Random(args.seed).shuffle(eligible)
        eligible = eligible[: args.max_heritages]
        print(f"  --max-heritages 적용: {len(eligible)}")

    # 3. by_cat_region 인덱스 (name distractor 용)
    by_cat_region = defaultdict(list)
    for h in eligible:
        by_cat_region[(h["category"], h["region"])].append(h["name"])
    all_names = [h["name"] for h in eligible]

    # 4. 샘플 생성
    print(f"=== 샘플 생성 (heritage {len(eligible)}, N={args.n_paraphrase}) ===")
    rng = random.Random(args.seed)
    letter_balance = defaultdict(lambda: Counter())
    train_records, test_records = [], []

    for idx, h in enumerate(eligible):
        tr, te = build_heritage_samples(h, args.n_paraphrase, rng,
                                        by_cat_region, all_names,
                                        overlap_set, letter_balance)
        train_records.extend(tr)
        test_records.extend(te)
        if (idx + 1) % 500 == 0:
            print(f"  [{idx+1}/{len(eligible)}] train={len(train_records)} test={len(test_records)}")

    # 5. shuffle 각 블록 (instruction §7)
    rng.shuffle(train_records)
    rng.shuffle(test_records)

    # 6. JSONL 출력 ([train 블록][test 블록])
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for r in test_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(train_records) + len(test_records)
    print(f"\n=== 완료 ===")
    print(f"  train: {len(train_records)} / test: {len(test_records)} / total: {total}")
    print(f"  test 비율: {len(test_records)/total:.4f}")
    print(f"  출력: {args.out_jsonl}")

    # 7. letter 분포 sanity check
    print(f"\n=== MC letter 분포 (per attr) ===")
    for attr, cnt in letter_balance.items():
        total_mc = sum(cnt.values())
        if total_mc > 0:
            pct = {L: cnt[L] / total_mc * 100 for L in "ABCD"}
            print(f"  {attr}: " + " ".join(f"{L}={pct[L]:.1f}%" for L in "ABCD"))


if __name__ == "__main__":
    main()

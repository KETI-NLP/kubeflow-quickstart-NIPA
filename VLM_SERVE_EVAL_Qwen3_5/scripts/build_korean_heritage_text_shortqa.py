#!/usr/bin/env python3
"""
Build benchmark-ready Korean heritage text short-answer QA datasets from the large
DPO-style text source JSON.

Outputs:
- candidate pool JSON with all high-confidence extracted short-answer facts
- benchmark JSON with one best short-answer QA per heritage item
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import ijson


DEFAULT_SOURCE_PATH = Path(
    "/workspace/2026_llm_data_generation/"
    "korean_heritage_multimodal_hallucination_dpo/dpo_dataset_generated_text.json"
)
DEFAULT_CANDIDATE_OUTPUT = Path(
    "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/tmp_eval_datasets/"
    "korean_heritage_text_shortqa_candidates.json"
)
DEFAULT_BENCHMARK_OUTPUT = Path(
    "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/tmp_eval_datasets/"
    "korean_heritage_text_shortqa_benchmark.json"
)


SAFE_LEGAL_TYPES = [
    "국보",
    "보물",
    "사적",
    "명승",
    "천연기념물",
    "국가민속문화재",
    "등록문화재",
    "경기도 유형문화재",
    "경기도 문화재자료",
    "경상북도 유형문화재",
    "경상북도 문화재자료",
    "전라남도 유형문화재",
    "전라남도 문화재자료",
    "전북특별자치도 유형문화재",
    "전북특별자치도 문화재자료",
    "충청남도 유형문화재",
    "충청남도 문화재자료",
    "충청북도 유형문화재",
    "충청북도 문화재자료",
    "부산광역시 유형문화재",
    "대구광역시 유형문화재",
    "대전광역시 유형문화재",
    "광주광역시 유형문화재",
    "울산광역시 유형문화재",
    "서울특별시 유형문화재",
    "제주특별자치도 유형문화재",
]


QUESTION_TEMPLATES = {
    "designation_date": "{heritage_name}의 문화재 지정일은 언제인가?",
    "creation_year": "{heritage_name}의 제작 또는 조성 시기는 언제인가?",
    "quantity": "{heritage_name}의 지정 수량은 얼마인가?",
    "holding_institution": "{heritage_name}의 소장 기관은 어디인가?",
    "designation_name": "{heritage_name}의 문화재 지정 명칭은 무엇인가?",
    "designation_type": "{heritage_name}의 지정 종별은 무엇인가?",
    "material": "{heritage_name}의 바탕 재질은 무엇인가?",
    "mounting_format": "{heritage_name}의 장황 형식은 무엇인가?",
}


MATERIAL_MANUAL_OVERRIDES: dict[str, str] = {
    "성안의 초상": "명주",
    "최문병 의병장 안장": "고슴도치 가죽",
    "복검관행차시하인식료기": "한지(닥종이)",
    "신숙주 초상": "모시",
    "예산 삽교읍 석조보살입상": "화강암",
    "봉원사 지장시왕도": "목재 패널",
    "창덕궁 인정전": "나무 마루",
    "순천 낙안읍성 서문성벽집": "나무와 대나무",
    "경주 천마총 장니 천마도": "자작나무 껍질",
    "고흥 김붕만 선무원종공신녹권과 신위단비": "목재",
    "김회련 고신왕지": "한지",
    "단양 청련암 목조보살좌상": "은행나무",
    "곤재우득록목판": "감나무",
    "전주향교소장완영책판": "자작나무과 목재",
    "오명항 초상 및 양무공신교서": "닥종이",
    "친목회회보": "양지",
    "하동 금남사 이색 초상": "명주",
    "대구 비산동 청동기 일괄 - 검 및 칼집 부속": "청동",
    "강세황 행초 표암유채": "죽청지",
    "창녕 계성 고분군": "나무",
    "봉화 대각사 석조석가여래좌상": "불석재",
}


MATERIAL_MANUAL_EXCLUDES = {
    "정충신장군 유품",
    "조순 장군비",
}


MATERIAL_QUESTION_OVERRIDES: dict[str, str] = {
    "강세황 행초 표암유채": "강세황 행초 표암유채의 바탕 재질은 무엇인가?",
    "경주 천마총 장니 천마도": "경주 천마총 장니 천마도의 주된 바탕 재료는 무엇인가?",
    "고흥 김붕만 선무원종공신녹권과 신위단비": "고흥 김붕만 선무원종공신녹권과 신위단비의 선무원종공신녹권 인쇄에 사용된 활자의 재질은 무엇인가?",
    "곤재우득록목판": "곤재우득록목판의 판목 재질은 무엇인가?",
    "공주영은사목조관음보살좌상": "공주영은사목조관음보살좌상의 재질은 무엇인가?",
    "관음보살도": "관음보살도의 바탕 재질은 무엇인가?",
    "권태사신도비": "권태사신도비의 재질은 무엇인가?",
    "김구 서명문 태극기": "김구 서명문 태극기의 바탕 재료는 무엇인가?",
    "김회련 고신왕지": "김회련 고신왕지에 사용된 종이의 재질은 무엇인가?",
    "나주 영천사 목조아미타여래좌상": "나주 영천사 목조아미타여래좌상의 재질은 무엇인가?",
    "단양 청련암 목조보살좌상": "단양 청련암 목조보살좌상에 사용된 나무 재질은 무엇인가?",
    "대구 비산동 청동기 일괄 - 검 및 칼집 부속": "대구 비산동 청동기 일괄 - 검 및 칼집 부속의 칼집 보강용 고리 재질은 무엇인가?",
    "덕원사 석가설법도": "덕원사 석가설법도의 바탕 재질은 무엇인가?",
    "도촌조응인선생유서": "도촌조응인선생유서의 재질은 무엇인가?",
    "밀양 용궁사 대송당 묘원 진영": "밀양 용궁사 대송당 묘원 진영의 바탕 재질은 무엇인가?",
    "밀양 표충사 목조석가삼불좌상": "밀양 표충사 목조석가삼불좌상의 재질은 무엇인가?",
    "범어사 사자암 칠성도": "범어사 사자암 칠성도의 바탕 재질은 무엇인가?",
    "범어사목조석가여래위패": "범어사목조석가여래위패의 재질은 무엇인가?",
    "보림사목조관음보살좌상": "보림사목조관음보살좌상의 재질은 무엇인가?",
    "보문사 신중도": "보문사 신중도의 바탕 재질은 무엇인가?",
    "복검관행차시하인식료기": "복검관행차시하인식료기에 사용된 종이의 재질은 무엇인가?",
    "봉원사 지장시왕도": "봉원사 지장시왕도의 바탕 재질은 무엇인가?",
    "봉화 대각사 석조석가여래좌상": "봉화 대각사 석조석가여래좌상의 재질은 무엇인가?",
    "분류두공부시(언해) 권21": "분류두공부시(언해) 권21에 사용된 종이의 재질은 무엇인가?",
    "비안면자락동석조여래좌상": "비안면자락동석조여래좌상의 조각 재질은 무엇인가?",
    "상지은니 대지도론 권28": "상지은니 대지도론 권28의 바탕 재질은 무엇인가?",
    "서산 보원사지 법인국사탑비": "서산 보원사지 법인국사탑비의 재질은 무엇인가?",
    "선광사 석가여래성도기": "선광사 석가여래성도기에 사용된 종이의 재질은 무엇인가?",
    "성안의 초상": "성안의 초상의 바탕 재질은 무엇인가?",
    "소상팔경도": "소상팔경도의 바탕 재질은 무엇인가?",
    "순천 낙안읍성 서문성벽집": "순천 낙안읍성 서문성벽집의 서까래 재질은 무엇인가?",
    "신라백지묵서 대방광불화엄경 주본 권1~10, 44~50": "신라백지묵서 대방광불화엄경 주본 권1~10, 44~50의 바탕 재질은 무엇인가?",
    "신숙주 초상": "신숙주 초상의 바탕 재질은 무엇인가?",
    "쌍계사아미타회상도": "쌍계사아미타회상도의 바탕 재질은 무엇인가?",
    "안동대사동모전석탑": "안동대사동모전석탑의 재질은 무엇인가?",
    "안시명 교첩": "안시명 교첩에 사용된 종이의 재질은 무엇인가?",
    "영덕 구 영해버스터미널": "영덕 구 영해버스터미널의 주된 건축 재질은 무엇인가?",
    "영동 반야사 삼층석탑": "영동 반야사 삼층석탑의 재질은 무엇인가?",
    "영월 보덕사 지장시왕도": "영월 보덕사 지장시왕도의 바탕 재질은 무엇인가?",
    "예산 삽교읍 석조보살입상": "예산 삽교읍 석조보살입상의 재질은 무엇인가?",
    "오명항 초상 및 양무공신교서": "오명항 초상 및 양무공신교서의 교서 바탕 재질은 무엇인가?",
    "우수영전진도첩": "우수영전진도첩의 재질은 무엇인가?",
    "원주봉산동석불좌상": "원주봉산동석불좌상의 대좌 재질은 무엇인가?",
    "이형 좌명원종공신녹권 및 함": "이형 좌명원종공신녹권 및 함의 녹권 재질은 무엇인가?",
    "자수 '아미타불'명 번": "자수 '아미타불'명 번의 바탕 재질은 무엇인가?",
    "자수 초충도 병풍": "자수 초충도 병풍의 바탕 재질은 무엇인가?",
    "전 이순신 초상": "전 이순신 초상의 바탕 재질은 무엇인가?",
    "전주향교소장완영책판": "전주향교소장완영책판의 판목 재질은 무엇인가?",
    "정조어필 - 신제학정민시출안호남": "정조어필 - 신제학정민시출안호남의 바탕 재질은 무엇인가?",
    "창녕 계성 고분군": "창녕 계성 고분군 무덤 상부 구조의 재질은 무엇인가?",
    "창녕 사리 석조광배": "창녕 사리 석조광배의 재질은 무엇인가?",
    "창덕궁 인정전": "창덕궁 인정전 바닥의 재질은 무엇인가?",
    "청양 정혜사 혜림암 목조보살좌상": "청양 정혜사 혜림암 목조보살좌상의 재질은 무엇인가?",
    "청허당 진영": "청허당 진영의 바탕 재질은 무엇인가?",
    "초조본 유가사지론 권32": "초조본 유가사지론 권32의 바탕 재료는 무엇인가?",
    "최문병 의병장 안장": "최문병 의병장 안장의 안교 부위 재질은 무엇인가?",
    "친목회회보": "친목회회보에 사용된 종이의 재질은 무엇인가?",
    "칠성여래도": "칠성여래도의 바탕 재질은 무엇인가?",
    "태안 태국사 목조관음보살좌상": "태안 태국사 목조관음보살좌상의 재질은 무엇인가?",
    "태인신잠선생영상": "태인신잠선생영상의 바탕 재료는 무엇인가?",
    "포항 원각사 소장 아미타불회도": "포항 원각사 소장 아미타불회도의 바탕 재질은 무엇인가?",
    "하동 금남사 이색 초상": "하동 금남사 이색 초상의 바탕 재질은 무엇인가?",
    "함양 금대사 삼층석탑": "함양 금대사 삼층석탑의 재질은 무엇인가?",
    "함양 벽송사 벽송당 지엄 진영": "함양 벽송사 벽송당 지엄 진영의 바탕 재질은 무엇인가?",
    "합천 해인사 대적광전 중건 상량문": "합천 해인사 대적광전 중건 상량문의 바탕 재질은 무엇인가?",
    "해학반도도 병풍": "해학반도도 병풍의 화폭 재질은 무엇인가?",
    "허목 전서 애군우국": "허목 전서 애군우국의 바탕 재질은 무엇인가?",
    "홍천 수타사 괘불": "홍천 수타사 괘불의 바탕 재질은 무엇인가?",
    "화순운주사거북바위교차문칠층석탑": "화순운주사거북바위교차문칠층석탑의 재질은 무엇인가?",
    "후원한담도": "후원한담도의 바탕 재질은 무엇인가?",
}


TYPE_PRIORITY = {
    "designation_date": 100,
    "creation_year": 90,
    "quantity": 80,
    "holding_institution": 75,
    "designation_name": 70,
    "designation_type": 65,
    "material": 60,
    "mounting_format": 55,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--candidate-output", type=Path, default=DEFAULT_CANDIDATE_OUTPUT)
    parser.add_argument("--benchmark-output", type=Path, default=DEFAULT_BENCHMARK_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_quotes(text: str) -> str:
    return text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")


def normalize_heritage_name(text: str) -> str:
    normalized = collapse_ws(text)
    while True:
        stripped = strip_last_parenthetical(normalized)
        if stripped == normalized:
            return normalized
        normalized = stripped


def strip_last_parenthetical(text: str) -> str:
    text = text.rstrip()
    if not text.endswith(")"):
        return text

    depth = 0
    for idx in range(len(text) - 1, -1, -1):
        ch = text[idx]
        if ch == ")":
            depth += 1
        elif ch == "(":
            depth -= 1
            if depth == 0:
                return text[:idx].rstrip()
    return text


def normalize_date(text: str) -> str:
    text = collapse_ws(text)
    text = re.sub(r"\s*년\s*", "년 ", text)
    text = re.sub(r"\s*월\s*", "월 ", text)
    text = re.sub(r"\s*일\s*", "일", text)
    return collapse_ws(text)


def normalize_quantity(text: str) -> str:
    text = collapse_ws(text)
    text = re.sub(r"^총\s*", "", text)
    return text


def normalize_answer(text: str) -> str:
    text = normalize_quotes(text)
    text = collapse_ws(text)
    return text.strip(" \t\n\r\"'.")


@dataclass
class Candidate:
    heritage_name: str
    heritage_name_full: str
    answer_type: str
    answer: str
    benchmark_question: str
    source_query: str
    source_chosen: str
    source_rejected: str
    qa_error_type: str
    hallucination_type: str
    paraphrase_count: int = 1
    source_queries: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "heritage_name": self.heritage_name,
            "heritage_name_full": self.heritage_name_full,
            "answer_type": self.answer_type,
            "answer": self.answer,
            "benchmark_question": self.benchmark_question,
            "source_query": self.source_query,
            "source_chosen": self.source_chosen,
            "source_rejected": self.source_rejected,
            "qa_error_type": self.qa_error_type,
            "hallucination_type": self.hallucination_type,
            "paraphrase_count": self.paraphrase_count,
            "source_queries": self.source_queries or [self.source_query],
        }


def extract_designation_date(query: str, chosen: str) -> str | None:
    if not re.search(r"(지정|등록).*(날짜|일)|지정일|등록일", query):
        return None
    match = re.search(r"(\d{4}년\s*\d{1,2}월\s*\d{1,2}일)", chosen)
    if not match:
        return None
    return normalize_date(match.group(1))


def extract_creation_year(query: str, chosen: str) -> str | None:
    if not re.search(
        r"(언제|몇 년|연도|시기).*(제작|조성|간행|건립|창건|창작|완공)|"
        r"(제작|조성|간행|건립|창건|창작|완공).*(언제|연도|시기)|"
        r"그려진 정확한 연도|작성 연도",
        query,
    ):
        return None
    match = re.search(r"(\d{4}년)", chosen)
    if not match:
        return None
    return normalize_answer(match.group(1))


def extract_quantity(query: str, chosen: str) -> str | None:
    if not re.search(r"수량|개수|몇 권|몇 점|몇 책|몇 구|몇 동|몇 기|몇 그루|총 점수", query):
        return None
    match = re.search(r"(총\s*\d+\s*(?:점|권|책|구|동|기|그루)|\d+\s*(?:점|권|책|구|동|기|그루))", chosen)
    if not match:
        return None
    return normalize_quantity(match.group(1))


def extract_holding_institution(query: str, chosen: str) -> str | None:
    if not re.search(r"소장된 구체적인 기관|소장 기관|소장.*기관 명칭", query):
        return None
    match = re.search(r"([가-힣A-Za-z0-9·ㆍ\-\(\)]+?)에서\s+소장", chosen)
    if not match:
        return None
    answer = normalize_answer(match.group(1))
    if len(answer) < 2:
        return None
    return answer


def extract_owner(query: str, chosen: str) -> str | None:
    if not re.search(r"소유자", query):
        return None
    match = re.search(r"소유자(?:는)?\s*([^,.]+?)(?:이며|이고|입니다|\.|,)", chosen)
    if not match:
        return None
    answer = normalize_answer(match.group(1))
    answer = re.sub(r"(이며|이고|입니다)$", "", answer).strip()
    if len(answer) < 2 or answer in {"가", "나", "다"}:
        return None
    return answer


def extract_manager(query: str, chosen: str) -> str | None:
    if not re.search(r"관리 단체|관리자는 누구|누가 관리", query):
        return None
    match = re.search(
        r"(?:관리자(?:\(관리단체\))?|관리 단체)(?:는)?\s*([^,.]+?)(?:에서\s+담당|이고|입니다|\.|,)",
        chosen,
    )
    if not match:
        return None
    answer = normalize_answer(match.group(1))
    answer = re.sub(r"^(역시)\s*", "", answer).strip()
    answer = re.sub(r"(에서|이고|입니다|담당하고)$", "", answer).strip()
    if len(answer) < 2 or answer in {"역시", "가", "나", "다"}:
        return None
    return answer


def extract_designation_name(query: str, chosen: str) -> str | None:
    if not re.search(r"지정 명칭", query):
        return None
    match = re.search(r"지정 명칭(?:은)?\s*[\"']?([^\"'이며,.]+)", chosen)
    if not match:
        return None
    return normalize_answer(match.group(1))


def extract_designation_type(query: str, chosen: str) -> str | None:
    if not re.search(r"분류|지정.*종별|유형문화재|보물|국보|사적|명승", query):
        return None
    for item in SAFE_LEGAL_TYPES:
        if item in chosen:
            return item
    return None


def extract_material(query: str, chosen: str) -> str | None:
    if not re.search(r"바탕 재질|바탕 재료|재질은 무엇|어떤 재료 위에", query):
        return None
    for material in ["비단", "종이", "한지", "목재", "나무", "석재", "가죽"]:
        if material in chosen:
            return material
    return None


def extract_mounting_format(query: str, chosen: str) -> str | None:
    if not re.search(r"장황 형식|장황.*무엇|형식은 무엇", query):
        return None
    match = re.search(r"([가-힣A-Za-z0-9·ㆍ\-\(\)]+형\s*장황)", chosen)
    if not match:
        return None
    return normalize_answer(match.group(1))


EXTRACTORS = OrderedDict(
    [
        ("designation_date", extract_designation_date),
        ("creation_year", extract_creation_year),
        ("quantity", extract_quantity),
        ("holding_institution", extract_holding_institution),
        ("designation_type", extract_designation_type),
        ("material", extract_material),
        ("mounting_format", extract_mounting_format),
    ]
)


def iter_items(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb") as f:
        yield from ijson.items(f, "item")


def build_benchmark_question(heritage_name: str, answer_type: str) -> str:
    template = QUESTION_TEMPLATES[answer_type]
    return template.format(heritage_name=heritage_name)


def choose_best_candidate(candidates: list[Candidate]) -> Candidate:
    def score(item: Candidate) -> tuple[int, int, int]:
        return (
            TYPE_PRIORITY.get(item.answer_type, 0),
            item.paraphrase_count,
            -len(item.answer),
        )

    return sorted(candidates, key=score, reverse=True)[0]


def main() -> None:
    args = parse_args()

    candidate_map: dict[tuple[str, str, str], Candidate] = {}
    candidates_by_heritage: dict[str, list[Candidate]] = defaultdict(list)
    stats = Counter()

    for idx, item in enumerate(iter_items(args.source), start=1):
        if args.limit is not None and idx > args.limit:
            break

        stats["rows_seen"] += 1
        if item.get("qa_error_type") != "type0_text_normal":
            continue
        stats["rows_type0_text_normal"] += 1

        heritage_name_full = collapse_ws(str(item.get("heritage_name", "") or ""))
        heritage_name = normalize_heritage_name(heritage_name_full)
        query = collapse_ws(str(item.get("user_query", "") or ""))
        chosen = collapse_ws(str(item.get("chosen", "") or ""))
        rejected = collapse_ws(str(item.get("rejected", "") or ""))

        if not heritage_name or not query or not chosen:
            continue

        for answer_type, extractor in EXTRACTORS.items():
            answer = extractor(query, chosen)
            if not answer:
                continue
            if answer_type == "material":
                if heritage_name in MATERIAL_MANUAL_EXCLUDES:
                    continue
                if heritage_name in MATERIAL_MANUAL_OVERRIDES:
                    answer = MATERIAL_MANUAL_OVERRIDES[heritage_name]
            answer = normalize_answer(answer)
            if not answer:
                continue

            stats[f"candidate_{answer_type}"] += 1
            key = (heritage_name, answer_type, answer)
            existing = candidate_map.get(key)
            if existing is None:
                benchmark_question = (
                    MATERIAL_QUESTION_OVERRIDES.get(heritage_name, build_benchmark_question(heritage_name, answer_type))
                    if answer_type == "material"
                    else build_benchmark_question(heritage_name, answer_type)
                )
                candidate = Candidate(
                    heritage_name=heritage_name,
                    heritage_name_full=heritage_name_full,
                    answer_type=answer_type,
                    answer=answer,
                    benchmark_question=benchmark_question,
                    source_query=query,
                    source_chosen=chosen,
                    source_rejected=rejected,
                    qa_error_type=item.get("qa_error_type", ""),
                    hallucination_type=item.get("hallucination_type", ""),
                    paraphrase_count=1,
                    source_queries=[query],
                )
                candidate_map[key] = candidate
                candidates_by_heritage[heritage_name].append(candidate)
            else:
                existing.paraphrase_count += 1
                if existing.source_queries is not None and query not in existing.source_queries:
                    if len(existing.source_queries) < 10:
                        existing.source_queries.append(query)

    candidate_list = [candidate.to_dict() for candidate in candidate_map.values()]
    best_list = []
    for heritage_name, candidates in candidates_by_heritage.items():
        best = choose_best_candidate(candidates)
        row = best.to_dict()
        row["candidate_count_for_heritage"] = len(candidates)
        best_list.append(row)

    candidate_type_counts = Counter(item["answer_type"] for item in candidate_list)
    best_type_counts = Counter(item["answer_type"] for item in best_list)

    candidate_payload = {
        "metadata": {
            "task_name": "korean_heritage_text_shortqa_candidates",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_path": str(args.source),
            "selection_policy": "high-confidence short-answer fact extraction from type0_text_normal",
            "row_stats": dict(stats),
            "candidate_count": len(candidate_list),
            "candidate_type_counts": dict(candidate_type_counts),
            "unique_heritage_count": len(candidates_by_heritage),
        },
        "samples": candidate_list,
    }

    benchmark_payload = {
        "metadata": {
            "task_name": "korean_heritage_text_shortqa_benchmark",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_path": str(args.source),
            "selection_policy": "one best high-confidence short-answer fact per heritage",
            "candidate_count": len(candidate_list),
            "benchmark_count": len(best_list),
            "unique_heritage_count": len(candidates_by_heritage),
            "benchmark_type_counts": dict(best_type_counts),
            "type_priority": TYPE_PRIORITY,
        },
        "samples": best_list,
    }

    args.candidate_output.parent.mkdir(parents=True, exist_ok=True)
    with args.candidate_output.open("w", encoding="utf-8") as f:
        json.dump(candidate_payload, f, ensure_ascii=False, indent=2)
    with args.benchmark_output.open("w", encoding="utf-8") as f:
        json.dump(benchmark_payload, f, ensure_ascii=False, indent=2)

    print(f"Candidate samples: {len(candidate_list)}")
    print(f"Benchmark samples: {len(best_list)}")
    print(f"Candidate output: {args.candidate_output}")
    print(f"Benchmark output: {args.benchmark_output}")
    print(f"Candidate type counts: {dict(candidate_type_counts)}")
    print(f"Benchmark type counts: {dict(best_type_counts)}")


if __name__ == "__main__":
    main()

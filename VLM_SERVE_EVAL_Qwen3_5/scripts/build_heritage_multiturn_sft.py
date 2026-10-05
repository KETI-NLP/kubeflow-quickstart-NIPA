"""
Korean Heritage 멀티턴 SFT 데이터셋 빌더 (raw -> JSONL).

두 종류의 학습 데이터셋을 같은 멀티턴 골격으로 생성한다 (--mode 로 선택):

  --mode knowledge  (text 지식)
      turn1 (user, 이미지 없음): "{문화재명}은 어느 시대의 문화재인가요?" 처럼
        문화재 '이름'을 주고 속성(시대/연도/소재지/분류)을 묻는 text-only QA.
      turn1 (assistant): 정답 키워드를 포함한 자연문.
      -> 이미지 없이 name->attribute 사실을 '텍스트로' 주입한다
         (image->attribute 를 가르치는 multi_attr 와 상보적).

  --mode reverse  (reverse QA)
      turn1 (user, 이미지 없음): 문화재 '설명'에서 이름을 가린 뒤
        "다음 설명에 해당하는 문화재의 이름은?" 으로 묻는 text-only QA.
      turn1 (assistant): "{문화재명}입니다."

두 모드 공통:
  turn2 (user): "그렇다면 {문화재명}의 실제 사진은? A.<image> B.<image> ..."
    -> 정답 main_photo + (num_choices-1) 개 distractor heritage 사진.
  turn2 (assistant): 정답 letter 를 담은 짧은 자연문.

왜 멀티턴인가:
  text-only 샘플을 그냥 섞으면 FSDP 에서 vision encoder 가 일부 rank 에서만
  실행돼 NCCL sync 가 깨진다(데드락). 모든 샘플이 turn2 에 이미지를 가지므로
  vision encoder 가 항상 돌고, turn1 은 causal mask 덕에 text-only 신호를 유지한다.
  학습 collator 는 모든 assistant 턴에 loss 를 건다(멀티턴 마스킹).

출력 JSONL 레코드 스키마 (build_multiturn_sft_hf.py 가 소비):
  {
    "images": ["<A경로>", "<B경로>", ...],   # turn2 <image> 마커 순서와 일치
    "messages": [{role, content}, ...],       # content 의 <image> 마커 개수 = len(images)
    "split": "train" | "test",
    "metadata": {...}
  }

출력:
  tmp_eval_datasets/korean_heritage_text_knowledge_mt.jsonl   (knowledge)
  tmp_eval_datasets/korean_heritage_reverse_mt.jsonl          (reverse)
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
    extract_top_category, extract_year,
)

HERITAGE_DATA_PATH = os.path.join(HERITAGE_BASE_DIR, "heritage_data.json")
OUT_DIR = "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/tmp_eval_datasets"
DEFAULT_OUT = {
    "knowledge": os.path.join(OUT_DIR, "korean_heritage_text_knowledge_mt.jsonl"),
    # leak-free + MC/free-form mix 적용 시 v2 사용 권장 (기본은 그대로 v1 호환을 위해 유지)
    "reverse": os.path.join(OUT_DIR, "korean_heritage_reverse_mt.jsonl"),
    "reverse_v2": os.path.join(OUT_DIR, "korean_heritage_reverse_mt_v2.jsonl"),
}

LETTERS = ["A", "B", "C", "D", "E", "F"]


# ==================== 조사 / 이름 처리 ====================

def josa(word: str, with_b: str, without_b: str) -> str:
    """받침 유무에 따른 조사 (예: 은/는, 이/가)."""
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
    return "로" if (jong == 0 or jong == 8) else "으로"


def display_name(name: str) -> str:
    """노출용 이름. 끝의 한자 괄호 (...) 제거."""
    s = re.sub(r"\s*\([^()]*\)\s*$", "", name).strip()
    return s if s else name


# ==================== knowledge: 이름을 주고 속성을 묻는 Q/A 풀 ====================
#   {N} = 문화재명, {NB}/{NG} = name 받침 기준 은/는·이/가 자동 치환은 render 단계에서.

KN_QUESTIONS = {
    "era": [
        "{N}{eun} 어느 시대의 문화재인가요?",
        "{N}{eun} 한국사에서 어느 시대에 속하나요?",
        "{N}{i} 만들어진 시대는 언제입니까?",
        "{N}의 제작 시대를 알려주세요.",
        "{N}{eun} 어느 왕조 또는 시대의 유산인가요?",
        "{N}{eun} 어느 시기에 만들어진 문화재입니까?",
        "{N}{i} 속하는 시대명을 한 단어로 알려주세요.",
        "{N}{eun} 언제 시대의 작품으로 분류되나요?",
        "{N}의 시대 구분은 어떻게 됩니까?",
        "{N}{i} 제작된 한국사 시대를 말씀해 주세요.",
        "{N}{eun} 어느 시대에 조성되었나요?",
        "{N}{i} 만들어진 시기는 어느 시대입니까?",
        "{N}의 조성 시대가 궁금합니다.",
        "{N}{eun} 역사적으로 어느 시대의 것인가요?",
        "{N}{i} 어느 시대에 세워졌는지 알려주세요.",
        "{N}{eun} 어느 시대를 대표하는 문화재인가요?",
        "{N}의 제작 시기를 시대명으로 답해주세요.",
        "{N}{eun} 어떤 시대에 만들어졌습니까?",
        "{N}{i} 속한 시대가 언제인지 알려주세요.",
        "{N}{eun} 시대상 언제의 유물인가요?",
        "{N}의 시대적 배경은 어느 시대입니까?",
        "{N}{i} 건립된 시대를 알려주세요.",
        "{N}{eun} 어느 시대 유산으로 분류됩니까?",
        "{N}의 제작 연대를 시대명으로 말해주세요.",
        "{N}{eun} 한국사 시대 구분상 어디에 해당하나요?",
        "{N}{i} 어느 시대의 산물인지 알려주세요.",
        "{N}{eun} 무슨 시대의 문화재입니까?",
        "{N}의 시대는 무엇인가요?",
        "{N}{eun} 어느 시대에 처음 만들어졌나요?",
        "{N}{i} 어느 시대 작품인지 궁금합니다.",
    ],
    "year": [
        "{N}{eun} 몇 년에 문화재로 지정되었나요?",
        "{N}의 문화재 지정 연도(서기 4자리)는 언제입니까?",
        "{N}{i} 국가 문화재로 등재된 해는 언제인가요?",
        "{N}{eun} 어느 해에 문화재로 등록되었습니까?",
        "{N}{i} 처음 지정·등록된 연도를 알려주세요.",
        "{N}의 지정 연도를 4자리 서기로 답해주세요.",
        "{N}{eun} 몇 년도에 정식 문화재가 되었나요?",
        "{N}{i} 문화재로 인정받은 해는 언제입니까?",
        "{N}의 문화재 지정 시점(연도)을 알려주세요.",
        "{N}{eun} 어느 연도에 문화재로 지정·등록되었습니까?",
        "{N}{i} 문화재 목록에 오른 해는 언제인가요?",
        "{N}의 지정 연도가 궁금합니다.",
        "{N}{eun} 언제 문화재로 지정됐는지 연도로 알려주세요.",
        "{N}{i} 국가유산으로 지정된 연도는 몇 년입니까?",
        "{N}의 지정·등록 연도를 말씀해 주세요.",
        "{N}{eun} 몇 년에 국가유산으로 등록되었나요?",
        "{N}{i} 처음 문화재 지정을 받은 해를 알려주세요.",
        "{N}의 문화재 등재 연도는 서기 몇 년인가요?",
        "{N}{eun} 어느 시점(연도)에 지정되었습니까?",
        "{N}{i} 지정된 해를 4자리 숫자로 답해주세요.",
        "{N}의 공식 지정 연도를 알려주세요.",
        "{N}{eun} 문화재 지정이 이뤄진 해가 언제인가요?",
        "{N}{i} 등록문화재가 된 연도는 언제입니까?",
        "{N}의 지정 연도(예: 1985)를 알려주세요.",
        "{N}{eun} 몇 년에 보호 대상으로 지정되었나요?",
        "{N}{i} 문화재로 등재된 시기를 연도로 알려주세요.",
        "{N}의 최초 지정 연도는 몇 년입니까?",
        "{N}{eun} 어느 해에 국가 지정을 받았나요?",
        "{N}의 지정 연도를 서기로 답해주세요.",
        "{N}{eun} 문화재 지정 연도가 몇 년인가요?",
    ],
    "location": [
        "{N}{eun} 어느 광역시·도에 있나요?",
        "{N}의 소재지(시/도)는 어디입니까?",
        "{N}{eun} 현재 어느 지역(시·도)에 위치해 있나요?",
        "{N}{i} 자리한 광역 행정구역은 어디인가요?",
        "{N}{eun} 어느 시·도에 보존되어 있습니까?",
        "{N}의 위치를 광역시·도 단위로 알려주세요.",
        "{N}{eun} 어느 시·도에 있는 문화재인가요?",
        "{N}{i} 속한 행정 소재지(시/도)는 어디입니까?",
        "{N}의 소재 광역시·도를 한 단어로 답해주세요.",
        "{N}{eun} 어디(시·도)에 위치하는지 알려주세요.",
        "{N}{i} 있는 광역시 또는 도는 어디인가요?",
        "{N}의 행정구역상 소재지(시/도)가 궁금합니다.",
        "{N}{eun} 어느 지방(시·도)에 자리하고 있나요?",
        "{N}{i} 보관·관리되는 시·도는 어디입니까?",
        "{N}의 소재 지역을 광역 단위로 알려주세요.",
        "{N}{eun} 어느 시·도 관할에 있습니까?",
        "{N}{i} 위치한 광역자치단체는 어디인가요?",
        "{N}의 소재지를 시/도 이름으로 답해주세요.",
        "{N}{eun} 지리적으로 어느 시·도에 있나요?",
        "{N}{i} 어느 광역시·도 소속인지 알려주세요.",
        "{N}의 위치(시·도)는 어디입니까?",
        "{N}{eun} 어느 시·도에 세워져 있나요?",
        "{N}{i} 자리잡은 시·도를 알려주세요.",
        "{N}의 광역 소재지가 어디인지 궁금합니다.",
        "{N}{eun} 어느 행정구역(시/도)에 속하나요?",
        "{N}{i} 있는 곳은 어느 시·도입니까?",
        "{N}의 소재 시·도를 말씀해 주세요.",
        "{N}{eun} 어느 지역에 위치한 문화재인가요? 시/도로 답해주세요.",
        "{N}의 소재지는 어느 광역시·도인가요?",
        "{N}{eun} 시·도 기준으로 어디에 있습니까?",
    ],
    "category": [
        "{N}{eun} 어떤 분류(최상위 카테고리)에 속하나요?",
        "{N}의 가장 큰 분류는 무엇입니까?",
        "{N}{eun} 어느 상위 카테고리로 분류되나요?",
        "{N}{i} 속하는 1차 분류명을 알려주세요.",
        "{N}{eun} 분류 체계상 어떤 최상위 항목에 들어갑니까?",
        "{N}의 상위 분류(예: 유적건조물, 유물)는 무엇인가요?",
        "{N}{eun} 어떤 큰 분류에 해당하는 문화재인가요?",
        "{N}{i} 들어가는 최상위 카테고리를 한 단어로 답해주세요.",
        "{N}의 문화재 분류명(상위)을 알려주세요.",
        "{N}{eun} 어느 분류에 속하는 유산입니까?",
        "{N}{i} 어느 대분류에 포함되나요?",
        "{N}의 최상위 분류가 궁금합니다.",
        "{N}{eun} 문화재 유형상 어떤 분류인가요?",
        "{N}{i} 속한 큰 갈래(분류)는 무엇입니까?",
        "{N}의 상위 카테고리를 한 단어로 말해주세요.",
        "{N}{eun} 어떤 종류의 문화재로 분류되나요?",
        "{N}{i} 분류 체계에서 어느 상위 항목에 속하는지 알려주세요.",
        "{N}의 1차 분류는 무엇인가요?",
        "{N}{eun} 가장 상위의 분류명이 무엇입니까?",
        "{N}{i} 어느 분류군에 속하는지 알려주세요.",
        "{N}의 대분류(최상위)를 답해주세요.",
        "{N}{eun} 어떤 상위 분류에 들어가는 문화재인가요?",
        "{N}{i} 속하는 최상위 분류는 무엇입니까?",
        "{N}의 분류상 가장 큰 범주는 무엇인가요?",
        "{N}{eun} 문화재 분류상 어디에 해당하나요?",
        "{N}{i} 어느 카테고리(상위)로 묶이는지 알려주세요.",
        "{N}의 상위 분류명을 알려주세요.",
        "{N}{eun} 어떤 분류 항목에 속하는 유산입니까?",
        "{N}의 최상위 카테고리는 무엇인가요?",
        "{N}{eun} 분류상 어느 큰 갈래에 들어가나요?",
    ],
}

KN_ANSWERS = {
    "era": [
        "{N}{eun} {V} 시대의 문화재입니다.",
        "{N}{eun} {V} 시대에 만들어졌습니다.",
        "{V} 시대의 유산입니다.",
        "{N}{eun} {V} 시대에 제작된 작품입니다.",
        "{V} 시대에 속하는 문화재입니다.",
        "{N}{eun} {V} 시대에 조성되었습니다.",
        "시대로는 {V} 시대에 해당합니다.",
        "{N}의 제작 시대는 {V} 시대입니다.",
        "{V} 시대에 만들어진 것으로 분류됩니다.",
        "{N}{eun} {V} 시대를 대표하는 문화재입니다.",
    ],
    "year": [
        "{N}{eun} {V}년에 문화재로 지정되었습니다.",
        "{N}의 지정 연도는 {V}년입니다.",
        "{V}년에 문화재로 등록되었습니다.",
        "{N}{eun} {V}년에 국가 문화재로 등재되었습니다.",
        "지정 연도는 {V}년입니다.",
        "{N}{eun} {V}년에 정식으로 지정되었습니다.",
        "{V}년에 국가유산으로 지정되었습니다.",
        "{N}의 문화재 지정은 {V}년에 이뤄졌습니다.",
        "{V}년에 처음 지정·등록되었습니다.",
        "{N}{eun} {V}년에 문화재로 인정받았습니다.",
    ],
    "location": [
        "{N}{eun} {V}에 있습니다.",
        "{N}의 소재지는 {V}입니다.",
        "{V}에 위치한 문화재입니다.",
        "{N}{eun} {V}에 보존되어 있습니다.",
        "소재지는 {V}입니다.",
        "{N}{eun} {V}에 자리하고 있습니다.",
        "{V}에 자리한 유산입니다.",
        "{N}의 위치는 {V}입니다.",
        "{V} 지역에 있습니다.",
        "{N}{eun} {V}에 소재합니다.",
    ],
    "category": [
        "{N}{eun} {V}에 속하는 문화재입니다.",
        "{N}의 상위 분류는 {V}입니다.",
        "{Vero} 분류됩니다.",
        "{N}{eun} {V}에 해당합니다.",
        "최상위 분류는 {V}입니다.",
        "{N}{eun} {Vero} 분류되는 유산입니다.",
        "분류상 {V}에 속합니다.",
        "{N}의 대분류는 {V}입니다.",
        "{V} 범주에 들어갑니다.",
        "{N}{eun} {V} 분류의 문화재입니다.",
    ],
}

KN_ATTR_LABEL = {"era": "시대", "year": "지정 연도", "location": "소재지", "category": "분류"}


def render_kn(tpl: str, name: str, value: str | None = None) -> str:
    out = tpl.replace("{eun}", josa(name, "은", "는"))
    out = out.replace("{i}", josa(name, "이", "가"))
    out = out.replace("{N}", name)
    if value is not None:
        out = out.replace("{Vero}", value + josa_euro_ro(value))  # 예: 유적건조물 + 으로/로
        out = out.replace("{V}", value)
    return out


# ==================== reverse: 설명에서 이름 가리기 ====================

_MASK_TOKEN = "(이 문화재)"

# heritage.go.kr 스크랩 시 섞여 들어온 UI 보일러플레이트 토큰 (명백한 잡음만).
#   안내판형 예: "설명비교안내판 설명국가유산 설명닫기안내판 설명개별안내판개별안내판{이름}\n{한자}\n국보\n\n{본문}"
#   일반형 예:   "국가유산 설명국가유산 설명{본문}"
_UI_NOISE_RE = re.compile(
    r"(?:설명비교안내판|설명국가유산|설명닫기안내판|설명개별안내판|개별안내판|국가유산\s*설명|유물\s*설명)+"
)

# reverse 는 객관식(MC). 설명을 주고 보기 중 알맞은 문화재명을 letter 로 고르게 한다.
# (주관식으로 이름 전체를 생성하는 건 매우 어려워 학습 신호가 약함 -> MC 로 학습 용이화.)
REVERSE_MC_Q_TEMPLATES = [
    "다음은 어떤 한국 문화재에 대한 설명입니다. 보기 중 이 문화재는 무엇입니까?\n\n{D}",
    "아래 설명에 해당하는 한국 문화재를 보기에서 고르세요.\n\n{D}",
    "다음 설명을 읽고, 보기 중 알맞은 문화재를 선택하세요.\n\n{D}",
    "이 설명이 가리키는 문화재는 보기 중 무엇인가요?\n\n{D}",
    "다음 설명에 부합하는 문화재를 보기에서 골라주세요.\n\n{D}",
    "아래 내용은 한 문화재를 설명한 것입니다. 보기 중 그 문화재를 고르세요.\n\n{D}",
    "다음 설명과 일치하는 문화재를 보기에서 선택하세요.\n\n{D}",
    "아래 설명을 보고 어떤 문화재인지 보기에서 고르시오.\n\n{D}",
    "다음은 한 문화재의 설명입니다. 알맞은 것을 보기에서 고르세요.\n\n{D}",
    "이 설명에 맞는 문화재를 아래 보기 중에서 선택해 주세요.\n\n{D}",
    "다음 설명이 묘사하는 문화재는 보기 중 어느 것입니까?\n\n{D}",
    "아래 설명에 해당하는 문화재를 보기 A~D 중에서 고르세요.\n\n{D}",
    "다음 글이 설명하는 한국 문화재를 보기에서 찾아 고르세요.\n\n{D}",
    "설명을 읽고 보기 중 알맞은 문화재를 하나 고르시오.\n\n{D}",
    "다음 설명에 해당하는 문화재로 가장 알맞은 것을 고르세요.\n\n{D}",
    "아래 문화재 설명을 보고 정답을 보기에서 선택하세요.\n\n{D}",
    "다음은 어떤 문화재에 대한 설명입니다. 보기 중 옳은 것을 고르세요.\n\n{D}",
    "이 설명에 부합하는 문화재를 보기에서 하나 선택하세요.\n\n{D}",
    "다음 설명을 근거로 알맞은 문화재를 보기에서 고르시오.\n\n{D}",
    "아래 설명이 가리키는 문화재를 보기 중에서 골라주세요.\n\n{D}",
]

# MC 답변은 'Answer: <letter>' 로 끝내 채점을 쉽게 한다(eval 힌트와 동일).
# 앞에 짧은 자연문(reasoning)을 붙여 출력이 letter 하나로 collapse 되는 것을 막고,
# desc/name → 정답 연결을 강화한다. (이름은 보기/질문에서 복사 가능하므로 생성 부담 적음)
MC_HINT = "응답의 마지막 줄은 반드시 'Answer: <보기 letter>' 형식이어야 합니다. (예: Answer: A)"

# turn1(이름 MC) reasoning — 이름 포함 자연문
REVERSE_T1_REASON = [
    "이 설명은 {N}에 대한 것입니다.",
    "정답은 {N}입니다.",
    "설명에 해당하는 문화재는 {N}입니다.",
    "보기 중 {N}{i} 맞습니다.",
    "{N}에 대한 설명으로 보입니다.",
]


def make_name_distractors(correct_name: str, all_names: list[str],
                          rng: random.Random, num: int) -> list[str]:
    """correct_name 외 서로 다른 문화재명 num개를 무작위 추출(오답 보기용)."""
    out = []
    guard = 0
    while len(out) < num and guard < 2000:
        guard += 1
        cand = rng.choice(all_names)
        if cand != correct_name and cand not in out:
            out.append(cand)
    return out


def clean_description_ko(text: str) -> str:
    """description_ko 에서 스크랩 UI 보일러플레이트와 헤더(이름/한자/지정종류)를 제거하고
    실제 설명 본문만 남긴다.

    안내판형 헤더는 UI 토큰 뒤에 '{이름}\\n{한자}\\n{지정종류}' 같은 짧은 줄들이 이어진다.
    UI 토큰을 먼저 지운 뒤, 앞쪽의 짧은(<20자) 헤더 잔여 줄들을 건너뛰고 첫 실제 문장부터 취한다.
    일반형('국가유산 설명...{본문}')은 본문이 한 줄로 길어 그대로 유지된다."""
    if not text:
        return ""
    s = _UI_NOISE_RE.sub(" ", text)
    lines = s.split("\n")
    out_start = 0
    for i, ln in enumerate(lines):
        if len(ln.strip()) >= 20:   # 실제 설명 문장으로 간주
            out_start = i
            break
    s = "\n".join(lines[out_start:])
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def split_sentences(text: str) -> list[str]:
    # 마침표/물음표/느낌표 뒤(공백 유무 무관) 또는 '다/요' + 공백 뒤에서 분리.
    # 스크랩 데이터엔 '보인다.대한천일은행은' 처럼 마침표 뒤 공백이 없는 경우가 많아,
    # 공백을 요구하면 문단 전체가 한 '문장'으로 뭉쳐 buf 가 폭증한다.
    sents = re.split(r"(?<=[.!?])\s*|(?<=[다요])\s+", text)
    return [x.strip() for x in sents if x and x.strip()]


def _ws_insensitive_pattern(s: str) -> str:
    """문자 사이 임의 공백을 허용하는 정규식 (예: '류홍부자' 가 '류홍 부자' 도 매칭)."""
    chars = [re.escape(c) for c in s if not c.isspace()]
    return r"\s*".join(chars)


def _strip_ws(s: str) -> str:
    return re.sub(r"\s+", "", s)


_HANGUL_RUN_RE = re.compile(r"[가-힣]+")
_LEAK_NOISE = {"이 문화재", "문화재", "이문화재"}


def _has_partial_leak(name: str, desc: str, min_len: int = 2) -> bool:
    """이름의 한글 런에서 min_len 이상 연속 substring 중 하나라도 desc 에 등장하면 leaky.
    reverse_mt_mc eval 의 leak-free 필터와 동일 정의. min_len=0 이면 필터 비활성."""
    if min_len <= 0:
        return False
    for run in _HANGUL_RUN_RE.findall(name):
        L = len(run)
        for s in range(L):
            for e in range(s + min_len, L + 1):
                sub = run[s:e]
                if sub in _LEAK_NOISE:
                    continue
                if sub in desc:
                    return True
    return False


def mask_name_in_text(text: str, name_full: str, name_disp: str) -> str:
    """설명 안의 문화재 이름(노출형/전체/한자)을 마스킹 토큰으로 치환.
    공백 변형('류홍부자묘역' vs '류홍 부자 묘역')까지 잡도록 공백 무시 매칭한다."""
    # 마스킹 대상: 노출형(한글), 전체명, 한자/중점 런(괄호 안 한자 포함)
    variants = {name_disp, name_full}
    variants.update(re.findall(r"[一-鿿·]{2,}", name_full))
    out = text
    for v in sorted((x for x in variants if x), key=lambda x: -len(x)):
        pat = _ws_insensitive_pattern(v)
        if pat:
            out = re.sub(pat, _MASK_TOKEN, out)
    return out


def _generate_leak_free_snippet(heritage: dict, rng: random.Random,
                                leak_min_len: int = 2, max_tries: int = 30) -> str | None:
    """leak-free + 마스킹된 설명 snippet 반환. 여러 (start, length) 조합을 시도해서
    이름의 어떤 min_len+ 한글 substring도 등장하지 않는 첫 후보를 반환한다.
    못 찾으면 None (해당 paraphrase 슬롯 스킵)."""
    desc = clean_description_ko(heritage["description_ko"])
    if not desc:
        return None
    sents = split_sentences(desc)
    if not sents:
        return None
    L = len(sents)
    # 후보: 다양한 길이(1~4문장)+시작 위치를 셔플 시도.
    candidates = []
    for k in range(1, min(5, L + 1)):
        for start in range(L - k + 1):
            candidates.append((start, k))
    rng.shuffle(candidates)
    name_disp = heritage["name"]
    name_full = heritage["name_full"]
    for start, k in candidates[:max_tries]:
        sel = sents[start:start + k]
        # 길이 상한
        snippet = ""
        for s in sel:
            if snippet and len(snippet) + len(s) > 500:
                break
            snippet = (snippet + " " + s).strip()
        if len(snippet) < 25:
            continue
        masked = mask_name_in_text(snippet, name_full, name_disp)
        if len(masked) < 25:
            continue
        # 전체 이름이 (공백 변형 포함) 남았으면 skip
        if _strip_ws(name_disp) in _strip_ws(masked):
            continue
        # 부분 토큰 leak 체크
        if _has_partial_leak(name_disp, masked, leak_min_len):
            continue
        return masked
    return None


def build_reverse_turn1_mc(heritage: dict, all_names: list[str], num_choices: int,
                           rng: random.Random, name_letter_balance: Counter,
                           leak_min_len: int = 2):
    """reverse turn1 (객관식): 이름 가린 설명(+leak-free 필터) + 이름 보기 -> 정답 letter.
    반환 (user, assistant, correct_letter) 또는 적절한 snippet 없거나 distractor 부족 시 None."""
    masked = _generate_leak_free_snippet(heritage, rng, leak_min_len=leak_min_len)
    if masked is None:
        return None

    distractors = make_name_distractors(heritage["name"], all_names, rng, num_choices - 1)
    if len(distractors) < num_choices - 1:
        return None

    # letter 균등 분포: 가장 적게 쓰인 위치에 정답 배치
    target_letter = min(LETTERS[:num_choices], key=lambda L: name_letter_balance[L])
    target_idx = LETTERS.index(target_letter)
    name_letter_balance[target_letter] += 1

    choices = [None] * num_choices
    choices[target_idx] = heritage["name"]
    di = 0
    for j in range(num_choices):
        if choices[j] is None:
            choices[j] = distractors[di]
            di += 1

    q = rng.choice(REVERSE_MC_Q_TEMPLATES).replace("{D}", masked)
    body = "\n".join(f"{LETTERS[j]}. {choices[j]}" for j in range(num_choices))
    user = f"{q}\n{body}\n\n{MC_HINT}"
    reason = render_kn(rng.choice(REVERSE_T1_REASON), heritage["name"])
    assistant = f"{reason}\n\nAnswer: {target_letter}"
    return user, assistant, target_letter


# ==================== reverse 주관식(free-form) ====================
# 보기 없이 설명 → 이름을 직접 생성. 마지막 줄 'Answer: <문화재 이름>' 형식.
REVERSE_FREE_Q_TEMPLATES = [
    "다음은 어떤 한국 문화재에 대한 설명입니다. 이 문화재의 이름은 무엇입니까?\n\n{D}",
    "아래 설명에 해당하는 한국 문화재의 명칭을 알려주세요.\n\n{D}",
    "다음 설명을 읽고, 어떤 문화재인지 그 이름을 답해주세요.\n\n{D}",
    "이 설명이 가리키는 문화재는 무엇인가요? 이름을 답해주세요.\n\n{D}",
    "다음 설명에 부합하는 문화재의 정식 명칭을 알려주세요.\n\n{D}",
    "아래 내용은 한 문화재를 설명한 것입니다. 그 문화재의 이름을 말씀해 주세요.\n\n{D}",
    "다음 설명과 일치하는 문화재의 이름은 무엇입니까?\n\n{D}",
    "아래 설명을 보고 어떤 문화재인지 이름을 답해주세요.\n\n{D}",
    "다음은 한 문화재의 설명입니다. 이 문화재의 정식 명칭을 알려주세요.\n\n{D}",
    "이 설명에 맞는 문화재의 이름을 적어주세요.\n\n{D}",
    "다음 설명이 묘사하는 문화재의 이름은 무엇인가요?\n\n{D}",
    "아래 설명에 해당하는 문화재의 이름을 답하세요.\n\n{D}",
    "다음 글이 설명하는 한국 문화재의 명칭을 알려주세요.\n\n{D}",
    "설명을 읽고 알맞은 문화재의 이름을 답하세요.\n\n{D}",
    "다음 설명에 해당하는 문화재로 가장 알맞은 것의 이름을 말해주세요.\n\n{D}",
    "아래 문화재 설명을 보고 그 정식 이름을 답하세요.\n\n{D}",
    "다음은 어떤 문화재에 대한 설명입니다. 정확한 명칭을 답해주세요.\n\n{D}",
    "이 설명에 부합하는 문화재의 이름은 무엇입니까?\n\n{D}",
    "다음 설명을 근거로 알맞은 문화재의 이름을 답하세요.\n\n{D}",
    "아래 설명이 가리키는 문화재의 이름을 적어주세요.\n\n{D}",
]

FREE_HINT = "응답의 마지막 줄은 반드시 'Answer: <문화재 이름>' 형식이어야 합니다."


def build_reverse_turn1_free(heritage: dict, rng: random.Random, leak_min_len: int = 2):
    """reverse turn1 (주관식 free-form): 이름 가린 설명 -> 'Answer: <이름>' 직접 생성.
    반환 (user, assistant) 또는 적절한 snippet 없으면 None."""
    masked = _generate_leak_free_snippet(heritage, rng, leak_min_len=leak_min_len)
    if masked is None:
        return None
    q = rng.choice(REVERSE_FREE_Q_TEMPLATES).replace("{D}", masked)
    user = f"{q}\n\n{FREE_HINT}"
    reason = render_kn(rng.choice(REVERSE_T1_REASON), heritage["name"])
    assistant = f"{reason}\n\nAnswer: {heritage['name']}"
    return user, assistant


# ==================== turn2: 이미지 선택 ====================

T2_QUESTIONS = [
    "그렇다면 {N}의 실제 사진은 다음 중 무엇입니까?",
    "위에서 다룬 {N}의 사진을 아래에서 고르세요.",
    "다음 사진 중 {N}{eul} 촬영한 것은 어느 것인가요?",
    "그럼 {N}의 모습이 담긴 사진을 골라주세요.",
    "아래 보기에서 {N}에 해당하는 사진을 선택하세요.",
    "다음 중 {N}{eul} 보여주는 사진은 무엇입니까?",
    "그러면 {N}의 사진은 보기 중 어느 것인가요?",
    "아래 네 사진 중 {N}{eul} 찍은 것을 고르세요.",
    "다음 사진들 가운데 {N}{eun} 어느 것입니까?",
    "{N}의 실제 모습이 담긴 사진을 보기에서 선택하세요.",
    "위 문화재({N})의 사진을 아래에서 골라주세요.",
    "다음 중 {N}의 사진으로 알맞은 것은 무엇인가요?",
]

# turn2(이미지 선택) reasoning — 이름 포함, letter 없이. 뒤에 'Answer: <letter>' 부착.
T2_REASON = [
    "{N}의 사진을 찾았습니다.",
    "사진 속에서 {N}{eul} 확인했습니다.",
    "{N}에 해당하는 사진입니다.",
    "보기에서 {N}{eul} 골랐습니다.",
    "{N}의 모습이 담긴 사진입니다.",
]


def render_t2(tpl: str, name: str, letter: str | None = None) -> str:
    out = tpl.replace("{eul}", josa(name, "을", "를"))
    out = out.replace("{eun}", josa(name, "은", "는"))
    out = out.replace("{N}", name)
    if letter is not None:
        out = out.replace("{ga}", josa(letter, "이", "가"))
        out = out.replace("{L}", letter)
    return out


def build_image_selection_turn(heritage: dict, all_heritages: list[dict],
                               num_choices: int, rng: random.Random,
                               letter_balance: Counter):
    """turn2 (이미지 선택) 생성.
    반환: (user_content, assistant_content, ordered_image_paths, ordered_image_urls, correct_letter)
    ordered_image_urls 는 ordered_image_paths 와 1:1 대응(배포 시 이미지 제거하고 URL 재다운로드용).
    """
    name = heritage["name"]
    # distractor heritage 무작위 선정 (정답과 다른 것)
    distractors = []
    seen = {heritage["name_full"]}
    tries = 0
    while len(distractors) < num_choices - 1 and tries < 500:
        tries += 1
        cand = rng.choice(all_heritages)
        if cand["name_full"] in seen:
            continue
        seen.add(cand["name_full"])
        distractors.append(cand)

    # letter 균등 분포: 가장 적게 쓰인 위치에 정답 배치
    target_letter = min(LETTERS[:num_choices], key=lambda L: letter_balance[L])
    target_idx = LETTERS.index(target_letter)
    letter_balance[target_letter] += 1

    ordered = [None] * num_choices
    ordered_urls = [None] * num_choices
    ordered[target_idx] = heritage["image_path"]
    ordered_urls[target_idx] = heritage["image_url"]
    di = 0
    for j in range(num_choices):
        if ordered[j] is None:
            ordered[j] = distractors[di]["image_path"]
            ordered_urls[j] = distractors[di]["image_url"]
            di += 1

    q = render_t2(rng.choice(T2_QUESTIONS), name)
    body = "\n".join(f"{LETTERS[j]}. <image>" for j in range(num_choices))
    user = f"{q}\n{body}\n\n{MC_HINT}"
    reason = render_t2(rng.choice(T2_REASON), name)
    assistant = f"{reason}\n\nAnswer: {target_letter}"
    return user, assistant, ordered, ordered_urls, target_letter


# ==================== heritage 풀 로드 ====================

def load_heritages(mode: str):
    """모드별 가용 heritage 로드.
      knowledge: 4-필드(시대/연도/소재지/분류) 전부 추출 + 이미지 캐시 보유.
      reverse:   설명(description_ko) + 이미지 캐시 보유 (속성은 불필요).
    """
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
        desc = item.get("description_ko", "")

        rec = {
            "name_full": name,
            "name": display_name(name),
            "image_path": img_path,
            "image_url": url,
            "answers": {
                "era": eras[0] if eras else "",
                "year": year,
                "location": region,
                "category": top_cat,
            },
            "description_ko": desc,
        }

        if mode == "knowledge":
            if not eras: stats["era_fail"] += 1; continue
            if not region: stats["location_fail"] += 1; continue
            if not top_cat: stats["category_fail"] += 1; continue
            if not year: stats["year_fail"] += 1; continue
        else:  # reverse
            if not clean_description_ko(desc) or len(clean_description_ko(desc)) < 30:
                stats["desc_fail"] += 1
                continue

        eligible.append(rec)
    return eligible, stats


# ==================== heritage 1개 -> 멀티턴 샘플들 ====================

def build_samples_for_heritage(heritage: dict, mode: str, n: int,
                               all_heritages: list[dict], num_choices: int,
                               rng: random.Random,
                               letter_balance: Counter,
                               all_names: list[str],
                               name_letter_balance: Counter,
                               n_mc: int = 0, n_free: int = 0,
                               leak_min_len: int = 0) -> tuple[list, list]:
    """heritage 1개에 대해 멀티턴 샘플 생성. (train_records, test_records) 반환.
    각 slot 의 마지막 1개를 test 블록으로 분리(heritage-aware split).

    reverse 모드:
      - leak_min_len > 0 이면 leak-free 필터 적용 (이름의 N+ 한글 substring 누수 차단).
      - n_mc > 0 이면 객관식 슬롯 n_mc 개, n_free > 0 이면 주관식 슬롯 n_free 개 생성.
      - 둘 다 0 이면 하위 호환: n 으로 객관식만 (legacy).
    knowledge 모드는 n (=속성별 paraphrase 수) 그대로 사용.
    """
    name = heritage["name"]
    name_full = heritage["name_full"]
    samples_by_slot: dict[str, list[dict]] = defaultdict(list)

    if mode == "knowledge":
        # 4 attr × n paraphrase
        for attr in ("era", "year", "location", "category"):
            value = heritage["answers"][attr]
            if not value:
                continue
            q_pool = KN_QUESTIONS[attr]
            a_pool = KN_ANSWERS[attr]
            chosen_q = rng.sample(q_pool, min(n, len(q_pool)))
            while len(chosen_q) < n:
                chosen_q.append(rng.choice(q_pool))
            for i in range(n):
                t1_user = render_kn(chosen_q[i], name)
                t1_asst = render_kn(rng.choice(a_pool), name, value)
                t2_user, t2_asst, imgs, img_urls, letter = build_image_selection_turn(
                    heritage, all_heritages, num_choices, rng, letter_balance)
                samples_by_slot[attr].append({
                    "images": imgs,
                    "image_urls": img_urls,
                    "messages": [
                        {"role": "user", "content": t1_user},
                        {"role": "assistant", "content": t1_asst},
                        {"role": "user", "content": t2_user},
                        {"role": "assistant", "content": t2_asst},
                    ],
                    "metadata": {
                        "mode": mode, "heritage_id": name, "heritage_name_full": name_full,
                        "attr_type": attr, "answer_keyword": value, "answer_letter": letter,
                    },
                })
    else:  # reverse
        # 하위 호환: n_mc/n_free 둘 다 0 이면 legacy n (객관식만, 기존 동작)
        if n_mc == 0 and n_free == 0:
            n_mc_eff, n_free_eff = n, 0
        else:
            n_mc_eff, n_free_eff = n_mc, n_free

        # --- 객관식 슬롯 ---
        for i in range(n_mc_eff):
            t1 = build_reverse_turn1_mc(heritage, all_names, num_choices, rng,
                                        name_letter_balance, leak_min_len=leak_min_len)
            if t1 is None:
                continue
            t1_user, t1_asst, t1_letter = t1
            t2_user, t2_asst, imgs, img_urls, letter = build_image_selection_turn(
                heritage, all_heritages, num_choices, rng, letter_balance)
            samples_by_slot["reverse_mc"].append({
                "images": imgs, "image_urls": img_urls,
                "messages": [
                    {"role": "user", "content": t1_user},
                    {"role": "assistant", "content": t1_asst},
                    {"role": "user", "content": t2_user},
                    {"role": "assistant", "content": t2_asst},
                ],
                "metadata": {
                    "mode": mode, "heritage_id": name, "heritage_name_full": name_full,
                    "attr_type": "reverse", "qa_format": "mc", "answer_keyword": name,
                    "name_mc_letter": t1_letter, "answer_letter": letter,
                    "leak_min_len": leak_min_len,
                },
            })

        # --- 주관식(free-form) 슬롯 ---
        for i in range(n_free_eff):
            t1 = build_reverse_turn1_free(heritage, rng, leak_min_len=leak_min_len)
            if t1 is None:
                continue
            t1_user, t1_asst = t1
            t2_user, t2_asst, imgs, img_urls, letter = build_image_selection_turn(
                heritage, all_heritages, num_choices, rng, letter_balance)
            samples_by_slot["reverse_free"].append({
                "images": imgs, "image_urls": img_urls,
                "messages": [
                    {"role": "user", "content": t1_user},
                    {"role": "assistant", "content": t1_asst},
                    {"role": "user", "content": t2_user},
                    {"role": "assistant", "content": t2_asst},
                ],
                "metadata": {
                    "mode": mode, "heritage_id": name, "heritage_name_full": name_full,
                    "attr_type": "reverse", "qa_format": "free", "answer_keyword": name,
                    "name_mc_letter": None, "answer_letter": letter,
                    "leak_min_len": leak_min_len,
                },
            })

    train, test = [], []
    for slot, recs in samples_by_slot.items():
        if len(recs) <= 1:
            train.extend(recs)
            continue
        train.extend(recs[:-1])
        test.extend(recs[-1:])
    return train, test


# ==================== 메인 ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["knowledge", "reverse"], required=True)
    parser.add_argument("--n-paraphrase", type=int, default=10,
                        help="knowledge: 속성당 paraphrase 수. reverse legacy: 객관식 슬롯 수.")
    parser.add_argument("--n-mc", type=int, default=0,
                        help="(reverse) 객관식 슬롯 수. >0 이면 legacy --n-paraphrase 무시.")
    parser.add_argument("--n-free", type=int, default=0,
                        help="(reverse) 주관식 free-form 슬롯 수. >0 이면 free-form 같이 생성.")
    parser.add_argument("--leak-min-len", type=int, default=0,
                        help="(reverse) 이름의 한글 substring 최소 길이로 leak-free 필터. "
                             "0=비활성(기존 v1), 2=엄격(eval과 동일).")
    parser.add_argument("--num-choices", type=int, default=4,
                        help="이미지 선택 turn 의 보기 수 (A..). 메모리상 4 권장.")
    parser.add_argument("--max-heritages", type=int, default=None,
                        help="테스트용 heritage 수 제한")
    parser.add_argument("--out", type=str, default=None,
                        help="기본: DEFAULT_OUT[mode]. reverse v2(leak-free+free-form)는 v2 경로 권장.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_path = args.out or DEFAULT_OUT[args.mode]

    print(f"=== heritage 풀 로드 (mode={args.mode}) ===")
    eligible, stats = load_heritages(args.mode)
    print(f"  제외 통계: {dict(stats)}")
    print(f"  가용 heritage: {len(eligible)}")

    if args.max_heritages:
        random.Random(args.seed).shuffle(eligible)
        eligible = eligible[: args.max_heritages]
        print(f"  --max-heritages 적용: {len(eligible)}")

    if len(eligible) < args.num_choices:
        print("[에러] 가용 heritage 가 보기 수보다 적습니다.")
        return

    rng = random.Random(args.seed)
    letter_balance = Counter()            # turn2 이미지 선택 letter
    name_letter_balance = Counter()       # reverse turn1 이름 MC letter
    all_names = [h["name"] for h in eligible]
    train_records, test_records = [], []

    print(f"=== 샘플 생성 (mode={args.mode}, N={args.n_paraphrase}, n_mc={args.n_mc}, "
          f"n_free={args.n_free}, leak_min_len={args.leak_min_len}, choices={args.num_choices}) ===")
    for idx, h in enumerate(eligible):
        tr, te = build_samples_for_heritage(
            h, args.mode, args.n_paraphrase, eligible, args.num_choices, rng, letter_balance,
            all_names, name_letter_balance,
            n_mc=args.n_mc, n_free=args.n_free, leak_min_len=args.leak_min_len)
        train_records.extend(tr)
        test_records.extend(te)
        if (idx + 1) % 1000 == 0:
            print(f"  [{idx+1}/{len(eligible)}] train={len(train_records)} test={len(test_records)}")

    rng.shuffle(train_records)
    rng.shuffle(test_records)
    for r in train_records:
        r["split"] = "train"
    for r in test_records:
        r["split"] = "test"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for r in test_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(train_records) + len(test_records)
    print(f"\n=== 완료 ===")
    print(f"  train: {len(train_records)} / test: {len(test_records)} / total: {total}")
    if total:
        print(f"  test 비율: {len(test_records)/total:.4f}")
    print(f"  출력: {out_path}")
    print(f"\n=== turn2 이미지선택 letter 분포 ===")
    tot_l = sum(letter_balance.values())
    if tot_l:
        print("  " + " ".join(f"{L}={letter_balance[L]/tot_l*100:.1f}%" for L in LETTERS[:args.num_choices]))
    if args.mode == "reverse":
        print(f"=== turn1 이름MC letter 분포 ===")
        tot_n = sum(name_letter_balance.values())
        if tot_n:
            print("  " + " ".join(f"{L}={name_letter_balance[L]/tot_n*100:.1f}%" for L in LETTERS[:args.num_choices]))

    # 샘플 미리보기
    if train_records:
        print(f"\n=== 샘플[0] 미리보기 ===")
        s = train_records[0]
        for m in s["messages"]:
            print(f"  [{m['role']}] {m['content'][:90]!r}")
        print(f"  images: {len(s['images'])}개  metadata: {s['metadata']}")


if __name__ == "__main__":
    main()

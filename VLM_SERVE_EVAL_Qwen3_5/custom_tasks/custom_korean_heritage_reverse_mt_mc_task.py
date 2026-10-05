"""
Korean Heritage Reverse MT — 객관식(MC) 벤치마크.

학습에 쓴 data_kr_heri_reverse_mt_hf 의 **test 스플릿(14,395 샘플)** 으로
설명→이름 MC 정확도를 측정한다. 학습 형식과 정확히 동일(MC + Answer: <letter>)
이라 학습/평가 미러링이 깔끔.

- 입력: turn1 user content (설명 + A/B/C/D 이름 보기 + MC_HINT) 그대로 사용.
- 정답 letter: turn1 assistant 의 "Answer: X" 라인을 정규식으로 파싱.
- 메트릭: strict_mc_metric (custom_mc_metric.StrictMultipleChoiceMatch) 재사용.
- 이미지 컬럼은 일부러 만지지 않아 (record_to_sample이 messages만 반환) 디코드 비용 회피.
"""

import os
import re
from typing import Any

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

from custom_tasks.custom_mc_metric import strict_mc_metric


DEFAULT_DATASET_PATH = (
    "/workspace/2026_llm_data_generation/llm_training_ready/"
    "data_korean_heritage_reverse_mt_hf"
)
DATASET_PATH = os.environ.get("KOREAN_HERITAGE_REVERSE_MT_MC_DATASET_PATH", DEFAULT_DATASET_PATH)

LETTER_INDICES = ["A", "B", "C", "D"]
LETTER_CHOICES = [" A", " B", " C", " D"]
_ANS_RE = re.compile(r"Answer:\s*([A-D])")
_CHOICE_RE = re.compile(r"^([A-D])\.\s+(.+?)\s*$", re.MULTILINE)
_HANGUL_RUN_RE = re.compile(r"[가-힣]+")
_MASK_NOISE = {"이 문화재", "문화재", "이문화재"}

# env var 로 leak-free 서브셋만 평가 (기본 off — 전체 evaluate).
# REVERSE_MT_MC_LEAK_FREE_ONLY=1 + 옵션 REVERSE_MT_MC_LEAK_MIN_LEN=2|3|4 (기본 2).
LEAK_FREE_ONLY = os.environ.get("REVERSE_MT_MC_LEAK_FREE_ONLY", "0") == "1"
LEAK_MIN_LEN = int(os.environ.get("REVERSE_MT_MC_LEAK_MIN_LEN", "2"))


def _has_name_leak(name: str, desc: str, min_len: int) -> bool:
    """이름의 한글 런에서 min_len 이상 연속 substring 중 하나라도 desc 에 등장하면 leaky."""
    for run in _HANGUL_RUN_RE.findall(name):
        L = len(run)
        for s in range(L):
            for e in range(s + min_len, L + 1):
                sub = run[s:e]
                if sub in _MASK_NOISE:
                    continue
                if sub in desc:
                    return True
    return False


def reverse_mt_mc_prompt(line: dict[str, Any], task_name: str | None = None):
    """messages 컬럼에서 query/정답letter 추출. LEAK_FREE_ONLY 모드면 누수 샘플은 None 반환 → 스킵."""
    msgs = line["messages"]
    user_turn1 = msgs[0]["content"]
    asst_turn1 = msgs[1]["content"] if len(msgs) > 1 else ""
    m = _ANS_RE.search(asst_turn1)
    letter = m.group(1) if m else "A"

    if LEAK_FREE_ONLY:
        # 보기 중 letter 위치의 이름 = 정답 이름
        choices = {mm.group(1): mm.group(2) for mm in _CHOICE_RE.finditer(user_turn1)}
        correct_name = choices.get(letter, "")
        desc_part = user_turn1.split("\nA.")[0]
        if _has_name_leak(correct_name, desc_part, LEAK_MIN_LEN):
            return None  # lighteval 이 None 인 doc 은 건너뜀

    return Doc(
        task_name=task_name,
        query=user_turn1,
        choices=LETTER_CHOICES,
        gold_index=LETTER_INDICES.index(letter),
        instruction="",
        specific={"gold_letter": letter},
    )


CUSTOM_KOREAN_HERITAGE_REVERSE_MT_MC_TASK = LightevalTaskConfig(
    name="korean_heritage_reverse_mt_mc",
    prompt_function=reverse_mt_mc_prompt,
    hf_repo=DATASET_PATH,
    hf_subset="default",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    metrics=[strict_mc_metric],
    stop_sequence=[],
    version=1,
    generation_size=512,
)


TASKS_TABLE = [CUSTOM_KOREAN_HERITAGE_REVERSE_MT_MC_TASK]

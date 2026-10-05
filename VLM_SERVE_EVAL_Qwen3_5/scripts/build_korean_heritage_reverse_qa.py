#!/usr/bin/env python3
"""
Build a reverse QA benchmark from Korean heritage item data.

기존 ShortQA: 문화재명이 주어지면 → 속성(지정일, 재질 등)을 답변
이번 ReverseQA: 속성/설명이 주어지면 → 문화재명을 답변

Seed: data_korean_object_image_3_generation.json
  - 최상위 키: heritage_id (e.g. "HF010461")
  - 각 값: {"title": str, "qa_pairs": [...], "references": [...], ...}
  - qa_pairs[i]["type"]: "text_knowledge" | "vision_recognition" | "vision_knowledge"

처리 방식:
  1. text_knowledge 타입이면서 answer에 title이 포함된 QA만 선택
  2. answer에서 title을 "[이 문화재]"로 마스킹
  3. 마스킹된 answer 1~MAX_CLUES개를 단서(clue)로 조합
  4. 모델에게 "위 설명의 문화재 명칭은?" 질문 → 정답은 title

Outputs:
  - korean_heritage_reverse_qa_benchmark.json (메인 벤치마크)
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_SOURCE_PATH = Path(
    "/workspace/2026_llm_data_generation/"
    "ai_hub_to_korean_qa/converted_datasets/"
    "data_korean_object_image_3_generation.json"
)
DEFAULT_BENCHMARK_OUTPUT = Path(
    "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/tmp_eval_datasets/"
    "korean_heritage_reverse_qa_benchmark.json"
)

MIN_CLUES = 1   # 단서가 이 수 이상인 항목만 포함
MAX_CLUES = 4   # 최대 단서 수 (너무 많으면 너무 쉬워짐)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_BENCHMARK_OUTPUT)
    parser.add_argument("--min-clues", type=int, default=MIN_CLUES)
    parser.add_argument("--max-clues", type=int, default=MAX_CLUES)
    parser.add_argument("--limit", type=int, default=None, help="항목 수 제한 (디버깅용)")
    return parser.parse_args()


def collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def mask_title(text: str, title: str) -> str:
    """answer 텍스트에서 title을 '[이 문화재]'로 치환한다."""
    if not title or not text:
        return text
    masked = re.sub(re.escape(title), "[이 문화재]", text)
    return masked


def build_question(masked_clues: list[str]) -> str:
    """마스킹된 단서들을 조합해 역방향 QA 질문 프롬프트를 구성한다."""
    lines: list[str] = [
        "아래는 어떤 한국 문화재에 대한 설명입니다.",
        "설명 속 \"[이 문화재]\"는 찾아야 할 문화재의 이름 대신 사용된 표현입니다.",
        "",
    ]
    for i, clue in enumerate(masked_clues, 1):
        lines.append(f"{i}. {clue}")
    lines.extend([
        "",
        "위 설명에 해당하는 한국 문화재의 명칭은 무엇인가요?",
        "",
        "반드시 마지막 줄에 다음 형식으로만 답하세요:",
        "Answer: <문화재 명칭>",
        "",
        "설명이나 추가 문장은 쓰지 마세요.",
    ])
    return "\n".join(lines)


def process_item(
    heritage_id: str,
    item: dict[str, Any],
    idx: int,
    max_clues: int,
) -> dict[str, Any] | None:
    """단일 heritage 항목을 ReverseQA 샘플로 변환한다."""
    title = collapse_ws(item.get("title", "") or "")
    if not title:
        return None

    qa_pairs = item.get("qa_pairs", [])

    # text_knowledge 타입 + answer에 title이 포함된 QA만 선택
    eligible_answers: list[str] = []
    for qa in qa_pairs:
        if qa.get("type") != "text_knowledge":
            continue
        answer = collapse_ws(qa.get("answer", "") or "")
        if not answer:
            continue
        if title not in answer:
            continue
        eligible_answers.append(answer)

    if not eligible_answers:
        return None

    selected = eligible_answers[:max_clues]
    masked_clues = [mask_title(ans, title) for ans in selected]
    question = build_question(masked_clues)

    return {
        "sample_id": f"korean_heritage_reverse_qa_{idx:06d}",
        "heritage_id": heritage_id.strip(),
        "heritage_name": title,
        "question": question,
        "answer": title,
        "num_clues": len(selected),
        "source_answers": selected,
    }


def main() -> None:
    args = parse_args()

    with args.source.open(encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)

    samples: list[dict[str, Any]] = []
    stats = Counter()

    # 중복 title 방지 (key에 trailing space 등이 있는 경우 존재)
    seen_titles: set[str] = set()

    for heritage_id, item in raw.items():
        stats["items_seen"] += 1
        if args.limit is not None and stats["items_seen"] > args.limit:
            break

        title = collapse_ws(item.get("title", "") or "")
        if not title:
            stats["skip_no_title"] += 1
            continue

        # 같은 title이 이미 처리된 경우 스킵 (중복 heritage_id 방어)
        if title in seen_titles:
            stats["skip_duplicate_title"] += 1
            continue

        sample = process_item(heritage_id, item, len(samples), args.max_clues)
        if sample is None:
            stats["skip_no_eligible_qa"] += 1
            continue

        if sample["num_clues"] < args.min_clues:
            stats["skip_insufficient_clues"] += 1
            continue

        seen_titles.add(title)
        samples.append(sample)
        stats[f"clues_{sample['num_clues']}"] += 1
        stats["accepted"] += 1

    clue_count_dist = {k: v for k, v in sorted(stats.items()) if k.startswith("clues_")}

    payload = {
        "metadata": {
            "task_name": "korean_heritage_reverse_qa_benchmark",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_path": str(args.source),
            "description": (
                "속성/설명이 주어지면 문화재명을 맞추는 역방향 QA 벤치마크. "
                "단서(clue)는 text_knowledge QA 답변에서 문화재명을 마스킹해 구성."
            ),
            "min_clues": args.min_clues,
            "max_clues": args.max_clues,
            "benchmark_count": len(samples),
            "clue_count_distribution": clue_count_dist,
            "stats": dict(stats),
        },
        "samples": samples,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Benchmark samples : {len(samples)}")
    print(f"Output            : {args.output}")
    print(f"Clue distribution : {clue_count_dist}")
    print(f"Stats             : {dict(stats)}")


if __name__ == "__main__":
    main()

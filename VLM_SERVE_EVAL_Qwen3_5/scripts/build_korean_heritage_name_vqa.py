#!/usr/bin/env python3
"""
Build a Korean heritage name VQA dataset from the large DPO-style source JSON.

The generated dataset:
- keeps only `type0_vqa_name_identification` rows
- strips trailing parenthetical aliases/Hanja from `heritage_name`
- deduplicates by `(local_image_path, normalized_heritage_name)`
- expands each unique image into several controlled name-question paraphrases

Output format:
{
  "metadata": {...},
  "samples": [
    {
      "sample_id": "...",
      "base_sample_id": "...",
      "template_id": "...",
      "question": "...",
      "answer": "...",
      "heritage_name": "...",
      "local_image_path": "...",
      "image_url": "...",
      ...
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


DEFAULT_SOURCE_PATH = Path(
    "/workspace/2026_llm_data_generation/"
    "korean_heritage_multimodal_hallucination_dpo/dpo_dataset_generated_vqa.json"
)
DEFAULT_OUTPUT_PATH = Path(
    "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/tmp_eval_datasets/"
    "korean_heritage_name_vqa_paraphrase.json"
)
TARGET_QA_ERROR_TYPE = "type0_vqa_name_identification"

QUESTION_TEMPLATES = [
    {
        "id": "name_01",
        "text": "이 이미지에 나온 한국 문화재의 이름은 무엇인가요? 문화재명만 답하세요.",
    },
    {
        "id": "name_02",
        "text": "사진 속 문화재의 명칭을 적어 주세요. 설명 없이 이름만 출력하세요.",
    },
    {
        "id": "name_03",
        "text": "이 사진의 문화재 이름을 한 줄로 답하세요.",
    },
    {
        "id": "name_04",
        "text": "이미지에 보이는 유물 또는 유적의 공식 한국어 명칭만 써 주세요.",
    },
    {
        "id": "name_05",
        "text": "사진 속 한국 문화재가 무엇인지 이름만 답해 주세요.",
    },
    {
        "id": "name_06",
        "text": "이 문화재의 이름을 답하세요. 다른 설명은 쓰지 마세요.",
    },
    {
        "id": "name_07",
        "text": "이미지에 나온 문화재의 정식 명칭을 한국어로만 적어 주세요.",
    },
    {
        "id": "name_08",
        "text": "사진 속 대상의 문화재명을 간단히 답하세요. 문화재 이름 한 줄만 출력하세요.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_PATH, help="Source JSON path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output JSON path")
    parser.add_argument(
        "--qa-error-type",
        default=TARGET_QA_ERROR_TYPE,
        help="Only rows with this qa_error_type are used",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of unique base samples for quick tests",
    )
    return parser.parse_args()


def iter_json_array(path: Path, chunk_size: int = 1 << 20) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    buffer = ""
    index = 0
    started = False

    with path.open("r", encoding="utf-8") as f:
        while True:
            if index >= len(buffer):
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                buffer = chunk
                index = 0

            if not started:
                while index < len(buffer) and buffer[index].isspace():
                    index += 1
                if index >= len(buffer):
                    continue
                if buffer[index] != "[":
                    raise ValueError(f"Expected '[' at start of JSON array in {path}")
                started = True
                index += 1

            while True:
                while index < len(buffer) and buffer[index].isspace():
                    index += 1

                if index >= len(buffer):
                    chunk = f.read(chunk_size)
                    if not chunk:
                        raise ValueError(f"Unexpected EOF while reading {path}")
                    buffer = buffer[index:] + chunk
                    index = 0
                    continue

                if buffer[index] == "]":
                    return

                if buffer[index] == ",":
                    index += 1
                    continue

                try:
                    item, end = decoder.raw_decode(buffer, index)
                except json.JSONDecodeError:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        raise
                    buffer = buffer[index:] + chunk
                    index = 0
                    continue

                yield item
                index = end

                if index > chunk_size:
                    buffer = buffer[index:]
                    index = 0
                break


def collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_heritage_name(text: str) -> str:
    normalized = collapse_ws(text)
    normalized = re.sub(r"[∼〜～]", "~", normalized)
    normalized = re.sub(r"\s*~\s*", "~", normalized)
    normalized = re.sub(r"\s*,\s*", ", ", normalized)
    while True:
        stripped = re.sub(r"\s*\([^()]*\)\s*$", "", normalized).strip()
        if stripped == normalized:
            return normalized
        normalized = stripped


def build_base_samples(
    source_path: Path,
    qa_error_type: str,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    deduped: "OrderedDict[tuple[str, str], dict[str, Any]]" = OrderedDict()
    stats = {
        "rows_seen": 0,
        "rows_kept": 0,
        "rows_skipped_missing_name": 0,
        "rows_skipped_missing_image": 0,
        "duplicate_rows_merged": 0,
    }

    for row in iter_json_array(source_path):
        stats["rows_seen"] += 1
        if row.get("qa_error_type") != qa_error_type:
            continue

        heritage_name_raw = collapse_ws(str(row.get("heritage_name", "") or ""))
        heritage_name = normalize_heritage_name(heritage_name_raw)
        if not heritage_name:
            stats["rows_skipped_missing_name"] += 1
            continue

        local_image_path = collapse_ws(str(row.get("local_image_path", "") or ""))
        if not local_image_path:
            stats["rows_skipped_missing_image"] += 1
            continue

        key = (local_image_path, heritage_name)
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = {
                "heritage_name": heritage_name,
                "heritage_name_full": heritage_name_raw,
                "local_image_path": local_image_path,
                "image_url": collapse_ws(str(row.get("image_url", "") or "")),
                "target_image_info": collapse_ws(str(row.get("target_image_info", "") or "")),
                "source_qa_error_type": row.get("qa_error_type", ""),
                "source_hallucination_type": row.get("hallucination_type", ""),
                "source_row_count": 1,
            }
            stats["rows_kept"] += 1
            if limit is not None and len(deduped) >= limit:
                break
        else:
            existing["source_row_count"] += 1
            stats["duplicate_rows_merged"] += 1

    return list(deduped.values()), stats


def expand_with_templates(base_samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for index, base_sample in enumerate(base_samples, start=1):
        base_sample_id = f"name_vqa_{index:05d}"
        for template in QUESTION_TEMPLATES:
            expanded.append(
                {
                    "sample_id": f"{base_sample_id}_{template['id']}",
                    "base_sample_id": base_sample_id,
                    "template_id": template["id"],
                    "question": template["text"],
                    "answer": base_sample["heritage_name"],
                    "heritage_name": base_sample["heritage_name"],
                    "heritage_name_full": base_sample["heritage_name_full"],
                    "local_image_path": base_sample["local_image_path"],
                    "image_url": base_sample["image_url"],
                    "target_image_info": base_sample["target_image_info"],
                    "source_qa_error_type": base_sample["source_qa_error_type"],
                    "source_hallucination_type": base_sample["source_hallucination_type"],
                    "source_row_count": base_sample["source_row_count"],
                }
            )
    return expanded


def main() -> None:
    args = parse_args()
    base_samples, stats = build_base_samples(
        source_path=args.source,
        qa_error_type=args.qa_error_type,
        limit=args.limit,
    )
    expanded_samples = expand_with_templates(base_samples)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "task_name": "korean_heritage_name_vqa",
            "version": "v1_paraphrase_controlled",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_path": str(args.source),
            "qa_error_type": args.qa_error_type,
            "answer_policy": "Korean heritage name only, with trailing parenthetical aliases removed",
            "dedupe_key": ["local_image_path", "heritage_name"],
            "question_templates": QUESTION_TEMPLATES,
            "base_sample_count": len(base_samples),
            "sample_count": len(expanded_samples),
            "stats": stats,
        },
        "samples": expanded_samples,
    }

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(expanded_samples)} samples from {len(base_samples)} base images")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()

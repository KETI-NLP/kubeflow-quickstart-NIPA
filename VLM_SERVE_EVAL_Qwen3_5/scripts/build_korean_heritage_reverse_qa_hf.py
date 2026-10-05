#!/usr/bin/env python3
"""
Convert the Korean heritage reverse QA benchmark JSON into a local
Hugging Face `save_to_disk` dataset for LightEval.

Input : korean_heritage_reverse_qa_benchmark.json
Output: korean_heritage_reverse_qa_benchmark_hf/  (HF Dataset 형식)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, Features, Value


DEFAULT_JSON_PATH = Path(
    "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/tmp_eval_datasets/"
    "korean_heritage_reverse_qa_benchmark.json"
)
DEFAULT_HF_PATH = Path(
    "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/tmp_eval_datasets/"
    "korean_heritage_reverse_qa_benchmark_hf"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_HF_PATH)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with args.input_json.open(encoding="utf-8") as f:
        payload = json.load(f)

    samples = payload["samples"]
    if args.limit is not None:
        samples = samples[: args.limit]

    rows = []
    for sample in samples:
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "heritage_id": sample["heritage_id"],
                "heritage_name": sample["heritage_name"],
                "question": sample["question"],
                "answer": sample["answer"],
                "num_clues": int(sample["num_clues"]),
            }
        )

    features = Features(
        {
            "sample_id": Value("string"),
            "heritage_id": Value("string"),
            "heritage_name": Value("string"),
            "question": Value("string"),
            "answer": Value("string"),
            "num_clues": Value("int32"),
        }
    )

    dataset = Dataset.from_list(rows, features=features)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(args.output_dir))

    print(f"Wrote dataset with {len(dataset)} rows")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()

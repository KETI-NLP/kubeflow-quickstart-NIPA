#!/usr/bin/env python3
"""
Convert the generated Korean heritage text short-answer benchmark JSON into a
local Hugging Face `save_to_disk` dataset for LightEval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, Features, Value


DEFAULT_JSON_PATH = Path(
    "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/tmp_eval_datasets/"
    "korean_heritage_text_shortqa_benchmark.json"
)
DEFAULT_HF_PATH = Path(
    "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/tmp_eval_datasets/"
    "korean_heritage_text_shortqa_benchmark_hf"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_HF_PATH)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with args.input_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    samples = payload["samples"]
    if args.limit is not None:
        samples = samples[: args.limit]

    rows = []
    for idx, sample in enumerate(samples):
        rows.append(
            {
                "sample_id": f"korean_heritage_text_shortqa_{idx:06d}",
                "heritage_name": sample["heritage_name"],
                "heritage_name_full": sample.get("heritage_name_full", ""),
                "question": sample["benchmark_question"],
                "answer": sample["answer"],
                "answer_type": sample["answer_type"],
                "source_query": sample.get("source_query", ""),
                "source_chosen": sample.get("source_chosen", ""),
                "source_rejected": sample.get("source_rejected", ""),
                "qa_error_type": sample.get("qa_error_type", ""),
                "hallucination_type": sample.get("hallucination_type", ""),
                "paraphrase_count": int(sample.get("paraphrase_count", 1)),
                "candidate_count_for_heritage": int(sample.get("candidate_count_for_heritage", 1)),
            }
        )

    features = Features(
        {
            "sample_id": Value("string"),
            "heritage_name": Value("string"),
            "heritage_name_full": Value("string"),
            "question": Value("string"),
            "answer": Value("string"),
            "answer_type": Value("string"),
            "source_query": Value("string"),
            "source_chosen": Value("string"),
            "source_rejected": Value("string"),
            "qa_error_type": Value("string"),
            "hallucination_type": Value("string"),
            "paraphrase_count": Value("int32"),
            "candidate_count_for_heritage": Value("int32"),
        }
    )

    dataset = Dataset.from_list(rows, features=features)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(args.output_dir))

    print(f"Wrote dataset with {len(dataset)} rows")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()

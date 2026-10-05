#!/usr/bin/env python3
"""
Convert the generated Korean heritage name VQA JSON into a local Hugging Face dataset.

The output dataset stores lightweight metadata and `image_path` strings.
Images are loaded lazily by the custom LightEval task at prompt-build time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, Features, Value


DEFAULT_JSON_PATH = Path(
    "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/tmp_eval_datasets/korean_heritage_name_vqa_paraphrase.json"
)
DEFAULT_HF_PATH = Path(
    "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/tmp_eval_datasets/korean_heritage_name_vqa_paraphrase_hf"
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
    for sample in samples:
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "base_sample_id": sample["base_sample_id"],
                "template_id": sample["template_id"],
                "question": sample["question"],
                "answer": sample["answer"],
                "heritage_name": sample["heritage_name"],
                "heritage_name_full": sample.get("heritage_name_full", ""),
                "image_path": sample["local_image_path"],
                "image_url": sample.get("image_url", ""),
                "target_image_info": sample.get("target_image_info", ""),
                "source_qa_error_type": sample.get("source_qa_error_type", ""),
                "source_hallucination_type": sample.get("source_hallucination_type", ""),
                "source_row_count": int(sample.get("source_row_count", 1)),
            }
        )

    features = Features(
        {
            "sample_id": Value("string"),
            "base_sample_id": Value("string"),
            "template_id": Value("string"),
            "question": Value("string"),
            "answer": Value("string"),
            "heritage_name": Value("string"),
            "heritage_name_full": Value("string"),
            "image_path": Value("string"),
            "image_url": Value("string"),
            "target_image_info": Value("string"),
            "source_qa_error_type": Value("string"),
            "source_hallucination_type": Value("string"),
            "source_row_count": Value("int32"),
        }
    )

    dataset = Dataset.from_list(rows, features=features)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(args.output_dir))

    print(f"Wrote dataset with {len(dataset)} rows")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()

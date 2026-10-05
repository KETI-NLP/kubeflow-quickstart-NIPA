"""
LightEval details parquet converter.

Reads a LightEval details parquet file and exports a flattened JSON or CSV file
that is easier to inspect in editors and spreadsheets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _to_builtin(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [_to_builtin(v) for v in value]
    return value


def flatten_row(row: pd.Series) -> dict[str, Any]:
    doc = _to_builtin(row.get("doc", {})) or {}
    metric = _to_builtin(row.get("metric", {})) or {}
    model_response = _to_builtin(row.get("model_response", {})) or {}

    texts = model_response.get("text") or []
    text_post_processed = model_response.get("text_post_processed") or []
    inputs = model_response.get("input") or []
    extracted = doc.get("specific") or {}

    return {
        "task_name": doc.get("task_name"),
        "sample_id": doc.get("id"),
        "query": doc.get("query"),
        "gold_index": doc.get("gold_index"),
        "gold": (doc.get("choices") or [None])[0],
        "prediction": texts[0] if texts else None,
        "prediction_post_processed": text_post_processed[0] if text_post_processed else None,
        "input_messages": json.dumps(inputs, ensure_ascii=False),
        "metric": json.dumps(metric, ensure_ascii=False),
        "extracted_golds": json.dumps(extracted.get("extracted_golds"), ensure_ascii=False),
        "extracted_predictions": json.dumps(extracted.get("extracted_predictions"), ensure_ascii=False),
        "model_response": json.dumps(model_response, ensure_ascii=False),
        "doc": json.dumps(doc, ensure_ascii=False),
    }


def convert_file(input_path: Path, output_path: Path, output_format: str) -> None:
    df = pd.read_parquet(input_path)
    flat_df = pd.DataFrame([flatten_row(row) for _, row in df.iterrows()])

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "json":
        output_path.write_text(
            json.dumps(flat_df.to_dict(orient="records"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    elif output_format == "csv":
        flat_df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {output_format}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert LightEval details parquet to JSON or CSV")
    parser.add_argument("input", type=Path, help="Input parquet file path")
    parser.add_argument(
        "--format",
        choices=["json", "csv", "both"],
        default="both",
        help="Output format (default: both)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path. If omitted, uses the input path with .json/.csv suffix.",
    )
    args = parser.parse_args()

    input_path = args.input
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if args.format == "both":
        json_path = args.output.with_suffix(".json") if args.output else input_path.with_suffix(".json")
        csv_path = args.output.with_suffix(".csv") if args.output else input_path.with_suffix(".csv")
        convert_file(input_path, json_path, "json")
        convert_file(input_path, csv_path, "csv")
        print(f"Saved JSON to {json_path}")
        print(f"Saved CSV to {csv_path}")
    else:
        if args.output:
            output_path = args.output
        else:
            output_path = input_path.with_suffix(f".{args.format}")
        convert_file(input_path, output_path, args.format)
        print(f"Saved {args.format.upper()} to {output_path}")


if __name__ == "__main__":
    main()

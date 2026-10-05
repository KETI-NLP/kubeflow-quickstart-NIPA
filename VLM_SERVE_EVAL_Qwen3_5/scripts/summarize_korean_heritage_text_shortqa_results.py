#!/usr/bin/env python3
"""
Summarize Korean heritage text shortqa LightEval results by answer_type.

This reads the latest details parquet under an evaluation output directory and
writes:
- per-answer-type JSON summary
- markdown report
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_EVAL_DIR = Path(
    "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/test_korean_heritage_text_shortqa_gpt4o_smoke_v2_prompt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def load_latest_results(eval_dir: Path) -> tuple[Path, dict[str, Any]]:
    results_dir = eval_dir / "results"
    latest_results = sorted(results_dir.glob("**/results_*.json"))[-1]
    with latest_results.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return latest_results, payload


def load_latest_details(eval_dir: Path) -> tuple[Path, pd.DataFrame]:
    details_dir = eval_dir / "details"
    latest_parquet = sorted(details_dir.glob("**/details_*.parquet"))[-1]
    return latest_parquet, pd.read_parquet(latest_parquet)


def extract_prediction(model_response: dict[str, Any]) -> str:
    texts = model_response.get("text_post_processed") or model_response.get("text") or []
    if len(texts):
        return str(texts[0])
    return ""


def build_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        doc = row["doc"]
        metric = row["metric"]
        model_response = row["model_response"]
        specific = doc["specific"]
        rows.append(
            {
                "answer_type": specific.get("answer_type", ""),
                "heritage_name": specific.get("heritage_name", ""),
                "question": doc.get("query", "").split("\n\n")[0],
                "gold": specific.get("answer", ""),
                "prediction": extract_prediction(model_response),
                "strict_em": float(metric.get("text_shortqa_strict_em", 0.0)),
                "relaxed_em": float(metric.get("text_shortqa_relaxed_em", 0.0)),
                "ordered_recall": float(metric.get("text_shortqa_ordered_recall", 0.0)),
                "ordered_precision": float(metric.get("text_shortqa_ordered_precision", 0.0)),
                "ordered_f1": float(metric.get("text_shortqa_ordered_f1", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def summarize_by_type(rows: pd.DataFrame) -> list[dict[str, Any]]:
    summary = []
    grouped = rows.groupby("answer_type", dropna=False)
    for answer_type, group in grouped:
        strict_examples = group[group["strict_em"] == 1].head(2)
        near_examples = group[(group["strict_em"] == 0) & (group["ordered_f1"] >= 0.5)].sort_values(
            "ordered_f1", ascending=False
        ).head(2)
        wrong_examples = group[(group["strict_em"] == 0) & (group["ordered_f1"] == 0)].head(2)

        def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
            return frame[
                ["heritage_name", "question", "gold", "prediction", "strict_em", "ordered_f1"]
            ].to_dict(orient="records")

        summary.append(
            {
                "answer_type": answer_type,
                "samples": int(len(group)),
                "strict_em": float(group["strict_em"].mean()),
                "relaxed_em": float(group["relaxed_em"].mean()),
                "ordered_recall": float(group["ordered_recall"].mean()),
                "ordered_precision": float(group["ordered_precision"].mean()),
                "ordered_f1": float(group["ordered_f1"].mean()),
                "correct_examples": records(strict_examples),
                "near_examples": records(near_examples),
                "wrong_examples": records(wrong_examples),
            }
        )
    summary.sort(key=lambda x: (-x["samples"], x["answer_type"]))
    return summary


def default_output_paths(eval_dir: Path) -> tuple[Path, Path]:
    summary_dir = eval_dir / "analysis"
    summary_dir.mkdir(parents=True, exist_ok=True)
    return (
        summary_dir / "per_answer_type_scores.json",
        summary_dir / "per_answer_type_scores.md",
    )


def format_score(x: float) -> str:
    return f"{x:.4f}"


def write_markdown(
    output_md: Path,
    eval_dir: Path,
    latest_results: Path,
    latest_details: Path,
    results_payload: dict[str, Any],
    summary: list[dict[str, Any]],
) -> None:
    result_key = next(k for k in results_payload["results"].keys() if k != "all")
    overall = results_payload["results"][result_key]

    lines = [
        "# Korean Heritage Text ShortQA Per-Answer-Type Scores",
        "",
        f"- eval dir: [{eval_dir.name}]({eval_dir})",
        f"- results: [{latest_results.name}]({latest_results})",
        f"- details: [{latest_details.name}]({latest_details})",
        "",
        "전체 점수:",
        f"- `text_shortqa_strict_em = {format_score(overall['text_shortqa_strict_em'])}`",
        f"- `text_shortqa_relaxed_em = {format_score(overall['text_shortqa_relaxed_em'])}`",
        f"- `text_shortqa_ordered_f1 = {format_score(overall['text_shortqa_ordered_f1'])}`",
        "",
    ]

    for item in summary:
        lines.extend(
            [
                f"## {item['answer_type']}",
                "",
                f"- samples: `{item['samples']}`",
                f"- `strict_em = {format_score(item['strict_em'])}`",
                f"- `relaxed_em = {format_score(item['relaxed_em'])}`",
                f"- `ordered_recall = {format_score(item['ordered_recall'])}`",
                f"- `ordered_precision = {format_score(item['ordered_precision'])}`",
                f"- `ordered_f1 = {format_score(item['ordered_f1'])}`",
                "",
            ]
        )

        for section_name, key in [
            ("Correct Examples", "correct_examples"),
            ("Near Examples", "near_examples"),
            ("Wrong Examples", "wrong_examples"),
        ]:
            examples = item[key]
            if not examples:
                continue
            lines.append(f"### {section_name}")
            lines.append("")
            for ex in examples:
                lines.extend(
                    [
                        "```text",
                        f"질문: {ex['question']}",
                        f"정답: {ex['gold']}",
                        f"예측: {ex['prediction']}",
                        f"strict_em: {ex['strict_em']}",
                        f"ordered_f1: {format_score(ex['ordered_f1'])}",
                        "```",
                        "",
                    ]
                )

    output_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    latest_results, results_payload = load_latest_results(args.eval_dir)
    latest_details, details_df = load_latest_details(args.eval_dir)
    rows = build_rows(details_df)
    summary = summarize_by_type(rows)

    default_json, default_md = default_output_paths(args.eval_dir)
    output_json = args.output_json or default_json
    output_md = args.output_md or default_md

    payload = {
        "eval_dir": str(args.eval_dir),
        "results_path": str(latest_results),
        "details_path": str(latest_details),
        "overall_results": results_payload["results"],
        "per_answer_type": summary,
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(output_md, args.eval_dir, latest_results, latest_details, results_payload, summary)

    print(f"JSON: {output_json}")
    print(f"MD: {output_md}")


if __name__ == "__main__":
    main()

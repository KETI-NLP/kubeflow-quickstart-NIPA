#!/usr/bin/env python3
"""
Korean heritage reverse QA 결과를 샘플별로 보기 좋게 저장한다.

출력:
  <eval_dir>/analysis/per_sample_results.md   - 마크다운 (입력/예측/정답/점수)
  <eval_dir>/analysis/per_sample_results.json - 구조화 JSON
  <eval_dir>/analysis/summary.md              - 전체 통계 요약
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True)
    return parser.parse_args()


def find_latest_parquet(eval_dir: Path) -> Path:
    files = sorted((eval_dir / "details").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet found in {eval_dir}/details/")
    return files[-1]


def extract_answer(text: str) -> str:
    text = (text or "").strip()
    matches = re.findall(r"^\s*Answer\s*:\s*(.+?)\s*$", text, flags=re.IGNORECASE | re.MULTILINE)
    return matches[-1].strip() if matches else ""


def main() -> None:
    args = parse_args()
    parquet_path = find_latest_parquet(args.eval_dir)
    df = pd.read_parquet(parquet_path)

    rows = []
    for _, row in df.iterrows():
        doc = row["doc"]
        gold = doc["choices"][0]
        query = doc.get("query", "")
        specific = doc.get("specific", {})

        pred_raw = (row["model_response"]["text"] or [""])[0]
        pred_answer = extract_answer(pred_raw)

        metric = row["metric"]
        f1        = metric.get("reverse_qa_ordered_f1", 0.0)
        recall    = metric.get("reverse_qa_ordered_recall", 0.0)
        precision = metric.get("reverse_qa_ordered_precision", 0.0)
        em        = metric.get("reverse_qa_strict_em", 0.0)
        r_em      = metric.get("reverse_qa_relaxed_em", 0.0)
        fmt       = metric.get("reverse_qa_format_valid", 0.0)

        rows.append({
            "sample_id":         specific.get("sample_id", ""),
            "heritage_id":       specific.get("heritage_id", ""),
            "num_clues":         specific.get("num_clues", 0),
            "query":             query,
            "gold":              gold,
            "pred_raw":          pred_raw.strip(),
            "pred_answer":       pred_answer,
            "strict_em":         float(em),
            "relaxed_em":        float(r_em),
            "ordered_recall":    float(recall),
            "ordered_precision": float(precision),
            "ordered_f1":        float(f1),
            "format_valid":      float(fmt),
        })

    rows.sort(key=lambda r: r["ordered_f1"], reverse=True)

    analysis_dir = args.eval_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # --- JSON ---
    with (analysis_dir / "per_sample_results.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    # --- Markdown ---
    n = len(rows)
    n_em       = sum(r["strict_em"] for r in rows)
    n_r_em     = sum(r["relaxed_em"] for r in rows)
    avg_recall = sum(r["ordered_recall"] for r in rows) / n if n else 0
    avg_prec   = sum(r["ordered_precision"] for r in rows) / n if n else 0
    avg_f1     = sum(r["ordered_f1"] for r in rows) / n if n else 0
    avg_fmt    = sum(r["format_valid"] for r in rows) / n if n else 0

    md_lines = [
        "# Korean Heritage Reverse QA — 샘플별 결과",
        "",
        "## 전체 요약",
        "",
        f"| 메트릭 | 값 |",
        f"|--------|-----|",
        f"| 샘플 수 | {n} |",
        f"| strict_em | {n_em}/{n} ({n_em/n*100:.1f}%) |",
        f"| relaxed_em | {n_r_em}/{n} ({n_r_em/n*100:.1f}%) |",
        f"| ordered_recall (평균) | {avg_recall:.4f} |",
        f"| ordered_precision (평균) | {avg_prec:.4f} |",
        f"| ordered_f1 (평균) | {avg_f1:.4f} |",
        f"| format_valid (평균) | {avg_fmt:.4f} |",
        "",
        "---",
        "",
        "## 샘플별 결과 (f1 내림차순)",
        "",
    ]

    for i, r in enumerate(rows, 1):
        em_mark = "✓" if r["strict_em"] else ("~" if r["relaxed_em"] else "✗")
        md_lines += [
            f"### {i}. {r['gold']}  [{em_mark}]  f1={r['ordered_f1']:.3f}",
            "",
            f"- **heritage_id**: {r['heritage_id']}  |  **단서 수**: {r['num_clues']}",
            f"- **정답**: `{r['gold']}`",
            f"- **예측**: `{r['pred_answer']}`",
            f"- **strict_em**: {r['strict_em']:.0f}  |  **relaxed_em**: {r['relaxed_em']:.0f}  |  **recall**: {r['ordered_recall']:.3f}  |  **precision**: {r['ordered_precision']:.3f}  |  **f1**: {r['ordered_f1']:.3f}",
            "",
            "<details><summary>입력 프롬프트 보기</summary>",
            "",
            "```",
            r["query"],
            "```",
            "",
            "</details>",
            "",
            "<details><summary>모델 출력 보기</summary>",
            "",
            "```",
            r["pred_raw"] or "(empty)",
            "```",
            "",
            "</details>",
            "",
            "---",
            "",
        ]

    with (analysis_dir / "per_sample_results.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # --- summary.md ---
    clue_stats: dict[int, dict] = {}
    for r in rows:
        c = r["num_clues"]
        if c not in clue_stats:
            clue_stats[c] = {"n": 0, "em": 0, "recall_sum": 0.0, "prec_sum": 0.0, "f1_sum": 0.0}
        clue_stats[c]["n"]          += 1
        clue_stats[c]["em"]         += r["strict_em"]
        clue_stats[c]["recall_sum"] += r["ordered_recall"]
        clue_stats[c]["prec_sum"]   += r["ordered_precision"]
        clue_stats[c]["f1_sum"]     += r["ordered_f1"]

    summary_lines = [
        "# Korean Heritage Reverse QA — 요약",
        "",
        f"- 샘플 수: **{n}**",
        f"- strict_em: **{n_em}/{n}** ({n_em/n*100:.1f}%)",
        f"- relaxed_em: **{n_r_em}/{n}** ({n_r_em/n*100:.1f}%)",
        f"- ordered_recall 평균: **{avg_recall:.4f}**",
        f"- ordered_precision 평균: **{avg_prec:.4f}**",
        f"- ordered_f1 평균: **{avg_f1:.4f}**",
        f"- format_valid 평균: **{avg_fmt:.4f}**",
        "",
        "## 단서 수별 성능",
        "",
        "| 단서 수 | 샘플 수 | strict_em | avg_recall | avg_precision | avg_f1 |",
        "|:-------:|:-------:|:---------:|:----------:|:-------------:|:------:|",
    ]
    for c in sorted(clue_stats):
        s = clue_stats[c]
        summary_lines.append(
            f"| {c} | {s['n']} | {s['em']:.0f}/{s['n']} "
            f"| {s['recall_sum']/s['n']:.3f} "
            f"| {s['prec_sum']/s['n']:.3f} "
            f"| {s['f1_sum']/s['n']:.3f} |"
        )

    with (analysis_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"샘플 수           : {n}")
    print(f"strict_em        : {n_em}/{n} ({n_em/n*100:.1f}%)")
    print(f"relaxed_em       : {n_r_em}/{n} ({n_r_em/n*100:.1f}%)")
    print(f"ordered_recall   : {avg_recall:.4f}")
    print(f"ordered_precision: {avg_prec:.4f}")
    print(f"ordered_f1       : {avg_f1:.4f}")
    print(f"format_valid     : {avg_fmt:.4f}")
    print(f"저장 위치    : {analysis_dir}")


if __name__ == "__main__":
    main()

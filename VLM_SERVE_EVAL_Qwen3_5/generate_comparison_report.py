"""여러 모델의 벤치마크 결과를 비교하는 합본 리포트 생성기.

사용 예:
    python generate_comparison_report.py \
        --model-dir test_final_11_tasks/qwen3_5_9b_..._checkpoint-1253 \
        --model-dir test_final_11_tasks/Qwen_Qwen3.5-9B \
        --output reports/comparison_report.md
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

TASKS = [
    "ai2d_gen", "chartqa_gen", "hae_rae_bench_gen", "hallusionbench_gen",
    "ifeval_ko_gen", "kmmlu_gen", "korean_character_ocr",
    "korean_heritage_name_vqa", "mathvista_gen", "mmbench_gen", "scienceqa_gen",
]


def extract_metric(task, metric):
    if isinstance(metric, dict):
        if task == "ifeval_ko_gen" and "prompt_level_strict_acc" in metric:
            return "prompt_level_strict_acc", float(metric["prompt_level_strict_acc"])
        if task == "korean_heritage_name_vqa" and "name_vqa_relaxed_em" in metric:
            return "name_vqa_relaxed_em", float(metric["name_vqa_relaxed_em"])
        key = next(iter(metric))
        return key, float(metric[key])
    if isinstance(metric, (int, float)):
        return "Unknown", float(metric)
    return "Unknown", 0.0


def collect_scores(model_dir: str) -> dict:
    """모델 디렉터리에서 task -> {metric, avg, count} 수집."""
    result: dict = {}
    for task in TASKS:
        file_paths = glob.glob(f"{model_dir}/{task}/details/**/*.parquet", recursive=True)
        if not file_paths:
            continue
        file_paths.sort()
        latest_file = file_paths[-1]
        try:
            df = pd.read_parquet(latest_file)
            if len(df) == 0:
                continue
            scores = []
            metric_name = "Unknown"
            for metric in df["metric"]:
                metric_name, val = extract_metric(task, metric)
                scores.append(val)
            avg = sum(scores) / len(scores) if scores else 0.0
            result[task] = {"metric": metric_name, "avg": avg, "count": len(scores)}
        except Exception as e:
            result[task] = {"metric": "ERROR", "avg": float("nan"), "count": 0, "error": str(e)}
    return result


def model_label(model_dir: str) -> str:
    return os.path.basename(os.path.normpath(model_dir))


def build_comparison(model_dirs: list, output_path: str) -> None:
    per_model = {}
    for d in model_dirs:
        per_model[model_label(d)] = collect_scores(d)

    labels = list(per_model.keys())

    # task별 metric 이름은 모델 간 동일하다고 가정 (다르면 첫 번째로 발견된 값 사용)
    task_metric: dict = {}
    for label in labels:
        for task, info in per_model[label].items():
            task_metric.setdefault(task, info["metric"])

    # 정렬: TASKS 정의 순서 유지하되, 어디에도 결과가 없는 태스크는 제외
    task_order = [t for t in TASKS if t in task_metric]

    # 비교 테이블 생성 (열: 모델, 행: 태스크)
    header = "| 태스크 (Task) | 채점 기준 (Metric) | " + " | ".join(labels) + " |\n"
    sep = "| :--- | :--- | " + " | ".join([":---:"] * len(labels)) + " |\n"
    rows = ""
    # 모델별 평균을 위해 태스크별 점수 모음
    per_model_scores: dict = {label: [] for label in labels}

    for task in task_order:
        metric = task_metric[task]
        row = f"| **{task}** | `{metric}` |"
        # 태스크별 최고 점수 굵게
        values = {}
        for label in labels:
            info = per_model[label].get(task)
            if info is None or np.isnan(info.get("avg", float("nan"))):
                values[label] = None
            else:
                values[label] = info["avg"]
        valid_vals = [v for v in values.values() if v is not None]
        max_val = max(valid_vals) if valid_vals else None

        for label in labels:
            v = values[label]
            if v is None:
                cell = " - "
            else:
                per_model_scores[label].append(v)
                if max_val is not None and abs(v - max_val) < 1e-9 and len(valid_vals) > 1:
                    cell = f" **{v:.2f}** "
                else:
                    cell = f" {v:.2f} "
            row += f"{cell}|"
        rows += row + "\n"

    # 평균 행
    avg_row = "| **평균 (모든 태스크)** | `mean` |"
    avg_values = {}
    for label in labels:
        scores = per_model_scores[label]
        avg_values[label] = (sum(scores) / len(scores)) if scores else None
    valid_avgs = [v for v in avg_values.values() if v is not None]
    max_avg = max(valid_avgs) if valid_avgs else None
    for label in labels:
        v = avg_values[label]
        if v is None:
            avg_row += " - |"
        else:
            if max_avg is not None and abs(v - max_avg) < 1e-9 and len(valid_avgs) > 1:
                avg_row += f" **{v:.2f}** |"
            else:
                avg_row += f" {v:.2f} |"
    rows += avg_row + "\n"

    # 모델 메타 정보 섹션
    meta_lines = ["## 🧾 비교 대상 모델", ""]
    for label, d in zip(labels, model_dirs):
        meta_lines.append(f"- **{label}** — `{d}`")
    meta_lines.append("")

    final_report = (
        "# 📊 모델 간 비교 리포트\n\n"
        + "\n".join(meta_lines)
        + "\n## 비교 표\n\n"
        + "각 셀은 평균 점수입니다. 동일 태스크에서 최고 점수는 **굵게** 표시됩니다.\n\n"
        + header
        + sep
        + rows
        + "\n"
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(final_report)

    print(f"Comparison report generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="여러 모델의 결과를 비교하는 리포트 생성")
    parser.add_argument(
        "--model-dir",
        action="append",
        required=True,
        help="비교할 모델 결과 디렉터리. 여러 번 지정 가능.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="출력 마크다운 경로 (예: reports/comparison_report.md)",
    )
    args = parser.parse_args()
    if len(args.model_dir) < 1:
        parser.error("적어도 하나의 --model-dir 가 필요합니다.")
    build_comparison(args.model_dir, args.output)


if __name__ == "__main__":
    main()

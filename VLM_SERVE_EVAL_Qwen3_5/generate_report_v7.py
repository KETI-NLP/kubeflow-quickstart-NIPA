"""모델별 상세 리포트 생성기.

사용 예:
    python generate_report_v7.py \
        --model-dir test_final_11_tasks/qwen3_5_9b_..._checkpoint-1253 \
        --output reports/sample_report_qwen3_5_9b_..._checkpoint-1253.md
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

TASKS = [
    "ai2d_gen", "chartqa_gen", "hae_rae_bench_gen", "hallusionbench_gen",
    "ifeval_ko_gen", "kmmlu_gen", "korean_character_ocr",
    "korean_heritage_name_vqa", "mathvista_gen", "mmbench_gen", "scienceqa_gen",
]


def safe_str(val):
    if isinstance(val, np.ndarray):
        val = val.tolist()
    if isinstance(val, (list, dict)):
        try:
            return json.dumps(val, ensure_ascii=False)
        except Exception:
            return str(val)
    return str(val)


def extract_metric(task, metric):
    """parquet 행의 metric 컬럼에서 (대표 metric 이름, 값)을 추출."""
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


def build_report(model_dir: str, output_path: str) -> dict:
    """모델 1개에 대한 상세 리포트를 생성하고 (task -> avg_score) 딕셔너리를 반환."""
    report_body = ""
    summary_table = (
        "## 📊 종합 결과 요약 (100개 샘플 기준)\n\n"
        "| 태스크 (Task) | 채점 기준 (Metric) | 평균 점수 (Accuracy) | 샘플 수 |\n"
        "| :--- | :--- | :--- | :--- |\n"
    )
    model_name = os.path.basename(os.path.normpath(model_dir))
    found_model_name = None
    task_scores: dict = {}

    for task in TASKS:
        file_paths = glob.glob(f"{model_dir}/{task}/details/**/*.parquet", recursive=True)
        if not file_paths:
            continue

        file_paths.sort()
        latest_file = file_paths[-1]

        if found_model_name is None:
            try:
                found_model_name = latest_file.split("details/")[1].rsplit("/", 2)[0]
            except Exception:
                pass

        try:
            df = pd.read_parquet(latest_file)
            if len(df) == 0:
                continue

            report_body += f"## 📚 Task: `{task}`\n\n"

            scores = []
            metric_name_str = "Unknown"
            for metric in df["metric"]:
                metric_name_str, val = extract_metric(task, metric)
                scores.append(val)

            avg_score = sum(scores) / len(scores) if scores else 0.0
            task_scores[task] = {
                "metric": metric_name_str,
                "avg": avg_score,
                "count": len(scores),
            }
            summary_table += (
                f"| **{task}** | `{metric_name_str}` | **{avg_score:.2f}** | {len(scores)} |\n"
            )

            num_samples = min(1, len(df)) if task == "ifeval_ko_gen" else min(100, len(df))

            for i in range(num_samples):
                sample = df.iloc[i].to_dict()

                doc = sample.get("doc", {})
                if isinstance(doc, dict):
                    prompt = doc.get("query", doc.get("instruction", str(doc)))
                    if "gold_index" in doc and "choices" in doc:
                        try:
                            gold_idx = doc["gold_index"]
                            choices = doc["choices"]
                            if isinstance(gold_idx, list):
                                gold = [choices[idx] for idx in gold_idx]
                            elif isinstance(gold_idx, int):
                                gold = choices[gold_idx]
                            else:
                                gold = doc.get("target", doc.get("answer", doc.get("choices", str(doc))))
                        except Exception:
                            gold = doc.get("target", doc.get("answer", doc.get("choices", str(doc))))
                    else:
                        gold = doc.get("target", doc.get("answer", doc.get("choices", str(doc))))
                else:
                    prompt = str(doc)
                    gold = str(doc)

                prompt = safe_str(prompt)
                if isinstance(prompt, str):
                    prompt = prompt.replace("<|image_pad|>", "")

                gold = safe_str(gold)

                model_response = sample.get("model_response", {})
                if isinstance(model_response, dict):
                    prediction = model_response.get("text", model_response)
                else:
                    prediction = model_response

                prediction = safe_str(prediction)
                if isinstance(prediction, str):
                    prediction = prediction.replace("<|im_start|>assistant", "").strip()

                metrics = sample.get("metric", {})
                clean_metrics = {}
                if isinstance(metrics, dict):
                    for k, v in metrics.items():
                        if isinstance(v, (np.float32, np.float64)):
                            clean_metrics[k] = float(v)
                        elif isinstance(v, (np.int32, np.int64)):
                            clean_metrics[k] = int(v)
                        else:
                            clean_metrics[k] = str(v)
                    metrics_str = json.dumps(clean_metrics, ensure_ascii=False, indent=2)
                else:
                    metrics_str = safe_str(metrics)

                report_body += f"### 🔹 Sample {i+1}\n"
                report_body += f"**📥 Input (입력)**\n```text\n{prompt[:1000]}\n```\n\n"
                report_body += f"**🎯 Ground Truth (정답)**\n```text\n{gold}\n```\n\n"
                report_body += f"**🤖 Prediction (모델 출력)**\n```text\n{prediction}\n```\n\n"
                report_body += f"**📊 Metrics (채점 결과)**\n```json\n{metrics_str}\n```\n\n"
                report_body += "---\n\n"

        except Exception as e:
            report_body += f"Error processing task {task}: {e}\n\n"

    if found_model_name:
        model_name = found_model_name

    final_report = (
        f"# 벤치마크 샘플 결과 리포트\n\n"
        f"**평가 모델:** `{model_name}`\n\n"
        f"**결과 디렉터리:** `{model_dir}`\n\n"
        f"최근 벤치마크 실행 결과입니다.\n\n"
        + summary_table
        + "\n---\n\n"
        + report_body
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(final_report)

    print(f"Report generated: {output_path}")
    return {
        "model_name": model_name,
        "model_dir": model_dir,
        "task_scores": task_scores,
    }


def main():
    parser = argparse.ArgumentParser(description="모델별 상세 리포트 생성")
    parser.add_argument(
        "--model-dir",
        required=True,
        help="모델 결과가 저장된 디렉터리 (예: test_final_11_tasks/<MODEL_TAG>)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="출력 마크다운 경로 (예: reports/sample_report_<MODEL_TAG>.md)",
    )
    args = parser.parse_args()
    build_report(args.model_dir, args.output)


if __name__ == "__main__":
    main()

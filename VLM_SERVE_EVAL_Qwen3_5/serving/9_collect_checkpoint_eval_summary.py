import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect per-checkpoint JSON outputs into a summary.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-prefix", default="parallel_summary")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    result_files = sorted(input_dir.glob("checkpoint-*.json"))
    if not result_files:
        raise FileNotFoundError(f"No checkpoint-*.json files found under {input_dir}")

    summaries = []
    responses = []

    for path in result_files:
        with open(path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        summaries.append(payload["summary"])
        responses.extend(payload["results"])

    summaries = sorted(summaries, key=lambda item: (-item["accuracy"], item["checkpoint"]))

    summary_json = input_dir / f"{args.output_prefix}.json"
    summary_csv = input_dir / f"{args.output_prefix}.csv"
    responses_csv = input_dir / f"{args.output_prefix}_responses.csv"

    with open(summary_json, "w", encoding="utf-8") as fp:
        json.dump({"summaries": summaries}, fp, ensure_ascii=False, indent=2)

    with open(summary_csv, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["checkpoint", "checkpoint_path", "correct_count", "total_questions", "accuracy"])
        writer.writeheader()
        writer.writerows(summaries)

    with open(responses_csv, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "checkpoint",
                "checkpoint_path",
                "question_id",
                "question",
                "expected_answer",
                "predicted_answer",
                "is_correct",
                "raw_response",
                "prompt_tokens",
                "completion_tokens",
            ],
        )
        writer.writeheader()
        writer.writerows(responses)

    best = summaries[0]
    print(
        json.dumps(
            {
                "best_checkpoint": best["checkpoint"],
                "accuracy": best["accuracy"],
                "checkpoint_count": len(summaries),
                "summary_json": str(summary_json),
                "summary_csv": str(summary_csv),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

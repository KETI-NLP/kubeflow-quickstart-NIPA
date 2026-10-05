import argparse
import csv
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect prompt-grid summary CSV files into one ranking table.")
    parser.add_argument(
        "--input-dir",
        default="/data/eval_results/qwen3_5_9b_multimodal_sft_lr_1e-5_prompt_grid_parallel",
        help="Directory containing *_prompt_grid_summary.csv files.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Optional output CSV path. Defaults to <input-dir>/prompt_grid_ranking.csv",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_file = Path(args.output_file).resolve() if args.output_file else input_dir / "prompt_grid_ranking.csv"

    rows: list[dict[str, str]] = []
    for csv_path in sorted(input_dir.glob("*_prompt_grid_summary.csv")):
        rows.extend(_read_csv(csv_path))

    if not rows:
        raise SystemExit(f"No *_prompt_grid_summary.csv files found under {input_dir}")

    rows.sort(key=lambda row: (-int(row["total_score"]), row["checkpoint"], row["prompt_id"]))

    with output_file.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["checkpoint", "checkpoint_path", "prompt_id", "total_score"])
        writer.writeheader()
        writer.writerows(rows)

    print(output_file)
    print()
    print("TOP RESULTS")
    for row in rows[:10]:
        print(f'{row["checkpoint"]},{row["prompt_id"]},{row["total_score"]}')


if __name__ == "__main__":
    main()

import argparse
import os
import sys

from dotenv import load_dotenv
from mlx.sdk.data import load_dataset, login


DEFAULT_DATASETS = [
    "data_13_kr_char_qwen3_5_sft_hf",
    "data_13_kr_char_hcx_omni_sft_hf",
    "data_13_kr_char_hcx_think_sft_hf",
    "finevision_qwen3_5_sft_hf",
    "finevision_hcx_omni_sft_hf",
    "finevision_hcx_think_sft_hf",
]


def get_num_rows(split_obj):
    if split_obj is None:
        return 0
    if hasattr(split_obj, "num_rows"):
        return split_obj.num_rows
    return len(split_obj)


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Print train/test sample counts for MLXP datasets"
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default="YOUR_WORKSPACE",
        help="Target MLXP workspace name",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Dataset names to inspect",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="MLXP API key or set MLX_APIKEY / MLXP_API_KEY",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        help="MLXP endpoint URL or set MLXP_ENDPOINT_URL",
    )
    args = parser.parse_args()

    api_key = os.environ.get("MLX_APIKEY", os.environ.get("MLXP_API_KEY", args.api_key))
    endpoint = os.environ.get("MLXP_ENDPOINT_URL", args.endpoint)

    if not api_key:
        print("Error: API key is required. Pass --api-key or set MLX_APIKEY / MLXP_API_KEY.")
        sys.exit(1)

    if endpoint:
        print(f"Logging in to MLXP DataManager at endpoint: {endpoint}")
        login(api_key, endpoint)
    else:
        print("Logging in to MLXP DataManager (default endpoint)")
        login(api_key)

    print("")
    print("Dataset sample counts")
    print("=" * 80)

    for dataset_name in args.datasets:
        repo_id = f"{args.workspace}/{dataset_name}"
        print(f"[{repo_id}]")
        try:
            ds = load_dataset(repo_id)
        except Exception as exc:
            print(f"  load_failed: {exc}")
            print("")
            continue

        if hasattr(ds, "keys"):
            split_names = list(ds.keys())
            train_count = get_num_rows(ds["train"]) if "train" in ds else 0
            test_count = get_num_rows(ds["test"]) if "test" in ds else 0
            print(f"  train: {train_count}")
            print(f"  test:  {test_count}")
            for split_name in split_names:
                if split_name in {"train", "test"}:
                    continue
                print(f"  {split_name}: {get_num_rows(ds[split_name])}")
        else:
            print(f"  train: {get_num_rows(ds)}")
            print("  test:  0")

        print("")


if __name__ == "__main__":
    main()

import argparse
import os
import sys


DEFAULT_DATASETS = [
    "YOUR_WORKSPACE/data_kr_obj_img_4_hallu_txt_dpo_hf",
]


def get_num_rows(split_obj):
    if split_obj is None:
        return 0
    if hasattr(split_obj, "num_rows"):
        return split_obj.num_rows
    return len(split_obj)


def main():
    parser = argparse.ArgumentParser(description="Download MLXP datasets for HyperClovaX VLM DPO experiments")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, help="MLXP dataset IDs to download")
    parser.add_argument("--output_dir", type=str, default="./vlm_dpo_hcx_dataset", help="Local directory to save datasets")
    parser.add_argument("--api_key", type=str, default=os.environ.get("MLX_APIKEY", os.environ.get("MLXP_API_KEY")), help="MLXP API key")
    parser.add_argument("--endpoint", type=str, default=os.environ.get("MLXP_ENDPOINT_URL"), help="MLXP endpoint URL")
    args = parser.parse_args()

    if not args.api_key:
        print("MLXP API key is required. Pass --api_key or set MLX_APIKEY / MLXP_API_KEY.")
        sys.exit(1)

    from mlx.sdk.data import load_dataset, login

    if args.endpoint:
        login(args.api_key, args.endpoint)
    else:
        login(args.api_key)

    os.makedirs(args.output_dir, exist_ok=True)

    for repo_id in args.datasets:
        print(f"Downloading {repo_id} ...")
        ds = load_dataset(repo_id)
        dataset_name = repo_id.split("/")[-1]
        save_path = os.path.join(args.output_dir, dataset_name)
        ds.save_to_disk(save_path)

        if hasattr(ds, "keys"):
            split_names = list(ds.keys())
            train_count = get_num_rows(ds["train"]) if "train" in ds else 0
            print(f"  saved_to: {save_path}")
            print(f"  splits: {split_names}")
            print(f"  train_samples: {train_count:,}")
            if "train" in ds and train_count > 0:
                print(f"  train_columns: {ds['train'].column_names}")
        else:
            print(f"  saved_to: {save_path}")
            print(f"  train_samples: {get_num_rows(ds):,}")
            if len(ds) > 0:
                print(f"  train_columns: {ds.column_names}")


if __name__ == "__main__":
    main()

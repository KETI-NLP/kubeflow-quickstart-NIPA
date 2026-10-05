import argparse
import os
import sys
import time


DEFAULT_DATASETS = [
    "YOUR_WORKSPACE/data_kr_obj_img_4_hallu_txt_dpo_hf",
    "YOUR_WORKSPACE/data_kr_obj_img_5_hallu_wr_txt_dpo_hf",
    "YOUR_WORKSPACE/data_kr_heri_vqa_dpo_hf",
]


def get_num_rows(split_obj):
    if split_obj is None:
        return 0
    if hasattr(split_obj, "num_rows"):
        return split_obj.num_rows
    return len(split_obj)


def patch_datasets_arrow_metadata():
    import datasets

    original_from_arrow_schema = datasets.Features.from_arrow_schema

    def patched_from_arrow_schema(schema):
        if schema.metadata and b"huggingface" in schema.metadata:
            new_metadata = {k: v for k, v in schema.metadata.items() if k != b"huggingface"}
            schema = schema.with_metadata(new_metadata)
        return original_from_arrow_schema(schema)

    datasets.Features.from_arrow_schema = patched_from_arrow_schema
    if hasattr(datasets, "arrow_dataset"):
        datasets.arrow_dataset.Features.from_arrow_schema = patched_from_arrow_schema
    if hasattr(datasets, "features") and hasattr(datasets.features, "features"):
        datasets.features.features.Features.from_arrow_schema = patched_from_arrow_schema

    original_from_dict = datasets.DatasetInfo.from_dict

    def patched_from_dict(d):
        if "features" in d:
            del d["features"]
        return original_from_dict(d)

    datasets.DatasetInfo.from_dict = patched_from_dict
    if hasattr(datasets, "info"):
        datasets.info.DatasetInfo.from_dict = patched_from_dict


def main():
    parser = argparse.ArgumentParser(description="Download MLXP datasets for Qwen3.5 VLM DPO experiments")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, help="MLXP dataset IDs to download")
    parser.add_argument("--output_dir", type=str, default="./vlm_dpo_qwen3_5_dataset", help="Local directory to save datasets")
    parser.add_argument("--api_key", type=str, default=os.environ.get("MLX_APIKEY", os.environ.get("MLXP_API_KEY")), help="MLXP API key")
    parser.add_argument("--endpoint", type=str, default=os.environ.get("MLXP_ENDPOINT_URL"), help="MLXP endpoint URL")
    args = parser.parse_args()

    if not args.api_key:
        print("MLXP API key is required. Pass --api_key or set MLX_APIKEY / MLXP_API_KEY.")
        sys.exit(1)

    patch_datasets_arrow_metadata()

    from mlx.sdk.data import load_dataset, login

    if args.endpoint:
        login(args.api_key, args.endpoint)
    else:
        login(args.api_key)

    os.makedirs(args.output_dir, exist_ok=True)

    for repo_id in args.datasets:
        print(f"Downloading {repo_id} ...")
        last_error = None
        for attempt in range(1, 6):
            try:
                ds = load_dataset(repo_id)
                break
            except Exception as exc:
                last_error = exc
                if attempt == 5:
                    raise
                wait_seconds = min(8, 2 ** (attempt - 1))
                print(
                    f"[Dataset Download Retry] source={repo_id} | attempt={attempt}/5 failed with "
                    f"{type(exc).__name__}: {exc}. Retrying in {wait_seconds}s...",
                    flush=True,
                )
                time.sleep(wait_seconds)
        else:
            raise last_error
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

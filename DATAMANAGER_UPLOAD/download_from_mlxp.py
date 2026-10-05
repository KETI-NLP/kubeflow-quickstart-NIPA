import os
import argparse
import sys
from dotenv import load_dotenv
from mlx.sdk.data import load_dataset, login

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Download datasets from Naver Cloud MLXP DataManager")
    
    # Auth args
    parser.add_argument("--api-key", type=str, help="MLXP API Key (or set MLXP_API_KEY env var)")
    parser.add_argument("--endpoint", type=str, help="MLX endpoint URL (or set MLXP_ENDPOINT_URL env var)")
    
    # Target args
    parser.add_argument("--workspace", type=str, required=True, help="Target MLXP Workspace Name")
    parser.add_argument("--dataset-name", type=str, required=True, help="Name of the Dataset repository to download from")
    
    # Data args
    parser.add_argument("--local-dir", type=str, required=True, help="Local directory path to download the dataset to")
    parser.add_argument("--raw-download", action="store_true", help="Use huggingface_hub snapshot_download to preserve raw files (e.g., parquets)")

    args = parser.parse_args()

    # Extract Auth from env or args
    api_key = os.environ.get("MLX_APIKEY", os.environ.get("MLXP_API_KEY", args.api_key))
    endpoint = os.environ.get("MLXP_ENDPOINT_URL", args.endpoint)

    if not api_key:
        print("Error: API Key is required. Pass --api-key or set MLX_APIKEY environment variable in .env")
        sys.exit(1)
    
    if endpoint:
        print(f"Logging in to MLXP DataManager at endpoint: {endpoint}...")
        login(api_key, endpoint)
    else:
        print("Logging in to MLXP DataManager (default endpoint)...")
        login(api_key)

    repo_id = f"{args.workspace}/{args.dataset_name}"
    
    os.makedirs(args.local_dir, exist_ok=True)
    
    if args.raw_download:
        print(f"Loading raw files from MLXP DataManager '{repo_id}' using snapshot_download...")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id, repo_type="dataset", local_dir=args.local_dir, max_workers=8)
            print("\n✅ Raw Download completed successfully!")
        except Exception as e:
            print(f"\n❌ Raw Download failed: {e}")
            sys.exit(1)
    else:
        print(f"Loading dataset from MLXP DataManager '{repo_id}'...")
        try:
            ds = load_dataset(repo_id)
            print(f"Saving dataset to disk at '{args.local_dir}'...")
            ds.save_to_disk(args.local_dir)
            print("\n✅ Download and save completed successfully!")
        except Exception as e:
            print(f"\n❌ Download failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()

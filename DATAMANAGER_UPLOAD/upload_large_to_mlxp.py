import os
import argparse
from mlx.sdk.data import login
from huggingface_hub import create_repo, upload_large_folder
import sys
from dotenv import load_dotenv

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Upload large datasets to Naver Cloud MLXP DataManager using huggingface_hub SDK")
    
    # Auth args
    parser.add_argument("--api-key", type=str, help="MLXP API Key (or set MLXP_API_KEY env var)")
    parser.add_argument("--endpoint", type=str, help="MLX endpoint URL (or set MLXP_ENDPOINT_URL env var)")
    
    # Target args
    parser.add_argument("--workspace", type=str, required=True, help="Target MLXP Workspace Name")
    parser.add_argument("--dataset-name", type=str, required=True, help="Name of the Dataset repository to be created")
    
    # Data args
    parser.add_argument("--local-dir", type=str, 
                        default="/workspace/2026_llm_data_generation/convert_to_llm_training_ready", 
                        help="Local directory path containing the converted datasets")
    parser.add_argument("--repo-path", type=str, default=".", 
                        help="Remote directory path inside the MLXP dataset (default is root '.')")

    args = parser.parse_args()

    # Extract Auth from env or args (supports MLX_APIKEY or MLXP_API_KEY)
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

    # Repo validation
    repo_id = f"{args.workspace}/{args.dataset_name}"
    
    if not os.path.exists(args.local_dir):
        print(f"Error: Local directory '{args.local_dir}' does not exist.")
        sys.exit(1)

    print(f"1. Ensuring repository '{repo_id}' exists...")
    try:
        # Note: create_repo requires Workspace Admin permission
        create_repo(repo_id=repo_id, repo_type="dataset")
        print("   -> Repository successfully created (or currently exists).")
    except Exception as e:
        print(f"   -> Note on CreateRepo: {e}")

    print(f"\n2. Starting Upload:")
    print(f"   - Local Source:  '{args.local_dir}'")
    print(f"   - Remote Target: '{repo_id}'")
    
    try:
        upload_large_folder(
            repo_id=repo_id,
            folder_path=args.local_dir,
            repo_type="dataset",
        )
        print("\n✅ Upload completed successfully!")
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

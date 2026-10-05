import os
import argparse
from mlx.sdk.data import login
from huggingface_hub import delete_repo
import sys
from dotenv import load_dotenv

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Delete datasets from Naver Cloud MLXP DataManager using huggingface_hub SDK")
    
    # Auth args
    parser.add_argument("--api-key", type=str, help="MLXP API Key (or set MLXP_API_KEY env var)")
    parser.add_argument("--endpoint", type=str, help="MLX endpoint URL (or set MLXP_ENDPOINT_URL env var)")
    
    # Target args
    parser.add_argument("--workspace", type=str, required=True, help="Target MLXP Workspace Name")
    parser.add_argument("--dataset-names", type=str, nargs='+', required=True, help="One or more dataset repository names to delete. Separated by spaces.")
    parser.add_argument("--force", action="store_true", help="Skip the confirmation prompt")
    
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

    print("\n[Warning] You are about to delete the following datasets forever:")
    for ds in args.dataset_names:
        print(f" - {args.workspace}/{ds}")
    
    if not args.force:
        confirm = input("\nAre you sure you want to delete these datasets? (y/n): ")
        if confirm.lower() != 'y':
            print("Deletion cancelled by user.")
            sys.exit(0)

    print("\n🚀 Starting Deletion:")
    success_count = 0
    fail_count = 0
    for ds in args.dataset_names:
        repo_id = f"{args.workspace}/{ds}"
        print(f"Deleting repository '{repo_id}'...")
        try:
            delete_repo(repo_id=repo_id, repo_type="dataset")
            print("   -> ✅ Successfully deleted!")
            success_count += 1
        except Exception as e:
            print(f"   -> ❌ Deletion failed: {e}")
            fail_count += 1
            
    print(f"\n==========================================")
    print(f"Finished! Deleted: {success_count}, Failed: {fail_count}")
    print(f"==========================================")

if __name__ == "__main__":
    main()

import os
import argparse
import sys
from mlx.sdk.model_registry import ModelRegistryAPI
from mlx.api.model_registry import ModelRequest, VersionRequest

def progress_callback(progress_size: int, total_size: int):
    # Print progress on the same line
    percent = (progress_size / total_size) * 100 if total_size > 0 else 0
    sys.stdout.write(f"\rUploading... {progress_size}/{total_size} bytes ({percent:.2f}%)")
    sys.stdout.flush()
    if progress_size == total_size:
        print()

def completion_callback(file_path: str, size: int):
    print(f"✅ Upload completed: {file_path} ({size} bytes)")

def main():
    parser = argparse.ArgumentParser(description="Upload a model to MLXP Model Registry")
    
    # Auth args
    parser.add_argument("--api-key", type=str, help="MLXP API Key (or set MLX_APIKEY env var)")
    parser.add_argument("--endpoint", type=str, help="MLX endpoint URL (or set MLX_ENDPOINT_URL env var)")
    parser.add_argument("--project", type=str, required=True, help="Target MLXP Project Name")
    
    # Model args
    parser.add_argument("--model-name", type=str, required=True, help="Name of the Model (3~64 chars, a-z, 0-9, -)")
    parser.add_argument("--version", type=str, default="v1.0", help="Version of the Model (e.g., v1.0)")
    parser.add_argument("--local-path", type=str, required=True, help="Local directory path containing the model files (inside the pod)")
    parser.add_argument("--excludes", type=str, nargs="*", default=[], help="List of files/folders to exclude (e.g. checkpoint-*)")
    
    args = parser.parse_args()

    # Retrieve from Env or CLI args
    api_key = os.environ.get("MLX_APIKEY", os.environ.get("MLXP_API_KEY", args.api_key))
    endpoint = os.environ.get("MLX_ENDPOINT_URL", os.environ.get("MLXP_ENDPOINT_URL", args.endpoint))
    project = os.environ.get("MLX_PROJECT", os.environ.get("MLXP_PROJECT", args.project))

    if not api_key or not endpoint or not project:
        print("Error: --api-key, --endpoint, and --project are required (or via env vars).")
        sys.exit(1)
        
    if not os.path.exists(args.local_path):
        print(f"Error: The model directory '{args.local_path}' does not exist.")
        sys.exit(1)

    print(f"🚀 Initializing ModelRegistryAPI (Endpoint: {endpoint})")
    client = ModelRegistryAPI(endpoint, api_key)

    print(f"📦 Ensuring Model Registry '{args.model_name}' exists in project '{project}'...")
    try:
        model_request = ModelRequest(name=args.model_name)
        model = client.model_api.create(project, model_request)
        print(f"   -> Model '{model.name}' created.")
    except Exception as e:
        if "AlreadyExists" in str(e) or "already exists" in str(e).lower() or "409" in str(e):
            print(f"   -> Model '{args.model_name}' already exists. Proceeding...")
        else:
            print(f"   -> Warning on Create Model: {e}")

    print(f"🔖 Ensuring Version '{args.version}' exists...")
    try:
        version_request = VersionRequest(version=args.version)
        version = client.model_version_api.create(project, args.model_name, version_request)
        print(f"   -> Version '{version_request.version}' created.")
    except Exception as e:
        if "AlreadyExists" in str(e) or "already exists" in str(e).lower() or "409" in str(e):
            print(f"   -> Version '{args.version}' already exists. Proceeding...")
        else:
            print(f"   -> Warning on Create Version: {e}")

    print(f"\n🚀 Starting Upload:")
    print(f"   - Local Source: '{args.local_path}'")
    print(f"   - Target Model: '{args.model_name}:{args.version}'")
    
    try:
        client.file_api.upload_sync(
            project_name=project,
            model_name=args.model_name,
            version_name=args.version, 
            local_path=args.local_path,
            remote_path="/",
            excludes=args.excludes,
            overwrite=True,
            parallel=4, # Using 4 parallel workers for faster upload of large files like safetensors
            progress_callback_func=progress_callback,
            file_complete_callback_func=completion_callback,
        )
        print("\n🎉 Model upload successfully finished!")
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

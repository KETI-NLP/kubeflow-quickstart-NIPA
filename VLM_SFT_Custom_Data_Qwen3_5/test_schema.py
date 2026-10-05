import os
os.environ["MLX_API_KEY"] = os.environ["MLX_API_KEY"]  # Supply credentials through the environment.
os.environ["MLXP_API_KEY"] = os.environ["MLXP_API_KEY"]  # Supply credentials through the environment.
os.environ["MLXP_ENDPOINT_URL"] = "https://kpb4r.mlxp.ncloud.com/"
os.environ["MLX_ENDPOINT_URL"] = "https://kpb4r.mlxp.ncloud.com/"
import warnings
warnings.filterwarnings('ignore')

try:
    from mlx.sdk.data import load_dataset
    ds = load_dataset("YOUR_WORKSPACE/data_13_korean_character-qwen3-packed-sft-ko")
    if 'train' in getattr(ds, '__dict__', {}) or hasattr(ds, 'keys') and 'train' in ds: ds = ds['train']
    print(ds.features)
    print("Keys of first element:", ds[0].keys())
except Exception as e:
    import traceback
    traceback.print_exc()

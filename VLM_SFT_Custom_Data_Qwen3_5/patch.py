import re
with open('/app/3_vlm_sft_distributed.py', 'r') as f: code = f.read()
code = code.replace('from mlx.sdk.data import load_dataset as mlxp_load_dataset', 'from huggingface_hub import snapshot_download\n                    from datasets import load_from_disk as mlxp_load_dataset')
code = code.replace('ds = mlxp_load_dataset(dp)', 'local_dir = snapshot_download(repo_id=dp, repo_type="dataset")\n                    ds = mlxp_load_dataset(local_dir)')
code = code.replace('if isinstance(ds, DatasetDict) and "train" in ds:', 'if (hasattr(ds, "keys") or isinstance(ds, dict)) and "train" in ds:')
code = re.sub(r'fsdp_config=\{.*?\},', 'fsdp_config={"transformer_layer_cls_to_wrap": ["Qwen3_5DecoderLayer"]},', code, flags=re.DOTALL)
with open('/app/3_vlm_sft_distributed.py', 'w') as f: f.write(code)

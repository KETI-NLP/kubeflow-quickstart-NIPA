import os
import json
from datasets import load_dataset
from transformers import AutoProcessor
from mlx.sdk.data import load_dataset as mlxp_load_dataset

# Login to MLXP if needed
api_key = os.environ.get("MLX_APIKEY") or os.environ.get("MLXP_API_KEY")
endpoint = os.environ.get("MLX_ENDPOINT_URL") or os.environ.get("MLXP_ENDPOINT_URL")
if api_key and endpoint:
    from mlx.sdk.data import login
    login(api_key, endpoint)

def test_sample():
    dataset_name = "YOUR_WORKSPACE/data_kr_obj_img_5_hallu_wr_txt_dpo_hf"
    try:
        ds = mlxp_load_dataset(dataset_name)
        if "train" in ds:
            ds = ds["train"]
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    print("Columns:", ds.column_names)
    example = ds[0]
    print("Example sample keys:", example.keys())

    # We can just copy the relevant functions from 7_normalize_filter_to_pvc.py to see the error
    import sys
    sys.path.append("/tmp")
    from importlib.machinery import SourceFileLoader
    normalize_module = SourceFileLoader("normalize", "/tmp/7_normalize_filter_to_pvc.py").load_module()

    processor_name = "/data/checkpoints/qwen3_5_9b_multimodal_sft_lr_1e-5"
    
    # Do what normalize_single_dataset does roughly
    normalized = normalize_module.normalize_multimodal_dpo_example(example)
    print("Images count:", len(normalize_module.ensure_list_images(normalized.get("images"))))
    print("Prompt placeholders:", normalize_module.count_image_placeholders(normalized.get("prompt")))

    processor = normalize_module.get_processor(processor_name)
    prompt_text = normalize_module.build_prompt_text(processor, normalized.get("prompt"))
    print("Prompt text preview:", prompt_text[:200])

    pil_images = [normalize_module.load_image_as_pil(img) for img in normalize_module.ensure_list_images(normalized.get("images"))]
    
    try:
        processor(
            text=[prompt_text],
            images=pil_images,
            padding=False,
            return_tensors="pt",
        )
        print("Success")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sample()

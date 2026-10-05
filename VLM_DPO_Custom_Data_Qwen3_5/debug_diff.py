import sys
sys.path.append("/tmp")
from importlib.machinery import SourceFileLoader
normalize_module = SourceFileLoader("normalize", "/tmp/7_normalize_filter_to_pvc.py").load_module()
from mlx.sdk.data import load_dataset as mlxp_load_dataset
import os
from mlx.sdk.data import login
api_key = os.environ.get("MLX_APIKEY") or os.environ.get("MLXP_API_KEY")
endpoint = os.environ.get("MLX_ENDPOINT_URL") or os.environ.get("MLXP_ENDPOINT_URL")
if api_key and endpoint:
    login(api_key, endpoint)


def main():
    dataset_name = "YOUR_WORKSPACE/data_kr_obj_img_5_hallu_wr_txt_dpo_hf"
    ds = mlxp_load_dataset(dataset_name)["train"]
    ds = ds.select([0])
    train_ds = ds.map(normalize_module.normalize_multimodal_dpo_example)
    
    example = train_ds[0]
    processor = normalize_module.get_processor("/data/checkpoints/qwen3_5_9b_multimodal_sft_lr_1e-5")
    prompt_text = normalize_module.build_prompt_text(processor, example.get("prompt"))
    images = normalize_module.ensure_list_images(example.get("images"))
    pil_images = [normalize_module.load_image_as_pil(img) for img in images]
    
    print("MAPPED PROMPT TEXT:\n", prompt_text)
    print("MAPPED IMAGE COUNT:", len(pil_images))
    print("MAPPED PROMPT IMAGE PAD COUNT:", prompt_text.count("<|image_pad|>"))
    
    # Compare with manual
    example_manual = normalize_module.normalize_multimodal_dpo_example(ds[0])
    prompt_text_manual = normalize_module.build_prompt_text(processor, example_manual.get("prompt"))
    images_manual = normalize_module.ensure_list_images(example_manual.get("images"))
    pil_images_manual = [normalize_module.load_image_as_pil(img) for img in images_manual]
    
    print("MANUAL PROMPT TEXT:\n", prompt_text_manual)
    print("MANUAL IMAGE COUNT:", len(pil_images_manual))
    print("MANUAL PROMPT IMAGE PAD COUNT:", prompt_text_manual.count("<|image_pad|>"))

if __name__ == "__main__":
    main()

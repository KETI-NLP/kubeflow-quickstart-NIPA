import os
import sys

sys.path.append("/tmp")
from importlib.machinery import SourceFileLoader
normalize_module = SourceFileLoader("normalize", "/tmp/7_normalize_filter_to_pvc.py").load_module()

from mlx.sdk.data import load_dataset as mlxp_load_dataset

def main():
    dataset_name = "YOUR_WORKSPACE/data_kr_obj_img_5_hallu_wr_txt_dpo_hf"
    print("Loading dataset...")
    ds = mlxp_load_dataset(dataset_name)["train"]
    ds = ds.select(range(5))
    
    print("Normalizing...")
    train_ds = ds.map(normalize_module.normalize_multimodal_dpo_example, num_proc=1, remove_columns=ds.column_names)
    
    print("Filtering mismatch...")
    filtered_ds = train_ds.filter(normalize_module.has_matching_prompt_images, num_proc=1)
    
    print("Checking processor compatibility...")
    processor_name = "/data/checkpoints/qwen3_5_9b_multimodal_sft_lr_1e-5"
    
    # Manually run is_processor_compatible_example to see the exception
    for i in range(len(filtered_ds)):
        example = filtered_ds[i]
        try:
            processor = normalize_module.get_processor(processor_name)
            prompt_text = normalize_module.build_prompt_text(processor, example.get("prompt"))
            images = normalize_module.ensure_list_images(example.get("images"))
            pil_images = [normalize_module.load_image_as_pil(image_payload) for image_payload in images]
            processor(
                text=[prompt_text],
                images=pil_images,
                padding=False,
                return_tensors="pt",
            )
            print(f"Sample {i} Success")
        except Exception as e:
            import traceback
            print(f"Sample {i} Failed:")
            traceback.print_exc()

if __name__ == "__main__":
    main()

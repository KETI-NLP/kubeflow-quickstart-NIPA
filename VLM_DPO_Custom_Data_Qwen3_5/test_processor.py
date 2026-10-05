import os
import sys
from importlib.machinery import SourceFileLoader
from transformers import AutoProcessor

def main():
    normalize_module = SourceFileLoader('normalize', '/tmp/7_normalize_filter_to_pvc.py').load_module()

    from mlx.sdk.data import load_dataset as mlxp_load_dataset
    from mlx.sdk.data import login

    api_key = os.environ.get('MLXP_API_KEY')
    endpoint = os.environ.get('MLXP_ENDPOINT_URL')
    if api_key and endpoint:
        login(api_key, endpoint)

    ds = mlxp_load_dataset('YOUR_WORKSPACE/data_kr_obj_img_5_hallu_wr_txt_dpo_hf')['train'].select([0])
    
    def test_map(example):
        ex = normalize_module.normalize_multimodal_dpo_example(example)
        ex["prompt_new"] = ex.pop("prompt")
        ex["chosen_new"] = ex.pop("chosen")
        ex["rejected_new"] = ex.pop("rejected")
        ex["images_new"] = ex.pop("images", [])
        return ex

    train_ds = ds.map(test_map, remove_columns=ds.column_names)
    train_ds = train_ds.rename_columns({
        "prompt_new": "prompt",
        "chosen_new": "chosen",
        "rejected_new": "rejected",
        "images_new": "images",
    })
    
    processor = AutoProcessor.from_pretrained('/data/checkpoints/qwen3_5_9b_multimodal_sft_lr_1e-5', trust_remote_code=True)
    
    mapped_prompt = train_ds[0]['prompt']
    print("MAPPED PROMPT:")
    print(mapped_prompt)
    
    try:
        t = processor.apply_chat_template(mapped_prompt, tokenize=False, add_generation_prompt=False)
        print("CHAT TEMPLATE OUTPUT:")
        print(t)
        
        images = normalize_module.ensure_list_images(train_ds[0].get("images"))
        pil_images = [normalize_module.load_image_as_pil(img) for img in images]
        
        print(f"PIL Images count: {len(pil_images)}")
        
        processor(
            text=[t],
            images=pil_images,
            padding=False,
            return_tensors="pt",
        )
        print("PROCESSOR CALL SUCCESSFUL!")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

import os
import sys
from importlib.machinery import SourceFileLoader
from mlx.sdk.data import load_dataset as mlxp_load_dataset
from mlx.sdk.data import login

def main():
    normalize_module = SourceFileLoader('normalize', '/home/user/mlxp/VLM_DPO_Custom_Data_Qwen3_5/7_normalize_filter_to_pvc.py').load_module()

    api_key = os.environ.get('MLXP_API_KEY')
    endpoint = os.environ.get('MLXP_ENDPOINT_URL')
    if api_key and endpoint:
        login(api_key, endpoint)

    ds = mlxp_load_dataset('YOUR_WORKSPACE/data_kr_obj_img_5_hallu_wr_txt_dpo_hf')['train'].select([0])
    train_ds = ds.map(normalize_module.normalize_multimodal_dpo_example, remove_columns=ds.column_names)

    mapped_prompt = train_ds[0]['prompt']
    manual_prompt = normalize_module.normalize_multimodal_dpo_example(ds[0])['prompt']

    print("Are they equal?", mapped_prompt == manual_prompt)
    if mapped_prompt != manual_prompt:
        print("MAPPED:")
        print(mapped_prompt)
        print("MANUAL:")
        print(manual_prompt)

if __name__ == '__main__':
    main()

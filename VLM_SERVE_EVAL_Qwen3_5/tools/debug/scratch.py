import pandas as pd
import glob
import os

base_dir = "./test_final_11_tasks/local_qwen3.5_9b"
tasks = [
    "ai2d_gen",
    "hallusionbench_gen",
]

for task in tasks:
    pattern = f"{base_dir}/{task}/details/Qwen/Qwen3.5-9B/*/*.parquet"
    files = glob.glob(pattern)
    if not files:
        print(f"No files found for {task}")
        continue
    file = max(files, key=os.path.getctime)
    df = pd.read_parquet(file)
    print(f"\n================ {task} ================")
    for i, row in df.iterrows():
        mr = row["model_response"]
        pred_text = mr.get("text", "")
        if isinstance(pred_text, list) and len(pred_text) > 0:
            pred_string = str(pred_text[0])
        else:
            pred_string = str(pred_text)
            
        print(f"--- Sample {i} length: {len(pred_string)} ---")
        if len(pred_string) > 200:
            print("Model Response (Start):", pred_string[:200].replace('\n', ' '))
            print("Model Response (End):", pred_string[-100:].replace('\n', ' '))
        else:
            print("Model Response:", pred_string.replace('\n', ' '))

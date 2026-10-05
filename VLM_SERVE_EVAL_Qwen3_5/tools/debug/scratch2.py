import pandas as pd
import glob
import os

base_dir = "./test_final_11_tasks/local_qwen3.5_9b"
pattern = f"{base_dir}/hallusionbench_gen/details/Qwen/Qwen3.5-9B/*/*.parquet"
files = glob.glob(pattern)
file = max(files, key=os.path.getctime)
df = pd.read_parquet(file)
print(f"================ hallusionbench_gen ================")
for i, row in df.iterrows():
    gold = row["doc"]["gold_index"]
    gt = row["doc"]["choices"][gold]
    print(f"Sample {i} GT: {gt}")

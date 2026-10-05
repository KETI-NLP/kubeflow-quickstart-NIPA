import json, glob, os
base_dir = "./test_final_11_tasks/local_qwen3.5_9b"
pattern = f"{base_dir}/hallusionbench_gen*/results/Qwen/Qwen3.5-9B/*.json"
files = glob.glob(pattern)
if files:
    with open(max(files, key=os.path.getctime), "r") as f:
        data = json.load(f)
    print("Hallusionbench results:", data.get("results"))

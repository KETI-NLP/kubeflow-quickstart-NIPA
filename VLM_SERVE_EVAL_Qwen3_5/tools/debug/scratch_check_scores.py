import json
import glob
import os

base_dir = "./test_final_11_tasks/local_qwen3.5_9b"
tasks = [
    "korean_heritage_name_vqa|0",
    "korean_character_ocr|0",
    "ifeval_ko_gen|0"
]

for task in tasks:
    pattern = f"{base_dir}/{task}/results/Qwen/Qwen3.5-9B/*.json"
    files = glob.glob(pattern)
    if not files:
        print(f"No results found for {task}")
        continue
    file = max(files, key=os.path.getctime)
    with open(file, "r") as f:
        data = json.load(f)
    print(f"========== {task} ==========")
    for k, v in data.get("results", {}).items():
        if k != "all":
            for m, s in v.items():
                if isinstance(s, float) or isinstance(s, int):
                    print(f"  {m}: {s}")

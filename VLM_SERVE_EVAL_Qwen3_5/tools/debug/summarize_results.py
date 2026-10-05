import json
import glob
import os

base_dir = "./final_11_tasks_run/local_qwen3.5_9b"
tasks = [
    "mmbench_gen",
    "scienceqa_gen",
    "mathvista_gen",
    "chartqa_gen",
    "ai2d_gen",
    "hallusionbench_gen",
    "korean_heritage_name_vqa",
    "korean_character_ocr",
    "kmmlu_gen",
    "hae_rae_bench_gen",
    "ifeval_ko_gen"
]

report = "========== RESULTS ==========\n"

for task in tasks:
    # Use glob to match task directory with or without |0
    pattern = f"{base_dir}/{task}*/results/Qwen/Qwen3.5-9B/*.json"
    files = glob.glob(pattern)
    if not files:
        continue
    file = max(files, key=os.path.getctime)
    with open(file, "r") as f:
        data = json.load(f)
        
    for k, v in data.get("results", {}).items():
        if k != "all":
            report += f"Task: {k}\n"
            for m, s in v.items():
                if isinstance(s, float) or isinstance(s, int):
                    report += f"  {m}: {s}\n"
            report += "------------------------------\n"

with open("final_report.txt", "w") as f:
    f.write(report)
print(report)

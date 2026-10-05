import json
import glob
import os

base_dir = "./test_3_samples_run/local_qwen3.5_9b_vlm"
results = {}

for task_dir in os.listdir(base_dir):
    full_dir = os.path.join(base_dir, task_dir, "results", "Qwen", "Qwen3.5-9B")
    if os.path.exists(full_dir):
        json_files = glob.glob(f"{full_dir}/*.json")
        if json_files:
            latest_file = max(json_files, key=os.path.getctime)
            try:
                with open(latest_file, "r") as f:
                    data = json.load(f)
                    for task, metrics in data.get("results", {}).items():
                        if task != "all":
                            results[task] = metrics
            except Exception as e:
                pass

for task, metrics in sorted(results.items()):
    print(f"=== {task} ===")
    for metric_name, value in metrics.items():
        if not metric_name.endswith("_stderr"):
            print(f"  {metric_name}: {value}")

import json
import glob
import os
from collections import defaultdict
from datetime import datetime

base_dir = "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/test_final_11_tasks/local_qwen3.5_9b"
results = defaultdict(list)

for file_path in glob.glob(f"{base_dir}/*/results/Qwen/Qwen3.5-9B/results_*.json"):
    task_dir = file_path.split("/")[-5]
    filename = os.path.basename(file_path)
    try:
        ts_str = filename.replace("results_", "").replace(".json", "")
        dt = datetime.strptime(ts_str.split(".")[0], "%Y-%m-%dT%H-%M-%S")
    except Exception:
        continue
        
    with open(file_path, "r") as f:
        data = json.load(f)
    
    metrics = data.get("results", {})
    if not metrics:
        continue
        
    task_name = list(metrics.keys())[0]
    task_metrics = metrics[task_name]
    
    primary_metric = None
    primary_value = None
    for k, v in task_metrics.items():
        if isinstance(v, float) and "stderr" not in k and "std" not in k:
            primary_metric = k
            primary_value = v
            break
            
    if primary_metric:
        results[task_dir].append((dt, primary_value, primary_metric))

print("COMPARISON OF RECENT RUN (NO-THINK) VS PREVIOUS RUNS:")
print(f"| {'Task':<28} | {'Previous':<10} | {'No-Think':<10} | {'Metric':<25} |")
print("-" * 85)

for task_dir, runs in sorted(results.items()):
    runs.sort(key=lambda x: x[0])
    if len(runs) >= 2:
        latest_run = runs[-1]
        previous_run = runs[-2]
        print(f"| {task_dir:<28} | {previous_run[1]:<10.4f} | {latest_run[1]:<10.4f} | {latest_run[2]:<25} |")
    elif len(runs) == 1:
        latest_run = runs[-1]
        print(f"| {task_dir:<28} | {'N/A':<10} | {latest_run[1]:<10.4f} | {latest_run[2]:<25} |")

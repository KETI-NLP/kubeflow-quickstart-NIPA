import pandas as pd
import glob
import os
import json
import numpy as np

base_dir = "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/test_final_11_tasks/local_qwen3.5_9b"
tasks = [
    "ai2d_gen", "chartqa_gen", "hae_rae_bench_gen", "hallusionbench_gen",
    "ifeval_ko_gen", "kmmlu_gen", "korean_character_ocr", 
    "korean_heritage_name_vqa", "mathvista_gen", "mmbench_gen", "scienceqa_gen"
]

report = "# 벤치마크 샘플 결과 리포트 (no-think 적용)\n\n"
report += "아래는 최근 벤치마크 실행 결과에서 각 태스크별로 하나의 샘플을 추출하여 정리한 마크다운 리포트입니다.\n\n"

def safe_str(val):
    if isinstance(val, np.ndarray):
        val = val.tolist()
    if isinstance(val, (dict, list)):
        return json.dumps(val, ensure_ascii=False, indent=2)
    return str(val)

for task in tasks:
    file_paths = glob.glob(f"{base_dir}/{task}/details/Qwen/Qwen3.5-9B/*/*.parquet")
    if not file_paths:
        continue
    
    file_paths.sort()
    latest_file = file_paths[-1]
    
    try:
        df = pd.read_parquet(latest_file)
        if len(df) == 0:
            continue
            
        sample = df.iloc[0]
        
        report += f"## Task: `{task}`\n\n"
        
        prompt = safe_str(sample.get("prompt", "N/A"))
        gold = safe_str(sample.get("gold", "N/A"))
        
        # Output is often in 'outputs'
        outputs = sample.get("outputs", {})
        if isinstance(outputs, dict) and "text" in outputs:
            prediction = safe_str(outputs["text"])
        elif isinstance(outputs, (list, np.ndarray)) and len(outputs) > 0:
            if isinstance(outputs[0], dict) and "text" in outputs[0]:
                prediction = safe_str(outputs[0]["text"])
            else:
                prediction = safe_str(outputs[0])
        else:
            prediction = safe_str(outputs)
            
        # Metrics
        metrics = sample.get("metrics", {})
        if isinstance(metrics, dict):
            metrics_str = "\n".join([f"- **{k}**: {v}" for k, v in metrics.items()])
        else:
            metrics_str = safe_str(metrics)
            
        report += f"### 📥 Input (입력)\n```text\n{prompt}\n```\n\n"
        report += f"### 🎯 Ground Truth (정답)\n```text\n{gold}\n```\n\n"
        report += f"### 🤖 Prediction (모델 출력)\n```text\n{prediction}\n```\n\n"
        report += f"### 📊 Metrics (채점 결과)\n{metrics_str}\n\n"
        report += "---\n\n"
        
    except Exception as e:
        report += f"Error processing task {task}: {e}\n\n"

with open("/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/sample_report.md", "w") as f:
    f.write(report)

print("Report generated successfully.")

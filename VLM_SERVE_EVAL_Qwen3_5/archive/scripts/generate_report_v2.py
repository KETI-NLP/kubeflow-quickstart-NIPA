import pandas as pd
import glob
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

def default_encoder(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)

def safe_str(val):
    if isinstance(val, (dict, list, tuple)):
        return json.dumps(val, ensure_ascii=False, indent=2, default=default_encoder)
    elif isinstance(val, np.ndarray):
        return json.dumps(val.tolist(), ensure_ascii=False, indent=2, default=default_encoder)
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
            
        sample = df.iloc[0].to_dict()
        
        report += f"## Task: `{task}`\n\n"
        
        doc = sample.get("doc", {})
        if isinstance(doc, dict):
            prompt = safe_str(doc.get("query", doc.get("instruction", doc)))
            gold = safe_str(doc.get("choices", doc.get("target", doc.get("answer", doc))))
        else:
            prompt = safe_str(doc)
            gold = safe_str(doc)
            
        prediction = safe_str(sample.get("model_response", "N/A"))
        metrics_str = safe_str(sample.get("metric", "N/A"))
            
        report += f"### 📥 Input (입력)\n```text\n{prompt}\n```\n\n"
        report += f"### 🎯 Ground Truth (정답)\n```text\n{gold}\n```\n\n"
        report += f"### 🤖 Prediction (모델 출력)\n```json\n{prediction}\n```\n\n"
        report += f"### 📊 Metrics (채점 결과)\n```json\n{metrics_str}\n```\n\n"
        report += "---\n\n"
        
    except Exception as e:
        report += f"Error processing task {task}: {e}\n\n"

with open("/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/sample_report_v2.md", "w") as f:
    f.write(report)

print("Report generated successfully.")

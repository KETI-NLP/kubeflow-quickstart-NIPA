import pandas as pd
import glob
import json
import numpy as np
import re

base_dir = "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/test_final_11_tasks/local_qwen3.5_9b"
tasks = [
    "ai2d_gen", "chartqa_gen", "hae_rae_bench_gen", "hallusionbench_gen",
    "ifeval_ko_gen", "kmmlu_gen", "korean_character_ocr", 
    "korean_heritage_name_vqa", "mathvista_gen", "mmbench_gen", "scienceqa_gen"
]

report = "# 벤치마크 샘플 결과 리포트 (no-think 적용)\n\n"
report += "최근 벤치마크 실행 결과에서 각 태스크별로 최대 3개의 샘플을 추출하여 정리한 리포트입니다.\n\n"

def safe_str(val):
    if isinstance(val, np.ndarray):
        val = val.tolist()
    if isinstance(val, (list, dict)):
        try:
            return json.dumps(val, ensure_ascii=False)
        except:
            return str(val)
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
            
        report += f"## 📚 Task: `{task}`\n\n"
        
        num_samples = min(3, len(df))
        
        for i in range(num_samples):
            sample = df.iloc[i].to_dict()
            
            doc = sample.get("doc", {})
            if isinstance(doc, dict):
                prompt = doc.get("query", doc.get("instruction", str(doc)))
                gold = doc.get("choices", doc.get("target", doc.get("answer", str(doc))))
            else:
                prompt = str(doc)
                gold = str(doc)
                
            prompt = safe_str(prompt)
            prompt = re.sub(r'<\|image_pad\|>', '', prompt)
            
            gold = safe_str(gold)
                
            model_response = sample.get("model_response", {})
            if isinstance(model_response, dict):
                prediction = model_response.get("text", model_response)
            else:
                prediction = model_response
            
            prediction = safe_str(prediction)    
            prediction = prediction.replace('<|im_start|>assistant', '').strip()
                
            metrics = sample.get("metric", {})
            clean_metrics = {}
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    if isinstance(v, (np.float32, np.float64)):
                        clean_metrics[k] = float(v)
                    elif isinstance(v, (np.int32, np.int64)):
                        clean_metrics[k] = int(v)
                    else:
                        clean_metrics[k] = v
                metrics_str = json.dumps(clean_metrics, ensure_ascii=False, indent=2)
            else:
                metrics_str = safe_str(metrics)
                
            report += f"### 🔹 Sample {i+1}\n"
            report += f"**📥 Input (입력)**\n```text\n{prompt[:1000]}\n```\n\n"
            report += f"**🎯 Ground Truth (정답)**\n```text\n{gold}\n```\n\n"
            report += f"**🤖 Prediction (모델 출력)**\n```text\n{prediction}\n```\n\n"
            report += f"**📊 Metrics (채점 결과)**\n```json\n{metrics_str}\n```\n\n"
            report += "---\n\n"
            
    except Exception as e:
        report += f"Error processing task {task}: {e}\n\n"

with open("/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/sample_report_v5.md", "w") as f:
    f.write(report)

print("Report generated successfully.")

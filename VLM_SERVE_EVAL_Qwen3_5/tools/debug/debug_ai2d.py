import pandas as pd
import glob
import json

base_dir = "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/test_final_11_tasks/local_qwen3.5_9b"
file_paths = glob.glob(f"{base_dir}/ai2d_gen/details/Qwen/Qwen3.5-9B/*/*.parquet")
file_paths.sort()
latest_file = file_paths[-1]

print(f"Reading from {latest_file}...")
df = pd.read_parquet(latest_file)
print("Loaded. Inspecting sample 1:")

sample = df.iloc[0].to_dict()
doc = sample.get("doc", {})

print(f"doc keys: {doc.keys()}")
print(f"choices: {doc.get('choices')}")
print(f"gold_index: {doc.get('gold_index')} (type: {type(doc.get('gold_index'))})")
print(f"target: {doc.get('target')}")
print(f"answer: {doc.get('answer')}")

metrics = sample.get("metric", {})
print(f"metric: {metrics}")

model_response = sample.get("model_response", {})
if isinstance(model_response, dict):
    prediction = model_response.get("text", model_response)
else:
    prediction = model_response
print(f"prediction: {prediction}")

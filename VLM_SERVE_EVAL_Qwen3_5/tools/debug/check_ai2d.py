import pandas as pd

file_path = "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/test_new_benchmarks_run/local_qwen3.5_9b_vlm/ai2d_gen/details/Qwen/Qwen3.5-9B/2026-04-30T10-32-24.722779/details_ai2d_gen:default|0_2026-04-30T10-32-24.722779.parquet"
df = pd.read_parquet(file_path)

# find where metrics.extractive_match == 0
failed = []
for i in range(len(df)):
    metric = df.iloc[i]['metrics']
    if metric.get('extractive_match', 1) == 0:
        failed.append(i)
        if len(failed) >= 3:
            break

for idx in failed:
    print(f"--- Sample {idx} ---")
    print(f"Gold: {df.iloc[idx]['gold']}")
    print(f"Choices: {df.iloc[idx].get('choices', 'N/A')}")
    try:
        print(f"Model Response: {df.iloc[idx]['model_response']}")
    except:
        pass
    print("="*40)

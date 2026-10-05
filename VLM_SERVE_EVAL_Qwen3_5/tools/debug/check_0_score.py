import pyarrow.parquet as pq

chartqa_path = "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/test_new_benchmarks_run/local_qwen3.5_9b/chartqa_gen/details/Qwen/Qwen3.5-9B/2026-04-30T05-00-11.340445/details_chartqa_gen:default|0_2026-04-30T05-00-11.340445.parquet"
hallusion_path = "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/test_new_benchmarks_run/local_qwen3.5_9b/hallusionbench_gen/details/Qwen/Qwen3.5-9B/2026-04-30T07-16-57.035858/details_hallusionbench_gen:default|0_2026-04-30T07-16-57.035858.parquet"

print("=== ChartQA ===")
table = pq.read_table(chartqa_path)
df = table.to_pandas()
print(f"Total rows: {len(df)}")
cols = df.columns.tolist()
print(f"Columns: {cols}")

print("\nSample 5 rows:")
for idx, row in df.head(5).iterrows():
    print(f"--- Row {idx} ---")
    print(f"Target: {row.get('target', 'N/A')}")
    
    resps = row.get('resps', [])
    if isinstance(resps, list) and len(resps) > 0:
        if isinstance(resps[0], list) and len(resps[0]) > 0:
            print(f"Response: {resps[0][0]}")
        else:
            print(f"Response: {resps[0]}")
    else:
        print(f"Response (raw): {resps}")
        
    print(f"Filtered Resps: {row.get('filtered_resps', 'N/A')}")
    print(f"Exact Match: {row.get('exact_match', 'N/A')}")
    print("")

print("=== HallusionBench ===")
table = pq.read_table(hallusion_path)
df = table.to_pandas()
print(f"Total rows: {len(df)}")
cols = df.columns.tolist()
print(f"Columns: {cols}")

print("\nSample 5 rows:")
for idx, row in df.head(5).iterrows():
    print(f"--- Row {idx} ---")
    print(f"Target: {row.get('target', 'N/A')}")
    
    resps = row.get('resps', [])
    if isinstance(resps, list) and len(resps) > 0:
        if isinstance(resps[0], list) and len(resps[0]) > 0:
            print(f"Response: {resps[0][0]}")
        else:
            print(f"Response: {resps[0]}")
    else:
        print(f"Response (raw): {resps}")
        
    print(f"Filtered Resps: {row.get('filtered_resps', 'N/A')}")
    print(f"Extractive Match: {row.get('extractive_match', 'N/A')}")
    print("")

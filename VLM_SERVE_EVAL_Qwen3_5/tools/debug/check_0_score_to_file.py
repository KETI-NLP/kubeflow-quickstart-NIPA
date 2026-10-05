import pyarrow.parquet as pq

chartqa_path = "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/test_new_benchmarks_run/local_qwen3.5_9b_vlm/chartqa_gen/details/Qwen/Qwen3.5-9B/2026-05-01T03-58-14.476340/details_chartqa_gen:default|0_2026-05-01T03-58-14.476340.parquet"
hallusion_path = "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/test_new_benchmarks_run/local_qwen3.5_9b_vlm/hallusionbench_gen/details/Qwen/Qwen3.5-9B/2026-05-01T04-01-09.216896/details_hallusionbench_gen:default|0_2026-05-01T04-01-09.216896.parquet"

with open("output_0_score.txt", "w") as f:
    f.write("=== ChartQA ===\n")
    table = pq.read_table(chartqa_path)
    df = table.to_pandas()
    f.write(f"Total rows: {len(df)}\n")
    cols = df.columns.tolist()
    f.write(f"Columns: {cols}\n\n")

    f.write("Sample 5 rows:\n")
    for idx, row in df.head(5).iterrows():
        f.write(f"--- Row {idx} ---\n")
        f.write(f"Doc: {row.get('doc', 'N/A')}\n")
        
        resps = row.get('model_response', [])
        if isinstance(resps, list) and len(resps) > 0:
            if isinstance(resps[0], list) and len(resps[0]) > 0:
                f.write(f"Response: {resps[0][0]}\n")
            else:
                f.write(f"Response: {resps[0]}\n")
        else:
            f.write(f"Response (raw): {resps}\n")
            
        f.write(f"Metric: {row.get('metric', 'N/A')}\n\n")

    f.write("=== HallusionBench ===\n")
    table = pq.read_table(hallusion_path)
    df = table.to_pandas()
    f.write(f"Total rows: {len(df)}\n")
    cols = df.columns.tolist()
    f.write(f"Columns: {cols}\n\n")

    f.write("Sample 5 rows:\n")
    for idx, row in df.head(5).iterrows():
        f.write(f"--- Row {idx} ---\n")
        f.write(f"Doc: {row.get('doc', 'N/A')}\n")
        
        resps = row.get('model_response', [])
        if isinstance(resps, list) and len(resps) > 0:
            if isinstance(resps[0], list) and len(resps[0]) > 0:
                f.write(f"Response: {resps[0][0]}\n")
            else:
                f.write(f"Response: {resps[0]}\n")
        else:
            f.write(f"Response (raw): {resps}\n")
            
        f.write(f"Metric: {row.get('metric', 'N/A')}\n\n")

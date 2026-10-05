import pyarrow.parquet as pq

chartqa_path = "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/test_new_benchmarks_run/local_qwen3.5_9b/chartqa_gen/details/Qwen/Qwen3.5-9B/2026-04-30T05-00-11.340445/details_chartqa_gen:default|0_2026-04-30T05-00-11.340445.parquet"
hallusion_path = "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/test_new_benchmarks_run/local_qwen3.5_9b/hallusionbench_gen/details/Qwen/Qwen3.5-9B/2026-04-30T07-16-57.035858/details_hallusionbench_gen:default|0_2026-04-30T07-16-57.035858.parquet"

with open("output_0_score.txt", "w") as f:
    f.write("=== ChartQA ===\n")
    table = pq.read_table(chartqa_path)
    df = table.to_pandas()
    for idx, row in df.head(5).iterrows():
        f.write(f"--- Row {idx} ---\n")
        f.write(f"Model Response: {row.get('model_response')}\n")
        f.write(f"Metric: {row.get('metric')}\n\n")

    f.write("=== HallusionBench ===\n")
    table = pq.read_table(hallusion_path)
    df = table.to_pandas()
    for idx, row in df.head(5).iterrows():
        f.write(f"--- Row {idx} ---\n")
        f.write(f"Model Response: {row.get('model_response')}\n")
        f.write(f"Metric: {row.get('metric')}\n\n")

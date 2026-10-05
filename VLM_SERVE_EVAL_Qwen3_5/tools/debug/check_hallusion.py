import pyarrow.parquet as pq
hallusion_path = '/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/test_new_benchmarks_run/local_qwen3.5_9b_vlm/hallusionbench_gen/details/Qwen/Qwen3.5-9B/2026-05-01T03-20-00.395358/details_hallusionbench_gen:default|0_2026-05-01T03-20-00.395358.parquet'
df = pq.read_table(hallusion_path).to_pandas()
for i, row in df.iterrows():
    print(f'Row {i}:', row['model_response'][0]['text'])

import pyarrow.parquet as pq

file_path = "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/test_new_benchmarks_run/local_qwen3.5_9b_vlm/ai2d_gen/details/Qwen/Qwen3.5-9B/2026-04-30T10-32-24.722779/details_ai2d_gen:default|0_2026-04-30T10-32-24.722779.parquet"
table = pq.read_table(file_path, columns=["gold", "choices", "model_response", "metrics"])

failed = 0
for i in range(table.num_rows):
    metrics = table["metrics"][i].as_py()
    if metrics.get("extractive_match", 1) == 0:
        print(f"--- Sample {i} ---")
        print(f"Gold: {table['gold'][i].as_py()}")
        print(f"Choices: {table['choices'][i].as_py()}")
        print(f"Model Response: {table['model_response'][i].as_py()}")
        print("="*40)
        failed += 1
        if failed >= 3:
            break

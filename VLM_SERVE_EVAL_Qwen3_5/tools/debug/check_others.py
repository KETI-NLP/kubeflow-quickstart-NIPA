import pyarrow.parquet as pq

def check_responses(file_path):
    try:
        table = pq.read_table(file_path)
        df = table.to_pandas()
        sample = df['model_response'].head(5).tolist()
        print(f"File: {file_path.split('/')[-1]}")
        for i, resp in enumerate(sample):
            print(f"  {i}: {resp}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

check_responses("/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/test_new_benchmarks_run/local_qwen3.5_9b_vlm/ai2d_gen/details/Qwen/Qwen3.5-9B/2026-04-30T10-32-24.722779/details_ai2d_gen:default|0_2026-04-30T10-32-24.722779.parquet")
check_responses("/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/test_new_benchmarks_run/local_qwen3.5_9b_vlm/mathvista_gen/details/Qwen/Qwen3.5-9B/2026-04-30T10-27-40.382966/details_mathvista_gen:default|0_2026-04-30T10-27-40.382966.parquet")

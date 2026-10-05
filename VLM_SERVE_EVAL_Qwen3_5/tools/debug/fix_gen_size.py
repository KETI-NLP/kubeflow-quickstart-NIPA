import os
import glob

files = [
    "custom_ai2d_task.py",
    "custom_mathvista_task.py",
    "custom_mmbench_task.py",
    "custom_scienceqa_task.py",
    "custom_mmmu_task.py",
]

for fname in files:
    path = f"/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/custom_tasks/{fname}"
    if not os.path.exists(path): continue
    
    with open(path, "r") as f:
        content = f.read()
        
    if "generation_size=" not in content:
        content = content.replace("version=1,", "version=1,\n    generation_size=2048,")
        with open(path, "w") as f:
            f.write(content)
        print(f"Fixed {fname}")

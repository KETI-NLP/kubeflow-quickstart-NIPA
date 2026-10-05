import os
import glob

files = glob.glob("/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/custom_tasks/custom_*_task.py")
for f in files:
    with open(f, "r") as file:
        content = file.read()
    
    if "import custom_tasks.force_no_think_patch" not in content:
        # Prepend after from __future__ import annotations or at the top
        if "from __future__ import annotations" in content:
            content = content.replace("from __future__ import annotations\n", "from __future__ import annotations\nimport custom_tasks.force_no_think_patch\n")
        else:
            content = "import custom_tasks.force_no_think_patch\n" + content
            
        with open(f, "w") as file:
            file.write(content)
        print(f"Patched {f}")

# Fix ScienceQA
sqa_path = "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/custom_tasks/custom_scienceqa_task.py"
with open(sqa_path, "r") as f:
    sqa_content = f.read()

# Remove filter_fn
sqa_content = sqa_content.replace("    filter_fn=filter_images,\n", "")

# Add dummy image logic
old_image_logic = """    images = []
    if line.get("image") is not None:
        img = line["image"]
        if isinstance(img, str):
            img = Image.open(img)
        images.append(img)"""

new_image_logic = """    images = []
    if line.get("image") is not None:
        img = line["image"]
        if isinstance(img, str):
            img = Image.open(img)
        images.append(img)
    else:
        # Dummy 1x1 image for text-only samples so VLM pipeline doesn't crash
        images.append(Image.new('RGB', (1, 1), color='white'))"""

sqa_content = sqa_content.replace(old_image_logic, new_image_logic)

with open(sqa_path, "w") as f:
    f.write(sqa_content)
print("Fixed custom_scienceqa_task.py dummy image logic.")


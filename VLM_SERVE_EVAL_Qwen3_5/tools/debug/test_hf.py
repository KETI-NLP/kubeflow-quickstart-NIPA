from huggingface_hub import list_repo_files
files = list_repo_files("AI4Math/MathVista", repo_type="dataset")
print([f for f in files if "images/34.jpg" in f])

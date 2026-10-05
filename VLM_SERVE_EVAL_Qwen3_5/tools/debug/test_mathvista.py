from datasets import load_dataset
ds = load_dataset("AI4Math/MathVista", "default", split="testmini", trust_remote_code=True)
for i in range(100):
    if ds[i]['decoded_image'] is None:
        print(f"Index {i} has None decoded_image")

from datasets import load_dataset
ds_h = load_dataset("lmms-lab/HallusionBench", "default", split="image", trust_remote_code=True)
print("HallusionBench features:", ds_h.features)
print("HallusionBench row 0:", ds_h[0])

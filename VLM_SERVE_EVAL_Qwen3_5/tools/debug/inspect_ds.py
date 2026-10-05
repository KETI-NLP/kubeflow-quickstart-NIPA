from datasets import load_dataset
ds_h = load_dataset("lmms-lab/HallusionBench", "default", split="image", trust_remote_code=True)
print("HallusionBench features:", ds_h.features)
print("HallusionBench sample:", ds_h[0].keys())

ds_m = load_dataset("AI4Math/MathVista", "default", split="testmini", trust_remote_code=True)
print("MathVista features:", ds_m.features)
print("MathVista sample keys:", ds_m[0].keys())
print("MathVista sample image type:", type(ds_m[0]['image']))
print("MathVista sample image:", ds_m[0]['image'])

from mlx.sdk.data import load_dataset
ds = load_dataset("YOUR_WORKSPACE/data_13_korean_character-qwen3-packed-sft-ko")
if hasattr(ds, "keys") and "train" in ds:
    train_ds = ds["train"]
else:
    train_ds = ds
print("Type:", type(train_ds))
print("Len:", len(train_ds))
try:
    print("Item 0 Keys:", train_ds[0].keys())
except Exception as e:
    print("Error getting Item 0 Keys:", e)
from datasets import concatenate_datasets
combined = concatenate_datasets([train_ds])
print("Combined Type:", type(combined))
try:
    print("Combined Item 0 Keys:", combined[0].keys())
except Exception as e:
    print("Error getting Combined Item 0 Keys:", e)

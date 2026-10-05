import glob
from pathlib import Path

for path in glob.glob("custom_tasks/custom_*_task.py"):
    text = Path(path).read_text()
    
    # Remove trust_dataset
    text = text.replace("trust_dataset=True,", "")
    text = text.replace("trust_dataset=True", "")
    
    # Fix metric
    text = text.replace("Metrics.quasi_exact_match_gsm8k", "Metrics.exact_match")
    
    Path(path).write_text(text)
    print(f"Patched {path}")

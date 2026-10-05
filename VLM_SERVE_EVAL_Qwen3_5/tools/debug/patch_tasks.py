import glob
from pathlib import Path

for path in glob.glob("custom_tasks/custom_*_task.py"):
    text = Path(path).read_text()
    
    # Add trust_dataset=True to LightevalTaskConfig
    if "trust_dataset=" not in text:
        text = text.replace("version=1,", "version=1,\n    trust_dataset=True,")
        
    # Ensure image is PIL
    if "from PIL import Image" not in text:
        text = "from PIL import Image\n" + text
        
    text = text.replace("images.append(line[\"image\"])", "img = line[\"image\"]\n        if isinstance(img, str):\n            img = Image.open(img)\n        images.append(img)")
    
    Path(path).write_text(text)
    print(f"Patched {path}")

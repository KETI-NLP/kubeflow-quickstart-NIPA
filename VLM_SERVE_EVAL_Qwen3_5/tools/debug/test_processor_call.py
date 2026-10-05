import traceback
from transformers import AutoProcessor
from PIL import Image

processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-9B", trust_remote_code=True)
texts = ["<|im_start|>user\nLook at the chart.<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n"]
images = [[Image.new("RGB", (224, 224))]]

try:
    inputs = processor(text=texts, images=images, return_tensors="pt")
    print("Inputs keys:", inputs.keys())
except Exception as e:
    print("Error processing:")
    traceback.print_exc()

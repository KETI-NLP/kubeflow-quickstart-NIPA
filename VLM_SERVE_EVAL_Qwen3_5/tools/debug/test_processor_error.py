import traceback
from transformers import AutoProcessor

try:
    processor = AutoProcessor.from_pretrained('Qwen/Qwen3.5-9B', trust_remote_code=True)
    print("Processor loaded successfully!")
except Exception as e:
    print("Error loading processor:")
    traceback.print_exc()

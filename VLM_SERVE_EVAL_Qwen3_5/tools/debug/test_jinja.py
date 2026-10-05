import json
from jinja2 import Environment, meta

with open('/workspace/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a/tokenizer_config.json', 'r') as f:
    config = json.load(f)
    
template = config.get('chat_template')
print("Contains 'enable_thinking':", "enable_thinking" in template)

# Let's extract the part at the very end
print("\nEnd of template:")
print(template[-200:])

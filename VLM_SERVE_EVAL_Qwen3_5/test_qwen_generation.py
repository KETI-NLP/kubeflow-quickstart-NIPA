import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "/workspace/local_models/qwen3_5_9b_multimodal_sft_dpo_normalized/checkpoint-1253"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="cuda:0", 
    torch_dtype=torch.float16, 
    trust_remote_code=True
)

template_path = "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/no_think_chat_template.jinja"
with open(template_path, "r") as f:
    tokenizer.chat_template = f.read()

messages = [
    {"role": "user", "content": [{"type": "text", "text": "안녕? 반가워!"}]}
]

prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
print("--- PROMPT ---")
print(prompt)
print("--------------")

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=False)

print("--- RESPONSE ---")
print(response)
print("----------------")

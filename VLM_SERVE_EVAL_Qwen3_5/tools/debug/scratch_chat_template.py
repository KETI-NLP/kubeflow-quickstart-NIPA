from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-9B')

messages = [
    {"role": "user", "content": "hello\n\nCRITICAL INSTRUCTION: DO NOT output any thinking process, reasoning, or explanations. DO NOT output 'Thinking Process:'. Output ONLY the final answer."}
]

print("--- Default chat template applied ---")
print(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))


import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
import sys

def main():
    model_name = "Qwen/Qwen3-VL-8B-Instruct"
    print(f"Loading processor...")
    processor = AutoProcessor.from_pretrained(model_name)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        
    print(f"Loading model (bfloat16, eager)...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="cuda:0"
    )

    text = "Instruction: hello\nAssistant: hi"
    print("Processing text...")
    inputs = processor(text=[text], padding="max_length", max_length=128, truncation=True, return_tensors="pt").to("cuda:0")
    
    print(f"Processor keys: {list(inputs.keys())}")
    
    labels = inputs["input_ids"].clone()
    labels[inputs["attention_mask"] == 0] = -100
    inputs["labels"] = labels
    
    model.gradient_checkpointing_enable()
    model.train()
    print("Running forward pass...")
    try:
        outputs = model(**inputs)
        loss = outputs.loss
        print(f"Loss: {loss.item()}")
        
        print("Running backward pass...")
        loss.backward()
        
        has_nan = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"NaN gradient found in: {name}")
                    has_nan = True
                    break
        if not has_nan:
            print("No NaN gradients found!")
    except Exception as e:
        print(f"Exception during forward/backward: {e}")

if __name__ == "__main__":
    main()

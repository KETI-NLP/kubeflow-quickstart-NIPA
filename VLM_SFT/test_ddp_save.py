import os
import torch.distributed as dist
from transformers import TrainingArguments

# Mock DDP
os.environ["RANK"] = "1"
os.environ["WORLD_SIZE"] = "2"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"
dist.init_process_group("gloo")

args = TrainingArguments(
    output_dir="./test_ddp", 
    save_strategy="epoch", 
    save_on_each_node=False
)
print(f"Rank 1 should_save: {args.should_save}")

import argparse
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
import torch.distributed as dist
import torch.distributed.fsdp as fsdp
if not hasattr(fsdp, "FSDPModule"):
    fsdp.FSDPModule = type("FSDPModule", (object,), {})
if not hasattr(fsdp, "register_fsdp_forward_method"):
    fsdp.register_fsdp_forward_method = lambda *args, **kwargs: None

import transformers.models.auto.modeling_auto as modeling_auto
if not hasattr(modeling_auto, "MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES"):
    modeling_auto.MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = {}

from datasets import load_from_disk
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, TrainingArguments
from trl import DPOTrainer, DPOConfig

def main():
    parser = argparse.ArgumentParser(description="LLM DPO Distributed Training")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen1.5-0.5B", help="HuggingFace model name")
    parser.add_argument("--data_path", type=str, default="/data/dpo_dataset", help="Path to local or PVC dataset")
    parser.add_argument("--output_path", type=str, default="/data/result/llm-dpo", help="Path to save model")
    args = parser.parse_args()

    # 1. Load dataset from PVC
    print(f"Loading dataset from {args.data_path} ...")
    dataset = load_from_disk(args.data_path)
    
    # Filter anomalously long samples to absolutely prevent OOM SIGABRT
    dataset["train"] = dataset["train"].filter(lambda x: len(x["prompt"]) + len(x["chosen"]) < 3000)
    dataset["test"] = dataset["test"].filter(lambda x: len(x["prompt"]) + len(x["chosen"]) < 3000)

    # Distributed setup
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
    else:
        rank = 0

    # Configure Training Arguments for FSDP
    training_args = DPOConfig(
        output_dir=args.output_path,
        per_device_train_batch_size=1,            # 2 -> 1 (OOM 방지)
        gradient_accumulation_steps=16,           # 8 -> 16 (배치사이즈 유지)
        num_train_epochs=1,
        logging_dir="/data/log/llm-dpo",
        logging_steps=10,
        save_strategy="epoch",
        fsdp="full_shard auto_wrap",              # CPU offload removed to prevent 10m NCCL ALLREDUCE PCIe Timeout bottlenecks
        fsdp_config={
            "transformer_layer_cls_to_wrap": "Qwen3VLTextDecoderLayer",
            "activation_checkpointing": True      # Proper FSDP-compatible Checkpointing (ignores standard TrainingArguments)
        },
        gradient_checkpointing=False,             # Explicitly suppress HF Trainer overlap crash
        ddp_find_unused_parameters=False,
        beta=0.1,
    )

    # Instantiate components simultaneously across all ranks to prevent PyTorch synchronization timeouts
    processor = AutoProcessor.from_pretrained(args.model_name)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "bos_token", None) is None:
        tokenizer.bos_token = tokenizer.eos_token
        
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"
    )
    ref_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"
    )
    
    # Freeze visual layers to definitively bypass FSDP/DDP 'unused parameter' crashes during backward reduction
    for name, param in model.named_parameters():
        if "visual" in name:
            param.requires_grad = False
    for name, param in ref_model.named_parameters():
        if "visual" in name:
            param.requires_grad = False
    
    # Move models to the correct device manually before wrapping to prevent FSDP device drift.
    model = model.to(torch.cuda.current_device())
    ref_model = ref_model.to(torch.cuda.current_device())

    # Deprecated: standard gradient_checkpointing is redundant with FSDP activation_checkpointing
    # model.gradient_checkpointing_enable()

    # Initialize DPOTrainer outside the blocking context manager so all ranks hit the NCCL wrappers simultaneously
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        # max_length kwargs not supported natively by this TRL version
    )

    # Move ref_model to correct device to prevent "cpu and cuda" FSDP device errors
    if getattr(trainer, "ref_model", None) is not None:
        trainer.ref_model = trainer.ref_model.to(trainer.accelerator.device)

    # 4. Safe shutil.rmtree monkey-patch to prevent PVC cascading deletion errors
    import shutil
    original_rmtree = shutil.rmtree
    def safe_rmtree(path, *args, **kwargs):
        try:
            original_rmtree(path, *args, **kwargs)
        except FileNotFoundError:
            pass
    shutil.rmtree = safe_rmtree

    # 5. Start distributed training
    print("Starting DPO training...")
    trainer.train()
    
    # 6. Save model (FSDP requires ALL ranks to participate in gathering sharded weights!)
    trainer.save_model(args.output_path)
    if rank == 0:
        print(f"Model successfully saved to {args.output_path}")

    # Prevents NCCL watchdog crash by ensuring all workers wait for I/O completion
    dist.barrier()

if __name__ == "__main__":
    main()

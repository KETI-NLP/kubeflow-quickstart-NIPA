import argparse
import os
import datetime
import torch
import torch.distributed as dist
from datasets import load_from_disk
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, TrainingArguments
from qwen_vl_utils import process_vision_info

# Fix trl>=0.15.0 import error on PyTorch 2.5.0
import torch.distributed.fsdp
if not hasattr(torch.distributed.fsdp, "FSDPModule"):
    torch.distributed.fsdp.FSDPModule = type("FSDPModule", (object,), {})
if not hasattr(torch.distributed.fsdp, "register_fsdp_forward_method"):
    torch.distributed.fsdp.register_fsdp_forward_method = lambda *args, **kwargs: None

from trl import DPOTrainer, DPOConfig

def main():
    parser = argparse.ArgumentParser(description="LLM DPO Distributed Training")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="HuggingFace model name")
    parser.add_argument("--data_path", type=str, default="/data/vlm_dpo_dataset", help="Path to local or PVC dataset")
    parser.add_argument("--output_path", type=str, default="/data/result/vlm-dpo", help="Path to save model")
    parser.add_argument("--max_steps", type=int, default=50, help="Max steps for testing")
    args = parser.parse_args()

    # 1. Load dataset from PVC
    print(f"Loading dataset from {args.data_path} ...")
    dataset = load_from_disk(args.data_path)

    # Distributed setup
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", timeout=datetime.timedelta(seconds=7200))
        rank = dist.get_rank()
    else:
        rank = 0

    # TRL DPOTrainer는 DPOConfig를 사용합니다
    training_args = DPOConfig(
        output_dir=args.output_path,
        per_device_train_batch_size=1, # 이미지 때문에 메모리를 많이 소모하므로 1설정
        gradient_accumulation_steps=16,
        max_steps=args.max_steps, # 빠른 테스트
        logging_dir="/data/log/vlm-dpo",
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        fsdp="full_shard auto_wrap",
        fsdp_config={"transformer_layer_cls_to_wrap": ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"]},
        ddp_find_unused_parameters=False,
        remove_unused_columns=False, # VLM 이미지 컬럼 유지
        dataset_num_proc=16,         # Speed up tokenization map
        ddp_timeout=7200,            # Prevent NCCL timeout during long maps
    )

    # Rank 0만 먼저 모델/토크나이저를 다운로드 하도록 다른 노드들은 대기
    if rank != 0:
        dist.barrier()

    local_only = (rank != 0)
    processor = AutoProcessor.from_pretrained(args.model_name, local_files_only=local_only)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, local_files_only=local_only, attn_implementation="flash_attention_2")
    ref_model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, local_files_only=local_only, attn_implementation="flash_attention_2")

    if rank == 0:
        dist.barrier()

    # Initialize DPOTrainer MUST BE OUTSIDE main_process_first because FSDP requires ALL ranks
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset["train"],
        processing_class=processor,
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

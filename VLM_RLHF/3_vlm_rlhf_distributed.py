import argparse
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import datetime
import torch
import torch.distributed as dist
from datasets import load_from_disk, Dataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from trl import GRPOTrainer, GRPOConfig


def build_dataset(data_path, processor):
    """Load dataset and format for GRPO (multimodal)."""
    dataset = load_from_disk(data_path)

    def format_sample(sample):
        # prompt is already a list of messages with {"type": "image"}
        # images is a list of PIL Images
        return {
            "prompt": sample["prompt"],
            "images": sample["images"]
        }

    train_dataset = dataset["train"] if isinstance(dataset, dict) else dataset
    train_dataset = train_dataset.map(format_sample, batched=False)
    return train_dataset


def compute_dummy_vlm_reward(completions, **kwargs):
    """Dummy reward function: random float in [-1, 1] per completion."""
    import random
    return [random.uniform(-1.0, 1.0) for _ in completions]


def main():
    parser = argparse.ArgumentParser(description="VLM RLHF Distributed Training (GRPO)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--data_path", type=str, default="/data/vlm_rlhf_dataset")
    parser.add_argument("--output_path", type=str, default="/data/result/vlm-rlhf")
    parser.add_argument("--max_steps", type=int, default=3, help="Max steps for testing")
    args = parser.parse_args()

    # Distributed setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                timeout=datetime.timedelta(seconds=7200)
            )
        rank = dist.get_rank()
    else:
        rank = 0

    # Only rank 0 downloads; others wait
    if rank != 0:
        dist.barrier()

    local_only = (rank != 0)
    processor = AutoProcessor.from_pretrained(args.model_name, local_files_only=local_only)
    
    # trl>=0.15.0 GRPOTrainer handles pad_token_id from processing_class automatically,
    # but we ensure processor.tokenizer has it.
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    processor.pad_token_id = processor.tokenizer.pad_token_id
    processor.bos_token_id = getattr(processor.tokenizer, "bos_token_id", None)
    processor.eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)

    if rank == 0:
        dist.barrier()

    # Load dataset
    print(f"[Rank {rank}] Loading dataset from {args.data_path} ...")
    train_dataset = build_dataset(args.data_path, processor)

    # GRPO config - trl>=0.15.0 supports num_generations and max_completion_length
    config = GRPOConfig(
        output_dir=args.output_path,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-6,
        bf16=True,
        logging_steps=1,
        save_steps=args.max_steps,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0,
        ddp_find_unused_parameters=True, 
        num_generations=8,               # GRPO generations per prompt
        max_completion_length=64,       # Limit generation length to save VRAM
    )

    # Load model
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        local_files_only=local_only,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False

    if processor.tokenizer.padding_side != "left":
        processor.tokenizer.padding_side = "left"
    
    # trl GRPOTrainer may still expect these attributes on non-PEFT models
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    
    if not hasattr(model, "disable_adapter"):
        from contextlib import contextmanager
        @contextmanager
        def _disable_adapter_noop(self):
            yield
        import types
        model.disable_adapter = types.MethodType(_disable_adapter_noop, model)

    # Monkey-patch GRPOTrainer to be compatible with latest transformers Trainer API
    original_get_train_sampler = getattr(GRPOTrainer, "_get_train_sampler", None)
    if original_get_train_sampler is not None:
        def patched_get_train_sampler(self, dataset=None):
            return original_get_train_sampler(self)
        try:
            GRPOTrainer._get_train_sampler = patched_get_train_sampler
        except Exception:
            pass

    trainer = GRPOTrainer(
        model=model,
        args=config,
        processing_class=processor,   # Passing full processor for multimodal support
        reward_funcs=compute_dummy_vlm_reward,
        train_dataset=train_dataset,
    )

    # [FIXED] Do NOT delete ref_model in GRPO. It requires the ref_model to compute KL divergence penalty.
    # If OOM occurs, you must use PEFT (LoRA) instead of full fine-tuning, or reduce batch size.
    # if hasattr(trainer, "ref_model") and trainer.ref_model is not None:
    #     del trainer.ref_model
    #     trainer.ref_model = None
    #     torch.cuda.empty_cache()

    print(f"[Rank {rank}] Starting VLM GRPO training for {args.max_steps} steps...")
    trainer.train()

    if rank == 0:
        trainer.save_model(args.output_path)
        processor.save_pretrained(args.output_path)
        print(f"[Rank 0] Model saved to {args.output_path}")

    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()

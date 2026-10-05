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

from datasets import load_from_disk, load_dataset
from transformers import AutoProcessor, AutoTokenizer, pipeline, AutoModelForSequenceClassification, Qwen3VLForConditionalGeneration
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, set_seed, RewardTrainer, RewardConfig

# Hot-patch the legacy RewardTrainer to ignore the `num_items_in_batch` param injected by transformers>=4.49.0
_old_reward_compute_loss = RewardTrainer.compute_loss
def _new_reward_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
    return _old_reward_compute_loss(self, model, inputs, return_outputs=return_outputs)
RewardTrainer.compute_loss = _new_reward_compute_loss

def build_dataset(data_path, tokenizer):
    """
    Load and tokenize dataset. PPOTrainer expects dataset with `input_ids` and `query`.
    """
    dataset = load_from_disk(data_path)
    
    def tokenize(sample):
        # Truncate prompt length to prevent PPO step() OOM on huge vocabularies!
        sample["input_ids"] = tokenizer.encode(sample["query"], truncation=True, max_length=512)
        return sample

    dataset = dataset.map(tokenize, batched=False)
    return dataset["train"]

def train_reward_model_pipeline(rank):
    print("=== Phase 1: Inline Reward Model Training ===")
    rm_name = "Qwen/Qwen1.5-0.5B"
    rm_tokenizer = AutoTokenizer.from_pretrained(rm_name)
    if rm_tokenizer.pad_token is None:
        rm_tokenizer.pad_token = rm_tokenizer.eos_token
    # Qwen typically lacks a cls/bos token for sequence classification boundaries
    if rm_tokenizer.bos_token is None:
        rm_tokenizer.bos_token = rm_tokenizer.eos_token
    
    print("Loading Anthropic/hh-rlhf for Reward Model Phase...")
    dataset = load_dataset("Anthropic/hh-rlhf", split="train[:500]") # Subset to simulate process quickly
    
    def preprocess_rm(examples):
        new_examples = {
            "input_ids_chosen": [], "attention_mask_chosen": [],
            "input_ids_rejected": [], "attention_mask_rejected": [],
        }
        for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
            tokenized_chosen = rm_tokenizer(chosen, truncation=True, padding="max_length", max_length=512)
            tokenized_rejected = rm_tokenizer(rejected, truncation=True, padding="max_length", max_length=512)
            
            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])
        return new_examples
        
    dataset = dataset.map(preprocess_rm, batched=True, num_proc=1, remove_columns=dataset.column_names)
    
    rm_model = AutoModelForSequenceClassification.from_pretrained(rm_name, num_labels=1, torch_dtype=torch.bfloat16)
    rm_model.config.pad_token_id = rm_tokenizer.pad_token_id
    
    from transformers import TrainerCallback
    class RMLogCallback(TrainerCallback):
        def on_log(self, trainer_args, state, control, logs=None, **kwargs):
            if state.is_local_process_zero and logs is not None and "loss" in logs:
                print(f"🔥 [Reward Model (Phase 1)] Step {state.global_step} - Loss: {logs['loss']:.4f}")

    training_args = RewardConfig(
        output_dir="/tmp/rm_checkpoint",
        per_device_train_batch_size=4,
        max_steps=50, # Fast simulation to prove pipeline mechanics
        logging_steps=1, # Print every step!
        report_to="none",
        bf16=True, # Critical for stable Qwen gradients
        ddp_find_unused_parameters=False # Important fix for PyTorch DDP inplace buffer errors
    )
    
    trainer = RewardTrainer(
        model=rm_model,
        args=training_args,
        tokenizer=rm_tokenizer,
        train_dataset=dataset,
        callbacks=[RMLogCallback()]
    )
    
    trainer.train()
    rm_model.eval()
    print("Successfully trained Reward Model!")
    return rm_model, rm_tokenizer

def main():
    parser = argparse.ArgumentParser(description="LLM RLHF Distributed Training")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen1.5-0.5B", help="HuggingFace model name")
    parser.add_argument("--data_path", type=str, default="/data/rlhf_dataset", help="Path to local or PVC prompt dataset")
    parser.add_argument("--output_path", type=str, default="/data/result/llm-rlhf", help="Path to save model")
    args = parser.parse_args()

    # Distributed setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank = dist.get_rank()
    else:
        rank = 0

    if rank != 0:
        dist.barrier()

    tokenizer = AutoProcessor.from_pretrained(args.model_name).tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Instantiate the base model normally using its native Vision2Seq class
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)

    # For text-only RLHF over a Multimodal VLM, the visual tower is never traversed.
    # We MUST freeze the visual parameters to prevent PyTorch DDP/FSDP from crashing due to 'unused parameters' during gradient reductions.
    for name, param in base_model.named_parameters():
        if "visual" in name:
            param.requires_grad = False

    # TRL's ValueHead (__init__) strictly looks for config.hidden_size to make the reward linear layer.
    # Since Qwen3-VL is a VLM, the hidden_size resides in text_config. Inject it at the root to prevent UnboundLocalError.
    base_model.config.hidden_size = getattr(base_model.config, "hidden_size", base_model.config.text_config.hidden_size)
    base_model.gradient_checkpointing_enable()
    # Wrap the created base model instance with the TRL Value Head wrapper directly
    model = AutoModelForCausalLMWithValueHead(pretrained_model=base_model)
    # Hot-patch missing peft attribute that 0.11.4 modeling_value_head relies on
    model.is_peft_model = False
    
    # Do the exact same for the reference model
    ref_base_model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    ref_base_model.config.hidden_size = ref_base_model.config.text_config.hidden_size
    ref_model = AutoModelForCausalLMWithValueHead(pretrained_model=ref_base_model)
    ref_model.is_peft_model = False

    # 1. Load dataset from PVC
    print(f"Loading dataset from {args.data_path} ...")
    dataset = build_dataset(args.data_path, tokenizer)

    if rank == 0:
        dist.barrier()

    # 2. Configure PPO Arguments
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Qwen의 초대형 단어사전(15만 개)과 8192 토큰 시퀀스가 결합되면, 역전파 (Backward) 시
    # PPO가 메모리에 올리는 행렬 1개당 엄청난 수십 기가의 VRAM을 집어삼킵니다.
    # GPU 1대당 처리할 데이터 총량(batch_size)은 GPU 개수와 맞추지만,
    # 한 틱에 욱여넣을 최대 메모리 단위(mini_batch_size)는 '2'로 완전히 박제하여
    # 수십 기가의 softmax 폭발을 원천 차단합니다! (world_size가 짝수면 항상 완벽하게 분배됨)
    
    config = PPOConfig(
        model_name=args.model_name,
        learning_rate=1e-6,
        log_with=None,
        batch_size=world_size, 
        mini_batch_size=1,                # VRAM 터짐 방지용 1 문장 제한 하드캡 (고정)
        gradient_accumulation_steps=1,
        optimize_cuda_cache=True,
        early_stopping=True,
        target_kl=0.1,
        # PPO interacts with FSDP via Accelerate natively
    )

    def collator(data):
        return {
            "input_ids": [d["input_ids"] if isinstance(d["input_ids"], torch.Tensor) else torch.tensor(d["input_ids"], dtype=torch.long) for d in data],
            "query": [d["query"] for d in data]
        }

    ppo_trainer = PPOTrainer(
        config=config, 
        model=model, 
        ref_model=ref_model, 
        tokenizer=tokenizer, 
        dataset=dataset,
        data_collator=collator
    )

    # 3. Setup Real Reward Model Pipeline
    rm_model, rm_tokenizer = train_reward_model_pipeline(rank)
    # MUST use LOCAL_RANK instead of global rank for physical GPU mapping!
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    reward_device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    rm_model.to(reward_device)
    
    # 🧹 Clear memory heavily before PPO starts! (Frees ~10GB of RM optimizer states)
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    dist.barrier() # Sync before PPO begins

    def compute_reward(query_list, response_list):
        """[]
        Calculates real rewards using the inline trained RM model.
        """
        texts = [q + r for q, r in zip(query_list, response_list)]
        inputs = rm_tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(reward_device)
        with torch.no_grad():
            outputs = rm_model(**inputs)
            scores = outputs.logits.squeeze(-1)
            # Clamp scores to prevent PPO gradient explosion (NaNs)
            scores = torch.clamp(scores, min=-2.0, max=2.0)
        return [s for s in scores]

    # 4. Safe shutil.rmtree monkey-patch to prevent PVC cascading deletion errors
    import shutil
    original_rmtree = shutil.rmtree
    def safe_rmtree(path, *arguments, **kwargs):
        try:
            original_rmtree(path, *arguments, **kwargs)
        except (FileNotFoundError, OSError):
            pass
    shutil.rmtree = safe_rmtree

    # 5. Start RLHF Loop
    print("Starting RLHF (PPO) training...")
    generation_kwargs = {
        "top_k": 50,
        "top_p": 0.9,
        "temperature": 0.7,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "max_new_tokens": 128,
    }

    # For demonstration, we run a short 10-step epoch. Real applications iterate over the dataloader completely.
    epochs = 1
    for epoch in range(epochs):
        for step, batch in enumerate(ppo_trainer.dataloader):
            query_tensors = batch["input_ids"]
            
            # Generate responses from the current policy
            model.eval()
            response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
            model.train()
            batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
            
            # Compute Rewards using the real pipeline
            rewards = compute_reward(batch["query"], batch["response"])
            
            # Run PPO step (computes KL divergence, trains Value Head and LM Head)
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)
            
            if step % 10 == 0 and rank == 0:
                print(f"Step {step}: Reward Mean: {torch.stack(rewards).mean().item()}")

    # 6. Save model to the shared block level storage
    if rank == 0:
        ppo_trainer.save_pretrained(args.output_path)
        tokenizer.save_pretrained(args.output_path)
        print(f"Model successfully saved to {args.output_path}")

    # Prevents NCCL watchdog crash by ensuring all workers wait for I/O completion
    dist.barrier()

if __name__ == "__main__":
    main()

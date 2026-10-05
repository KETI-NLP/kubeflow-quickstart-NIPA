import argparse
import os
import torch
import torch.distributed as dist
import torch.distributed.fsdp as fsdp
if not hasattr(fsdp, "register_fsdp_forward_method"):
    fsdp.register_fsdp_forward_method = lambda *args, **kwargs: None
import shutil
import time
from datasets import load_from_disk, concatenate_datasets, DatasetDict
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    AutoConfig,
)

def main():
    parser = argparse.ArgumentParser(description="LLM SFT Distributed Training")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen1.5-0.5B", help="HuggingFace model name")
    parser.add_argument("--data_path", type=str, default="/data/chatqa2_dataset", help="Path to local or PVC dataset")
    parser.add_argument("--output_path", type=str, default="/data/result/llm-sft-long", help="Path to save model")
    parser.add_argument("--max_length", type=int, default=131072, help="Max sequence length for SP training (default 128K)")
    parser.add_argument("--max_steps", type=int, default=5, help="Max steps for training")
    parser.add_argument("--max_samples", type=int, default=None, help="Number of samples to use for training (None for all)")
    parser.add_argument("--tokenized_data_path", type=str, default="/data/tokenized_chatqa2_long_v7_qwen3vl", help="Path to save/load tokenized dataset")
    args = parser.parse_args()

    # Append max_samples to tokenized_data_path for isolated caching
    if args.max_samples is not None:
        args.tokenized_data_path = f"{args.tokenized_data_path}_{args.max_samples}"


    # 1. PVC(디스크)에서 전처리된 데이터 불러오기
    print(f"Arguments: {args}")
    print(f"Loading dataset from {args.data_path} ...")
    dataset = load_from_disk(args.data_path)

    # 분산 학습 환경에서 모든 워커 중복 다운로드를 막기 위해 (HF 429 Too Many Requests 방지)
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank = dist.get_rank()
    else:
        rank = 0

    # Rank 0만 먼저 모델/토크나이저를 다운로드 하도록 다른 노드들은 대기
    if dist.is_initialized():
        if rank != 0:
            print(f"Rank {rank}: Waiting for Rank 0 to download model/tokenizer...")
            dist.barrier()
        else:
            print(f"Rank 0: Pre-downloading model/tokenizer to cache...")
            # AutoTokenizer.from_pretrained(args.model_name) # Removed as processor handles it
            AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True) # Use AutoProcessor
            # from transformers import AutoConfig # Removed as AutoConfig is now in main import block
            AutoConfig.from_pretrained(args.model_name, trust_remote_code=True) # Added trust_remote_code
            dist.barrier()
    else:
        rank = 0

    # Ring attention setup
    group = dist.new_group(ranks=range(dist.get_world_size()), backend="nccl")
    try:
        from ring_flash_attn import substitute_hf_flash_attn, update_ring_flash_attn_params
        substitute_hf_flash_attn(group, heads_k_stride=1)
        device = torch.device(f"cuda:{rank % torch.cuda.device_count() if torch.cuda.is_available() else 'cpu'}")
        cu_seqlens = torch.cat([torch.tensor([0], device=device), torch.tensor([args.max_length], device=device)]).to(torch.int32)
        update_ring_flash_attn_params(cu_seqlens, group)
    except ImportError:
        print("Warning: ring-flash-attention not installed. SP mapping will fail if model relies on it.")

    # tokenizer = AutoTokenizer.from_pretrained(args.model_name) # Removed, use processor.tokenizer
    # tokenizer.padding_side = "left" # Removed, use processor.tokenizer
    # if tokenizer.pad_token is None: # Removed, use processor.tokenizer
    #     tokenizer.pad_token = tokenizer.eos_token # Removed, use processor.tokenizer
        
    # from transformers import AutoConfig # Removed, now in main import block
    # config = AutoConfig.from_pretrained(args.model_name) # Replaced
    # config.padding_side = "left" # Triple-Lock 1: Config Lock # Replaced
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True, local_files_only=(rank!=0))
    config.padding_side = "left" 
    # config.rope_scaling = {"type": "linear", "factor": 4.0} # Removed, specific to original model
    # config.max_position_embeddings = args.max_length # Removed, specific to original model
    
    # VL 모델은 Processor를 사용합니다.
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True, local_files_only=(rank!=0))
    processor.tokenizer.padding_side = "left" # Tokenizer Lock
    
    print(f"Rank {rank}: Starting to load model {args.model_name}...")
    # 8B 모델은 bfloat16과 flash_attention_2를 권장합니다.
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        local_files_only=(rank!=0),
    )
    
    # 8B 128K를 위해 체크포인팅 활성화
    model.gradient_checkpointing_enable()
    print(f"Rank {rank}: Model load complete.")

    if dist.is_initialized():
        print(f"Rank {rank}: Waiting at post-load barrier...")
        dist.barrier()
        print(f"Rank {rank}: Passed post-load barrier.")

    # 3. 데이터 프롬프트 전처리 (SFT 형식으로 문자열 조립 후 Tokenizing)
    def preprocess_function(examples):
        texts = []
        for idx in range(len(examples["question"])):
            context = examples.get("sub-paragraphs", [])[idx] if "sub-paragraphs" in examples else ""
            question = examples["question"][idx]
            answers = examples["answer"][idx]
            answer_text = answers[0] if isinstance(answers, list) and len(answers) > 0 else str(answers)
            
            prompt = f"Context:\n{context}\n\nQuestion:\n{question}"
            msg = [
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
                {"role": "assistant", "content": [{"type": "text", "text": answer_text}]}
            ]
            text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            texts.append(text)
            
        # Processor로 토큰화 (Left padding 강제)
        processor.tokenizer.padding_side = "left"
        model_inputs = processor(
            text=texts, 
            padding="max_length", 
            max_length=args.max_length, 
            truncation=True,
        )
        
        # Labels는 input_ids를 복사 (차후 Prompt 부분 마스킹 고도화 가능)
        model_inputs["labels"] = [list(ids) for ids in model_inputs["input_ids"]]
        
        # [Validation Logging] 패딩 확인용
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            sample_ids = model_inputs["input_ids"][0]
            print(f"Rank {rank}: [Validation] First 15 tokens: {sample_ids[:15]}", flush=True)
            print(f"Rank {rank}: [Validation] Last 15 tokens: {sample_ids[-15:]}", flush=True)
            if sample_ids[0] == processor.tokenizer.pad_token_id:
                print(f"Rank {rank}: [SUCCESS] LEFT-PADDING CONFIRMED.", flush=True)
            else:
                print(f"Rank {rank}: [WARNING] RIGHT-PADDING DETECTED IN v7!", flush=True)

        return model_inputs

    print(f"Rank {rank}: Tokenizing dataset step started.")
    print(f"Rank {rank}: Accessing column_names...")
    column_names = dataset["train"].column_names
    print(f"Rank {rank}: column_names accessed: {column_names}")

    if args.max_samples is not None:
        print(f"Rank {rank}: Getting dataset length...")
        total_samples = len(dataset["train"])
        num_samples = min(total_samples, args.max_samples)
        if rank == 0:
            print(f"Sampling {num_samples} examples out of {total_samples}...")
        
        print(f"Rank {rank}: Selecting samples (range {num_samples})...")
        dataset["train"] = dataset["train"].select(range(num_samples))
        print(f"Rank {rank}: Samples selected.")
    
    print(f"Rank {rank}: Reached world_size check.")
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    print(f"Rank {rank}: world_size={world_size}. Entering distributed tokenization block...")
    
    # 3.1 Distributed Tokenization
    # Only Rank 0 checks if the dataset exists to avoid inconsistent states across nodes
    do_tokenize = False
    if rank == 0:
        print(f"Rank 0: Checking if {args.tokenized_data_path} exists and is left-padded...", flush=True)
        marker = os.path.join(args.tokenized_data_path, ".left_padded")
        if os.path.exists(args.tokenized_data_path):
            if os.path.exists(marker):
                print(f"Rank 0: {args.tokenized_data_path} already exists with left-padding marker. Skipping tokenization.", flush=True)
                do_tokenize = False
            else:
                print(f"Rank 0: {args.tokenized_data_path} exists but is NOT left-padded. FORCING RE-TOKENIZATION...", flush=True)
                os.system(f"rm -rf {args.tokenized_data_path}")
                do_tokenize = True
        else:
            print(f"Rank 0: {args.tokenized_data_path} not found. Starting distributed tokenization...", flush=True)
            do_tokenize = True

    # Broadcast decision to all ranks
    if dist.is_initialized():
        print(f"Rank {rank}: Reached broadcast point.", flush=True)
        # 텐서 기반 브로드캐스트로 변경 (더 견고함)
        device = torch.device(f"cuda:{rank % torch.cuda.device_count() if torch.cuda.is_available() else 'cpu'}")
        do_tokenize_tensor = torch.tensor([1 if do_tokenize else 0], dtype=torch.int, device=device)
        
        if rank == 0: print("Rank 0: Broadcasting tokenization decision via tensor...", flush=True)
        dist.broadcast(do_tokenize_tensor, src=0)
        do_tokenize = (do_tokenize_tensor.item() == 1)
        print(f"Rank {rank}: Broadcast complete. do_tokenize={do_tokenize}", flush=True)

    if do_tokenize:
        # 각 랭크가 완전히 독립된 디렉토리에 저장하도록 변경 (NFS 경합 및 메타데이터 이슈 원천 차단)
        # 예: /data/tokenized_..._shards/rank_0, /data/tokenized_..._shards/rank_1 ...
        base_shard_dir = args.tokenized_data_path + "_shards"
        my_shard_dir = os.path.join(base_shard_dir, f"rank_{rank}")
        
        # 자신이 사용할 디렉토리 직접 생성
        os.makedirs(my_shard_dir, exist_ok=True)
        # NFS 전파를 위한 아주 짧은 대기
        time.sleep(1)
        
        if dist.is_initialized():
            print(f"Rank {rank}: Isolated shard directory ready at {my_shard_dir}. Waiting at shard-init barrier...", flush=True)
            dist.barrier()
            print(f"Rank {rank}: Passed shard-init barrier.", flush=True)
        
        # 각 Rank가 자신의 샤드만 토크나이징
        print(f"Rank {rank}: Sharding dataset (shard {rank}/{world_size})...", flush=True)
        rank_shard = dataset["train"].shard(num_shards=world_size, index=rank, contiguous=True)
        print(f"Rank {rank}: Shard size: {len(rank_shard)}. Starting map...", flush=True)
        
        # tqdm 대신 rank별 메시지 출력을 위해 간단한 콜백 스타일 또는 num_proc 활용
        tokenized_shard = rank_shard.map(
            preprocess_function, 
            batched=True, 
            remove_columns=column_names, 
            num_proc=4,
            desc=f"Rank {rank} tokenizing"
        )
        
        print(f"Rank {rank}: Saving shard to {my_shard_dir}...", flush=True)
        tokenized_shard.save_to_disk(my_shard_dir)
        
        if dist.is_initialized():
            print(f"Rank {rank}: Shard saved. Waiting at merge barrier...", flush=True)
            dist.barrier()
        
        if rank == 0:
            try:
                print(f"Rank 0: Starting merge process for {world_size} isolated shards...", flush=True)
                shards = []
                for i in range(world_size):
                    shard_path = os.path.join(base_shard_dir, f"rank_{i}")
                    if i % 10 == 0 or i == world_size - 1:
                        print(f"Rank 0: Audit - Checking shard {i}/{world_size} at {shard_path}...", flush=True)
                    
                    if not os.path.exists(shard_path):
                        print(f"Rank 0: WARNING - Shard {i} not found at {shard_path}. Dataset might be incomplete.", flush=True)
                        continue
                    
                    try:
                        shards.append(load_from_disk(shard_path))
                    except Exception as shard_e:
                        print(f"Rank 0: ERROR loading shard {i}: {shard_e}. Skipping.", flush=True)

                print(f"Rank 0: Concatenating {len(shards)} valid shards...", flush=True)
                if len(shards) == 0:
                    raise ValueError("No valid shards found to merge!")
                
                final_train_dataset = concatenate_datasets(shards)
                tokenized_datasets = DatasetDict({"train": final_train_dataset})
                
                print(f"Rank 0: Saving merged dataset to {args.tokenized_data_path}...", flush=True)
                tokenized_datasets.save_to_disk(args.tokenized_data_path)
                
                # 병합 후 임시 샤드 디렉토리 전체 정리
                os.system(f"rm -rf {base_shard_dir}")

                # 저장 완료 플래그 생성 (NFS 일관성 보장용)
                done_flag = os.path.join(args.tokenized_data_path, ".done")
                with open(done_flag, "w") as f:
                    f.write("done")
                
                # 좌측 패딩 마커 생성 (재토크나이징 방지용)
                marker = os.path.join(args.tokenized_data_path, ".left_padded")
                with open(marker, "w") as f:
                    f.write("left")
                
                print(f"Rank 0: Created success flags at {done_flag} and {marker}", flush=True)
            except Exception as merge_e:
                print(f"Rank 0: CRITICAL ERROR during merge process: {merge_e}", flush=True)
                import traceback
                traceback.print_exc()
                # 모든 워커가 알 수 있도록(대기 루프 탈출) 에러 플래그를 남겨둘 수도 있지만, 
                # 일단은 에러 로그를 남기는 데 집중합니다.
                raise merge_e

        if dist.is_initialized():
            print(f"Rank {rank}: Reaching final tokenization barrier...", flush=True)
            dist.barrier()
            print(f"Rank {rank}: Passed final tokenization barrier.", flush=True)

    # Load result (with robust retry for NFS visibility)
    success = False
    done_flag = os.path.join(args.tokenized_data_path, ".done")
    max_retries = 20
    for i in range(max_retries):
        if os.path.exists(done_flag):
            try:
                tokenized_datasets = load_from_disk(args.tokenized_data_path)
                print(f"Rank {rank}: Successfully loaded tokenized dataset.", flush=True)
                success = True
                break
            except Exception as e:
                print(f"Rank {rank}: Found .done flag but load failed: {e}. Retrying...", flush=True)
        else:
            if rank == 0 and not do_tokenize: # 이미 있었어야 하는 경우
                # 가끔 NFS 문제로 본인도 못볼 수 있으므로 일단 대기
                pass
            print(f"Rank {rank}: Waiting for {done_flag} (attempt {i+1}/{max_retries})...", flush=True)
        
        time.sleep(5)
    
    if not success:
        raise FileNotFoundError(f"Rank {rank}: Failed to load tokenized dataset after {max_retries} attempts at {args.tokenized_data_path}")

    class SPDataCollator:
        def __init__(self, world_size, rank, seq_len):
            self.world_size = world_size
            self.rank = rank
            self.seq_len = seq_len
            self.chunk_size = seq_len // world_size

        def __call__(self, features):
            batch = {
                "input_ids": torch.tensor([f["input_ids"] for f in features]),
                "labels": torch.tensor([f["labels"] for f in features]),
            }
            if "attention_mask" in features[0]:
                batch["attention_mask"] = torch.tensor([f["attention_mask"] for f in features])
            
            # Create global position_ids
            batch["position_ids"] = torch.arange(self.seq_len).unsqueeze(0).expand(len(features), -1)

            # Sequence Parallelism Chunking
            start = self.rank * self.chunk_size
            end = (self.rank + 1) * self.chunk_size

            for k, v in batch.items():
                batch[k] = v[:, start:end]
            return batch

    class SPTrainer(Trainer):
        def _get_train_sampler(self, *args, **kwargs):
            # All ranks use the same random sampler so they see the exact same batch, 
            # and sequence parallelism chunks it correctly across ranks.
            return torch.utils.data.RandomSampler(self.train_dataset)
            
        def train(self, *args, **kwargs):
            # Final Force: Ensure 'left' padding right before the core training loop
            print(f"Rank {rank}: [FORCE] Setting model.config.padding_side to 'left' IN trainer.train()", flush=True)
            self.model.config.padding_side = "left"
            if hasattr(self.model, "module"): # Handle FSDP/DDP wrapping
                self.model.module.config.padding_side = "left"
            
            # Check processor's tokenizer too
            if hasattr(self, "processor") and self.processor is not None:
                self.processor.tokenizer.padding_side = "left"
            elif self.tokenizer is not None:
                self.tokenizer.padding_side = "left"
                
            return super().train(*args, **kwargs)

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    sp_collator = SPDataCollator(world_size=world_size, rank=rank, seq_len=args.max_length)

    # 4. HuggingFace Trainer 설정
    training_args = TrainingArguments(
        output_dir=args.output_path,
        per_device_train_batch_size=1,  # Required to be 1 for ring_flash_attn varlen integration unless specifically managed
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        logging_dir="/data/log/llm-sft-long",
        logging_steps=10,
        max_steps=args.max_steps,
        save_strategy="epoch",
        fsdp="full_shard auto_wrap",
        fsdp_config={"transformer_layer_cls_to_wrap": ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"]},
        ddp_find_unused_parameters=False,
    )

    trainer = SPTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        data_collator=sp_collator,
        processing_class=processor,
    )
    trainer.processor = processor # Store processor for the train() override

    original_rmtree = shutil.rmtree
    def safe_rmtree(path, *args, **kwargs):
        try:
            original_rmtree(path, *args, **kwargs)
        except FileNotFoundError:
            pass
    shutil.rmtree = safe_rmtree

    # 5. 분산 학습 시작
    print(f"Rank {rank}: [DEBUG] model.config.padding_side before force: {getattr(model.config, 'padding_side', 'N/A')}", flush=True)
    print(f"Rank {rank}: [DEBUG] tokenizer.padding_side before force: {processor.tokenizer.padding_side}", flush=True)
    
    # Runtime Force: Ensure both model and tokenizer are set to 'left'
    model.config.padding_side = "left"
    processor.tokenizer.padding_side = "left"
    
    print(f"Rank {rank}: [DEBUG] model.config.padding_side AFTER force: {model.config.padding_side}", flush=True)
    print(f"Rank {rank}: [DEBUG] tokenizer.padding_side AFTER force: {processor.tokenizer.padding_side}", flush=True)
    
    print("Starting training...", flush=True)
    trainer.train()
    
    # 6. 학습 결과 로컬 저장 (Rank 0 워커만 최종 모델을 쓰도록 하여 충돌 방지)
    trainer.save_model(args.output_path)
    if rank == 0:
        print(f"Model successfully saved to {args.output_path}")

    # 모든 워커가 Rank 0의 저장이 끝날 때까지 대기하여 NCCL Watchdog Timeout 에러 방지
    dist.barrier()

if __name__ == "__main__":
    main()

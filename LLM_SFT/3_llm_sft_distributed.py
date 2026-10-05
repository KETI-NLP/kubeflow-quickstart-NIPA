import argparse
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
import torch.distributed as dist
import torch.distributed.fsdp as fsdp
if not hasattr(fsdp, "register_fsdp_forward_method"):
    fsdp.register_fsdp_forward_method = lambda *args, **kwargs: None
from datasets import load_from_disk
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, TrainingArguments, Trainer

def main():
    parser = argparse.ArgumentParser(description="LLM SFT Distributed Training")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen1.5-0.5B", help="HuggingFace model name")
    parser.add_argument("--data_path", type=str, default="/data/alpaca_dataset", help="Path to local or PVC dataset")
    parser.add_argument("--output_path", type=str, default="/data/result/llm-sft", help="Path to save model")
    args = parser.parse_args()

    # 1. PVC(디스크)에서 전처리된 데이터 불러오기
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
    if rank != 0:
        dist.barrier()

    processor = AutoProcessor.from_pretrained(args.model_name)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    
    model.gradient_checkpointing_enable()

    # Rank 0가 다운로드를 끝내면 다른 노드들도 로컬 캐시에서 로드하도록 진행
    if rank == 0:
        dist.barrier()

    # 3. 데이터 프롬프트 전처리 (SFT 형식으로 문자열 조립 후 Tokenizing)
    def preprocess_function(examples):
        inputs = []
        # Alpaca 파인튜닝 데이터 포맷
        insts = examples.get('instruction', [])
        inps = examples.get('input', [])
        outs = examples.get('output', [])
        
        for inst, inp, out in zip(insts, inps, outs):
            prompt = f"Instruction: {inst}\n"
            if inp: 
                prompt += f"Input: {inp}\n"
            
            msg = [
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
                {"role": "assistant", "content": [{"type": "text", "text": out}]}
            ]
            text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            inputs.append(text)
            
        model_inputs = processor(text=inputs, max_length=1024, truncation=True, padding="max_length")
        
        # Next Token Prediction을 위해 labels 설계 (CrossEntropyLoss가 패딩을 무시하도록 attention_mask==0인 곳을 -100으로 치환)
        labels = []
        for ids, mask in zip(model_inputs["input_ids"], model_inputs["attention_mask"]):
            labels.append([token_id if m == 1 else -100 for token_id, m in zip(ids, mask)])
        model_inputs["labels"] = labels
        
        return model_inputs

    print("Tokenizing dataset...")
    column_names = dataset["train"].column_names
    tokenized_datasets = dataset.map(preprocess_function, batched=True, num_proc=16, remove_columns=column_names)

    # 4. HuggingFace Trainer 설정
    training_args = TrainingArguments(
        output_dir=args.output_path,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        logging_dir="/data/log/llm-sft",
        logging_steps=10,
        save_strategy="epoch",
        fsdp="full_shard auto_wrap",
        fsdp_config={"transformer_layer_cls_to_wrap": "Qwen3VLTextDecoderLayer"},
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
    )

    import shutil
    original_rmtree = shutil.rmtree
    def safe_rmtree(path, *args, **kwargs):
        try:
            original_rmtree(path, *args, **kwargs)
        except FileNotFoundError:
            pass
    shutil.rmtree = safe_rmtree

    # 5. 분산 학습 시작
    print("Starting training...")
    trainer.train()
    
    # 6. 학습 결과 로컬 저장 (Rank 0 워커만 최종 모델을 쓰도록 하여 충돌 방지)
    # FSDP 환경에서는 모든 워커가 통신에 참여하여 모델 가중치를 Gather 해야 하므로 if문을 제거합니다.
    trainer.save_model(args.output_path)
    if rank == 0:
        print(f"Model successfully saved to {args.output_path}")

    # 모든 워커가 Rank 0의 저장이 끝날 때까지 대기하여 NCCL Watchdog Timeout 에러 방지
    dist.barrier()

if __name__ == "__main__":
    main()

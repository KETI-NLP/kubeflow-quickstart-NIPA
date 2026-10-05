import argparse
import os
import datetime
import torch
import torch.distributed as dist
import torch.distributed.fsdp
if not hasattr(torch.distributed.fsdp, "FSDPModule"):
    torch.distributed.fsdp.FSDPModule = type("FSDPModule", (object,), {})
if not hasattr(torch.distributed.fsdp, "register_fsdp_forward_method"):
    torch.distributed.fsdp.register_fsdp_forward_method = lambda *args, **kwargs: None

from datasets import load_from_disk
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, TrainingArguments, Trainer
from qwen_vl_utils import process_vision_info

def main():
    parser = argparse.ArgumentParser(description="LLM SFT Distributed Training")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="HuggingFace model name")
    parser.add_argument("--data_path", type=str, default="/data/vlm_sft_dataset", help="Path to local or PVC dataset")
    parser.add_argument("--output_path", type=str, default="/data/result/vlm-sft", help="Path to save model")
    parser.add_argument("--max_steps", type=int, default=3, help="Max steps for testing")
    args = parser.parse_args()

    # 1. PVC(디스크)에서 전처리된 데이터 불러오기
    print(f"Loading dataset from {args.data_path} ...")
    dataset = load_from_disk(args.data_path)

    # 분산 학습 환경에서 모든 워커 중복 다운로드를 막기 위해 (HF 429 Too Many Requests 방지)
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", timeout=datetime.timedelta(seconds=7200))
        rank = dist.get_rank()
    else:
        rank = 0

    # Rank 0만 먼저 모델/토크나이저를 다운로드 하도록 다른 노드들은 대기
    if rank != 0:
        dist.barrier()

    local_only = (rank != 0)
    processor = AutoProcessor.from_pretrained(args.model_name, local_files_only=local_only)
    
    # Qwen-VL은 AutoModelForCausalLM 혹은 Qwen2VLForConditionalGeneration을 사용합니다.
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        local_files_only=local_only,
    )

    # Rank 0가 다운로드를 끝내면 다른 노드들도 로컬 캐시에서 로드하도록 진행
    if rank == 0:
        dist.barrier()

    # 3. 데이터 프롬프트 전처리 (Data Collator에서 실시간 처리)
    def vlm_data_collator(features):
        texts = []
        prompt_texts = []
        image_inputs = []
        
        for feature in features:
            messages = feature.get("messages", [])
            images = feature.get("images", []) # PIL Image 리스트
            
            # HuggingFaceH4/llava-instruct-mix-vsft 포맷을 Qwen 방식으로 변환
            qwen_messages = []
            img_idx = 0
            for msg in messages:
                role = msg["role"]
                content_str = msg["content"]
                
                # <image> 토큰을 실제 이미지 입력 형식으로 변환
                if "<image>" in content_str and img_idx < len(images):
                    text_part = content_str.replace("<image>", "").strip()
                    content = [{"type": "image", "image": images[img_idx]}]
                    if text_part:
                        content.append({"type": "text", "text": text_part})
                    img_idx += 1
                else:
                    content = [{"type": "text", "text": content_str}]
                
                qwen_messages.append({"role": role, "content": content})
                
            text = processor.apply_chat_template(qwen_messages, tokenize=False, add_generation_prompt=False)
            
            if len(qwen_messages) > 0 and qwen_messages[-1]["role"] == "assistant":
                prompt_text = processor.apply_chat_template(qwen_messages[:-1], tokenize=False, add_generation_prompt=True)
            else:
                prompt_text = text
                
            prompt_texts.append(prompt_text)
            image_inputs.append(process_vision_info(qwen_messages)[0])
            texts.append(text)

        clean_image_inputs = [img for img in image_inputs if img is not None]
        if len(clean_image_inputs) == 0:
            clean_image_inputs = None

        batch = processor(
            text=texts,
            images=clean_image_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        prompt_batch = processor(
            text=prompt_texts,
            images=clean_image_inputs,
            padding=True,
            return_tensors="pt",
        )

        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][batch["attention_mask"] == 0] = -100 
        
        # Mask user prompts from labels so loss is only calculated on assistant responses
        for i in range(len(texts)):
            prompt_len = prompt_batch["attention_mask"][i].sum().item()
            non_pad_indices = batch["attention_mask"][i].nonzero(as_tuple=True)[0]
            if len(non_pad_indices) > prompt_len:
                batch["labels"][i, non_pad_indices[:prompt_len]] = -100
            else:
                batch["labels"][i, non_pad_indices] = -100
                
        return batch

    print("Dataset Ready. Training will collate on the fly.")

    # 4. HuggingFace Trainer 설정
    training_args = TrainingArguments(
        output_dir=args.output_path,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        max_steps=args.max_steps, # 빠른 테스트용
        logging_dir="/data/log/vlm-sft",
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        fsdp="full_shard auto_wrap",
        fsdp_config={"transformer_layer_cls_to_wrap": ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"]},
        ddp_find_unused_parameters=False,
        remove_unused_columns=False, # VLM에서는 image가 사용되므로 False 필수
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=vlm_data_collator,
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
    
    # 6. 학습 결과 로컬 저장 (FSDP 환경에서는 모든 워커가 통신에 참여하여 모델 가중치를 Gather 해야 하므로 if문을 사용하지 않습니다)
    trainer.save_model(args.output_path)
    if rank == 0:
        print(f"Model successfully saved to {args.output_path}")

    dist.barrier()

if __name__ == "__main__":
    main()

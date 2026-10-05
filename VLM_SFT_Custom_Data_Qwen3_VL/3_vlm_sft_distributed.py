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
    parser.add_argument("--max_steps", type=int, default=None, help="Max steps for training (None means full epochs)")
    parser.add_argument("--use_test_dataset", action="store_true", help="Redirects data paths to the convert_to_llm_training_ready_test folder for fast debugging")
    args = parser.parse_args()

    if args.use_test_dataset:
        args.data_path = args.data_path.replace("convert_to_llm_training_ready", "convert_to_llm_training_ready_test")
        print(f"Test dataset flag activated. Overriding data paths to: {args.data_path}")


    # 분산 환경 초기화를 최상단으로 끌어올림 (NFS 동시 캐시 쓰기 충돌 방지)
    if "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", timeout=datetime.timedelta(seconds=7200))
        rank = dist.get_rank()
    else:
        rank = 0

    from datasets import concatenate_datasets

    import contextlib
    @contextlib.contextmanager
    def main_process_first():
        if rank != 0:
            dist.barrier()
        yield
        if rank == 0:
            dist.barrier()

    with main_process_first():
        # 1. PVC(디스크)에서 전처리된 다중 데이터 불러오기
        data_paths = [dp.strip() for dp in args.data_path.split(",")]
        dataset_list = []

        for dp in data_paths:
            print(f"Loading dataset from {dp} ...")
            ds = load_from_disk(dp)

            # DPO 형태의 데이터셋을 SFT로 변환하는 자동 로직
            from datasets import DatasetDict
            if isinstance(ds, DatasetDict) and "train" in ds:
                train_ds = ds["train"]
            else:
                train_ds = ds

            if "chosen" in train_ds.column_names:
                print(f"Detected 'chosen' column in {dp}. Using .map() to align to SFT features...")

                def convert_dpo_to_sft(example):
                    messages = []
                    images = []

                    user_msg = example.get("prompt", [{}])[0]
                    user_content_str = ""
                    for part in user_msg.get("content", []):
                        if part.get("type") == "image":
                            user_content_str += "<image>\n"
                            if part.get("image") is not None:
                                images.append(part["image"])
                        elif part.get("type") == "text" and part.get("text"):
                            user_content_str += part["text"] + "\n"

                    messages.append({"role": "user", "content": user_content_str.strip()})

                    ast_msg = example.get("chosen", [{}])[0]
                    ast_content = ast_msg.get("content", [])
                    ast_content_str = ast_content[0]["text"] if len(ast_content) > 0 else ""
                    messages.append({"role": "assistant", "content": ast_content_str.strip()})

                    return {"messages": messages, "images": images}

                train_ds = train_ds.map(convert_dpo_to_sft, remove_columns=train_ds.column_names, num_proc=4, desc="DPO to SFT alignment")

            dataset_list.append(train_ds)

        print(f"Combining {len(dataset_list)} datasets...")
        combined_train = concatenate_datasets(dataset_list)

        # 텍스트 전용 데이터로 인한 FSDP 데드락 방지용 필터
        def has_image(example):
            if "pixel_values" in example:
                return len(example["pixel_values"]) > 0
            return len(example.get("images", [])) > 0

        print("Filtering out text-only examples to prevent FSDP deadlock...")
        combined_train = combined_train.filter(has_image, num_proc=4, desc="Filtering out text-only examples")

        dataset = {"train": combined_train}

        local_only = (rank != 0)
        processor = AutoProcessor.from_pretrained(args.model_name, local_files_only=local_only)

        # Qwen-VL은 AutoModelForCausalLM 혹은 Qwen2VLForConditionalGeneration을 사용합니다.
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_name,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            local_files_only=local_only,
        )


    # # Rank 0가 다운로드를 끝내면 다른 노드들도 로컬 캐시에서 로드하도록 진행
    if rank == 0:
        dist.barrier()

    # 3. 데이터 프롬프트 전처리 (Data Collator에서 실시간 처리)
    def vlm_data_collator(features):
        # Check if the features are already packed
        if isinstance(features[0], dict) and "input_ids" in features[0] and "pixel_values" in features[0]:
            from torch.nn.utils.rnn import pad_sequence
            
            batch_input_ids = [torch.tensor(f["input_ids"]) if not isinstance(f["input_ids"], torch.Tensor) else f["input_ids"] for f in features]
            batch_labels = [torch.tensor(f["labels"]) if not isinstance(f["labels"], torch.Tensor) else f["labels"] for f in features]
            
            input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
            labels = pad_sequence(batch_labels, batch_first=True, padding_value=-100)
            attention_mask = input_ids.ne(processor.tokenizer.pad_token_id)
            
            pixel_values = []
            image_grid_thw = []
            for f in features:
                pv = f.get("pixel_values", None)
                grid = f.get("image_grid_thw", None)
                if pv is not None and len(pv) > 0:
                    pixel_values.append(torch.tensor(pv) if not isinstance(pv, torch.Tensor) else pv)
                if grid is not None and len(grid) > 0:
                    image_grid_thw.append(torch.tensor(grid) if not isinstance(grid, torch.Tensor) else grid)
                    
            batch = {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask
            }
            if len(pixel_values) > 0:
                batch["pixel_values"] = torch.cat(pixel_values, dim=0)
                batch["image_grid_thw"] = torch.cat(image_grid_thw, dim=0)
            return batch

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
        max_steps=args.max_steps if args.max_steps is not None else -1, # 인자가 있을 때만 적용, 없으면 전체 데이터 학습
        logging_dir="/data/log/vlm-sft",
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        bf16=True,                          # VRAM 절반 감소
        gradient_checkpointing=True,        # Forward 중간 캐시 삭제 (OOM 완벽 해결)
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

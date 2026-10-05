import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.distributed.fsdp.fully_sharded_data_parallel")

import argparse
import os
import datetime
import torch
import torch.nn.functional as F

# 최신 transformers가 구형 PyTorch의 SDPA에 지원하지 않는 enable_gqa 인자를 넘기는 에러를 방지합니다.
_original_sdpa = F.scaled_dot_product_attention
def _patched_sdpa(*args, **kwargs):
    kwargs.pop("enable_gqa", None)
    return _original_sdpa(*args, **kwargs)
F.scaled_dot_product_attention = _patched_sdpa

import torch.distributed as dist
import torch.distributed.fsdp
if not hasattr(torch.distributed.fsdp, "FSDPModule"):
    torch.distributed.fsdp.FSDPModule = type("FSDPModule", (object,), {})
if not hasattr(torch.distributed.fsdp, "register_fsdp_forward_method"):
    torch.distributed.fsdp.register_fsdp_forward_method = lambda *args, **kwargs: None

from datasets import load_from_disk
from transformers import AutoProcessor, AutoModelForCausalLM, TrainingArguments, Trainer

# 최신 transformers 버전에서 제거된 no_init_weights 함수를 HyperCLOVAX 커스텀 코드가 찾을 수 있도록 동적 주입합니다.
import transformers.modeling_utils
import contextlib
if not hasattr(transformers.modeling_utils, "no_init_weights"):
    @contextlib.contextmanager
    def no_init_weights(_enable=True):
        yield
    transformers.modeling_utils.no_init_weights = no_init_weights

# 최신 transformers 모듈이 요구하는 tied weights 속성이 구형 커스텀 모델에 없어 뻗는 에러를 우회합니다.
from transformers import PreTrainedModel
if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
    def _get_tied_weights(self): return getattr(self, "_mock_tied_weights_keys", {})
    def _set_tied_weights(self, value): self._mock_tied_weights_keys = value
    PreTrainedModel.all_tied_weights_keys = property(_get_tied_weights, _set_tied_weights)

# 구형 커스텀 모델의 tie_weights() 함수가 최신 transformers의 추가 인자(missing_keys 등)를 받지 못해 발생하는 TypeError 방지
import inspect
if hasattr(PreTrainedModel, "_finalize_model_loading") and not hasattr(PreTrainedModel, "_is_tie_weights_patched"):
    PreTrainedModel._is_tie_weights_patched = True
    original_finalize = PreTrainedModel._finalize_model_loading
    
    @classmethod
    def patched_finalize(cls, model, *args, **kwargs):
        if hasattr(model, "tie_weights"):
            sig = inspect.signature(model.tie_weights)
            accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            if not accepts_kwargs and "missing_keys" not in sig.parameters:
                original_tie = model.tie_weights
                def safe_tie(*t_args, **t_kwargs):
                    valid_kwargs = {k: v for k, v in t_kwargs.items() if k in sig.parameters}
                    return original_tie(*t_args, **valid_kwargs)
                model.tie_weights = safe_tie
        return original_finalize(model, *args, **kwargs)
            
    PreTrainedModel._finalize_model_loading = patched_finalize

from qwen_vl_utils import process_vision_info

def main():
    parser = argparse.ArgumentParser(description="LLM SFT Distributed Training")
    parser.add_argument("--model_name", type=str, default="naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B", help="HuggingFace model name")
    parser.add_argument("--data_path", type=str, default="/data/vlm_sft_dataset", help="Path to local or PVC dataset")
    parser.add_argument("--output_path", type=str, default="/data/result/vlm-sft-hcx-seed-omni-8b", help="Path to save model")
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

        # 텍스트 전용 데이터 모델 학습 시 FSDP 데드락 방지용 필터 (이제 Dummy Image를 사용하므로 필터링 보류)
        # def has_image(example):
        #     return len(example.get("images", [])) > 0
        # print("Filtering out text-only examples to prevent FSDP deadlock...")
        # combined_train = combined_train.filter(has_image, num_proc=4, desc="Filtering out text-only examples")

        dataset = {"train": combined_train}

        local_only = (rank != 0)
        processor = AutoProcessor.from_pretrained(args.model_name, local_files_only=local_only, trust_remote_code=True)

        # HyperCLOVAX-SEED-Omni-8B는 AutoModelForCausalLM을 사용합니다.
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            local_files_only=local_only,
            trust_remote_code=True,
        )

        # 모델의 모든 파라미터를 bfloat16으로 강제 변환하여 FSDP 파라미터 Flatten 에러 방지
        model.to(torch.bfloat16)

        # 최신 transformers 라이브러리에서 vision_model이 BaseModelOutput을 반환하게 되어
        # mm_projector 입력 타입 오류(TypeError: linear argument must be Tensor)가 생기는 것을 방지하는 몽키 패치
        class MMProjectorWrapper(torch.nn.Module):
            def __init__(self, original_projector):
                super().__init__()
                self.original_projector = original_projector
                
            def forward(self, x, *args, **kwargs):
                if not isinstance(x, torch.Tensor):
                    if hasattr(x, 'last_hidden_state'):
                        x = x.last_hidden_state
                    elif isinstance(x, (tuple, list)):
                        x = x[0]
                    else:
                        x = x[0]
                return self.original_projector(x, *args, **kwargs)

        def patch_all_projectors(module):
            for name, child in module.named_children():
                if name == "mm_projector":
                    setattr(module, name, MMProjectorWrapper(child))
                    print(f"Patched mm_projector in {module.__class__.__name__}")
                else:
                    patch_all_projectors(child)

        patch_all_projectors(model)

        # 2. Vision Model Merger 누락 패치 (transformers >= 4.48)
        # 최신 transformers에서는 Qwen2.5-VL VisionModel이 내부에서 merger를 호출하지 않아 1280 차원이 그대로 나옴
        if hasattr(model, "model") and hasattr(model.model, "vision_model") and hasattr(model.model.vision_model, "merger"):
            orig_vision_forward = model.model.vision_model.forward
            def patched_vision_forward(*args, **kwargs):
                out = orig_vision_forward(*args, **kwargs)
                
                hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else (out[0] if isinstance(out, tuple) else out)
                
                if hasattr(hidden, "shape") and hidden.shape[-1] == 1280:
                    merger = model.model.vision_model.merger
                    try:
                        hidden = merger(hidden)
                    except Exception as e:
                        print(f"merger call failed: {e}")
                        pass
                    
                    if hasattr(out, "last_hidden_state"):
                        from transformers.modeling_outputs import BaseModelOutput
                        if isinstance(out, BaseModelOutput):
                            out = BaseModelOutput(
                                last_hidden_state=hidden,
                                hidden_states=out.hidden_states if hasattr(out, "hidden_states") else None,
                                attentions=out.attentions if hasattr(out, "attentions") else None
                            )
                        else:
                            out.last_hidden_state = hidden
                    elif isinstance(out, tuple):
                        out = (hidden,) + out[1:]
                    else:
                        out = hidden
                return out
                
            model.model.vision_model.forward = patched_vision_forward
            print("Successfully patched vision_model to manually call patch merger")


    # Rank 0와 나머지 노드들의 동기화는 위에서 사용한 main_process_first() 컨텍스트 매니저로 이미 완벽하게 처리되었습니다.
    # 여기에 있던 중복된 dist.barrier()는 Rank 0 혼자만 두 번 기다리게 만들어 무한 데드락을 유발하므로 삭제합니다.
    # 3. 데이터 프롬프트 전처리 (Data Collator에서 실시간 처리)
    def vlm_data_collator(features):
        texts = []
        prompt_texts = []
        image_inputs = []
        
        for feature in features:
            messages = feature.get("messages", [])
            images = feature.get("images", []) # PIL Image 리스트
            
            # 텍스트 전용 데이터일 경우 FSDP 데드락을 피하기 위해 28x28 픽셀의 검은색 Dummy 이미지 삽입
            if len(images) == 0:
                from PIL import Image
                dummy_image = Image.new('RGB', (28, 28), (0, 0, 0))
                images = [dummy_image]
                
                # 첫 번째 유저 발화 제일 앞에 <image> 토큰 추가
                for msg in messages:
                    if msg["role"] == "user":
                        msg["content"] = "<image>\n" + msg["content"]
                        break
            
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
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=args.max_steps if args.max_steps is not None else -1, # 인자가 있을 때만 적용, 없으면 전체 데이터 학습
        logging_dir="/data/log/vlm-sft",
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        bf16=True,                          # VRAM 절반 감소
        gradient_checkpointing=True,        # Forward 중간 캐시 삭제 (OOM 완벽 해결)
        fsdp="full_shard auto_wrap",
        fsdp_config={"transformer_layer_cls_to_wrap": ["LlamaDecoderLayer", "Qwen2_5_VLVisionBlock"]},
        ddp_find_unused_parameters=False,
        remove_unused_columns=False, # VLM에서는 image가 사용되므로 False 필수
    )

    # 4.5. 모델 저장 시 JSON 직렬화 에러 대응 (torch.dtype 객체를 문자열로 치환)
    def clean_config_dtype(config_obj):
        for k, v in config_obj.__dict__.items():
            if isinstance(v, torch.dtype):
                setattr(config_obj, k, str(v).split('.')[-1])
            elif hasattr(v, '__dict__'):
                clean_config_dtype(v)
            elif isinstance(v, dict):
                for dk, dv in v.items():
                    if isinstance(dv, torch.dtype):
                        v[dk] = str(dv).split('.')[-1]
                    elif hasattr(dv, '__dict__'):
                        clean_config_dtype(dv)
    clean_config_dtype(model.config)

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

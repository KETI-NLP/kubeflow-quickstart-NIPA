import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.distributed.fsdp.fully_sharded_data_parallel")

import argparse
import os
import datetime
import torch
import torch.nn.functional as F

# ==========================================
# 몽키 패치 & 핫픽스 (Monkey Patches) 모음
# ==========================================
def apply_monkey_patches():
    # 1. 최신 transformers가 구형 PyTorch의 SDPA에 지원하지 않는 enable_gqa 인자를 넘기는 에러 방지
    _original_sdpa = F.scaled_dot_product_attention
    def _patched_sdpa(*args, **kwargs):
        kwargs.pop("enable_gqa", None)
        return _original_sdpa(*args, **kwargs)
    F.scaled_dot_product_attention = _patched_sdpa

    # 2. FSDP 관련 속성 더미 주입 (버전 호환성 문제 우회)
    import torch.distributed.fsdp
    if not hasattr(torch.distributed.fsdp, "FSDPModule"):
        torch.distributed.fsdp.FSDPModule = type("FSDPModule", (object,), {})
    if not hasattr(torch.distributed.fsdp, "register_fsdp_forward_method"):
        torch.distributed.fsdp.register_fsdp_forward_method = lambda *args, **kwargs: None

    # 3. 최신 transformers에서 제거된 no_init_weights 함수 복원
    import transformers.modeling_utils
    import contextlib
    if not hasattr(transformers.modeling_utils, "no_init_weights"):
        @contextlib.contextmanager
        def no_init_weights(_enable=True): yield
        transformers.modeling_utils.no_init_weights = no_init_weights

    # 3.5. [HOTFIX] save_pretrained 직렬화 시 tied_weights 구조 에러(AttributeError: 'list' object has no attribute 'keys') 방지
    transformers.modeling_utils.remove_tied_weights_from_state_dict = lambda state_dict, *args, **kwargs: state_dict

    # 4. 구형 커스텀 모델의 tie_weights() 파라미터 에러 우회
    from transformers import PreTrainedModel
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        def _get_tied_weights(self): return getattr(self, "_mock_tied_weights_keys", {})
        def _set_tied_weights(self, value): self._mock_tied_weights_keys = value
        PreTrainedModel.all_tied_weights_keys = property(_get_tied_weights, _set_tied_weights)

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

    # 5. [HOTFIX] Naver MLX Namespace Collision Bypass
    import transformers.utils.generic as generic_utils
    generic_utils._is_mlx_available = False
    generic_utils.is_mlx_available = lambda: False
    
    # 6. Fsspec Maxdepth KeyError 방지 패치
    try:
        import huggingface_hub.hf_file_system
        import fsspec.spec
        huggingface_hub.hf_file_system.HfFileSystem.glob = fsspec.spec.AbstractFileSystem.glob
    except Exception:
        pass

apply_monkey_patches()

from datasets import load_from_disk, concatenate_datasets
from transformers import AutoProcessor, AutoModelForCausalLM, TrainingArguments, Trainer
import torch.distributed as dist

def main():
    parser = argparse.ArgumentParser(description="Unified VLM SFT Distributed Training for HyperCLOVA X")
    parser.add_argument("--model_name", type=str, default="naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B", help="HuggingFace model name")
    parser.add_argument("--data_path", type=str, required=True, help="Path to local or PVC dataset (comma separated)")
    parser.add_argument("--output_path", type=str, default="/data/result/vlm-sft-unified", help="Path to save model")
    
    # 하드코딩되었던 변수들을 외부 인자로 추출
    parser.add_argument("--logging_dir", type=str, default="/data/log/vlm-sft-unified", help="Tensorboard logging directory")
    parser.add_argument("--num_train_epochs", type=float, default=3.0, help="Total number of training epochs to perform")
    parser.add_argument("--max_steps", type=int, default=None, help="Max steps for training (None means full epochs)")
    parser.add_argument("--max_sequence_len", type=int, default=8192, help="Max sequence length for data collator")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--save_steps", type=int, default=200, help="Save interval steps")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging interval steps")
    
    parser.add_argument("--use_test_dataset", action="store_true", help="Redirects data paths to the test folder for fast debugging")
    parser.add_argument("--apply_rope_patch", action="store_true", help="Force apply 32B RoPE patch bypass logic")
    args = parser.parse_args()

    if args.use_test_dataset:
        args.data_path = args.data_path.replace("convert_to_llm_training_ready", "convert_to_llm_training_ready_test")

    # 분산 환경 초기화
    if "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", timeout=datetime.timedelta(seconds=7200))
        rank = dist.get_rank()
    else:
        rank = 0

    if rank == 0:
        print(f"Loaded config: {args}")

    import contextlib
    @contextlib.contextmanager
    def main_process_first():
        if rank != 0: dist.barrier()
        yield
        if rank == 0: dist.barrier()

    with main_process_first():
        local_only = (rank != 0)
        
        orig_hf_endpoint = os.environ.pop("HF_ENDPOINT", None)
        os.environ["HF_ENDPOINT"] = "https://huggingface.co"
        
        processor = AutoProcessor.from_pretrained(args.model_name, local_files_only=local_only, trust_remote_code=True)
        
        # 모델 로딩 및 RoPE 에러 핸들링 (통합 스크립트화)
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(args.model_name, local_files_only=local_only, trust_remote_code=True)
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                config=config,
                attn_implementation="sdpa",
                torch_dtype=torch.bfloat16,
                local_files_only=local_only,
                trust_remote_code=True,
            )
        except KeyError as e:
            if rank == 0:
                print(f"Caught KeyError during load: {e}. Attempting HyperCLOVAX 32B RoPE Patch...")
            
            import sys
            for mod_name, mod in list(sys.modules.items()):
                if "modeling_hyperclovax" in mod_name and hasattr(mod, "ROPE_INIT_FUNCTIONS"):
                    rope_dict = getattr(mod, "ROPE_INIT_FUNCTIONS")
                    if "default" not in rope_dict:
                        import transformers.modeling_rope_utils as rope_utils
                        rope_fn = getattr(rope_utils, "_compute_default_rope_parameters", getattr(rope_utils, "compute_default_rope_parameters", None))
                        
                        if not rope_fn and hasattr(rope_utils, "ROPE_INIT_FUNCTIONS"):
                            if "linear" in rope_utils.ROPE_INIT_FUNCTIONS:
                                base_fn = rope_utils.ROPE_INIT_FUNCTIONS["linear"]
                                def safe_rope_fn(c, d, **kwargs):
                                    if getattr(c, "rope_scaling", None) is None: c.rope_scaling = {}
                                    if "factor" not in c.rope_scaling: c.rope_scaling["factor"] = 1.0
                                    return base_fn(c, d, **kwargs)
                                rope_fn = safe_rope_fn

                        if rope_fn:
                            rope_dict["default"] = rope_fn
            
            # 패치 적용 후 다시 로딩 시도
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                config=config,
                attn_implementation="sdpa",
                torch_dtype=torch.bfloat16,
                local_files_only=local_only,
                trust_remote_code=True,
            )

        model.to(torch.bfloat16)

        if orig_hf_endpoint is not None: 
            os.environ["HF_ENDPOINT"] = orig_hf_endpoint

        # mm_projector patch (Type에러 방지)
        class MMProjectorWrapper(torch.nn.Module):
            def __init__(self, original_projector):
                super().__init__()
                self.original_projector = original_projector
                
            def forward(self, x, *args, **kwargs):
                if not isinstance(x, torch.Tensor):
                    x = x.last_hidden_state if hasattr(x, 'last_hidden_state') else (x[0] if isinstance(x, (tuple, list)) else x[0])
                return self.original_projector(x, *args, **kwargs)

        def patch_all_projectors(module):
            for name, child in module.named_children():
                if name == "mm_projector":
                    setattr(module, name, MMProjectorWrapper(child))
                else:
                    patch_all_projectors(child)
        patch_all_projectors(model)

        # Vision Model Merger 패치: 하드코딩된 1280 대신 config 동적 확인
        if hasattr(model, "model") and hasattr(model.model, "vision_model") and hasattr(model.model.vision_model, "merger"):
            # 동적 Hidden Size 추출
            vision_config = getattr(model.config, "vision_config", None)
            vision_hidden_size = getattr(vision_config, "hidden_size", 1280) if vision_config else 1280
            
            orig_vision_forward = model.model.vision_model.forward
            def patched_vision_forward(*args, **kwargs):
                out = orig_vision_forward(*args, **kwargs)
                hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else (out[0] if isinstance(out, tuple) else out)
                
                # 추출한 동적 vision dimension과 비교
                if hasattr(hidden, "shape") and hidden.shape[-1] == vision_hidden_size:
                    merger = model.model.vision_model.merger
                    try:
                        hidden = merger(hidden)
                    except Exception:
                        pass
                    
                    if hasattr(out, "last_hidden_state"):
                        from transformers.modeling_outputs import BaseModelOutput
                        if isinstance(out, BaseModelOutput):
                            out = BaseModelOutput(
                                last_hidden_state=hidden,
                                hidden_states=getattr(out, "hidden_states", None),
                                attentions=getattr(out, "attentions", None)
                            )
                        else:
                            out.last_hidden_state = hidden
                    elif isinstance(out, tuple):
                        out = (hidden,) + out[1:]
                    else:
                        out = hidden
                return out
                
            model.model.vision_model.forward = patched_vision_forward

        # 데이터 부름 및 병합
        data_paths = [dp.strip() for dp in args.data_path.split(",")]
        dataset_list = []

        for dp in data_paths:
            if "/" in dp and not os.path.exists(dp) and not dp.startswith("/data/"):
                try:
                    from mlx.sdk.data import load_dataset as mlxp_load_dataset
                    ds = mlxp_load_dataset(dp)
                except ImportError:
                    ds = load_from_disk(dp)
            else:
                ds = load_from_disk(dp)

            from datasets import DatasetDict
            if isinstance(ds, DatasetDict) and "train" in ds:
                train_ds = ds["train"]
            else:
                train_ds = ds

            dataset_list.append(train_ds)

        combined_train = concatenate_datasets(dataset_list)

        # 디스크 IO 및 프로세스 잠금으로 인한 NFS 데드락 방지를 위해 num_proc=1로 하향조정
        combined_train = combined_train.filter(lambda x: x.get("input_ids") is not None, num_proc=1, load_from_cache_file=False)

        dataset = {"train": combined_train}

    # Data Collator 정의
    def vlm_data_collator(features):
        input_ids_list = []
        labels_list = []
        
        for f in features:
            if isinstance(f["input_ids"], list):
                input_ids_list.append(torch.tensor(f["input_ids"], dtype=torch.long))
            else:
                input_ids_list.append(torch.tensor(f["input_ids"]).long())
                
            if isinstance(f["labels"], list):
                labels_list.append(torch.tensor(f["labels"], dtype=torch.long))
            else:
                labels_list.append(torch.tensor(f["labels"]).long())
                
        pad_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else processor.tokenizer.eos_token_id
        if pad_id is None: pad_id = 0
            
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
        
        max_sequence_len = args.max_sequence_len
        current_len = input_ids.shape[1]
        if current_len < max_sequence_len:
            pad_len = max_sequence_len - current_len
            input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=pad_id)
            labels = torch.nn.functional.pad(labels, (0, pad_len), value=-100)
        elif current_len > max_sequence_len:
            input_ids = input_ids[:, :max_sequence_len]
            labels = labels[:, :max_sequence_len]
            
        attention_mask = (input_ids != pad_id).long()
        
        all_images = []
        for f in features:
            images = f.get("images", [])
            if images is not None and len(images) > 0:
                all_images.append(images)
            else:
                from PIL import Image
                dummy_image = Image.new('RGB', (28, 28), (0, 0, 0))
                all_images.append([dummy_image])
                
        image_batch = processor(
            text=[""] * len(all_images), # Processor의 NoneType 에러 우회를 위한 더미 텍스트 유지
            images=all_images,
            padding=False,
            return_tensors="pt"
        )
        
        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
        
        for k, v in image_batch.items():
            if k not in batch:
                batch[k] = v
            
        return batch

    if rank == 0:
        print("Dataset Ready. Training will collate on the fly.")

    valid_wrap_classes = list(set(
        m.__class__.__name__ for m in model.modules() 
        if "DecoderLayer" in m.__class__.__name__ or ("VisionBlock" in m.__class__.__name__ and "Qwen" in m.__class__.__name__)
    ))
    if not valid_wrap_classes: valid_wrap_classes = ["LlamaDecoderLayer"]
    
    if rank == 0:
        print(f"FSDP dynamically wrapping layer classes: {valid_wrap_classes}")

    training_args = TrainingArguments(
        output_dir=args.output_path,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        bf16=True,                          
        gradient_checkpointing=True,        
        fsdp="full_shard auto_wrap",
        fsdp_config={"transformer_layer_cls_to_wrap": valid_wrap_classes},
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )

    def clean_config_dtype(config_obj):
        for k, v in config_obj.__dict__.items():
            if isinstance(v, torch.dtype): getattr(config_obj, k); setattr(config_obj, k, str(v).split('.')[-1])
            elif hasattr(v, '__dict__'): clean_config_dtype(v)
            elif isinstance(v, dict):
                for dk, dv in v.items():
                    if isinstance(dv, torch.dtype): v[dk] = str(dv).split('.')[-1]
                    elif hasattr(dv, '__dict__'): clean_config_dtype(dv)
    clean_config_dtype(model.config)

    from transformers import TrainerCallback
    class GPUMemoryCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if torch.cuda.is_available():
                mem_alloc = torch.cuda.memory_allocated() / (1024**3)
                mem_max = torch.cuda.max_memory_allocated() / (1024**3)
                rank = os.environ.get("RANK", "0")
                if rank == "0":
                    print(f"\n[Step {state.global_step}] GPU Mem Allocated: {mem_alloc:.2f} GB | Max: {mem_max:.2f} GB\n", flush=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=vlm_data_collator,
        callbacks=[GPUMemoryCallback()],
    )

    # 7. 분산 학습 FSDP Save Model시 타 노드끼리 동시에 캐시 삭제/액세스를 하다 터지는 문제 핸들링 (기존 에러 삼키기 최소화)
    import shutil
    original_rmtree = shutil.rmtree
    def safe_rmtree(path, *args, **kwargs):
        try:
            original_rmtree(path, *args, **kwargs)
        except Exception as e:
            if rank == 0:
                print(f"[Warn] Non-critical error during rmtree {path}: {e}")
    shutil.rmtree = safe_rmtree

    # 학습 시작
    if rank == 0:
        print("Starting training...")
    trainer.train()
    
    # 모델 저장 (FSDP는 모든 랭크가 모여야 Gather 되므로 모든 워커가 호출)
    trainer.save_model(args.output_path)
    if rank == 0:
        print(f"Model successfully saved to {args.output_path}")
        try:
            if processor is not None:
                processor.save_pretrained(args.output_path)
                print(f"Processor/tokenizer successfully saved to {args.output_path}")
        except Exception as e:
            print(
                f"[Warn] Model save succeeded but processor/tokenizer save failed at {args.output_path}: {e}",
                flush=True,
            )

    dist.barrier()

if __name__ == "__main__":
    main()

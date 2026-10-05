import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.distributed.fsdp.fully_sharded_data_parallel")

import argparse
import contextlib
import datetime
import inspect
import os
import sys

import torch
import torch.distributed as dist
import torch.distributed.fsdp
import torch.nn.functional as F

from datasets import DatasetDict, concatenate_datasets, load_from_disk
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor


def apply_monkey_patches():
    _original_sdpa = F.scaled_dot_product_attention

    def _patched_sdpa(*args, **kwargs):
        kwargs.pop("enable_gqa", None)
        return _original_sdpa(*args, **kwargs)

    F.scaled_dot_product_attention = _patched_sdpa

    if not hasattr(torch.distributed.fsdp, "FSDPModule"):
        torch.distributed.fsdp.FSDPModule = type("FSDPModule", (object,), {})
    if not hasattr(torch.distributed.fsdp, "register_fsdp_forward_method"):
        torch.distributed.fsdp.register_fsdp_forward_method = lambda *args, **kwargs: None

    import transformers.modeling_utils

    if not hasattr(transformers.modeling_utils, "no_init_weights"):
        @contextlib.contextmanager
        def no_init_weights(_enable=True):
            yield

        transformers.modeling_utils.no_init_weights = no_init_weights

    transformers.modeling_utils.remove_tied_weights_from_state_dict = (
        lambda state_dict, *args, **kwargs: state_dict
    )

    from transformers import PreTrainedModel

    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        def _get_tied_weights(self):
            return getattr(self, "_mock_tied_weights_keys", {})

        def _set_tied_weights(self, value):
            self._mock_tied_weights_keys = value

        PreTrainedModel.all_tied_weights_keys = property(_get_tied_weights, _set_tied_weights)

    if hasattr(PreTrainedModel, "_finalize_model_loading") and not hasattr(PreTrainedModel, "_is_tie_weights_patched"):
        PreTrainedModel._is_tie_weights_patched = True
        original_finalize = PreTrainedModel._finalize_model_loading

        @classmethod
        def patched_finalize(cls, model, *args, **kwargs):
            if hasattr(model, "tie_weights"):
                sig = inspect.signature(model.tie_weights)
                accepts_kwargs = any(
                    param.kind == inspect.Parameter.VAR_KEYWORD
                    for param in sig.parameters.values()
                )
                if not accepts_kwargs and "missing_keys" not in sig.parameters:
                    original_tie = model.tie_weights

                    def safe_tie(*tie_args, **tie_kwargs):
                        valid_kwargs = {
                            key: value
                            for key, value in tie_kwargs.items()
                            if key in sig.parameters
                        }
                        return original_tie(*tie_args, **valid_kwargs)

                    model.tie_weights = safe_tie
            return original_finalize(model, *args, **kwargs)

        PreTrainedModel._finalize_model_loading = patched_finalize

    import transformers.utils.generic as generic_utils

    generic_utils._is_mlx_available = False
    generic_utils.is_mlx_available = lambda: False

    try:
        import huggingface_hub.hf_file_system
        import fsspec.spec

        huggingface_hub.hf_file_system.HfFileSystem.glob = fsspec.spec.AbstractFileSystem.glob
    except Exception:
        pass


apply_monkey_patches()

from trl import DPOConfig, DPOTrainer


DEFAULT_DATA_PATH = "YOUR_WORKSPACE/data_kr_obj_img_4_hallu_txt_dpo_hf"
DEFAULT_MODEL_NAME = "naver-hyperclovax/HyperCLOVAX-SEED-Think-32B"


def init_distributed():
    if "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                timeout=datetime.timedelta(seconds=7200),
            )
        return dist.get_rank()
    return 0


@contextlib.contextmanager
def main_process_first(rank):
    if dist.is_available() and dist.is_initialized() and rank != 0:
        dist.barrier()
    yield
    if dist.is_available() and dist.is_initialized() and rank == 0:
        dist.barrier()


def load_dataset_from_source(data_path):
    if "/" in data_path and not os.path.exists(data_path) and not data_path.startswith("/data/"):
        from mlx.sdk.data import load_dataset as mlxp_load_dataset

        return mlxp_load_dataset(data_path)
    return load_from_disk(data_path)


def normalize_train_split(ds):
    if isinstance(ds, DatasetDict) and "train" in ds:
        return ds["train"]
    return ds


def coerce_to_dpo_schema(train_ds, source_name):
    required_columns = {"prompt", "chosen", "rejected"}
    if required_columns.issubset(set(train_ds.column_names)):
        return train_ds

    raise ValueError(
        f"{source_name} does not look like a DPO dataset. "
        f"Required columns: {sorted(required_columns)} | actual: {sorted(train_ds.column_names)}"
    )


def summarize_example(example):
    summary = {}
    for key, value in example.items():
        value_type = type(value).__name__
        if isinstance(value, (list, tuple)):
            summary[key] = f"{value_type}[len={len(value)}]"
        elif isinstance(value, dict):
            summary[key] = f"dict(keys={list(value.keys())[:5]})"
        else:
            summary[key] = value_type
    return summary


def load_and_merge_datasets(data_path, rank):
    data_paths = [dp.strip() for dp in data_path.split(",") if dp.strip()]
    if not data_paths:
        raise ValueError("No dataset paths were provided. Pass at least one MLXP dataset ID or local path.")

    dataset_list = []
    dataset_stats = []

    for source_name in data_paths:
        ds = load_dataset_from_source(source_name)
        train_ds = normalize_train_split(ds)
        train_ds = coerce_to_dpo_schema(train_ds, source_name)
        dataset_list.append(train_ds)
        dataset_stats.append((source_name, len(train_ds)))

        if rank == 0:
            preview = summarize_example(train_ds[0]) if len(train_ds) > 0 else {}
            print(
                f"Loaded dataset: {source_name} | samples={len(train_ds):,} | "
                f"columns={train_ds.column_names} | sample0={preview}",
                flush=True,
            )

    merged = dataset_list[0] if len(dataset_list) == 1 else concatenate_datasets(dataset_list)

    if rank == 0:
        for source_name, sample_count in dataset_stats:
            print(f"  - merge_source: {source_name} | samples={sample_count:,}", flush=True)
        print(
            f"Combined {len(dataset_list)} datasets into train split with {len(merged):,} samples.",
            flush=True,
        )

    return {"train": merged}


def patch_mm_projector(model):
    class MMProjectorWrapper(torch.nn.Module):
        def __init__(self, original_projector):
            super().__init__()
            self.original_projector = original_projector

        def forward(self, x, *args, **kwargs):
            if not isinstance(x, torch.Tensor):
                if hasattr(x, "last_hidden_state"):
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
            else:
                patch_all_projectors(child)

    patch_all_projectors(model)


def patch_vision_merger(model):
    if not (hasattr(model, "model") and hasattr(model.model, "vision_model") and hasattr(model.model.vision_model, "merger")):
        return

    vision_config = getattr(model.config, "vision_config", None)
    vision_hidden_size = getattr(vision_config, "hidden_size", 1280) if vision_config else 1280
    orig_vision_forward = model.model.vision_model.forward

    def patched_vision_forward(*args, **kwargs):
        out = orig_vision_forward(*args, **kwargs)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else (out[0] if isinstance(out, tuple) else out)

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
                        attentions=getattr(out, "attentions", None),
                    )
                else:
                    out.last_hidden_state = hidden
            elif isinstance(out, tuple):
                out = (hidden,) + out[1:]
            else:
                out = hidden

        return out

    model.model.vision_model.forward = patched_vision_forward


def maybe_patch_hcx_rope(rank):
    for module_name, module in list(sys.modules.items()):
        if "modeling_hyperclovax" in module_name and hasattr(module, "ROPE_INIT_FUNCTIONS"):
            rope_dict = getattr(module, "ROPE_INIT_FUNCTIONS")
            if "default" in rope_dict:
                continue

            import transformers.modeling_rope_utils as rope_utils

            rope_fn = getattr(
                rope_utils,
                "_compute_default_rope_parameters",
                getattr(rope_utils, "compute_default_rope_parameters", None),
            )

            if not rope_fn and hasattr(rope_utils, "ROPE_INIT_FUNCTIONS") and "linear" in rope_utils.ROPE_INIT_FUNCTIONS:
                base_fn = rope_utils.ROPE_INIT_FUNCTIONS["linear"]

                def safe_rope_fn(config_obj, device, **kwargs):
                    if getattr(config_obj, "rope_scaling", None) is None:
                        config_obj.rope_scaling = {}
                    if "factor" not in config_obj.rope_scaling:
                        config_obj.rope_scaling["factor"] = 1.0
                    return base_fn(config_obj, device, **kwargs)

                rope_fn = safe_rope_fn

            if rope_fn:
                rope_dict["default"] = rope_fn
                if rank == 0:
                    print(f"Applied RoPE patch for module: {module_name}", flush=True)


def load_single_model(model_name, local_only, rank):
    import huggingface_hub.constants

    orig_hf_endpoint = os.environ.pop("HF_ENDPOINT", None)
    orig_hf_token = os.environ.pop("HF_TOKEN", None)
    huggingface_hub.constants.ENDPOINT = "https://huggingface.co"
    os.environ["HF_ENDPOINT"] = "https://huggingface.co"

    processor = AutoProcessor.from_pretrained(
        model_name,
        local_files_only=local_only,
        trust_remote_code=True,
    )

    config = AutoConfig.from_pretrained(
        model_name,
        local_files_only=local_only,
        trust_remote_code=True,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            local_files_only=local_only,
            trust_remote_code=True,
        )
    except KeyError as exc:
        if rank == 0:
            print(f"Caught KeyError during load: {exc}. Attempting HyperCLOVAX RoPE patch...", flush=True)
        maybe_patch_hcx_rope(rank)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            local_files_only=local_only,
            trust_remote_code=True,
        )

    if orig_hf_endpoint is not None:
        os.environ["HF_ENDPOINT"] = orig_hf_endpoint
    else:
        os.environ.pop("HF_ENDPOINT", None)

    if orig_hf_token is not None:
        os.environ["HF_TOKEN"] = orig_hf_token

    huggingface_hub.constants.ENDPOINT = orig_hf_endpoint if orig_hf_endpoint else "https://huggingface.co"

    model.to(torch.bfloat16)
    patch_mm_projector(model)
    patch_vision_merger(model)
    return processor, model


def load_model_and_processor(model_name, local_only, rank):
    processor, model = load_single_model(model_name, local_only=local_only, rank=rank)
    _, ref_model = load_single_model(model_name, local_only=local_only, rank=rank)
    return processor, model, ref_model


def main():
    parser = argparse.ArgumentParser(description="VLM DPO HyperCLOVAX Distributed Training")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="HuggingFace model name")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH, help="Comma-separated local paths or MLXP dataset IDs")
    parser.add_argument("--output_path", type=str, default="/data/result/vlm-dpo-hcx-think-32b", help="Path to save model")
    parser.add_argument("--max_steps", type=int, default=50, help="Max training steps")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Gradient accumulation steps")
    args = parser.parse_args()

    rank = init_distributed()

    with main_process_first(rank):
        dataset = load_and_merge_datasets(args.data_path, rank)
        local_only = rank != 0
        processor, model, ref_model = load_model_and_processor(args.model_name, local_only=local_only, rank=rank)

    valid_wrap_classes = list(set(
        module.__class__.__name__
        for module in model.modules()
        if "DecoderLayer" in module.__class__.__name__ or "VisionBlock" in module.__class__.__name__
    ))
    if not valid_wrap_classes:
        valid_wrap_classes = ["HyperCLOVAXDecoderLayer"]

    training_args = DPOConfig(
        output_dir=args.output_path,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        logging_dir="/data/log/vlm-dpo-hcx",
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        bf16=True,
        gradient_checkpointing=True,
        fsdp="full_shard auto_wrap",
        fsdp_config={"transformer_layer_cls_to_wrap": valid_wrap_classes},
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        dataset_num_proc=16,
        ddp_timeout=7200,
        max_prompt_length=4096,
        max_length=8192,
    )

    if rank == 0:
        print(
            f"Starting DPO with model={args.model_name} | "
            f"wrap_classes={valid_wrap_classes} | "
            f"train_samples={len(dataset['train']):,}",
            flush=True,
        )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset["train"],
        processing_class=processor,
    )

    if getattr(trainer, "ref_model", None) is not None:
        trainer.ref_model = trainer.ref_model.to(trainer.accelerator.device)

    import shutil

    original_rmtree = shutil.rmtree

    def safe_rmtree(path, *rmtree_args, **rmtree_kwargs):
        try:
            original_rmtree(path, *rmtree_args, **rmtree_kwargs)
        except FileNotFoundError:
            pass

    shutil.rmtree = safe_rmtree

    trainer.train()
    trainer.save_model(args.output_path)

    if rank == 0:
        print(f"Model successfully saved to {args.output_path}", flush=True)
        try:
            if processor is not None:
                processor.save_pretrained(args.output_path)
                print(f"Processor/tokenizer successfully saved to {args.output_path}", flush=True)
        except Exception as e:
            print(
                f"[Warn] Model save succeeded but processor/tokenizer save failed at {args.output_path}: {e}",
                flush=True,
            )

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()

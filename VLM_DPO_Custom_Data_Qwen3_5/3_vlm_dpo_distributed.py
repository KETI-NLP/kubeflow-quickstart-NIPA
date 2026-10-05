import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.distributed.fsdp.fully_sharded_data_parallel")

import argparse
import contextlib
import datetime
import glob
import inspect
import io
import os
import time
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.fsdp

from datasets import DatasetDict, concatenate_datasets, load_from_disk
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

# Prevent transformers from passing unsupported kwargs to older PyTorch SDPA.
_original_sdpa = F.scaled_dot_product_attention
def _patched_sdpa(*args, **kwargs):
    kwargs.pop("enable_gqa", None)
    return _original_sdpa(*args, **kwargs)
F.scaled_dot_product_attention = _patched_sdpa

# Fix trl>=0.15.0 import error on some PyTorch 2.5.x environments.
if not hasattr(torch.distributed.fsdp, "FSDPModule"):
    torch.distributed.fsdp.FSDPModule = type("FSDPModule", (object,), {})
if not hasattr(torch.distributed.fsdp, "register_fsdp_forward_method"):
    torch.distributed.fsdp.register_fsdp_forward_method = lambda *args, **kwargs: None

# [HOTFIX] HuggingFace MLX vs Naver MLX namespace collision bypass.
import transformers.utils.generic as generic_utils
generic_utils._is_mlx_available = False
generic_utils.is_mlx_available = lambda: False

from trl import DPOConfig, DPOTrainer


DEFAULT_DATA_PATH = ",".join([
    "YOUR_WORKSPACE/data_kr_obj_img_4_hallu_txt_dpo_hf",
    "YOUR_WORKSPACE/data_kr_obj_img_5_hallu_wr_txt_dpo_hf",
    "YOUR_WORKSPACE/data_kr_heri_vqa_dpo_hf",
])

DEFAULT_MODEL_PATH = "/data/checkpoints/qwen3_5_9b_multimodal_sft"
DEFAULT_OUTPUT_PATH = "/data/checkpoints/qwen3_5_9b_multimodal_sft_dpo"
DEFAULT_LOGGING_DIR = "/data/log/vlm-dpo-qwen3_5"
PROCESSOR_CONFIG_FILES = (
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer_config.json",
)
PREPROCESS_METADATA_FILENAME = "preprocess_metadata.json"


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


def load_dataset_from_source(data_path, max_retries=5, retry_sleep_seconds=5):
    if "/" in data_path and not os.path.exists(data_path) and not data_path.startswith("/data/"):
        from mlx.sdk.data import load_dataset as mlxp_load_dataset
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                return mlxp_load_dataset(data_path)
            except Exception as exc:
                last_error = exc
                if attempt == max_retries:
                    break
                print(
                    f"[Dataset Load Retry] source={data_path} | attempt={attempt}/{max_retries} failed with "
                    f"{type(exc).__name__}: {exc}. Retrying in {retry_sleep_seconds}s...",
                    flush=True,
                )
                time.sleep(retry_sleep_seconds)
        raise last_error

    try:
        return load_from_disk(data_path)
    except Exception as load_from_disk_error:
        from mlx.sdk.data import load_dataset as mlxp_load_dataset

        local_candidates = [data_path]
        wildcard_candidate = os.path.join(data_path, "*")
        if glob.glob(wildcard_candidate):
            local_candidates.append(wildcard_candidate)

        last_error = load_from_disk_error
        for candidate in local_candidates:
            try:
                return mlxp_load_dataset(candidate)
            except Exception as exc:
                last_error = exc

        raise RuntimeError(
            f"Failed to load local dataset from {data_path}. "
            f"Tried datasets.load_from_disk() and mlx.sdk.data.load_dataset() candidates={local_candidates}. "
            f"Last error: {type(last_error).__name__}: {last_error}"
        ) from load_from_disk_error


def normalize_train_split(ds):
    if isinstance(ds, DatasetDict) and "train" in ds:
        return ds["train"]
    return ds


def is_preprocessed_dataset_source(source_name):
    return os.path.isfile(os.path.join(source_name, PREPROCESS_METADATA_FILENAME))


def coerce_to_dpo_schema(train_ds, source_name):
    required_columns = {"prompt", "chosen", "rejected"}
    column_names = set(train_ds.column_names)

    if required_columns.issubset(column_names):
        return train_ds

    if "messages" in column_names and {"chosen", "rejected"}.issubset(column_names):
        def convert_messages_to_prompt(example):
            return {"prompt": example["messages"]}
        return train_ds.map(convert_messages_to_prompt, desc=f"add_prompt::{source_name}")

    raise ValueError(
        f"{source_name} does not look like a DPO dataset. "
        f"Required columns: {sorted(required_columns)} | actual: {sorted(train_ds.column_names)}"
    )


def normalize_image_payload(image_payload):
    if image_payload is None:
        return None
    if isinstance(image_payload, dict):
        img_bytes = image_payload.get("bytes")
        img_path = image_payload.get("path")
        image_value = image_payload.get("image")
        image_url = image_payload.get("url")
        if img_bytes is not None or img_path is not None:
            return {"bytes": img_bytes, "path": img_path}
        if image_value is not None and image_value is not image_payload:
            return normalize_image_payload(image_value)
        if image_url is not None:
            return image_url
        return image_payload
    return image_payload


def extract_images_from_messages(messages):
    if not isinstance(messages, list):
        return messages, []

    normalized_messages = []
    extracted_images = []

    for message in messages:
        if not isinstance(message, dict):
            normalized_messages.append(message)
            continue

        content = message.get("content")
        if not isinstance(content, list):
            normalized_messages.append(message)
            continue

        normalized_content = []
        for item in content:
            if not isinstance(item, dict):
                normalized_content.append(item)
                continue

            if item.get("type") == "image":
                image_payload = None
                for key in ("image", "images", "image_url", "url", "path", "bytes"):
                    if key in item and item[key] is not None:
                        image_payload = item[key]
                        break

                if isinstance(image_payload, list):
                    for payload in image_payload:
                        normalized_payload = normalize_image_payload(payload)
                        if normalized_payload is not None:
                            extracted_images.append(normalized_payload)
                else:
                    normalized_payload = normalize_image_payload(image_payload)
                    if normalized_payload is not None:
                        extracted_images.append(normalized_payload)

                normalized_item = {"type": "image", "text": ""}
                if item.get("text") not in (None, ""):
                    normalized_item["text"] = item["text"]
                normalized_content.append(normalized_item)
            else:
                item_type = item.get("type")
                if item_type == "text":
                    normalized_text_item = {"type": "text", "text": ""}
                    if item.get("text") is not None:
                        normalized_text_item["text"] = item.get("text")
                    normalized_content.append(normalized_text_item)
                else:
                    # For other types, ensure it at least has type and text to avoid schema conflicts
                    normalized_other = dict(item)
                    if "type" not in normalized_other:
                        normalized_other["type"] = item_type or "unknown"
                    if "text" not in normalized_other:
                        normalized_other["text"] = ""
                    normalized_content.append(normalized_other)

        normalized_message = dict(message)
        normalized_message["content"] = normalized_content
        normalized_messages.append(normalized_message)

    return normalized_messages, extracted_images


def ensure_list_images(images_value):
    if images_value is None:
        return []
    if isinstance(images_value, list):
        values = images_value
    else:
        values = [images_value]

    normalized_images = []
    for image_value in values:
        normalized_payload = normalize_image_payload(image_value)
        if normalized_payload is not None:
            normalized_images.append(normalized_payload)
    return normalized_images


def select_images_for_prompt(prompt_placeholder_count, *image_candidates):
    normalized_candidates = [ensure_list_images(candidate) for candidate in image_candidates]

    if prompt_placeholder_count <= 0:
        for candidate in normalized_candidates:
            if candidate:
                return candidate
        return []

    for candidate in normalized_candidates:
        if len(candidate) == prompt_placeholder_count:
            return candidate

    for candidate in normalized_candidates:
        if len(candidate) > prompt_placeholder_count:
            return candidate[:prompt_placeholder_count]

    for candidate in normalized_candidates:
        if candidate:
            return candidate

    return []


def normalize_multimodal_dpo_example(example):
    normalized_example = dict(example)

    prompt_messages, prompt_images = extract_images_from_messages(normalized_example.get("prompt"))
    chosen_messages, chosen_images = extract_images_from_messages(normalized_example.get("chosen"))
    rejected_messages, rejected_images = extract_images_from_messages(normalized_example.get("rejected"))

    normalized_example["prompt_new"] = prompt_messages
    normalized_example["chosen_new"] = chosen_messages
    normalized_example["rejected_new"] = rejected_messages

    existing_images = []
    for key in ("images", "image", "imgs"):
        if key in normalized_example:
            existing_images.extend(ensure_list_images(normalized_example.get(key)))

    prompt_placeholder_count = count_image_placeholders(prompt_messages)
    merged_images = select_images_for_prompt(
        prompt_placeholder_count,
        prompt_images,
        existing_images,
        chosen_images,
        rejected_images,
    )
    normalized_example["images_new"] = merged_images

    # Remove the original columns so HF Datasets doesn't cast back to their schema
    for key in ("prompt", "chosen", "rejected", "images", "image", "imgs", "messages"):
        normalized_example.pop(key, None)

    return normalized_example


def count_image_placeholders(messages):
    if not isinstance(messages, list):
        return 0

    placeholder_count = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                placeholder_count += 1
    return placeholder_count


def has_matching_prompt_images(example):
    prompt_placeholder_count = count_image_placeholders(example.get("prompt"))
    actual_images_count = len(ensure_list_images(example.get("images")))

    if prompt_placeholder_count == 0:
        return True

    return prompt_placeholder_count == actual_images_count


def filter_invalid_multimodal_dpo_dataset(train_ds, source_name, rank):
    original_count = len(train_ds)
    preview_limit = 5
    mismatch_previews = []

    for idx in range(original_count):
        example = train_ds[idx]
        prompt_placeholder_count = count_image_placeholders(example.get("prompt"))
        actual_images_count = len(ensure_list_images(example.get("images")))
        if prompt_placeholder_count > 0 and prompt_placeholder_count != actual_images_count:
            mismatch_previews.append(
                {
                    "index": idx,
                    "prompt_image_placeholders": prompt_placeholder_count,
                    "actual_images": actual_images_count,
                }
            )
            if len(mismatch_previews) >= preview_limit:
                break

    filtered_ds = train_ds.filter(
        has_matching_prompt_images,
        desc=f"filter_multimodal_mismatch::{source_name}",
        num_proc=16,
    )
    filtered_count = len(filtered_ds)
    dropped_count = original_count - filtered_count

    if rank == 0:
        print(
            f"[Dataset Filter] {source_name} | "
            f"kept_samples={filtered_count:,} | "
            f"dropped_mismatch_samples={dropped_count:,}",
            flush=True,
        )
        if mismatch_previews:
            print(f"[Dataset Filter] {source_name} mismatch_examples={mismatch_previews}", flush=True)

    return filtered_ds


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
        if is_preprocessed_dataset_source(source_name):
            if rank == 0:
                print(
                    f"[Dataset Preprocess] source={source_name} | "
                    f"detected={PREPROCESS_METADATA_FILENAME} | action=skip_normalize_and_filter",
                    flush=True,
                )
        else:
            train_ds = train_ds.map(
                normalize_multimodal_dpo_example,
                desc=f"normalize_multimodal::{source_name}",
                num_proc=16,
                remove_columns=train_ds.column_names,
            )
            train_ds = train_ds.rename_columns({
                "prompt_new": "prompt",
                "chosen_new": "chosen",
                "rejected_new": "rejected",
                "images_new": "images",
            })
            train_ds = filter_invalid_multimodal_dpo_dataset(train_ds, source_name, rank)
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


def has_processor_files(path):
    if not path or not os.path.isdir(path):
        return False
    return any(os.path.exists(os.path.join(path, file_name)) for file_name in PROCESSOR_CONFIG_FILES)


def resolve_processor_source(model_name, processor_name):
    if processor_name:
        return processor_name

    if os.path.isdir(model_name) and has_processor_files(model_name):
        return model_name

    normalized_model_name = model_name.rstrip("/\\")
    parent_dir = os.path.dirname(normalized_model_name)
    if parent_dir and parent_dir != normalized_model_name and has_processor_files(parent_dir):
        return parent_dir

    return model_name


def load_model_and_processor(model_name, processor_name, local_only):
    import huggingface_hub.constants

    orig_hf_endpoint = os.environ.pop("HF_ENDPOINT", None)
    orig_hf_token = os.environ.pop("HF_TOKEN", None)
    huggingface_hub.constants.ENDPOINT = "https://huggingface.co"
    processor_source = resolve_processor_source(model_name, processor_name)

    processor = AutoProcessor.from_pretrained(
        processor_source,
        local_files_only=local_only,
        trust_remote_code=True,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 512,
    )

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_name,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        local_files_only=local_only,
        trust_remote_code=True,
    )

    ref_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_name,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        local_files_only=local_only,
        trust_remote_code=True,
    )

    if orig_hf_endpoint:
        os.environ["HF_ENDPOINT"] = orig_hf_endpoint
    if orig_hf_token:
        os.environ["HF_TOKEN"] = orig_hf_token
    huggingface_hub.constants.ENDPOINT = orig_hf_endpoint if orig_hf_endpoint else "https://huggingface.co"

    model.to(torch.bfloat16)
    ref_model.to(torch.bfloat16)
    return processor, model, ref_model


def filter_supported_kwargs(callable_obj, kwargs):
    signature = inspect.signature(callable_obj)
    return {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters and value is not None
    }


def main():
    parser = argparse.ArgumentParser(description="VLM DPO Qwen3.5 Distributed Training")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_PATH, help="HuggingFace model name or local checkpoint path")
    parser.add_argument("--processor_name", type=str, default=None, help="Optional processor source; defaults to model_name or checkpoint parent if needed")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH, help="Comma-separated local paths or MLXP dataset IDs")
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT_PATH, help="Path to save model")
    parser.add_argument("--logging_dir", type=str, default=DEFAULT_LOGGING_DIR, help="TensorBoard log directory")
    parser.add_argument("--max_steps", type=int, default=None, help="Max training steps (None means full epochs)")
    parser.add_argument("--num_train_epochs", type=float, default=3.0, help="Number of training epochs when max_steps is not set")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-7, help="Learning rate")
    parser.add_argument("--save_steps", type=int, default=200, help="Checkpoint save interval in steps")
    parser.add_argument("--save_total_limit", type=int, default=None, help="Optional max number of checkpoints to keep")
    args = parser.parse_args()

    rank = init_distributed()

    try:
        import huggingface_hub.hf_file_system
        import fsspec.spec
        huggingface_hub.hf_file_system.HfFileSystem.glob = fsspec.spec.AbstractFileSystem.glob
    except Exception as exc:
        if rank == 0:
            print(f"Skipping hf_file_system patch: {exc}", flush=True)

    with main_process_first(rank):
        dataset = load_and_merge_datasets(args.data_path, rank)
        local_only = (rank != 0)
        processor, model, ref_model = load_model_and_processor(
            args.model_name,
            args.processor_name,
            local_only=local_only,
        )

    valid_wrap_classes = list(set(
        module.__class__.__name__
        for module in model.modules()
        if "DecoderLayer" in module.__class__.__name__ or ("VisionBlock" in module.__class__.__name__ and "Qwen" in module.__class__.__name__)
    ))
    if not valid_wrap_classes:
        valid_wrap_classes = ["Qwen2DecoderLayer"]

    dpo_config_kwargs = {
        "output_dir": args.output_path,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_steps": args.max_steps,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "logging_dir": args.logging_dir,
        "logging_steps": 10,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "bf16": True,
        "gradient_checkpointing": True,
        "fsdp": "full_shard auto_wrap",
        "fsdp_config": {"transformer_layer_cls_to_wrap": valid_wrap_classes},
        "ddp_find_unused_parameters": False,
        "remove_unused_columns": False,
        "dataset_num_proc": 16,
        "ddp_timeout": 7200,
        "max_prompt_length": 4096,
        "max_length": 8192,
    }
    training_args = DPOConfig(**filter_supported_kwargs(DPOConfig, dpo_config_kwargs))

    if rank == 0:
        print(
            f"Starting DPO with model={args.model_name} | "
            f"wrap_classes={valid_wrap_classes} | "
            f"train_samples={len(dataset['train']):,}",
            flush=True,
        )

    import time
    import math
    import subprocess
    from transformers import TrainerCallback

    def format_eta(seconds):
        if seconds is None or seconds < 0 or math.isinf(seconds):
            return "unknown"
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def get_nvidia_smi_stats(local_rank):
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={local_rank}",
                    "--query-gpu=memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            used_mb, total_mb, util_gpu = [x.strip() for x in result.stdout.strip().split(",")]
            return int(used_mb), int(total_mb), int(util_gpu)
        except Exception:
            return None, None, None

    class GPUMemoryCallback(TrainerCallback):
        def __init__(self):
            self.last_log_step = 0
            self.last_log_time = time.time()

        def on_log(self, args, state, control, logs=None, **kwargs):
            if torch.cuda.is_available():
                mem_alloc = torch.cuda.memory_allocated() / (1024**3)
                mem_max = torch.cuda.max_memory_allocated() / (1024**3)
                rank_str = os.environ.get("RANK", "0")
                
                current_time = time.time()
                steps_passed = state.global_step - self.last_log_step
                step_time = (current_time - self.last_log_time) / max(steps_passed, 1)
                max_steps = state.max_steps if state.max_steps and state.max_steps > 0 else getattr(state, "max_steps", 0)
                remaining_steps = max(0, max_steps - state.global_step)
                eta_seconds = remaining_steps * step_time
                
                if rank_str == "0" and state.global_step > 0:
                    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                    used_mb, total_mb, util_gpu = get_nvidia_smi_stats(local_rank)
                    nvidia_smi_msg = ""
                    if used_mb is not None and total_mb is not None and util_gpu is not None:
                        nvidia_smi_msg = f" | nvidia-smi Mem {used_mb}/{total_mb} MiB | GPU Util {util_gpu}%"

                    print(
                        f"\n[Step {state.global_step}/{max_steps}] "
                        f"{step_time:.2f} s/it | "
                        f"ETA {format_eta(eta_seconds)} | "
                        f"GPU Mem Allocated: {mem_alloc:.2f} GB | Max: {mem_max:.2f} GB"
                        f"{nvidia_smi_msg}\n",
                        flush=True,
                    )
                
                self.last_log_step = state.global_step
                self.last_log_time = current_time

    trainer_kwargs = {
        "model": model,
        "ref_model": ref_model,
        "args": training_args,
        "train_dataset": dataset["train"],
        "processing_class": processor,
        "max_prompt_length": 4096,
        "max_length": 8192,
    }
    trainer = DPOTrainer(**filter_supported_kwargs(DPOTrainer, trainer_kwargs))
    trainer.add_callback(GPUMemoryCallback())

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

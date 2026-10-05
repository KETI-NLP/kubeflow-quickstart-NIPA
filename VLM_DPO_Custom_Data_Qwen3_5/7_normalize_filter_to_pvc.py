import argparse
import contextlib
import glob
import io
import json
import os
import shutil
import threading
import time

from datasets import DatasetDict, concatenate_datasets, load_from_disk
DEFAULT_PROCESSOR_NAME = "/data/checkpoints/qwen3_5_9b_multimodal_sft_lr_1e-5"
PROCESSOR_CONFIG_FILES = (
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer_config.json",
)
PROCESSOR_PREVIEW_SCAN_LIMIT = 500
DEFAULT_PROCESSOR_VALIDATION_MODE = "full"
DEFAULT_PROCESSOR_VALIDATION_SAMPLE_SIZE = 1000

_PROCESSOR_CACHE = {}


@contextlib.contextmanager
def progress_heartbeat(label, interval_seconds=30):
    stop_event = threading.Event()
    start_time = time.time()

    def emit_heartbeat():
        while not stop_event.wait(interval_seconds):
            elapsed = int(time.time() - start_time)
            print(f"[Progress] {label} | elapsed={elapsed}s | still running...", flush=True)

    thread = threading.Thread(target=emit_heartbeat, daemon=True)
    thread.start()
    print(f"[Progress] {label} | started", flush=True)
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=1)
        elapsed = int(time.time() - start_time)
        print(f"[Progress] {label} | finished | elapsed={elapsed}s", flush=True)


def load_dataset_from_source(data_path, max_retries=5, retry_sleep_seconds=5):
    if "/" in data_path and not os.path.exists(data_path) and not data_path.startswith("/data/"):
        from mlx.sdk.data import load_dataset as mlxp_load_dataset

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                with progress_heartbeat(f"mlxp_load_dataset::{data_path}"):
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
        with progress_heartbeat(f"load_from_disk::{data_path}"):
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
                with progress_heartbeat(f"mlxp_load_dataset_fallback::{candidate}"):
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


def has_processor_files(path):
    if not path or not os.path.isdir(path):
        return False
    return any(os.path.exists(os.path.join(path, file_name)) for file_name in PROCESSOR_CONFIG_FILES)


def resolve_processor_source(processor_name):
    if os.path.isdir(processor_name) and has_processor_files(processor_name):
        return processor_name

    normalized_path = processor_name.rstrip("/\\")
    parent_dir = os.path.dirname(normalized_path)
    if parent_dir and parent_dir != normalized_path and has_processor_files(parent_dir):
        return parent_dir

    return processor_name


def get_processor(processor_name):
    processor_source = resolve_processor_source(processor_name)
    if processor_source not in _PROCESSOR_CACHE:
        from transformers import AutoProcessor

        _PROCESSOR_CACHE[processor_source] = AutoProcessor.from_pretrained(
            processor_source,
            local_files_only=os.path.exists(processor_source),
            trust_remote_code=True,
        )
    return _PROCESSOR_CACHE[processor_source]


def coerce_to_dpo_schema(train_ds, source_name, dataset_num_proc):
    required_columns = {"prompt", "chosen", "rejected"}
    column_names = set(train_ds.column_names)

    if required_columns.issubset(column_names):
        return train_ds

    if "messages" in column_names and {"chosen", "rejected"}.issubset(column_names):
        def convert_messages_to_prompt(example):
            return {"prompt": example["messages"]}

        return train_ds.map(
            convert_messages_to_prompt,
            desc=f"add_prompt::{source_name}",
            num_proc=dataset_num_proc,
        )

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


def load_image_as_pil(image_payload):
    from PIL import Image

    if hasattr(image_payload, "copy") and hasattr(image_payload, "convert"):
        return image_payload.copy().convert("RGB")

    if isinstance(image_payload, dict):
        nested_payload = image_payload.get("image")
        if nested_payload is not None and nested_payload is not image_payload:
            return load_image_as_pil(nested_payload)

        image_bytes = image_payload.get("bytes")
        image_path = image_payload.get("path")
        image_url = image_payload.get("url") or image_payload.get("image_url")

        if image_bytes is not None:
            with Image.open(io.BytesIO(image_bytes)) as img:
                return img.convert("RGB").copy()
        if isinstance(image_path, str) and os.path.exists(image_path):
            with Image.open(image_path) as img:
                return img.convert("RGB").copy()
        if isinstance(image_url, str) and image_url.startswith("file://") and os.path.exists(image_url[7:]):
            with Image.open(image_url[7:]) as img:
                return img.convert("RGB").copy()
        raise ValueError(f"Unsupported dict image payload keys: {sorted(image_payload.keys())}")

    if isinstance(image_payload, str):
        image_path = image_payload[7:] if image_payload.startswith("file://") else image_payload
        if os.path.exists(image_path):
            with Image.open(image_path) as img:
                return img.convert("RGB").copy()
        raise ValueError(f"Image path does not exist: {image_payload}")

    raise ValueError(f"Unsupported image payload type: {type(image_payload).__name__}")


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
    normalized_example["images_new"] = select_images_for_prompt(
        prompt_placeholder_count,
        prompt_images,
        existing_images,
        chosen_images,
        rejected_images,
    )
    
    for key in ("prompt", "chosen", "rejected", "images", "image", "imgs", "messages"):
        normalized_example.pop(key, None)

    return normalized_example


def has_matching_prompt_images(example):
    prompt_placeholder_count = count_image_placeholders(example.get("prompt"))
    actual_images_count = len(ensure_list_images(example.get("images")))
    if prompt_placeholder_count == 0:
        return True
    return prompt_placeholder_count == actual_images_count


def build_prompt_text(processor, messages):
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def is_processor_compatible_example(example, processor_name):
    prompt_placeholder_count = count_image_placeholders(example.get("prompt"))
    if prompt_placeholder_count == 0:
        return True

    images = ensure_list_images(example.get("images"))
    if len(images) != prompt_placeholder_count:
        return False

    try:
        processor = get_processor(processor_name)
        prompt_text = build_prompt_text(processor, example.get("prompt"))
        pil_images = [load_image_as_pil(image_payload) for image_payload in images]
        processor(
            text=[prompt_text],
            images=pil_images,
            padding=False,
            return_tensors="pt",
        )
        return True
    except Exception:
        return False


def collect_processor_failure_previews(train_ds, processor_name, preview_limit=5, scan_limit=PROCESSOR_PREVIEW_SCAN_LIMIT):
    previews = []
    total_to_scan = min(len(train_ds), scan_limit)
    for idx in range(total_to_scan):
        example = train_ds[idx]
        if not is_processor_compatible_example(example, processor_name):
            previews.append(
                {
                    "index": idx,
                    "prompt_image_placeholders": count_image_placeholders(example.get("prompt")),
                    "actual_images": len(ensure_list_images(example.get("images"))),
                }
            )
            if len(previews) >= preview_limit:
                break
    return previews


def build_evenly_spaced_sample(dataset, sample_size):
    if sample_size is None or sample_size <= 0 or len(dataset) <= sample_size:
        return dataset

    if sample_size == 1:
        return dataset.select([0])

    max_index = len(dataset) - 1
    indices = sorted({round(i * max_index / (sample_size - 1)) for i in range(sample_size)})
    return dataset.select(indices)


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


def preprocess_single_dataset(
    source_name,
    output_root,
    dataset_num_proc,
    processor_name,
    processor_validation_num_proc,
    processor_validation_mode,
    processor_validation_sample_size,
    max_samples=None,
    overwrite=False,
):
    dataset_name = source_name.rstrip("/").split("/")[-1]
    save_path = os.path.join(output_root, dataset_name)
    tmp_save_path = f"{save_path}.tmp"

    if overwrite and os.path.exists(save_path):
        print(f"[Preprocess] overwrite enabled, removing existing path: {save_path}", flush=True)
        shutil.rmtree(save_path)

    if os.path.exists(save_path):
        print(f"[Preprocess] SKIP {source_name} -> {save_path} already exists", flush=True)
        return save_path

    print(
        f"[Preprocess] source={source_name} | output={save_path} | dataset_num_proc={dataset_num_proc}",
        flush=True,
    )

    ds = load_dataset_from_source(source_name)
    train_ds = normalize_train_split(ds)
    train_ds = coerce_to_dpo_schema(train_ds, source_name, dataset_num_proc)
    if max_samples is not None:
        train_ds = train_ds.select(range(min(len(train_ds), max_samples)))
    print(
        f"[Preprocess] loaded {source_name} | samples={len(train_ds):,} | columns={train_ds.column_names}",
        flush=True,
    )

    train_ds = train_ds.map(
        normalize_multimodal_dpo_example,
        desc=f"normalize_multimodal::{source_name}",
        num_proc=dataset_num_proc,
        remove_columns=train_ds.column_names,
    )
    train_ds = train_ds.rename_columns({
        "prompt_new": "prompt",
        "chosen_new": "chosen",
        "rejected_new": "rejected",
        "images_new": "images",
    })

    filtered_ds = train_ds.filter(
        has_matching_prompt_images,
        desc=f"filter_multimodal_mismatch::{source_name}",
        num_proc=dataset_num_proc,
    )

    dropped_count = len(train_ds) - len(filtered_ds)
    validation_ds = filtered_ds
    if processor_validation_mode == "sample":
        validation_ds = build_evenly_spaced_sample(filtered_ds, processor_validation_sample_size)
        print(
            f"[Preprocess] processor_sample {source_name} | sample_size={len(validation_ds):,} / total={len(filtered_ds):,}",
            flush=True,
        )

    print(
        f"[Preprocess] processor_preview {source_name} | scan_limit={min(len(validation_ds), PROCESSOR_PREVIEW_SCAN_LIMIT):,}",
        flush=True,
    )
    with progress_heartbeat(f"processor_preview::{source_name}"):
        processor_failure_previews = collect_processor_failure_previews(validation_ds, processor_name)

    print(
        f"[Preprocess] processor_filter {source_name} | mode={processor_validation_mode} | num_proc={processor_validation_num_proc}",
        flush=True,
    )
    with progress_heartbeat(f"filter_processor_incompatible::{source_name}"):
        processor_filtered_validation_ds = validation_ds.filter(
            is_processor_compatible_example,
            fn_kwargs={"processor_name": processor_name},
            desc=f"filter_processor_incompatible::{source_name}",
            num_proc=processor_validation_num_proc,
        )
    processor_dropped_count = len(validation_ds) - len(processor_filtered_validation_ds)
    processor_filtered_ds = filtered_ds
    if processor_validation_mode == "full":
        processor_filtered_ds = processor_filtered_validation_ds
    elif processor_dropped_count > 0:
        raise RuntimeError(
            f"Processor sample validation failed for {source_name}. "
            f"sample_size={len(validation_ds):,}, failed_samples={processor_dropped_count:,}, "
            f"processor_failure_previews={processor_failure_previews}"
        )
    example_summary = summarize_example(filtered_ds[0]) if len(filtered_ds) > 0 else {}
    print(
        f"[Preprocess] filtered {source_name} | kept_samples={len(filtered_ds):,} | "
        f"dropped_mismatch_samples={dropped_count:,} | sample0={example_summary}",
        flush=True,
    )
    print(
        f"[Preprocess] processor_validated {source_name} | mode={processor_validation_mode} | "
        f"kept_samples={len(processor_filtered_ds):,} | dropped_processor_incompatible_samples={processor_dropped_count:,}",
        flush=True,
    )
    if processor_failure_previews:
        print(
            f"[Preprocess] processor_failure_examples {source_name} | previews={processor_failure_previews}",
            flush=True,
        )

    filtered_ds = processor_filtered_ds
    example_summary = summarize_example(filtered_ds[0]) if len(filtered_ds) > 0 else {}

    if len(filtered_ds) == 0:
        raise RuntimeError(
            f"All samples were filtered out for {source_name}. "
            f"dropped_mismatch_samples={dropped_count:,}, "
            f"dropped_processor_incompatible_samples={processor_dropped_count:,}, "
            f"processor_failure_previews={processor_failure_previews}"
        )

    os.makedirs(output_root, exist_ok=True)
    if os.path.exists(tmp_save_path):
        shutil.rmtree(tmp_save_path)

    with progress_heartbeat(f"save_to_disk::{save_path}", interval_seconds=60):
        DatasetDict({"train": filtered_ds}).save_to_disk(tmp_save_path, num_proc=dataset_num_proc)

    os.rename(tmp_save_path, save_path)

    metadata = {
        "source_name": source_name,
        "save_path": save_path,
        "dataset_num_proc": dataset_num_proc,
        "processor_name": resolve_processor_source(processor_name),
        "processor_validation_num_proc": processor_validation_num_proc,
        "processor_validation_mode": processor_validation_mode,
        "processor_validation_sample_size": processor_validation_sample_size,
        "kept_samples": len(filtered_ds),
        "dropped_mismatch_samples": dropped_count,
        "dropped_processor_incompatible_samples": processor_dropped_count,
        "columns": filtered_ds.column_names,
        "example_summary": example_summary,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(save_path, "preprocess_metadata.json"), "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, ensure_ascii=True, indent=2)

    print(f"[Preprocess] DONE {source_name} -> {save_path}", flush=True)
    return save_path


def main():
    parser = argparse.ArgumentParser(description="Normalize and cache multimodal DPO datasets to PVC")
    parser.add_argument("--data_paths", type=str, required=True, help="Comma-separated dataset IDs or local paths")
    parser.add_argument("--output_root", type=str, required=True, help="Directory where normalized datasets will be saved")
    parser.add_argument("--dataset_num_proc", type=int, default=max(1, min(48, (os.cpu_count() or 16) - 4)), help="num_proc for map/filter/save_to_disk")
    parser.add_argument("--processor_name", type=str, default=DEFAULT_PROCESSOR_NAME, help="Processor source used for strict multimodal dry-run validation")
    parser.add_argument("--processor_validation_num_proc", type=int, default=4, help="num_proc for processor dry-run compatibility filtering")
    parser.add_argument("--processor_validation_mode", type=str, choices=["full", "sample"], default=DEFAULT_PROCESSOR_VALIDATION_MODE, help="Use full filtering or sample-only processor validation before saving")
    parser.add_argument("--processor_validation_sample_size", type=int, default=DEFAULT_PROCESSOR_VALIDATION_SAMPLE_SIZE, help="How many evenly spaced samples to check when processor_validation_mode=sample")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap for quick subset validation runs")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing normalized datasets")
    args = parser.parse_args()

    data_paths = [path.strip() for path in args.data_paths.split(",") if path.strip()]
    if not data_paths:
        raise ValueError("No data_paths provided.")

    os.makedirs(args.output_root, exist_ok=True)
    print(
        f"[Preprocess] start | datasets={data_paths} | output_root={args.output_root} | "
        f"dataset_num_proc={args.dataset_num_proc} | cpu_count={os.cpu_count()}",
        flush=True,
    )

    if any("/" in path and not os.path.exists(path) and not path.startswith("/data/") for path in data_paths):
        from mlx.sdk.data import login

        api_key = os.environ.get("MLX_APIKEY") or os.environ.get("MLXP_API_KEY")
        endpoint = os.environ.get("MLX_ENDPOINT_URL") or os.environ.get("MLXP_ENDPOINT_URL")
        login(api_key, endpoint)
        print("[Preprocess] MLXP login complete", flush=True)

    saved_paths = []
    for source_name in data_paths:
        saved_paths.append(
            preprocess_single_dataset(
                source_name,
                args.output_root,
                args.dataset_num_proc,
                args.processor_name,
                args.processor_validation_num_proc,
                args.processor_validation_mode,
                args.processor_validation_sample_size,
                max_samples=args.max_samples,
                overwrite=args.overwrite,
            )
        )

    manifest = {
        "saved_paths": saved_paths,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_num_proc": args.dataset_num_proc,
        "processor_name": resolve_processor_source(args.processor_name),
        "processor_validation_num_proc": args.processor_validation_num_proc,
        "processor_validation_mode": args.processor_validation_mode,
        "processor_validation_sample_size": args.processor_validation_sample_size,
    }
    manifest_path = os.path.join(args.output_root, "preprocess_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=True, indent=2)
    print(f"[Preprocess] manifest saved to {manifest_path}", flush=True)


if __name__ == "__main__":
    main()

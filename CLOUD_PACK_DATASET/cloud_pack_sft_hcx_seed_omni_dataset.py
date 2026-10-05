import argparse
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import logging
import torch
import multiprocessing
from datasets import load_from_disk, Dataset, Features, Sequence, Value, Image
from transformers import AutoProcessor
from PIL import Image as PILImage
import pyarrow.ipc as ipc
import pyarrow as pa
import io

def safe_load_from_disk(ds_path):
    """Bypasses HF Datasets feature dataclass issues while preserving zero-copy Memory Mapping."""
    import datasets
    
    original_from_arrow_schema = datasets.Features.from_arrow_schema
    def patched_from_arrow_schema(schema):
        if schema.metadata and b'huggingface' in schema.metadata:
            new_metadata = {k: v for k, v in schema.metadata.items() if k != b'huggingface'}
            schema = schema.with_metadata(new_metadata)
        return original_from_arrow_schema(schema)
    
    datasets.Features.from_arrow_schema = patched_from_arrow_schema
    if hasattr(datasets, 'arrow_dataset'):
        datasets.arrow_dataset.Features.from_arrow_schema = patched_from_arrow_schema
    if hasattr(datasets, 'features') and hasattr(datasets.features, 'features'):
        datasets.features.features.Features.from_arrow_schema = patched_from_arrow_schema
        
    original_from_dict = datasets.DatasetInfo.from_dict
    def patched_from_dict(d):
        if 'features' in d:
            del d['features']
        return original_from_dict(d)
        
    datasets.DatasetInfo.from_dict = patched_from_dict
    if hasattr(datasets, 'info'):
        datasets.info.DatasetInfo.from_dict = patched_from_dict
        
    return load_from_disk(ds_path)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    logging.error("qwen_vl_utils not found. Please pip install qwen-vl-utils")
    import sys; sys.exit(1)

# Global variables for processor sharing in multiprocessing
global_processor = None

def init_processor(model_name, trust_remote_code=False, local_files_only=False):
    global global_processor

    import sys
    try:
        import transformers.video_utils
    except ImportError:
        try:
            import transformers.image_utils
            sys.modules['transformers.video_utils'] = type('video_utils', (), {'VideoInput': transformers.image_utils.VideoInput})
        except ImportError:
            pass

    if global_processor is None:
        global_processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code, local_files_only=local_files_only)
        if hasattr(global_processor, "image_token") and global_processor.tokenizer is not None:
            if "<|IMAGE_PAD|>" in global_processor.tokenizer.get_vocab():
                global_processor.image_token = "<|IMAGE_PAD|>"
                global_processor.image_token_id = global_processor.tokenizer.encode("<|IMAGE_PAD|>")[0]
    return global_processor

def map_tokenize_fn(example, model_name=None, trust_remote_code=False, local_files_only=False):
    processor = init_processor(model_name, trust_remote_code, local_files_only)
    
    messages = example.get("messages", [])
    images_raw = example.get("images", [])
    images = []
    for img in images_raw:
        if isinstance(img, dict) and "bytes" in img and img["bytes"] is not None:
            images.append(PILImage.open(io.BytesIO(img["bytes"])))
        else:
            images.append(img)
    
    # Auto-convert DPO format to SFT format if needed
    if len(messages) == 0 and "prompt" in example and "chosen" in example:
        user_msg = example.get("prompt", [{}])[0]
        user_content_str = ""
        images = []
        for part in user_msg.get("content", []):
            if part.get("type") == "image":
                user_content_str += "<image>\n"
                if part.get("image") is not None:
                    img_data = part["image"]
                    if isinstance(img_data, dict) and "bytes" in img_data and img_data["bytes"] is not None:
                        img_data = PILImage.open(io.BytesIO(img_data["bytes"]))
                    images.append(img_data)
            elif part.get("type") == "text" and part.get("text"):
                user_content_str += part["text"] + "\n"
                
        ast_msg = example.get("chosen", [{}])[0]
        ast_content = ast_msg.get("content", [])
        ast_content_str = ast_content[0]["text"] if len(ast_content) > 0 else ""
        
        messages = [
            {"role": "user", "content": user_content_str.strip()},
            {"role": "assistant", "content": ast_content_str.strip()}
        ]
    
    # Resize images to 512px max to save space and calculation time
    images_to_pack = []
    for img in images:
        width, height = img.size
        # RGBA or other formats to RGB safely first
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        if max(width, height) > 512:
            scale = 512.0 / max(width, height)
            new_width = max(1, int(width * scale))
            new_height = max(1, int(height * scale))
            img = img.resize((new_width, new_height), PILImage.Resampling.LANCZOS)
            
        images_to_pack.append(img)
    
    # Dummy Image Injection for FSDP Deadlock Prevention
    if len(images_to_pack) == 0:
        dummy_image = PILImage.new('RGB', (28, 28), (0, 0, 0))
        images_to_pack = [dummy_image]
        for msg in messages:
            if msg["role"] == "user":
                if isinstance(msg["content"], list):
                    msg["content"] = [{"type": "image", "image": dummy_image}] + msg["content"]
                else:
                    msg["content"] = "<image>\n" + str(msg["content"])
                break
    
    qwen_messages = []
    img_idx = 0
    for msg in messages:
        role = msg["role"]
        raw_content = msg["content"]
        
        if isinstance(raw_content, list):
            qwen_messages.append({"role": role, "content": raw_content})
        else:
            content_str = str(raw_content)
            if "<image>" in content_str and img_idx < len(images_to_pack):
                parts = content_str.split("<image>")
                content = []
                for idx, text_chunk in enumerate(parts):
                    if text_chunk:
                        content.append({"type": "text", "text": text_chunk})
                    # Add an image after each split part, except the last one
                    if idx < len(parts) - 1 and img_idx < len(images_to_pack):
                        content.append({"type": "image", "image": images_to_pack[img_idx]})
                        img_idx += 1
            else:
                content = [{"type": "text", "text": content_str}]
            qwen_messages.append({"role": role, "content": content})
        
    # HCX Chat Template 호환성을 위해 qwen_messages(VLM 파싱용)를 template_messages(Jinja용) 포맷으로 변환
    # (Processor의 buggy jinja dict 변환 우회를 위해 이미지는 바로 raw string 변환)
    template_messages = []
    _local_img_idx = 0
    for msg in qwen_messages:
        role, content = msg["role"], msg["content"]
        if isinstance(content, list):
            if role == "assistant":
                text_only = "".join([c.get("text", "") for c in content if c.get("type") == "text"])
                template_messages.append({"role": role, "content": text_only})
            else:
                text_content = ""
                for part in content:
                    if part.get("type") == "image":
                        image_id = f"image_{_local_img_idx:02d}"
                        text_content += f'<|mime_start|>{{"id": "{image_id}", "type": "image/jpeg", "filename": "image.jpg"}}<|mime_end|>\n'
                        text_content += '<|discrete_image_start|><|DISCRETE_IMAGE_PAD|><|discrete_image_end|>\n'
                        text_content += '<|image_start|><|IMAGE_PAD|><|image_end|>'
                        _local_img_idx += 1
                    else:
                        text_content += str(part.get("text", ""))
                template_messages.append({"role": role, "content": text_content})
        else:
            template_messages.append({"role": role, "content": content})

    text = processor.apply_chat_template(template_messages, tokenize=False, add_generation_prompt=False)
    if len(template_messages) > 0 and template_messages[-1]["role"] == "assistant":
        prompt_text = processor.apply_chat_template(template_messages[:-1], tokenize=False, add_generation_prompt=True)
    else:
        prompt_text = text
        
    image_inputs = process_vision_info(qwen_messages)[0]
    clean_image_inputs = [img for img in image_inputs if img is not None] if image_inputs else None

    # Processor dynamically determines length including <|image_pad|> based on resized inputs
    inputs = processor(
        text=[text],
        images=clean_image_inputs,
        padding=False,
        return_tensors="pt"
    )
    
    prompt_inputs = processor(
        text=[prompt_text],
        images=clean_image_inputs,
        padding=False,
        return_tensors="pt"
    )
    
    input_ids = inputs["input_ids"][0].tolist()
    seq_len = len(input_ids)
    
    prompt_len = prompt_inputs["input_ids"][0].shape[0]
    labels = input_ids.copy()
    for i in range(min(prompt_len, seq_len)):
        labels[i] = -100

    return {
        "input_ids": input_ids,
        "labels": labels,
        "images_for_pack": images_to_pack,
        "seq_len": seq_len
    }

def main():
    parser = argparse.ArgumentParser(description="Parallel Offline Packing for HyperCLOVA X Seed Omni")
    parser.add_argument("--data_path", type=str, default="/data/vlm_sft_dataset", help="Path to preprocessed SFT or DPO dataset")
    parser.add_argument("--output_path", type=str, default="/data/result/packed_dataset", help="Path to save packed dataset")
    parser.add_argument("--model_name", type=str, default="naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B", help="Model name for tokenizer/processor")
    parser.add_argument("--max_seq_len", type=int, default=8192, help="Maximum sequence length for packing")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of samples to process (for testing)")
    parser.add_argument("--trust_remote_code", action="store_true", default=True, help="Trust remote code for processor")
    args = parser.parse_args()

    print(f"Loading dataset via PyArrow fallback from {args.data_path}")
    dataset = safe_load_from_disk(args.data_path)
    
    if hasattr(dataset, 'keys') and 'train' in dataset:
        ds_iter = dataset['train']
    else:
        ds_iter = dataset

    if args.limit is not None:
        logging.info(f"Limiting dataset to {args.limit} samples for testing")
        if hasattr(ds_iter, 'select'):
            ds_iter = ds_iter.select(range(min(args.limit, len(ds_iter))))

    # Stage 1: Parallel Tokenization (Multiprocessing map)
    num_cores = multiprocessing.cpu_count()
    num_proc = min(64, max(1, num_cores - 1))
    
    logging.info(f"Preloading Tokenizer locally to prevent Hugging Face Hub Rate Limiting (HTTP 429) across {num_proc} instances...")
    init_processor(args.model_name, trust_remote_code=args.trust_remote_code, local_files_only=False)
    
    logging.info(f"Stage 1: Parsing, Image Resizing, and Tokenizing using {num_proc} processor cores...")
    map_features = Features({
        "input_ids": Sequence(Value("int32")),
        "labels": Sequence(Value("int32")),
        "images_for_pack": Sequence(Image(decode=False)),
        "seq_len": Value("int32")
    })
    
    fn_kwargs = {
        "model_name": args.model_name,
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": True
    }
    
    tokenized_ds = ds_iter.map(
        map_tokenize_fn,
        num_proc=num_proc,
        batched=False,
        remove_columns=ds_iter.column_names,
        features=map_features,
        fn_kwargs=fn_kwargs,
        desc="Parallel Tokenization & Resizing"
    )
    
    # Stage 2: Offline FFD Allocation
    logging.info("Stage 2: Offline FFD Sequence Length Bin Allocation...")
    seq_lens = tokenized_ds["seq_len"]
    indices = list(range(len(seq_lens)))
    
    capped_lens = [min(l, args.max_seq_len) for l in seq_lens]
    sorted_items = sorted(zip(indices, capped_lens), key=lambda x: x[1], reverse=True)
    
    bins = []
    bin_capacities = []
    for idx, length in sorted_items:
        placed = False
        for i in range(len(bins)):
            if bin_capacities[i] >= length:
                bins[i].append(idx)
                bin_capacities[i] -= length
                placed = True
                break
        if not placed:
            bins.append([idx])
            bin_capacities.append(args.max_seq_len - length)
            
    logging.info(f"Sorted {len(indices)} samples into {len(bins)} perfectly packed 0-padding chunks!")

    # Stage 3: Fast Assembly
    logging.info("Stage 3: High-speed Assembly Generator extracting from disk cache...")
    def generate_ffb_chunks():
        for b_idx, b_indices in enumerate(bins):
            if b_idx > 0 and b_idx % 200 == 0:
                logging.info(f"Assembling chunk {b_idx}/{len(bins)}")
            
            chunk_data = tokenized_ds[b_indices]
            
            cur_input_ids = []
            cur_labels = []
            cur_images = []
            
            for i in range(len(b_indices)):
                ids = chunk_data["input_ids"][i]
                lbls = chunk_data["labels"][i]
                imgs = chunk_data["images_for_pack"][i]
                
                # Double safety bounds check
                if len(ids) > args.max_seq_len:
                    ids = ids[:args.max_seq_len]
                    lbls = lbls[:args.max_seq_len]
                    
                cur_input_ids.extend(ids)
                cur_labels.extend(lbls)
                if imgs is not None:
                    cur_images.extend(imgs)
                    
            yield {
                "input_ids": cur_input_ids,
                "labels": cur_labels,
                "images": cur_images
            }

    final_features = Features({
        "input_ids": Sequence(Value("int32")),
        "labels": Sequence(Value("int32")),
        "images": Sequence(Image(decode=True))
    })
    
    packed_dataset = Dataset.from_generator(generate_ffb_chunks, features=final_features)
    
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    logging.info(f"Saving explicitly packed {args.model_name} dataset to {args.output_path}...")
    packed_dataset.save_to_disk(args.output_path)
    logging.info("Multi-stage Parallel Offline packing complete! Ready for highly efficient SFT.")

if __name__ == "__main__":
    main()

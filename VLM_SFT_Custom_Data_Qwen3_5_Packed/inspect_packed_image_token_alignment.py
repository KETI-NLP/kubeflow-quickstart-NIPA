import argparse
import io
import os
from typing import Any

from datasets import load_dataset, load_from_disk
from transformers import AutoProcessor
from PIL import Image
from PIL import Image as PILImage
from qwen_vl_utils import process_vision_info


def load_any_dataset(path: str, split: str):
    if os.path.exists(path):
        ds = load_from_disk(path)
        if hasattr(ds, "keys"):
            if split in ds:
                return ds[split]
            if "train" in ds:
                return ds["train"]
        return ds

    try:
        from mlx.sdk.data import load_dataset as mlxp_load_dataset

        ds = mlxp_load_dataset(path)
        if hasattr(ds, "keys"):
            if split in ds:
                return ds[split]
            if "train" in ds:
                return ds["train"]
        return ds
    except Exception:
        return load_dataset(path, split=split)


def prepare_image_for_processor(img: Any):
    if not isinstance(img, Image.Image):
        return img

    if img.mode != "RGB":
        img = img.convert("RGB")

    width, height = img.size
    if max(width, height) > 512:
        scale = 512.0 / max(width, height)
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return img


def decode_images(images_value):
    if images_value is None:
        return []

    if not isinstance(images_value, list):
        images_value = [images_value]

    decoded = []
    for img in images_value:
        if isinstance(img, dict) and img.get("bytes") is not None:
            decoded.append(prepare_image_for_processor(Image.open(io.BytesIO(img["bytes"]))))
        elif isinstance(img, Image.Image):
            decoded.append(prepare_image_for_processor(img))
        else:
            decoded.append(img)
    return decoded


def build_synthetic_qwen_messages_from_packed_images(decoded_images):
    if not decoded_images:
        return None

    return [
        {
            "role": "user",
            "content": [{"type": "image", "image": img} for img in decoded_images],
        }
    ]


def build_qwen_messages_from_raw_example(example):
    messages = example.get("messages", [])
    images_raw = example.get("images", [])
    images = []
    for img in images_raw:
        if isinstance(img, dict) and img.get("bytes") is not None:
            images.append(PILImage.open(io.BytesIO(img["bytes"])))
        else:
            images.append(img)

    if len(messages) == 0 and "prompt" in example and "chosen" in example:
        user_msg = example.get("prompt", [{}])[0]
        user_content_str = ""
        images = []
        for part in user_msg.get("content", []):
            if part.get("type") == "image":
                user_content_str += "<image>\n"
                if part.get("image") is not None:
                    img_data = part["image"]
                    if isinstance(img_data, dict) and img_data.get("bytes") is not None:
                        img_data = PILImage.open(io.BytesIO(img_data["bytes"]))
                    images.append(img_data)
            elif part.get("type") == "text" and part.get("text"):
                user_content_str += part["text"] + "\n"

        ast_msg = example.get("chosen", [{}])[0]
        ast_content = ast_msg.get("content", [])
        ast_content_str = ast_content[0]["text"] if len(ast_content) > 0 else ""

        messages = [
            {"role": "user", "content": user_content_str.strip()},
            {"role": "assistant", "content": ast_content_str.strip()},
        ]

    if len(messages) == 0:
        return None

    images_to_pack = []
    for img in images:
        if not isinstance(img, PILImage.Image):
            images_to_pack.append(img)
            continue

        width, height = img.size
        if img.mode != "RGB":
            img = img.convert("RGB")
        if max(width, height) > 512:
            scale = 512.0 / max(width, height)
            new_width = max(1, int(width * scale))
            new_height = max(1, int(height * scale))
            img = img.resize((new_width, new_height), PILImage.Resampling.LANCZOS)
        images_to_pack.append(img)

    if len(images_to_pack) == 0:
        dummy_image = PILImage.new("RGB", (28, 28), (0, 0, 0))
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
            continue

        content_str = str(raw_content)
        if "<image>" in content_str and img_idx < len(images_to_pack):
            parts = content_str.split("<image>")
            content = []
            for idx, text_chunk in enumerate(parts):
                if text_chunk:
                    content.append({"type": "text", "text": text_chunk})
                if idx < len(parts) - 1 and img_idx < len(images_to_pack):
                    content.append({"type": "image", "image": images_to_pack[img_idx]})
                    img_idx += 1
        else:
            content = [{"type": "text", "text": content_str}]
        qwen_messages.append({"role": role, "content": content})

    return qwen_messages


def main():
    parser = argparse.ArgumentParser(description="Compare packed image token counts with processor-derived image token counts.")
    parser.add_argument("--data_path", required=True, help="Comma-separated local dataset paths or MLXP/HF dataset names")
    parser.add_argument("--processor_name", required=True, help="Processor name/path to use for recomputing image tokens")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=20, help="How many samples to inspect")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.processor_name, trust_remote_code=True)
    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(getattr(processor, "tokenizer", None), "image_token_id", None)
    if image_token_id is None:
        image_token_id = 248056
    image_processor = getattr(processor, "image_processor", None)
    spatial_merge_size = getattr(image_processor, "merge_size", None)
    if spatial_merge_size is None:
        spatial_merge_size = getattr(image_processor, "spatial_merge_size", None)
    if spatial_merge_size is None:
        spatial_merge_size = 2

    data_paths = [dp.strip() for dp in args.data_path.split(",") if dp.strip()]

    print(f"processor={args.processor_name}")
    print(f"image_token_id={image_token_id}")
    print(f"spatial_merge_size={spatial_merge_size}")
    print(f"dataset_count={len(data_paths)}")

    for data_path in data_paths:
        ds = load_any_dataset(data_path, args.split)
        total = min(len(ds), args.limit)
        mismatches_nested = 0
        mismatches_flat = 0
        mismatches_qwen_smart = 0

        print("")
        print(f"dataset={data_path}")
        print(f"samples_to_check={total}")

        if total > 0:
            replay_messages = build_qwen_messages_from_raw_example(ds[0])
            if replay_messages is None:
                print("raw_replay_check=unavailable (packed dataset has no messages/prompt/chosen)")
            else:
                replay_text = processor.apply_chat_template(
                    replay_messages, tokenize=False, add_generation_prompt=False
                )
                replay_images = process_vision_info(replay_messages)[0]
                replay_clean_images = [img for img in replay_images if img is not None] if replay_images else None
                replay_batch = processor(
                    text=[replay_text],
                    images=replay_clean_images,
                    padding=False,
                    return_tensors="pt",
                )
                replay_tokens = sum(
                    1 for tok in replay_batch["input_ids"][0].tolist() if tok == image_token_id
                )
                packed0 = sum(1 for tok in ds[0].get("input_ids", []) if tok == image_token_id)
                print(
                    f"raw_replay_check=available replay_image_tokens={replay_tokens} "
                    f"packed_image_tokens={packed0}"
                )

        for idx in range(total):
            ex = ds[idx]
            packed_input_ids = ex.get("input_ids", [])
            packed_image_tokens = sum(1 for tok in packed_input_ids if tok == image_token_id)
            decoded_images = decode_images(ex.get("images"))

            if decoded_images:
                nested_batch = processor(
                    text=[""],
                    images=[decoded_images],
                    padding=False,
                    return_tensors="pt",
                )
                nested_grid = nested_batch.get("image_grid_thw")
                nested_grid_rows = nested_grid.tolist() if nested_grid is not None else None
                if nested_grid is not None:
                    processor_feature_tokens_nested = sum(
                        int((t * h * w) // (spatial_merge_size ** 2))
                        for t, h, w in nested_grid_rows
                    )
                else:
                    processor_feature_tokens_nested = 0

                flat_batch = processor(
                    text=[""],
                    images=decoded_images,
                    padding=False,
                    return_tensors="pt",
                )
                flat_grid = flat_batch.get("image_grid_thw")
                flat_grid_rows = flat_grid.tolist() if flat_grid is not None else None
                if flat_grid is not None:
                    processor_feature_tokens_flat = sum(
                        int((t * h * w) // (spatial_merge_size ** 2))
                        for t, h, w in flat_grid_rows
                    )
                else:
                    processor_feature_tokens_flat = 0

                synthetic_messages = build_synthetic_qwen_messages_from_packed_images(decoded_images)
                synthetic_images = process_vision_info(synthetic_messages)[0] if synthetic_messages else None
                qwen_smart_batch = processor(
                    text=[""],
                    images=synthetic_images,
                    padding=False,
                    return_tensors="pt",
                )
                qwen_smart_grid = qwen_smart_batch.get("image_grid_thw")
                qwen_smart_grid_rows = qwen_smart_grid.tolist() if qwen_smart_grid is not None else None
                if qwen_smart_grid is not None:
                    processor_feature_tokens_qwen_smart = sum(
                        int((t * h * w) // (spatial_merge_size ** 2))
                        for t, h, w in qwen_smart_grid_rows
                    )
                else:
                    processor_feature_tokens_qwen_smart = 0
            else:
                processor_feature_tokens_nested = 0
                processor_feature_tokens_flat = 0
                processor_feature_tokens_qwen_smart = 0
                nested_grid_rows = None
                flat_grid_rows = None
                qwen_smart_grid_rows = None

            ok_nested = packed_image_tokens == processor_feature_tokens_nested
            ok_flat = packed_image_tokens == processor_feature_tokens_flat
            ok_qwen_smart = packed_image_tokens == processor_feature_tokens_qwen_smart
            if not ok_nested:
                mismatches_nested += 1
            if not ok_flat:
                mismatches_flat += 1
            if not ok_qwen_smart:
                mismatches_qwen_smart += 1

            print(
                f"idx={idx} num_images={len(decoded_images)} "
                f"packed_image_tokens={packed_image_tokens} "
                f"nested_feature_tokens={processor_feature_tokens_nested} ok_nested={ok_nested} "
                f"flat_feature_tokens={processor_feature_tokens_flat} ok_flat={ok_flat} "
                f"qwen_smart_feature_tokens={processor_feature_tokens_qwen_smart} ok_qwen_smart={ok_qwen_smart}"
            )
            if not ok_nested or not ok_flat or not ok_qwen_smart:
                print(f"  nested_image_grid_thw={nested_grid_rows}")
                print(f"  flat_image_grid_thw={flat_grid_rows}")
                print(f"  qwen_smart_image_grid_thw={qwen_smart_grid_rows}")

        print(f"mismatches_nested={mismatches_nested}/{total}")
        print(f"mismatches_flat={mismatches_flat}/{total}")
        print(f"mismatches_qwen_smart={mismatches_qwen_smart}/{total}")


if __name__ == "__main__":
    main()

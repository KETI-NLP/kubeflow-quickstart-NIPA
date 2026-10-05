import argparse
from PIL import Image, ImageDraw
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info


def make_dummy_image(width: int, height: int, label: str):
    image = Image.new("RGB", (width, height), (240, 240, 240))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width - 1, height - 1), outline=(0, 0, 0), width=3)
    draw.text((20, 20), label, fill=(20, 20, 20))
    return image


def maybe_resize_like_packing(img: Image.Image):
    if img.mode != "RGB":
        img = img.convert("RGB")

    width, height = img.size
    if max(width, height) > 512:
        scale = 512.0 / max(width, height)
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return img


def count_image_tokens(input_ids, image_token_id):
    return sum(1 for token in input_ids if token == image_token_id)


def count_feature_tokens(image_grid_thw, spatial_merge_size):
    if image_grid_thw is None:
        return 0
    rows = image_grid_thw.tolist()
    return sum(int((t * h * w) // (spatial_merge_size ** 2)) for t, h, w in rows)


def main():
    parser = argparse.ArgumentParser(description="Compare Qwen processor results between packing-style and training-style calls.")
    parser.add_argument("--processor_name", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--layout", default="middle", choices=["front", "middle", "end", "double"])
    parser.add_argument("--resize_like_packing", action="store_true")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.processor_name, trust_remote_code=True)
    image_processor = getattr(processor, "image_processor", None)
    spatial_merge_size = getattr(image_processor, "merge_size", None)
    if spatial_merge_size is None:
        spatial_merge_size = getattr(image_processor, "spatial_merge_size", None)
    if spatial_merge_size is None:
        spatial_merge_size = 2

    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(getattr(processor, "tokenizer", None), "image_token_id", 248056)

    img1 = make_dummy_image(1024, 768, "IMG-1")
    img2 = make_dummy_image(640, 960, "IMG-2")
    if args.resize_like_packing:
        img1 = maybe_resize_like_packing(img1)
        img2 = maybe_resize_like_packing(img2)

    if args.layout == "front":
        user_text = "<image>\nDescribe this scene briefly."
        images = [img1]
    elif args.layout == "middle":
        user_text = "First image:<image>\nNow describe it."
        images = [img1]
    elif args.layout == "end":
        user_text = "Describe this scene briefly.\n<image>"
        images = [img1]
    else:
        user_text = "Compare these two images.\n<image>\nThen this one.\n<image>"
        images = [img1, img2]

    qwen_messages = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": "Dummy answer."},
    ]

    # Packing-style path
    packing_messages = []
    img_idx = 0
    for msg in qwen_messages:
        if msg["role"] != "user":
            packing_messages.append({"role": msg["role"], "content": [{"type": "text", "text": msg["content"]}]})
            continue

        parts = msg["content"].split("<image>")
        content = []
        for idx, text_chunk in enumerate(parts):
            if text_chunk:
                content.append({"type": "text", "text": text_chunk})
            if idx < len(parts) - 1 and img_idx < len(images):
                content.append({"type": "image", "image": images[img_idx]})
                img_idx += 1
        packing_messages.append({"role": "user", "content": content})

    packing_text = processor.apply_chat_template(packing_messages, tokenize=False, add_generation_prompt=False)
    packing_image_inputs = process_vision_info(packing_messages)[0]
    packing_clean_images = [img for img in packing_image_inputs if img is not None] if packing_image_inputs else None
    packing_batch = processor(
        text=[packing_text],
        images=packing_clean_images,
        padding=False,
        return_tensors="pt",
    )

    # Training-style path
    training_batch = processor(
        text=[""],
        images=[images],
        padding=False,
        return_tensors="pt",
    )

    packing_input_ids = packing_batch["input_ids"][0].tolist()
    training_input_ids = training_batch["input_ids"][0].tolist()

    print(f"processor={args.processor_name}")
    print(f"layout={args.layout}")
    print(f"resize_like_packing={args.resize_like_packing}")
    print(f"image_token_id={image_token_id}")
    print(f"spatial_merge_size={spatial_merge_size}")
    print(f"num_images={len(images)}")
    print(f"image_sizes={[img.size for img in images]}")

    print("")
    print("[packing_style]")
    print(f"input_len={len(packing_input_ids)}")
    print(f"image_tokens={count_image_tokens(packing_input_ids, image_token_id)}")
    print(f"feature_tokens={count_feature_tokens(packing_batch.get('image_grid_thw'), spatial_merge_size)}")
    print(f"image_grid_thw={packing_batch.get('image_grid_thw').tolist() if packing_batch.get('image_grid_thw') is not None else None}")

    print("")
    print("[training_style]")
    print(f"input_len={len(training_input_ids)}")
    print(f"image_tokens={count_image_tokens(training_input_ids, image_token_id)}")
    print(f"feature_tokens={count_feature_tokens(training_batch.get('image_grid_thw'), spatial_merge_size)}")
    print(f"image_grid_thw={training_batch.get('image_grid_thw').tolist() if training_batch.get('image_grid_thw') is not None else None}")


if __name__ == "__main__":
    main()

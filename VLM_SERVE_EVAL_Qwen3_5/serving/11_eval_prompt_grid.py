import argparse
import csv
import gc
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from safetensors.torch import load_file
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from qwen_vl_utils import process_vision_info


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("prompt-grid-eval")


_ORIGINAL_SDPA = F.scaled_dot_product_attention


def _patched_sdpa(*args, **kwargs):
    try:
        return _ORIGINAL_SDPA(*args, **kwargs)
    except TypeError as exc:
        if "enable_gqa" not in str(exc):
            raise
        kwargs.pop("enable_gqa", None)
        return _ORIGINAL_SDPA(*args, **kwargs)


F.scaled_dot_product_attention = _patched_sdpa

OCR_STYLE_RE = re.compile(
    r"(사진|이미지).*(보입|쓰여|써|적혀|문구|문장)"
    r"|사진\s*속\s*텍스트"
    r"|사진에\s*쓰인\s*글"
    r"|[\"'“”‘’][^\"'“”‘’]{1,80}[\"'“”‘’]?(?:라고|이라고)\s*(?:되어|적혀|쓰여)\s*있"
)

PROMPT_VARIANTS = {
    "general_multimodal": (
        "You are a helpful assistant. "
        "Answer in Korean unless the user asks otherwise. "
        "If the conversation includes an image, use it when relevant. "
        "If it is a normal text-only conversation, answer naturally like a regular chat assistant."
    ),
    "universal_current": (
        "You are a helpful Korean multimodal assistant. "
        "Answer naturally and directly in Korean unless the user asks otherwise. "
        "When an image is actually provided or clearly relevant from the conversation, use it carefully and accurately. "
        "When the conversation is text-only, answer like a normal chat assistant and do not assume there is an unseen image, photo, screenshot, or OCR target."
    ),
    "universal_explicit_mode": (
        "You are a helpful Korean assistant for both text-only chat and image-grounded conversation. "
        "If the current user turn includes an image, use the image. "
        "If no image is provided in the current turn or prior conversation, treat the exchange as ordinary text chat and answer naturally "
        "without referring to images, photos, screenshots, visible text, or OCR."
    ),
    "universal_text_first": (
        "You are a helpful Korean assistant. Default to ordinary text conversation. "
        "Only switch to image understanding when the user actually provides an image or asks about a previously shared image. "
        "Do not describe imaginary images or pretend there is visible text when none was provided."
    ),
    "universal_directive": (
        "You are a Korean multimodal assistant. Follow this rule carefully: "
        "if an image is present, answer based on the image and text together; "
        "if no image is present, answer only as a text assistant. "
        "Never mention a photo, image, screenshot, or visible text unless one is actually present in the conversation."
    ),
    "universal_examples": (
        "You are a helpful Korean multimodal assistant. Respond naturally in Korean. "
        "Use image evidence only when an image is actually provided in the conversation. "
        "If no image is provided, answer as normal text chat. "
        "Examples: user: 안녕? assistant: 안녕하세요! 무엇을 도와드릴까요? "
        "user: 자기소개해줘. assistant: 안녕하세요. 저는 질문에 답하고 이미지도 이해할 수 있는 AI 어시스턴트입니다. "
        "user: [image] 이미지에 뭐라고 적혀 있어? assistant: 이미지에 보이는 텍스트를 읽어 답한다."
    ),
}


@dataclass
class TestCase:
    name: str
    user_content: Any
    expects_ocr_style: bool
    answer_hint: str | None = None


def _build_test_image(text: str) -> Image.Image:
    image = Image.new("RGB", (240, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 40), text, fill="black")
    return image


def _build_cases() -> list[TestCase]:
    test_image = _build_test_image("안녕")
    return [
        TestCase(name="text_greeting", user_content="안녕?", expects_ocr_style=False),
        TestCase(name="text_intro", user_content="자기소개해줘.", expects_ocr_style=False),
        TestCase(name="text_math", user_content="12 + 35는?", expects_ocr_style=False, answer_hint="47"),
        TestCase(
            name="image_ocr",
            user_content=[
                {"type": "text", "text": "이미지에 뭐라고 적혀 있어?"},
                {"type": "image", "image": test_image},
            ],
            expects_ocr_style=True,
            answer_hint="안녕",
        ),
    ]


def _patch_qwen_model_for_inference(model: Qwen3_5ForConditionalGeneration) -> None:
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
                LOGGER.info("Patched mm_projector in %s", module.__class__.__name__)
            else:
                patch_all_projectors(child)

    patch_all_projectors(model)

    if hasattr(model, "model") and hasattr(model.model, "vision_model") and hasattr(model.model.vision_model, "merger"):
        original_forward = model.model.vision_model.forward

        def patched_vision_forward(*args, **kwargs):
            out = original_forward(*args, **kwargs)
            hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else (out[0] if isinstance(out, tuple) else out)

            if hasattr(hidden, "shape") and hidden.shape[-1] == 1280:
                merger = model.model.vision_model.merger
                try:
                    hidden = merger(hidden)
                except Exception as exc:
                    LOGGER.warning("vision merger call failed: %s", exc)
            return out

        model.model.vision_model.forward = patched_vision_forward
        LOGGER.info("Patched Qwen vision merger path")


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    table = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in table:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return table[dtype_name]


def _pick_model_source(model_path: Path, base_model_name: str | None) -> str:
    if (model_path / "config.json").exists():
        return str(model_path)
    if not base_model_name:
        raise ValueError(f"{model_path} does not contain config.json, so --base-model-name is required.")
    return base_model_name


def _pick_processor_source(model_path: Path, processor_path: str | None, base_model_name: str | None) -> str:
    if processor_path:
        return processor_path
    if (model_path / "processor_config.json").exists():
        return str(model_path)
    if not base_model_name:
        raise ValueError(
            f"{model_path} does not contain processor_config.json, so --processor-path or --base-model-name is required."
        )
    return base_model_name


def _sanitize_response_text(text: str) -> str:
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>", 1)[0]
    if "<|im_start|>" in text:
        text = text.split("<|im_start|>", 1)[0]
    if "</think>" in text:
        text = text.split("</think>", 1)[0]
    return text.strip()


def _contains_ocr_style(text: str) -> bool:
    return bool(OCR_STYLE_RE.search(text))


def _score_response(case: TestCase, response: str) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    has_ocr_style = _contains_ocr_style(response)

    if case.expects_ocr_style:
        if case.answer_hint and case.answer_hint in response:
            score += 1
            reasons.append("image-answer-ok")
        if has_ocr_style:
            score += 1
            reasons.append("image-style-ok")
    else:
        if not has_ocr_style:
            score += 1
            reasons.append("text-non-ocr")
        if case.answer_hint:
            if case.answer_hint in response:
                score += 1
                reasons.append("answer-hint-ok")
        else:
            lowered = response.casefold()
            if any(token in lowered for token in ["안녕", "도와", "소개", "어시스턴트"]):
                score += 1
                reasons.append("chat-natural")
    return score, reasons


def _to_messages(system_prompt: str, case: TestCase) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": case.user_content if isinstance(case.user_content, list) else [{"type": "text", "text": case.user_content}],
        },
    ]


def _has_vision_content(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        for block in message.get("content", []):
            if block.get("type") in {"image", "image_url", "video"}:
                return True
    return False


def _generate_response(
    model: Qwen3_5ForConditionalGeneration,
    processor: AutoProcessor,
    device: torch.device,
    system_prompt: str,
    case: TestCase,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    messages = _to_messages(system_prompt, case)
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    image_inputs = None
    video_inputs = None
    if _has_vision_content(messages):
        image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    input_token_count = int(inputs["input_ids"].shape[1])

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": temperature > 0,
    }
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)
    new_tokens = output_ids[0][input_token_count:]
    raw_text = processor.decode(new_tokens, skip_special_tokens=False)
    return _sanitize_response_text(raw_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate prompt variants on a single checkpoint.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--processor-path", default=None)
    parser.add_argument("--base-model-name", default=None)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=1.0)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_source = _pick_model_source(checkpoint_dir, args.base_model_name)
    processor_source = _pick_processor_source(checkpoint_dir, args.processor_path, args.base_model_name)
    dtype = _resolve_dtype(args.torch_dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LOGGER.info("Loading checkpoint=%s model_source=%s processor_source=%s", checkpoint_dir.name, model_source, processor_source)

    processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_source,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation=args.attn_implementation,
    )

    safetensors_path = checkpoint_dir / "model.safetensors"
    if safetensors_path.exists():
        state_dict = load_file(str(safetensors_path))
        model.load_state_dict(state_dict, strict=False)

    _patch_qwen_model_for_inference(model)
    model.eval()
    device = next(model.parameters()).device

    cases = _build_cases()
    results: list[dict[str, Any]] = []
    prompt_summary: list[dict[str, Any]] = []

    for prompt_id, prompt_text in PROMPT_VARIANTS.items():
        total_score = 0
        for case in cases:
            response = _generate_response(
                model=model,
                processor=processor,
                device=device,
                system_prompt=prompt_text,
                case=case,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            score, reasons = _score_response(case, response)
            total_score += score
            row = {
                "checkpoint": checkpoint_dir.name,
                "checkpoint_path": str(checkpoint_dir),
                "prompt_id": prompt_id,
                "case": case.name,
                "response": response,
                "score": score,
                "reasons": ";".join(reasons),
            }
            results.append(row)
            LOGGER.info("checkpoint=%s prompt=%s case=%s score=%s response=%r", checkpoint_dir.name, prompt_id, case.name, score, response)
        prompt_summary.append(
            {
                "checkpoint": checkpoint_dir.name,
                "checkpoint_path": str(checkpoint_dir),
                "prompt_id": prompt_id,
                "total_score": total_score,
            }
        )

    detail_path = output_dir / f"{checkpoint_dir.name}_prompt_grid_detail.csv"
    with detail_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["checkpoint", "checkpoint_path", "prompt_id", "case", "response", "score", "reasons"])
        writer.writeheader()
        writer.writerows(results)

    summary_path = output_dir / f"{checkpoint_dir.name}_prompt_grid_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["checkpoint", "checkpoint_path", "prompt_id", "total_score"])
        writer.writeheader()
        writer.writerows(sorted(prompt_summary, key=lambda row: (-row["total_score"], row["prompt_id"])))

    json_path = output_dir / f"{checkpoint_dir.name}_prompt_grid_summary.json"
    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "checkpoint": checkpoint_dir.name,
                "checkpoint_path": str(checkpoint_dir),
                "prompt_summary": sorted(prompt_summary, key=lambda row: (-row["total_score"], row["prompt_id"])),
                "results": results,
            },
            fp,
            ensure_ascii=False,
            indent=2,
        )

    del model
    del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    LOGGER.info("Saved prompt-grid eval to %s", output_dir)


if __name__ == "__main__":
    main()

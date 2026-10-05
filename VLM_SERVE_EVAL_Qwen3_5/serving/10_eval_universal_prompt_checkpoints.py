import argparse
import base64
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import request as urllib_request

from PIL import Image, ImageDraw


UNIVERSAL_PROMPT = (
    "You are a helpful Korean multimodal assistant. "
    "Answer naturally and directly in Korean unless the user asks otherwise. "
    "When an image is actually provided or clearly relevant from the conversation, use it carefully and accurately. "
    "When the conversation is text-only, answer like a normal chat assistant and do not assume there is an unseen image, photo, screenshot, or OCR target."
)

OCR_STYLE_RE = re.compile(
    r"(사진|이미지).*(보입|쓰여|써|적혀|문구|문장)"
    r"|사진\s*속\s*텍스트"
    r"|사진에\s*쓰인\s*글"
    r"|[\"'“”‘’][^\"'“”‘’]{1,80}[\"'“”‘’]?(?:라고|이라고)\s*(?:되어|적혀|쓰여)\s*있"
)


@dataclass
class TestCase:
    name: str
    messages: list[dict[str, Any]]
    expects_ocr_style: bool
    answer_hint: str | None = None


def _build_data_url(text: str) -> str:
    image = Image.new("RGB", (240, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 40), text, fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _build_cases() -> list[TestCase]:
    image_url = _build_data_url("안녕")
    return [
        TestCase(
            name="text_greeting",
            messages=[
                {"role": "system", "content": UNIVERSAL_PROMPT},
                {"role": "user", "content": "안녕?"},
            ],
            expects_ocr_style=False,
        ),
        TestCase(
            name="text_intro",
            messages=[
                {"role": "system", "content": UNIVERSAL_PROMPT},
                {"role": "user", "content": "자기소개해줘."},
            ],
            expects_ocr_style=False,
        ),
        TestCase(
            name="text_math",
            messages=[
                {"role": "system", "content": UNIVERSAL_PROMPT},
                {"role": "user", "content": "12 + 35는?"},
            ],
            expects_ocr_style=False,
            answer_hint="47",
        ),
        TestCase(
            name="image_ocr",
            messages=[
                {"role": "system", "content": UNIVERSAL_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "이미지에 뭐라고 적혀 있어?"},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            expects_ocr_style=True,
            answer_hint="안녕",
        ),
    ]


def _request_chat_completion(endpoint: str, model: str, messages: list[dict[str, Any]], max_tokens: int) -> str:
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib_request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib_request.urlopen(req, timeout=300) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


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


def _checkpoint_sort_key(model_id: str) -> tuple[int, str]:
    match = re.search(r"checkpoint-(\d+)$", model_id)
    if match:
        return int(match.group(1)), model_id
    return 10**9, model_id


def discover_checkpoints_from_dir(base_dir: str, base_model_id: str) -> list[str]:
    base_path = Path(base_dir)
    checkpoints = []
    for child in sorted(base_path.iterdir()):
        if child.is_dir() and child.name.startswith("checkpoint-"):
            checkpoints.append(f"{base_model_id}/{child.name}")
    return sorted(checkpoints, key=_checkpoint_sort_key)


def discover_checkpoints_from_api(endpoint: str, base_model_id: str) -> list[str]:
    req = urllib_request.Request(endpoint.rstrip("/") + "/v1/models")
    with urllib_request.urlopen(req, timeout=60) as response:
        data = json.loads(response.read().decode("utf-8"))
    checkpoints = []
    for item in data.get("data", []):
        model_id = item.get("id", "")
        if model_id.startswith(base_model_id + "/checkpoint-"):
            checkpoints.append(model_id)
    return sorted(checkpoints, key=_checkpoint_sort_key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare universal prompt behavior across checkpoint-* models.")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000", help="OpenAI-compatible endpoint base URL.")
    parser.add_argument(
        "--base-model-id",
        default="qwen3_5_9b_multimodal_sft_lr_1e-5",
        help="Base model id used by the serving API.",
    )
    parser.add_argument(
        "--base-model-dir",
        default=None,
        help="Optional local checkpoint root directory used to discover checkpoint-* folders.",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of checkpoints for quick smoke tests.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional explicit model ids such as qwen3_5_9b_multimodal_sft_lr_1e-5/checkpoint-300",
    )
    args = parser.parse_args()

    if args.models:
        checkpoints = sorted(args.models, key=_checkpoint_sort_key)
    elif args.base_model_dir:
        checkpoints = discover_checkpoints_from_dir(args.base_model_dir, args.base_model_id)
    else:
        checkpoints = discover_checkpoints_from_api(args.endpoint, args.base_model_id)
    if args.limit > 0:
        checkpoints = checkpoints[: args.limit]
    if not checkpoints:
        raise SystemExit("No checkpoint-* directories found.")

    cases = _build_cases()
    summary: list[dict[str, Any]] = []

    for model_id in checkpoints:
        model_score = 0
        model_results = []
        print(f"=== {model_id} ===")
        for case in cases:
            response = _request_chat_completion(args.endpoint, model_id, case.messages, args.max_tokens)
            score, reasons = _score_response(case, response)
            model_score += score
            result = {
                "case": case.name,
                "response": response,
                "score": score,
                "reasons": reasons,
            }
            model_results.append(result)
            print(f"{case.name}: score={score} reasons={','.join(reasons) or '-'}")
            print(response)
            print("---")
        summary.append({"model_id": model_id, "total_score": model_score, "results": model_results})
        print(f"TOTAL_SCORE={model_score}")
        print()

    print("=== RANKING ===")
    for item in sorted(summary, key=lambda x: (-x["total_score"], _checkpoint_sort_key(x["model_id"]))):
        print(f"{item['model_id']},{item['total_score']}")


if __name__ == "__main__":
    main()

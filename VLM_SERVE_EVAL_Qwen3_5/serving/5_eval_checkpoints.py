import argparse
import csv
import gc
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("checkpoint-eval")


_ORIGINAL_SDPA = F.scaled_dot_product_attention


def _patched_sdpa(*args, **kwargs):
    kwargs.pop("enable_gqa", None)
    return _ORIGINAL_SDPA(*args, **kwargs)


F.scaled_dot_product_attention = _patched_sdpa


DEFAULT_QUESTIONS = [
    {"id": "q1", "question": "12 + 35 = ?", "answer": "47"},
    {"id": "q2", "question": "144 / 12 = ?", "answer": "12"},
    {"id": "q3", "question": "17 * 19 = ?", "answer": "323"},
    {"id": "q4", "question": "250 - 87 = ?", "answer": "163"},
    {"id": "q5", "question": "(18 + 24) / 6 = ?", "answer": "7"},
    {"id": "q6", "question": "7 * 8 + 15 = ?", "answer": "71"},
    {"id": "q7", "question": "3^4 + 5 = ?", "answer": "86"},
    {"id": "q8", "question": "1.5 + 2.25 = ?", "answer": "3.75"},
]


@dataclass
class Question:
    question_id: str
    prompt: str
    expected_answer: str


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


def _load_questions(questions_file: str | None) -> list[Question]:
    raw_questions = DEFAULT_QUESTIONS
    if questions_file:
        with open(questions_file, "r", encoding="utf-8") as fp:
            raw_questions = json.load(fp)

    questions: list[Question] = []
    for idx, item in enumerate(raw_questions, start=1):
        question_id = item.get("id") or f"q{idx}"
        prompt = item["question"]
        expected_answer = str(item["answer"])
        questions.append(Question(question_id=question_id, prompt=prompt, expected_answer=expected_answer))
    return questions


def _discover_checkpoint_dirs(checkpoints_dir: Path, glob_pattern: str) -> list[Path]:
    candidates = [path for path in checkpoints_dir.glob(glob_pattern) if path.is_dir()]

    def sort_key(path: Path):
        match = re.search(r"(\d+)$", path.name)
        if match:
            return (0, int(match.group(1)))
        return (1, path.name)

    return sorted(candidates, key=sort_key)


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


def _unload_model(model, processor) -> None:
    if model is not None:
        del model
    if processor is not None:
        del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _extract_answer(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""

    answer_match = re.findall(r"정답\s*[:：]\s*(.+)", cleaned)
    if answer_match:
        return answer_match[-1].strip()

    number_match = re.findall(r"[-+]?\d+(?:\.\d+)?", cleaned.replace(",", ""))
    if number_match:
        return number_match[-1]

    return cleaned.splitlines()[-1].strip()


def _normalize_answer(text: str) -> str:
    value = text.strip()
    value = re.sub(r"^정답\s*[:：]\s*", "", value, flags=re.IGNORECASE).strip()
    value = value.strip("`'\"")
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[.!?。]+$", "", value)
    value = value.replace(",", "")
    if not value:
        return value
    try:
        number = float(value)
    except ValueError:
        return value.casefold()
    if number.is_integer():
        return str(int(number))
    return format(number, "g")


def _build_prompt(question: str) -> list[dict[str, object]]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You answer quiz questions accurately in Korean."}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "다음 질문에 답해주세요. "
                        "설명은 짧게 해도 되지만 마지막 줄은 반드시 `정답: <짧은 답>` 형식으로 끝내세요.\n"
                        f"질문: {question}"
                    ),
                }
            ],
        },
    ]


def _generate_answer(
    model: Qwen3_5ForConditionalGeneration,
    processor: AutoProcessor,
    question: Question,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[str, int, int]:
    messages = _build_prompt(question.prompt)
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = processor(text=[prompt], padding=True, return_tensors="pt")
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
    cleaned_text = raw_text.split("<|im_end|>", 1)[0].split("<|im_start|>", 1)[0].strip()
    return cleaned_text, input_token_count, int(new_tokens.shape[0])


def evaluate_checkpoint(
    checkpoint_dir: Path,
    questions: list[Question],
    base_model_name: str | None,
    processor_path: str | None,
    torch_dtype: torch.dtype,
    attn_implementation: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    model_source = _pick_model_source(checkpoint_dir, base_model_name)
    processor_source = _pick_processor_source(checkpoint_dir, processor_path, base_model_name)

    LOGGER.info("Loading checkpoint=%s processor=%s model=%s", checkpoint_dir, processor_source, model_source)
    processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_source,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation=attn_implementation,
    )

    local_weight_path = checkpoint_dir / "model.safetensors"
    if Path(model_source) != checkpoint_dir and local_weight_path.exists():
        LOGGER.info("Overlaying fine-tuned weights from %s", local_weight_path)
        state_dict = load_file(str(local_weight_path))
        incompatible = model.load_state_dict(state_dict, strict=False)
        LOGGER.info(
            "Loaded fine-tuned state dict. missing_keys=%d unexpected_keys=%d",
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
        )

    _patch_qwen_model_for_inference(model)
    model.eval()
    device = next(model.parameters()).device

    rows: list[dict[str, object]] = []
    correct_count = 0

    try:
        for question in questions:
            raw_text, prompt_tokens, completion_tokens = _generate_answer(
                model=model,
                processor=processor,
                question=question,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            predicted_answer = _normalize_answer(_extract_answer(raw_text))
            expected_answer = _normalize_answer(question.expected_answer)
            is_correct = predicted_answer == expected_answer
            correct_count += int(is_correct)

            row = {
                "checkpoint": checkpoint_dir.name,
                "checkpoint_path": str(checkpoint_dir),
                "question_id": question.question_id,
                "question": question.prompt,
                "expected_answer": expected_answer,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "raw_response": raw_text,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
            rows.append(row)
            LOGGER.info(
                "checkpoint=%s question=%s expected=%s predicted=%s correct=%s",
                checkpoint_dir.name,
                question.question_id,
                expected_answer,
                predicted_answer,
                is_correct,
            )
    finally:
        _unload_model(model, processor)

    summary = {
        "checkpoint": checkpoint_dir.name,
        "checkpoint_path": str(checkpoint_dir),
        "correct_count": correct_count,
        "total_questions": len(questions),
        "accuracy": correct_count / len(questions) if questions else 0.0,
    }
    return summary, rows


def _write_json(path: Path, payload: object) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3.5 checkpoints with simple math prompts.")
    parser.add_argument("--checkpoints-dir", required=True, help="Root directory containing checkpoint-* folders.")
    parser.add_argument("--output-dir", required=True, help="Directory where evaluation artifacts will be written.")
    parser.add_argument("--base-model-name", default=None, help="Optional base model when checkpoint lacks config.json.")
    parser.add_argument("--processor-path", default=None, help="Optional processor path. Defaults to checkpoint/root.")
    parser.add_argument("--questions-file", default=None, help="Optional JSON file containing evaluation questions.")
    parser.add_argument("--checkpoint-glob", default="checkpoint-*", help="Glob used to select checkpoint folders.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of checkpoints to evaluate.")
    parser.add_argument("--torch-dtype", default="bfloat16", help="float16, bfloat16, or float32.")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    args = parser.parse_args()

    checkpoints_dir = Path(args.checkpoints_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    questions = _load_questions(args.questions_file)
    checkpoint_dirs = _discover_checkpoint_dirs(checkpoints_dir, args.checkpoint_glob)
    if args.limit is not None:
        checkpoint_dirs = checkpoint_dirs[: args.limit]
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint directories matched {args.checkpoint_glob} under {checkpoints_dir}")

    torch_dtype = _resolve_dtype(args.torch_dtype)
    run_metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoints_dir": str(checkpoints_dir),
        "output_dir": str(output_dir),
        "base_model_name": args.base_model_name,
        "processor_path": args.processor_path,
        "checkpoint_glob": args.checkpoint_glob,
        "torch_dtype": args.torch_dtype,
        "attn_implementation": args.attn_implementation,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "question_count": len(questions),
        "checkpoints": [path.name for path in checkpoint_dirs],
    }
    _write_json(output_dir / "run_config.json", run_metadata)

    summaries: list[dict[str, object]] = []
    all_rows: list[dict[str, object]] = []

    for checkpoint_dir in checkpoint_dirs:
        summary, rows = evaluate_checkpoint(
            checkpoint_dir=checkpoint_dir,
            questions=questions,
            base_model_name=args.base_model_name,
            processor_path=args.processor_path,
            torch_dtype=torch_dtype,
            attn_implementation=args.attn_implementation,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        summaries.append(summary)
        all_rows.extend(rows)
        _write_json(output_dir / f"{checkpoint_dir.name}.json", {"summary": summary, "results": rows})

    summaries = sorted(summaries, key=lambda item: (-item["accuracy"], item["checkpoint"]))
    _write_json(output_dir / "summary.json", {"run": run_metadata, "summaries": summaries})
    _write_jsonl(output_dir / "responses.jsonl", all_rows)
    _write_csv(
        output_dir / "summary.csv",
        summaries,
        ["checkpoint", "checkpoint_path", "correct_count", "total_questions", "accuracy"],
    )
    _write_csv(
        output_dir / "responses.csv",
        all_rows,
        [
            "checkpoint",
            "checkpoint_path",
            "question_id",
            "question",
            "expected_answer",
            "predicted_answer",
            "is_correct",
            "raw_response",
            "prompt_tokens",
            "completion_tokens",
        ],
    )

    best = summaries[0]
    LOGGER.info(
        "Evaluation complete. best_checkpoint=%s accuracy=%.4f output_dir=%s",
        best["checkpoint"],
        best["accuracy"],
        output_dir,
    )


if __name__ == "__main__":
    main()

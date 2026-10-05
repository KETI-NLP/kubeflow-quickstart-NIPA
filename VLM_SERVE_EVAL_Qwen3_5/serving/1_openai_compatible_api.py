import argparse
import base64
import gc
import io
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from PIL import Image
from pydantic import BaseModel, Field
from safetensors.torch import load_file
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from transformers import StoppingCriteria, StoppingCriteriaList

from qwen_vl_utils import process_vision_info


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("vlm-openai-server")

_original_sdpa = F.scaled_dot_product_attention


def _patched_sdpa(*args, **kwargs):
    # Only drop `enable_gqa` if the installed torch's SDPA does not accept it
    # (older torch). torch >= 2.5 supports it and Qwen3.5's GQA REQUIRES it —
    # unconditionally popping it caused a (16 vs 4) head-count shape mismatch.
    try:
        return _original_sdpa(*args, **kwargs)
    except TypeError as exc:
        if "enable_gqa" not in str(exc):
            raise
        kwargs.pop("enable_gqa", None)
        return _original_sdpa(*args, **kwargs)


F.scaled_dot_product_attention = _patched_sdpa


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

                if hasattr(out, "last_hidden_state"):
                    from transformers.modeling_outputs import BaseModelOutput

                    if isinstance(out, BaseModelOutput):
                        out = BaseModelOutput(
                            last_hidden_state=hidden,
                            hidden_states=out.hidden_states if hasattr(out, "hidden_states") else None,
                            attentions=out.attentions if hasattr(out, "attentions") else None,
                        )
                    else:
                        out.last_hidden_state = hidden
                elif isinstance(out, tuple):
                    out = (hidden,) + out[1:]
                else:
                    out = hidden

            return out

        model.model.vision_model.forward = patched_vision_forward
        LOGGER.info("Patched Qwen vision merger path")


def _decode_data_url(url: str) -> Image.Image:
    header, data = url.split(",", 1)
    if ";base64" not in header:
        raise ValueError("Only base64 data URLs are supported.")
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


def _load_image_from_url(url: str) -> Image.Image:
    if url.startswith("data:"):
        return _decode_data_url(url)
    if url.startswith("file://"):
        return Image.open(url[len("file://") :]).convert("RGB")
    if url.startswith("/"):
        return Image.open(url).convert("RGB")
    if url.startswith("http://") or url.startswith("https://"):
        response = httpx.get(url, timeout=30.0)
        response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def _ensure_list_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return content
    raise ValueError("message.content must be a string or a list of content blocks")


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    stream: bool = False


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    stream: bool = False


class ModelServer:
    def __init__(
        self,
        model_path: str,
        model_id: str,
        base_model_name: str | None,
        processor_path: str | None,
        torch_dtype: str,
        attn_implementation: str,
        checkpoints_root: str | None,
        models_config_path: str | None,
    ):
        self.default_model_path = model_path
        self.default_model_id = model_id
        self.default_base_model_name = base_model_name
        self.default_processor_path = processor_path
        self.torch_dtype = self._resolve_dtype(torch_dtype)
        self.attn_implementation = attn_implementation
        self.checkpoints_root = Path(checkpoints_root).resolve() if checkpoints_root else None
        self.models_config_path = Path(models_config_path).resolve() if models_config_path else None
        self.lock = threading.Lock()

        self.processor = None
        self.model = None
        self.primary_device = None
        self.stop_token_sequences = None
        self.model_path = None
        self.model_id = None
        self.base_model_name = None
        self.processor_path = None
        self.available_models = self._load_configured_models() or self._discover_models()
        self.runtime_phase = "idle"
        self.loading_model_id = None
        self.generating_model_id = None

    @staticmethod
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

    def _discover_models(self) -> dict[str, dict[str, str | None]]:
        models: dict[str, dict[str, str | None]] = {
            self.default_model_id: {
                "id": self.default_model_id,
                "path": str(Path(self.default_model_path).resolve()),
                "base_model_name": self.default_base_model_name,
                "processor_path": self.default_processor_path,
            }
        }

        if not self.checkpoints_root or not self.checkpoints_root.exists():
            return models

        candidate_dirs: list[tuple[str, Path]] = []
        for entry in sorted(self.checkpoints_root.iterdir()):
            if not entry.is_dir():
                continue
            candidate_dirs.append((entry.name, entry))
            for child in sorted(entry.iterdir()):
                if child.is_dir():
                    candidate_dirs.append((f"{entry.name}/{child.name}", child))

        for model_id, directory in candidate_dirs:
            has_model_files = (
                (directory / "config.json").exists()
                or (directory / "model.safetensors").exists()
                or (directory / "model.safetensors.index.json").exists()
            )
            if not has_model_files:
                continue

            processor_path = None
            for candidate in (directory, directory.parent):
                if candidate == self.checkpoints_root.parent:
                    continue
                if (candidate / "processor_config.json").exists():
                    processor_path = str(candidate.resolve())
                    break

            models.setdefault(
                model_id,
                {
                    "id": model_id,
                    "path": str(directory.resolve()),
                    "base_model_name": self.default_base_model_name,
                    "processor_path": processor_path,
                },
            )

        return models

    def _load_configured_models(self) -> dict[str, dict[str, str | None]]:
        if not self.models_config_path or not self.models_config_path.exists():
            return {}

        with self.models_config_path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)

        items = payload.get("models", payload) if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            raise ValueError(f"models config must be a list or an object with a 'models' key: {self.models_config_path}")

        models: dict[str, dict[str, str | None]] = {}
        for item in items:
            model_id = item["id"]
            models[model_id] = {
                "id": model_id,
                "path": str(Path(item["path"]).resolve()),
                "base_model_name": item.get("base_model_name", self.default_base_model_name),
                "processor_path": item.get("processor_path", self.default_processor_path),
            }

        return models

    def _set_active_model(self, model_spec: dict[str, str | None]) -> None:
        self.model_id = model_spec["id"]
        self.model_path = model_spec["path"]
        self.base_model_name = model_spec.get("base_model_name")
        self.processor_path = model_spec.get("processor_path")

    def _pick_model_source(self) -> str:
        config_path = Path(self.model_path) / "config.json"
        if config_path.exists():
            return self.model_path
        if not self.base_model_name:
            raise ValueError(
                f"{self.model_path} does not contain config.json, so --base-model-name is required to load the architecture."
            )
        return self.base_model_name

    def _pick_processor_source(self) -> str:
        if self.processor_path:
            return self.processor_path
        processor_config = Path(self.model_path) / "processor_config.json"
        if processor_config.exists():
            return self.model_path
        if not self.base_model_name:
            raise ValueError(
                f"{self.model_path} does not contain processor_config.json, so --processor-path or --base-model-name is required."
            )
        return self.base_model_name

    def _unload_current_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self.primary_device = None
        self.stop_token_sequences = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load(self, requested_model_id: str | None = None) -> None:
        target_model_id = requested_model_id or self.default_model_id
        if target_model_id not in self.available_models:
            raise HTTPException(status_code=404, detail=f"Unknown model: {target_model_id}")

        model_spec = self.available_models[target_model_id]
        target_path = model_spec["path"]
        if self.model is not None and self.model_id == target_model_id and self.model_path == target_path:
            return

        self.runtime_phase = "loading"
        self.loading_model_id = target_model_id
        try:
            self._unload_current_model()
            self._set_active_model(model_spec)

            processor_source = self._pick_processor_source()
            model_source = self._pick_model_source()

            LOGGER.info("Loading requested model_id=%s processor=%s model=%s", self.model_id, processor_source, model_source)
            self.processor = AutoProcessor.from_pretrained(
                processor_source,
                trust_remote_code=True,
            )

            LOGGER.info("Loading model from %s", model_source)
            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_source,
                trust_remote_code=True,
                torch_dtype=self.torch_dtype,
                device_map="auto",
                attn_implementation=self.attn_implementation,
            )

            local_weight_path = Path(self.model_path) / "model.safetensors"
            if Path(model_source) != Path(self.model_path) and local_weight_path.exists():
                LOGGER.info("Overlaying fine-tuned weights from %s", local_weight_path)
                state_dict = load_file(str(local_weight_path))
                incompatible = self.model.load_state_dict(state_dict, strict=False)
                LOGGER.info(
                    "Loaded fine-tuned state dict. missing_keys=%d unexpected_keys=%d",
                    len(incompatible.missing_keys),
                    len(incompatible.unexpected_keys),
                )

            _patch_qwen_model_for_inference(self.model)
            self.model.eval()
            self.primary_device = next(self.model.parameters()).device
            self.stop_token_sequences = self._resolve_stop_token_sequences()
            LOGGER.info("Model ready on primary device %s", self.primary_device)
        finally:
            self.runtime_phase = "idle"
            self.loading_model_id = None

    def _to_qwen_messages(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        qwen_messages: list[dict[str, Any]] = []
        for message in messages:
            blocks = []
            for block in _ensure_list_content(message.content):
                block_type = block.get("type")
                if block_type in {"text", "input_text"}:
                    blocks.append({"type": "text", "text": block.get("text", "")})
                elif block_type == "image_url":
                    image_obj = block.get("image_url", {})
                    image_url = image_obj["url"] if isinstance(image_obj, dict) else image_obj
                    blocks.append({"type": "image", "image": _load_image_from_url(image_url)})
                elif block_type == "image":
                    image_value = block.get("image")
                    if isinstance(image_value, str):
                        image_value = _load_image_from_url(image_value)
                    blocks.append({"type": "image", "image": image_value})
                else:
                    raise ValueError(f"Unsupported content block type: {block_type}")

            qwen_messages.append({"role": message.role, "content": blocks})
        return qwen_messages

    def _prepare_inputs(self, messages: list[ChatMessage]) -> dict[str, torch.Tensor]:
        qwen_messages = self._to_qwen_messages(messages)
        prompt = self.processor.apply_chat_template(
            qwen_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        image_inputs, video_inputs = process_vision_info(qwen_messages)
        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        prepared = {}
        for key, value in inputs.items():
            prepared[key] = value.to(self.primary_device) if hasattr(value, "to") else value
        return prepared

    def _resolve_stop_token_sequences(self) -> list[list[int]]:
        tokenizer = self.processor.tokenizer
        sequences: list[list[int]] = []

        for token in ("<|im_end|>", "</think>"):
            token_id = tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id != tokenizer.unk_token_id:
                candidate = [token_id]
            else:
                candidate = tokenizer.encode(token, add_special_tokens=False)
            if candidate and candidate not in sequences:
                sequences.append(candidate)

        eos_token_id = tokenizer.eos_token_id
        if isinstance(eos_token_id, int) and [eos_token_id] not in sequences:
            sequences.append([eos_token_id])

        LOGGER.info("Resolved stop token sequences: %s", sequences)
        return sequences

    def _build_stopping_criteria(self) -> StoppingCriteriaList | None:
        if not self.stop_token_sequences:
            return None

        class StopOnTokenSequence(StoppingCriteria):
            def __init__(self, sequence: list[int]):
                self.sequence = sequence
                self.sequence_len = len(sequence)

            def __call__(self, input_ids, scores, **kwargs):
                if input_ids.shape[1] < self.sequence_len:
                    return False
                recent = input_ids[0, -self.sequence_len :].tolist()
                return recent == self.sequence

        return StoppingCriteriaList([StopOnTokenSequence(sequence) for sequence in self.stop_token_sequences])

    @staticmethod
    def _sanitize_response_text(text: str) -> str:
        if "<|im_end|>" in text:
            text = text.split("<|im_end|>", 1)[0]
        if "<|im_start|>" in text:
            text = text.split("<|im_start|>", 1)[0]
        if "</think>" in text:
            text = text.split("</think>", 1)[0]
        return text.strip()

    def generate_chat(self, request: ChatCompletionRequest) -> dict[str, Any]:
        if request.stream:
            raise HTTPException(status_code=400, detail="Streaming responses are not implemented in this server yet.")

        with self.lock:
            self.load(request.model or self.default_model_id)
            started_at = int(time.time())
            inputs = self._prepare_inputs(request.messages)
            input_token_count = int(inputs["input_ids"].shape[1])

            generation_kwargs = {
                "max_new_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "do_sample": request.temperature > 0,
            }
            stopping_criteria = self._build_stopping_criteria()
            if stopping_criteria is not None:
                generation_kwargs["stopping_criteria"] = stopping_criteria

            self.runtime_phase = "generating"
            self.generating_model_id = self.model_id
            try:
                with torch.inference_mode():
                    output_ids = self.model.generate(**inputs, **generation_kwargs)
            finally:
                self.runtime_phase = "idle"
                self.generating_model_id = None

            new_tokens = output_ids[0][input_token_count:]
            raw_text = self.processor.decode(new_tokens, skip_special_tokens=False)
            text = self._sanitize_response_text(raw_text)
            if not text:
                text = "[빈 응답] 모델이 종료 토큰 또는 </think>만 생성했습니다. 다른 체크포인트를 선택하거나 프롬프트를 다시 시도해보세요."
            completion_tokens = int(new_tokens.shape[0])
            LOGGER.info(
                "generation_debug token_count=%s contains_im_end=%s contains_think_end=%s raw_text=%r",
                completion_tokens,
                "<|im_end|>" in raw_text,
                "</think>" in raw_text,
                raw_text[:1200],
            )

        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        return {
            "id": response_id,
            "object": "chat.completion",
            "created": started_at,
            "model": self.model_id,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": text,
                    },
                }
            ],
            "usage": {
                "prompt_tokens": input_token_count,
                "completion_tokens": completion_tokens,
                "total_tokens": input_token_count + completion_tokens,
            },
        }

    def generate_completion(self, request: CompletionRequest) -> dict[str, Any]:
        chat_request = ChatCompletionRequest(
            model=request.model,
            messages=[ChatMessage(role="user", content=request.prompt)],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=request.stream,
        )
        chat_response = self.generate_chat(chat_request)
        text = chat_response["choices"][0]["message"]["content"]
        return {
            "id": chat_response["id"].replace("chatcmpl-", "cmpl-"),
            "object": "text_completion",
            "created": chat_response["created"],
            "model": self.model_id,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "text": text,
                }
            ],
            "usage": chat_response["usage"],
        }


def build_app(server: ModelServer, chat_ui_path: Path) -> FastAPI:
    app = FastAPI(title="Qwen3.5 VLM OpenAI-Compatible Server")

    @app.on_event("startup")
    def _startup():
        LOGGER.info("Server startup complete without preloading a model. The selected model will load on first request.")

    @app.get("/")
    def root():
        return RedirectResponse(url="/chat")

    @app.get("/healthz")
    def healthz():
        return {
            "status": "ok",
            "model_id": server.model_id,
            "model_loaded": server.model is not None,
            "runtime_phase": server.runtime_phase,
            "loading_model_id": server.loading_model_id,
            "generating_model_id": server.generating_model_id,
        }

    @app.get("/v1/models")
    def list_models():
        def display_name_for_path(path: str) -> str:
            model_path = Path(path)
            local_models_root = Path("/workspace/local_models")
            try:
                return str(model_path.relative_to(local_models_root))
            except ValueError:
                return model_path.name

        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "display_name": display_name_for_path(model_spec["path"]),
                    "path": model_spec["path"],
                    "object": "model",
                    "owned_by": "local",
                }
                for model_id, model_spec in server.available_models.items()
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest):
        return JSONResponse(server.generate_chat(request))

    @app.post("/v1/completions")
    def completions(request: CompletionRequest):
        return JSONResponse(server.generate_completion(request))

    @app.get("/chat", response_class=HTMLResponse)
    def chat_page():
        if not chat_ui_path.exists():
            raise HTTPException(status_code=500, detail=f"Missing chat UI file: {chat_ui_path}")
        return HTMLResponse(chat_ui_path.read_text(encoding="utf-8"))

    @app.get("/debug/config")
    def debug_config():
        return {
            "model_path": server.model_path,
            "model_id": server.model_id,
            "base_model_name": server.base_model_name,
            "processor_path": server.processor_path,
            "torch_dtype": str(server.torch_dtype),
            "checkpoints_root": str(server.checkpoints_root) if server.checkpoints_root else None,
            "available_models": list(server.available_models.keys()),
            "model_loaded": server.model is not None,
            "runtime_phase": server.runtime_phase,
            "loading_model_id": server.loading_model_id,
            "generating_model_id": server.generating_model_id,
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a fine-tuned Qwen3.5 VLM checkpoint with an OpenAI-compatible API and web chat UI.")
    parser.add_argument("--model-path", required=True, help="Local or PVC path containing the fine-tuned checkpoint.")
    parser.add_argument("--base-model-name", default=None, help="Base Hugging Face model name. Required if model-path lacks config.json.")
    parser.add_argument("--processor-path", default=None, help="Optional processor path. Defaults to model-path, then base-model-name.")
    parser.add_argument("--model-id", default="qwen3_5_vlm_ft", help="Model ID exposed through the OpenAI-compatible API.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--chat-ui-path", default=str(Path(__file__).resolve().parent / "chat_ui.html"))
    parser.add_argument("--checkpoints-root", default=None, help="Optional root directory to scan for selectable checkpoint folders.")
    parser.add_argument("--models-config", default=None, help="Optional JSON file describing selectable models.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ModelServer(
        model_path=args.model_path,
        model_id=args.model_id,
        base_model_name=args.base_model_name,
        processor_path=args.processor_path,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        checkpoints_root=args.checkpoints_root,
        models_config_path=args.models_config,
    )
    app = build_app(server, Path(args.chat_ui_path))

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

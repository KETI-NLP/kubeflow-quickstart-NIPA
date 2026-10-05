import sys
import os
import transformers
from transformers import AutoTokenizer

# ============================================================================
# Hook into AutoTokenizer to inject our custom chat template
# ============================================================================
orig_from_pretrained = AutoTokenizer.from_pretrained

def patched_from_pretrained(*args, **kwargs):
    tok = orig_from_pretrained(*args, **kwargs)

    template_path = os.environ.get("LIGHTEVAL_CUSTOM_TEMPLATE")
    if template_path and os.path.exists(template_path):
        print(f"--- INJECTING CUSTOM CHAT TEMPLATE FROM {template_path} ---")
        with open(template_path) as f:
            tok.chat_template = f.read()

    return tok

transformers.AutoTokenizer.from_pretrained = patched_from_pretrained


# ============================================================================
# Patch lighteval LightevalTask.download_dataset_worker so it (a) accepts
# `path1::path2` multi-path syntax and (b) falls back to load_from_disk when
# the dataset on disk was saved via Dataset.save_to_disk(). The PyPI 0.12.2
# release lacks both, but custom_korean_character_ocr_task.py and
# custom_korean_heritage_name_vqa_task.py rely on them.
# ============================================================================
from lighteval.tasks.lighteval_task import LightevalTask
from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)


def _patched_download_dataset_worker(task):
    dataset_paths = [p for p in task.dataset_path.split("::") if p]
    if len(dataset_paths) > 1:
        datasets_by_split: dict[str, list] = {}
        for dataset_path in dataset_paths:
            try:
                loaded = load_dataset(
                    path=dataset_path,
                    name=task.dataset_config_name,
                    revision=task.dataset_revision,
                )
            except ValueError as e:
                if "saved using `save_to_disk`" not in str(e):
                    raise
                loaded = load_from_disk(dataset_path)
                if isinstance(loaded, Dataset):
                    loaded = DatasetDict({"train": loaded})

            for split_name, split_dataset in loaded.items():
                if "__dataset_path" not in split_dataset.column_names:
                    split_dataset = split_dataset.add_column(
                        "__dataset_path", [dataset_path] * len(split_dataset)
                    )
                datasets_by_split.setdefault(split_name, []).append(split_dataset)

        dataset = DatasetDict(
            {
                split_name: parts[0] if len(parts) == 1 else concatenate_datasets(parts)
                for split_name, parts in datasets_by_split.items()
            }
        )
    else:
        try:
            dataset = load_dataset(
                path=task.dataset_path,
                name=task.dataset_config_name,
                revision=task.dataset_revision,
            )
        except ValueError as e:
            if "saved using `save_to_disk`" not in str(e):
                raise
            dataset = load_from_disk(task.dataset_path)
            if isinstance(dataset, Dataset):
                dataset = DatasetDict({"train": dataset})

    if task.dataset_filter is not None:
        dataset = dataset.filter(task.dataset_filter)

    return dataset


LightevalTask.download_dataset_worker = staticmethod(_patched_download_dataset_worker)


# ============================================================================
# Patch lighteval's `is_package_available` so the vllm version specifier
# (lighteval 0.13.0 pins vllm<0.10.2,>=0.10.0) does not block us. We intend
# to use vllm 0.19.x, which is the only line that ships sm_100 (Blackwell)
# wheels AND tolerates transformers 5.x. The vllm Python API is stable
# enough for lighteval's vllm_model.py — runtime errors will surface
# normally if any specific call has drifted, and we'll handle those if they
# appear.
# ============================================================================
import lighteval.utils.imports as _li_imports

_orig_is_package_available = _li_imports.is_package_available


def _patched_is_package_available(package):
    pkg_name = getattr(package, "name", str(package))
    if pkg_name == "vllm":
        try:
            from importlib.metadata import version as _v
            _v("vllm")
            return True
        except Exception:
            return False
    return _orig_is_package_available(package)


_li_imports.is_package_available = _patched_is_package_available


# ============================================================================
# Patch vLLM's Qwen3_5ForConditionalGeneration weight mapper so it accepts
# transformers v4.52+ checkpoints where the vision tower is nested under
# `model.language_model.visual.*` (our Qwen3.5 9B SFT checkpoints). The parent
# class (Qwen3VLForConditionalGeneration) only handles `model.visual.*`, which
# leaves all 333 vision-tower tensors uninitialized and aborts model load.
# Adding the more specific prefix BEFORE the generic `model.language_model.`
# mapping is sufficient — WeightsMapper applies prefix rules in dict order.
# ============================================================================
try:
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
    from vllm.model_executor.models.utils import WeightsMapper

    Qwen3_5ForConditionalGeneration.hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.visual.": "visual.",
            "model.visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
        }
    )
    print("--- PATCHED Qwen3_5ForConditionalGeneration.hf_to_vllm_mapper ---")
except ImportError:
    pass  # vllm not installed; running with accelerate backend


# ============================================================================
# Make lighteval's API/endpoint backends (litellm etc.) vision-aware.
# `PromptManager.prepare_prompt_api` returns chat messages whose user content
# is a plain string — it drops `doc.images`. The litellm endpoint's
# `greedy_until` calls only this method, so vision tasks would send no image.
# We wrap it: when the doc has images, the last user message's content becomes
# an OpenAI-style multimodal block list (text + base64 `image_url`), which
# litellm forwards verbatim and the OpenAI-compatible serve accepts.
# ============================================================================
import io as _io
import base64 as _base64
from lighteval.tasks.prompt_manager import PromptManager

_orig_prepare_prompt_api = PromptManager.prepare_prompt_api


def _patched_prepare_prompt_api(self, doc):
    messages = _orig_prepare_prompt_api(self, doc)
    images = getattr(doc, "images", None)
    if not images:
        return messages

    image_blocks = []
    for img in images:
        buf = _io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG")
        b64 = _base64.b64encode(buf.getvalue()).decode()
        image_blocks.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        )

    # Attach images to the last user message.
    for msg in reversed(messages):
        if msg.get("role") == "user":
            text = msg["content"]
            if isinstance(text, str):
                msg["content"] = [{"type": "text", "text": text}] + image_blocks
            elif isinstance(text, list):
                msg["content"] = text + image_blocks
            break
    return messages


PromptManager.prepare_prompt_api = _patched_prepare_prompt_api
print("--- PATCHED PromptManager.prepare_prompt_api (vision-aware endpoint) ---")


if __name__ == "__main__":
    from lighteval.__main__ import app
    # Pass arguments to the original lighteval CLI
    app()

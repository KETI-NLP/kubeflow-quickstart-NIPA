"""vLLM general plugin: weight-mapper patch for Qwen3.5 VLM SFT/DPO checkpoints.

Our fine-tuned Qwen3.5 9B checkpoints store the vision tower nested under
`model.language_model.visual.*` (transformers 5.x SFT layout), but vLLM's
`Qwen3_5ForConditionalGeneration` only maps `model.visual.*`. Without this
patch all 333 vision-tower tensors stay uninitialized and model load aborts.

vLLM calls every entry point registered in the `vllm.general_plugins` group
in *each* process (API server, EngineCore, TP workers) during init, before
the model is built — so the class attribute set here is in effect wherever
weights are actually loaded. This is the same mapping `run_lighteval_patched.py`
applies in-process for the eval-side vLLM backend.
"""

import sys


def register() -> None:
    try:
        from vllm.model_executor.models.qwen3_5 import (
            Qwen3_5ForConditionalGeneration,
        )
        from vllm.model_executor.models.utils import WeightsMapper
    except ImportError as exc:  # vLLM build without qwen3_5 support
        print(f"[qwen35-plugin] qwen3_5 model class unavailable: {exc}", file=sys.stderr)
        return

    # WeightsMapper applies prefix rules in dict order — the more specific
    # `model.language_model.visual.` MUST precede the generic
    # `model.language_model.` rule.
    Qwen3_5ForConditionalGeneration.hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.visual.": "visual.",
            "model.visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
        }
    )
    print(
        "[qwen35-plugin] patched Qwen3_5ForConditionalGeneration.hf_to_vllm_mapper",
        file=sys.stderr,
    )

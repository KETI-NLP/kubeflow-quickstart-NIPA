import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.distributed.fsdp.fully_sharded_data_parallel")

import argparse
import os
import datetime
import math
import subprocess
import logging
import io
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Dataset

# 최신 transformers가 구형 PyTorch의 SDPA에 지원하지 않는 enable_gqa 인자를 넘기는 에러를 방지합니다.
_original_sdpa = F.scaled_dot_product_attention
def _patched_sdpa(*args, **kwargs):
    kwargs.pop("enable_gqa", None)
    return _original_sdpa(*args, **kwargs)
F.scaled_dot_product_attention = _patched_sdpa

import torch.distributed as dist
import torch.distributed.fsdp
if not hasattr(torch.distributed.fsdp, "FSDPModule"):
    torch.distributed.fsdp.FSDPModule = type("FSDPModule", (object,), {})
if not hasattr(torch.distributed.fsdp, "register_fsdp_forward_method"):
    torch.distributed.fsdp.register_fsdp_forward_method = lambda *args, **kwargs: None

from datasets import load_from_disk
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration, TrainingArguments, Trainer

# [REMOVE]
# 최신 transformers 버전에서 제거된 no_init_weights 함수를 HyperCLOVAX 커스텀 코드가 찾을 수 있도록 동적 주입합니다.
import transformers.modeling_utils
import contextlib
if not hasattr(transformers.modeling_utils, "no_init_weights"):
    @contextlib.contextmanager
    def no_init_weights(_enable=True):
        yield
    transformers.modeling_utils.no_init_weights = no_init_weights

# 최신 transformers 모듈이 요구하는 tied weights 속성이 구형 커스텀 모델에 없어 뻗는 에러를 우회합니다.
from transformers import PreTrainedModel
if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
    def _get_tied_weights(self): return getattr(self, "_mock_tied_weights_keys", {})
    def _set_tied_weights(self, value): self._mock_tied_weights_keys = value
    PreTrainedModel.all_tied_weights_keys = property(_get_tied_weights, _set_tied_weights)

# 구형 커스텀 모델의 tie_weights() 함수가 최신 transformers의 추가 인자(missing_keys 등)를 받지 못해 발생하는 TypeError 방지
import inspect
if hasattr(PreTrainedModel, "_finalize_model_loading") and not hasattr(PreTrainedModel, "_is_tie_weights_patched"):
    PreTrainedModel._is_tie_weights_patched = True
    original_finalize = PreTrainedModel._finalize_model_loading
    
    @classmethod
    def patched_finalize(cls, model, *args, **kwargs):
        if hasattr(model, "tie_weights"):
            sig = inspect.signature(model.tie_weights)
            accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            if not accepts_kwargs and "missing_keys" not in sig.parameters:
                original_tie = model.tie_weights
                def safe_tie(*t_args, **t_kwargs):
                    valid_kwargs = {k: v for k, v in t_kwargs.items() if k in sig.parameters}
                    return original_tie(*t_args, **valid_kwargs)
                model.tie_weights = safe_tie
        return original_finalize(model, *args, **kwargs)
            
    PreTrainedModel._finalize_model_loading = patched_finalize

# [HOTFIX] HuggingFace MLX vs Naver MLX Namespace Collision Bypass
import transformers.utils.generic as generic_utils
generic_utils._is_mlx_available = False
generic_utils.is_mlx_available = lambda: False
print("Successfully monkey-patched transformers to ignore Apple MLX arrays to avoid Naver MLX collision.")

from qwen_vl_utils import process_vision_info


class _QwenFastPathWarningFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return "The fast path is not available because one of the required library is not installed." not in msg


logging.getLogger("transformers.models.qwen3_5.modeling_qwen3_5").addFilter(_QwenFastPathWarningFilter())
logging.getLogger("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe").addFilter(_QwenFastPathWarningFilter())

def main():
    parser = argparse.ArgumentParser(description="VLM SFT Qwen 3.5 Distributed Training")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-27B", help="HuggingFace model name")
    parser.add_argument("--processor_name", type=str, default=None, help="Optional processor source; defaults to model_name")
    parser.add_argument("--data_path", type=str, default="/data/vlm_sft_dataset", help="Path to local or PVC dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save model")
    parser.add_argument("--logging_dir", type=str, default="/data/log/vlm-sft", help="TensorBoard log directory")
    parser.add_argument("--skip_dataset_validation", action="store_true", help="Skip dataset-wide validation scan and issue counting")
    parser.add_argument("--max_steps", type=int, default=None, help="Max steps for training (None means full epochs)")
    parser.add_argument("--num_train_epochs", type=float, default=3.0, help="Number of training epochs when max_steps is not set")
    parser.add_argument("--max_sequence_len", type=int, default=8192, help="Sequence max length for padding")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--save_steps", type=int, default=2000, help="Checkpoint save interval in steps")
    parser.add_argument("--save_total_limit", type=int, default=None, help="Optional max number of checkpoints to keep")
    args = parser.parse_args()



    # 분산 환경 초기화를 최상단으로 끌어올림 (NFS 동시 캐시 쓰기 충돌 방지)
    if "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", timeout=datetime.timedelta(seconds=7200))
        rank = dist.get_rank()
    else:
        rank = 0

    from datasets import concatenate_datasets

    # ------------------
    # Fsspec Maxdepth KeyError 방지 패치
    # (ncloud-mlx-data-manager의 fsspec 2023.5.0 요구사항과 huggingface_hub 최신 버전 간의 충돌 우회)
    # ------------------
    try:
        import huggingface_hub.hf_file_system
        import fsspec.spec
        
        # HfFileSystem의 glob이 kwargs에 maxdepth를 강제 주입하여 발생하는 KeyError 우회
        huggingface_hub.hf_file_system.HfFileSystem.glob = fsspec.spec.AbstractFileSystem.glob
        print("Successfully patched HfFileSystem.glob to bypass fsspec maxdepth error.")
    except Exception as e:
        print(f"Failed to patch hf_file_system: {e}")

    import contextlib
    @contextlib.contextmanager
    def main_process_first():
        if rank != 0:
            dist.barrier()
        yield
        if rank == 0:
            dist.barrier()

    with main_process_first():
        local_only = (rank != 0)
        processor_name = args.processor_name or args.model_name
        
        # HuggingFace hub login token conflict 우회를 위해 Dataset(MLXP) Load 이전에 모델/토크나이저 미리 로드
        # HACK: ncloud-mlx가 파이썬 시작 시점에 HF_ENDPOINT를 강제로 MLXP로 돌려놓으므로, 다운로드 직전에 원상복구합니다.
        print(f"Loading processor from {processor_name} and model from {args.model_name} bypassing MLXP...")
        import huggingface_hub.constants
        orig_hf_endpoint = os.environ.pop("HF_ENDPOINT", None)
        orig_hf_token = os.environ.pop("HF_TOKEN", None)
        huggingface_hub.constants.ENDPOINT = "https://huggingface.co"
        
        processor = AutoProcessor.from_pretrained(processor_name, local_files_only=local_only, trust_remote_code=True)
        print(f"[RANK {rank}] Qwen3_5ForConditionalGeneration.from_pretrained!!!")
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            args.model_name,
            attn_implementation="flash_attention_2",
            dtype=torch.bfloat16,
            local_files_only=local_only,
            trust_remote_code=True,
        )

        # 원래 상태 복구 (이후 MLXP 데이터셋 로드를 위해 필수)
        if orig_hf_endpoint: os.environ["HF_ENDPOINT"] = orig_hf_endpoint
        if orig_hf_token: os.environ["HF_TOKEN"] = orig_hf_token
        huggingface_hub.constants.ENDPOINT = orig_hf_endpoint if orig_hf_endpoint else "https://huggingface.co"
        
        # 모델의 모든 파라미터를 bfloat16으로 강제 변환하여 FSDP 파라미터 Flatten 에러 방지
        model.to(torch.bfloat16)

        # 최신 transformers 라이브러리에서 vision_model이 BaseModelOutput을 반환하게 되어
        # mm_projector 입력 타입 오류(TypeError: linear argument must be Tensor)가 생기는 것을 방지하는 몽키 패치
        class MMProjectorWrapper(torch.nn.Module):
            def __init__(self, original_projector):
                super().__init__()
                self.original_projector = original_projector
                
            def forward(self, x, *args, **kwargs):
                if not isinstance(x, torch.Tensor):
                    if hasattr(x, 'last_hidden_state'):
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
                    print(f"Patched mm_projector in {module.__class__.__name__}")
                else:
                    patch_all_projectors(child)

        patch_all_projectors(model)

        # 2. Vision Model Merger 누락 패치 (transformers >= 4.48)
        # 최신 transformers에서는 Qwen2.5-VL VisionModel이 내부에서 merger를 호출하지 않아 1280 차원이 그대로 나옴
        if hasattr(model, "model") and hasattr(model.model, "vision_model") and hasattr(model.model.vision_model, "merger"):
            orig_vision_forward = model.model.vision_model.forward
            def patched_vision_forward(*args, **kwargs):
                out = orig_vision_forward(*args, **kwargs)
                
                hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else (out[0] if isinstance(out, tuple) else out)
                
                if hasattr(hidden, "shape") and hidden.shape[-1] == 1280:
                    merger = model.model.vision_model.merger
                    try:
                        hidden = merger(hidden)
                    except Exception as e:
                        print(f"merger call failed: {e}")
                        pass
                    
                    if hasattr(out, "last_hidden_state"):
                        from transformers.modeling_outputs import BaseModelOutput
                        if isinstance(out, BaseModelOutput):
                            out = BaseModelOutput(
                                last_hidden_state=hidden,
                                hidden_states=out.hidden_states if hasattr(out, "hidden_states") else None,
                                attentions=out.attentions if hasattr(out, "attentions") else None
                            )
                        else:
                            out.last_hidden_state = hidden
                    elif isinstance(out, tuple):
                        out = (hidden,) + out[1:]
                    else:
                        out = hidden
                return out
                
            model.model.vision_model.forward = patched_vision_forward
            print("Successfully patched vision_model to manually call patch merger")


        # 1. PVC(디스크)에서 전처리된 다중 데이터 불러오기
        data_paths = [dp.strip() for dp in args.data_path.split(",")]
        dataset_list = []
        dataset_stats = []

        for dp in data_paths:
            # MLXP Dataset Loading
            if "/" in dp and not os.path.exists(dp) and not dp.startswith("/data/"):
                try:
                    from mlx.sdk.data import load_dataset as mlxp_load_dataset
                    ds = mlxp_load_dataset(dp)
                except ImportError:
                    print("MLXP SDK is not installed but requested. Trying load_from_disk as fallback.")
                    ds = load_from_disk(dp)
            else:
                ds = load_from_disk(dp)

            # DPO 형태의 데이터셋을 SFT로 변환하는 자동 로직 (Packed Dataset은 이미 변환되어 있음)
            from datasets import DatasetDict
            if isinstance(ds, DatasetDict) and "train" in ds:
                train_ds = ds["train"]
            else:
                train_ds = ds

            dataset_list.append(train_ds)
            dataset_stats.append((dp, len(train_ds)))

            if rank == 0:
                print(f"Loaded dataset: {dp} | samples={len(train_ds):,}", flush=True)

        if rank == 0:
            total_loaded_samples = sum(sample_count for _, sample_count in dataset_stats)
            print(
                f"Combining {len(dataset_list)} datasets... total_samples={total_loaded_samples:,}",
                flush=True,
            )

        dataset_issue_stats = []

        def normalize_images_value(images):
            if images is None:
                return []

            if not isinstance(images, list):
                images = [images]

            normalized_images = []
            for img in images:
                if img is None:
                    continue
                if isinstance(img, dict):
                    img_bytes = img.get("bytes")
                    img_path = img.get("path")
                    if img_bytes is None and img_path is None:
                        continue
                    normalized_images.append({"bytes": img_bytes, "path": img_path})
                elif hasattr(img, "save"):
                    buffer = io.BytesIO()
                    img_format = getattr(img, "format", None) or "PNG"
                    img.save(buffer, format=img_format)
                    normalized_images.append({"bytes": buffer.getvalue(), "path": None})
                else:
                    normalized_images.append(img)

            return normalized_images

        class PackedVLMDataset(Dataset):
            def __init__(self, base_dataset, dataset_name, issues, rank, skip_validation=False):
                self.base_dataset = base_dataset
                self.dataset_name = dataset_name
                self.skip_validation = skip_validation
                self.valid_indices = None

                if self.skip_validation:
                    issues["valid_samples"] = len(base_dataset)
                    issues["filtered_invalid_samples"] = 0
                    if rank == 0:
                        print(
                            f"[Dataset Scan] {dataset_name} | skipped validation scan | samples={len(base_dataset):,}",
                            flush=True,
                        )
                    return

                self.valid_indices = []
                total_examples = len(base_dataset)
                progress_interval = max(1000, total_examples // 20)

                for idx in range(len(base_dataset)):
                    example = base_dataset[idx]
                    if example.get("input_ids") is None:
                        issues["missing_input_ids"] += 1
                    if example.get("labels") is None:
                        issues["missing_labels"] += 1

                    images = example.get("images")
                    if images is None:
                        issues["images_none"] += 1
                    elif not isinstance(images, list):
                        issues["images_non_list"] += 1
                        images = [images]

                    if images is not None:
                        for img in images:
                            if img is None:
                                continue
                            if isinstance(img, dict):
                                if img.get("bytes") is None and img.get("path") is None:
                                    issues["images_dict_missing_payload"] += 1
                            elif hasattr(img, "save"):
                                issues["images_pil_decoded"] += 1

                    if example.get("input_ids") is not None and example.get("labels") is not None:
                        self.valid_indices.append(idx)

                    if rank == 0 and ((idx + 1) % progress_interval == 0 or (idx + 1) == total_examples):
                        percent = ((idx + 1) / total_examples) * 100
                        print(
                            f"[Dataset Scan] {dataset_name} | "
                            f"processed={idx + 1:,}/{total_examples:,} ({percent:.1f}%) | "
                            f"valid={len(self.valid_indices):,} | "
                            f"missing_input_ids={issues['missing_input_ids']:,} | "
                            f"missing_labels={issues['missing_labels']:,}",
                            flush=True,
                        )

                issues["valid_samples"] = len(self.valid_indices)
                issues["filtered_invalid_samples"] = len(base_dataset) - len(self.valid_indices)

            def __len__(self):
                if self.valid_indices is None:
                    return len(self.base_dataset)
                return len(self.valid_indices)

            def __getitem__(self, idx):
                if self.valid_indices is None:
                    example = dict(self.base_dataset[idx])
                else:
                    example = dict(self.base_dataset[self.valid_indices[idx]])
                example["images"] = normalize_images_value(example.get("images"))
                example["dataset_name"] = self.dataset_name
                return example

        wrapped_datasets = []

        for i in range(len(dataset_list)):
            dataset_name = dataset_stats[i][0]
            train_ds = dataset_list[i]
            issues = {
                "dataset": dataset_name,
                "missing_input_ids": 0,
                "missing_labels": 0,
                "missing_images_column": "images" not in train_ds.features,
                "images_none": 0,
                "images_non_list": 0,
                "images_dict_missing_payload": 0,
                "images_pil_decoded": 0,
            }

            wrapped_dataset = PackedVLMDataset(
                train_ds,
                dataset_name,
                issues,
                rank,
                skip_validation=args.skip_dataset_validation,
            )
            wrapped_datasets.append(wrapped_dataset)
            dataset_issue_stats.append(issues)

        if rank == 0:
            for issues in dataset_issue_stats:
                print(
                    "[Dataset Check] "
                    f"{issues['dataset']} | "
                    f"missing_input_ids={issues['missing_input_ids']:,} | "
                    f"missing_labels={issues['missing_labels']:,} | "
                    f"missing_images_column={issues['missing_images_column']} | "
                    f"images_none={issues['images_none']:,} | "
                    f"images_non_list={issues['images_non_list']:,} | "
                    f"images_dict_missing_payload={issues['images_dict_missing_payload']:,} | "
                    f"images_pil_decoded={issues['images_pil_decoded']:,} | "
                    f"filtered_invalid_samples={issues['filtered_invalid_samples']:,} | "
                    f"valid_samples={issues['valid_samples']:,}",
                    flush=True,
                )

        combined_train = ConcatDataset(wrapped_datasets)
        if rank == 0:
            total_valid_samples = sum(item["valid_samples"] for item in dataset_issue_stats)
            total_filtered_samples = sum(item["filtered_invalid_samples"] for item in dataset_issue_stats)
            print(
                f"Prepared concatenated torch dataset: valid_samples={total_valid_samples:,} | "
                f"filtered_invalid_samples={total_filtered_samples:,}",
                flush=True,
            )

        # print("Selecting only 1 sample for debugging...")
        # combined_train = combined_train.select([0])

        # 텍스트 전용 데이터 모델 학습 시 FSDP 데드락 방지용 필터 (이제 Dummy Image를 사용하므로 필터링 보류)
        # def has_image(example):
        #     return len(example.get("images", [])) > 0
        # print("Filtering out text-only examples to prevent FSDP deadlock...")
        # combined_train = combined_train.filter(has_image, num_proc=4, desc="Filtering out text-only examples")

        dataset = {"train": combined_train}

    # Rank 0와 나머지 노드들의 동기화는 위에서 사용한 main_process_first() 컨텍스트 매니저로 이미 완벽하게 처리되었습니다.
    # 여기에 있던 중복된 dist.barrier()는 Rank 0 혼자만 두 번 기다리게 만들어 무한 데드락을 유발하므로 삭제합니다.
    # 3. 데이터 프롬프트 전처리 (Data Collator에서 실시간 처리)
    def vlm_data_collator(features):
        def decode_packed_images(images):
            from PIL import Image
            import io

            decoded_images = []
            for img in images:
                if isinstance(img, dict) and "bytes" in img and img["bytes"] is not None:
                    pil_image = Image.open(io.BytesIO(img["bytes"]))
                    pil_image.load()
                    decoded_images.append(pil_image)
                elif isinstance(img, Image.Image):
                    img.load()
                    decoded_images.append(img)
                else:
                    decoded_images.append(img)
            return decoded_images

        def build_synthetic_qwen_messages(images):
            return [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": img} for img in images],
                }
            ]

        input_ids_list = []
        labels_list = []
        
        for f in features:
            if isinstance(f["input_ids"], list):
                input_ids_list.append(torch.tensor(f["input_ids"], dtype=torch.long))
            else:
                input_ids_list.append(torch.tensor(f["input_ids"]).long())
                
            if isinstance(f["labels"], list):
                labels_list.append(torch.tensor(f["labels"], dtype=torch.long))
            else:
                labels_list.append(torch.tensor(f["labels"]).long())
                
        # pad input_ids and labels
        pad_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else processor.tokenizer.eos_token_id
        if pad_id is None: pad_id = 0
            
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
        
        # 특정 길이(max_sequence_len)로 고정 패딩 (초과 시 잘림)
        max_sequence_len = args.max_sequence_len
        current_len = input_ids.shape[1]
        if current_len < max_sequence_len:
            pad_len = max_sequence_len - current_len
            input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=pad_id)
            labels = torch.nn.functional.pad(labels, (0, pad_len), value=-100)
        elif current_len > max_sequence_len:
            input_ids = input_ids[:, :max_sequence_len]
            labels = labels[:, :max_sequence_len]
            
        attention_mask = (input_ids != pad_id).long()

        image_batches = []
        for f in features:
            images = f.get("images", [])
            if images is not None and len(images) > 0:
                decoded_images = decode_packed_images(images)
            else:
                from PIL import Image
                dummy_image = Image.new('RGB', (28, 28), (0, 0, 0))
                decoded_images = [dummy_image]

            synthetic_messages = build_synthetic_qwen_messages(decoded_images)
            qwen_images = process_vision_info(synthetic_messages)[0]
            qwen_images = [img for img in qwen_images if img is not None] if qwen_images else decoded_images

            # Packed image tokens were produced after qwen_vl_utils.fetch_image() applied smart_resize.
            # Replaying that path sample-wise keeps image_grid_thw aligned with the stored input_ids.
            sample_image_batch = processor(
                text=[""],
                images=qwen_images,
                padding=False,
                return_tensors="pt"
            )
            image_batches.append(sample_image_batch)
        
        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

        # Qwen3.5 M-RoPE는 토큰별 modality 타입 정보를 별도로 요구합니다.
        # 공식 processor 출력 규칙에 맞춰 text=0, image token=1, video token=2로 복원합니다.
        mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32)
        image_token_id = getattr(model.config, "image_token_id", None)
        video_token_id = getattr(model.config, "video_token_id", None)
        if image_token_id is not None:
            mm_token_type_ids[input_ids == image_token_id] = 1
        if video_token_id is not None:
            mm_token_type_ids[input_ids == video_token_id] = 2
        batch["mm_token_type_ids"] = mm_token_type_ids

        def concat_image_batches(key):
            values = [image_batch[key] for image_batch in image_batches if key in image_batch]
            if values:
                batch[key] = torch.cat(values, dim=0)

        concat_image_batches("pixel_values")
        concat_image_batches("pixel_values_2d")
        concat_image_batches("pixel_values_3d")
        concat_image_batches("image_grid_thw")
        concat_image_batches("cross_attention_mask")
            
        return batch

    print("Dataset Ready. Training will collate on the fly.")

    # 4. HuggingFace Trainer 설정
    valid_wrap_classes = list(set(
        m.__class__.__name__ for m in model.modules() 
        if "DecoderLayer" in m.__class__.__name__ or ("VisionBlock" in m.__class__.__name__ and "Qwen" in m.__class__.__name__)
    ))
    # 그래도 없으면 기본 폴백 추가
    if not valid_wrap_classes:
        valid_wrap_classes = ["Qwen2DecoderLayer"]
    print(f"FSDP dynamically wrapping layer classes: {valid_wrap_classes}")

    os.environ["TENSORBOARD_LOGGING_DIR"] = args.logging_dir

    training_args = TrainingArguments(
        output_dir=args.output_path,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps if args.max_steps is not None else -1, # 인자가 있을 때만 적용, 없으면 전체 데이터 학습
        logging_dir=args.logging_dir,
        logging_steps=1,
        report_to=["tensorboard"],
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=True,                          # VRAM 절반 감소
        gradient_checkpointing=True,        # Forward 중간 캐시 삭제 (OOM 완벽 해결)
        fsdp="full_shard auto_wrap",
        fsdp_config={"transformer_layer_cls_to_wrap": valid_wrap_classes},
        ddp_find_unused_parameters=False,
        remove_unused_columns=False, # VLM에서는 image가 사용되므로 False 필수
    )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    grad_accum_steps = training_args.gradient_accumulation_steps
    dataset_num_samples = len(dataset["train"])
    global_batch_size = training_args.per_device_train_batch_size * world_size * grad_accum_steps
    steps_per_epoch = max(1, math.ceil(dataset_num_samples / max(global_batch_size, 1)))
    uses_explicit_max_steps = training_args.max_steps is not None and training_args.max_steps > 0
    planned_max_steps = training_args.max_steps if uses_explicit_max_steps else math.ceil(steps_per_epoch * training_args.num_train_epochs)
    total_sample_exposures = planned_max_steps * global_batch_size if uses_explicit_max_steps else int(dataset_num_samples * training_args.num_train_epochs)

    if rank == 0:
        print(
            "Training plan: "
            f"dataset_samples={dataset_num_samples:,} | "
            f"world_size={world_size} | "
            f"global_batch_size={global_batch_size:,} | "
            f"steps_per_epoch={steps_per_epoch:,} | "
            f"planned_steps={planned_max_steps:,} | "
            f"planned_sample_exposures={total_sample_exposures:,}",
            flush=True,
        )

    # 4.5. 모델 저장 시 JSON 직렬화 에러 대응 (torch.dtype 객체를 문자열로 치환)
    def clean_config_dtype(config_obj):
        for k, v in config_obj.__dict__.items():
            if isinstance(v, torch.dtype):
                setattr(config_obj, k, str(v).split('.')[-1])
            elif hasattr(v, '__dict__'):
                clean_config_dtype(v)
            elif isinstance(v, dict):
                for dk, dv in v.items():
                    if isinstance(dv, torch.dtype):
                        v[dk] = str(dv).split('.')[-1]
                    elif hasattr(dv, '__dict__'):
                        clean_config_dtype(dv)
    clean_config_dtype(model.config)

    import time
    from transformers import TrainerCallback
    def format_eta(seconds):
        if seconds is None or seconds < 0 or math.isinf(seconds):
            return "unknown"
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def get_nvidia_smi_stats(local_rank):
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={local_rank}",
                    "--query-gpu=memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            used_mb, total_mb, util_gpu = [x.strip() for x in result.stdout.strip().split(",")]
            return int(used_mb), int(total_mb), int(util_gpu)
        except Exception:
            return None, None, None

    class GPUMemoryCallback(TrainerCallback):
        def __init__(self):
            self.last_log_step = 0
            self.last_log_time = time.time()

        def on_log(self, args, state, control, logs=None, **kwargs):
            if torch.cuda.is_available():
                mem_alloc = torch.cuda.memory_allocated() / (1024**3)
                mem_max = torch.cuda.max_memory_allocated() / (1024**3)
                rank = os.environ.get("RANK", "0")
                
                current_time = time.time()
                steps_passed = state.global_step - self.last_log_step
                step_time = (current_time - self.last_log_time) / max(steps_passed, 1)
                max_steps = state.max_steps if state.max_steps and state.max_steps > 0 else planned_max_steps
                remaining_steps = max(0, max_steps - state.global_step)
                eta_seconds = remaining_steps * step_time
                processed_samples = min(state.global_step * global_batch_size, total_sample_exposures)
                
                if rank == "0" and state.global_step > 0:
                    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                    used_mb, total_mb, util_gpu = get_nvidia_smi_stats(local_rank)
                    nvidia_smi_msg = ""
                    if used_mb is not None and total_mb is not None and util_gpu is not None:
                        nvidia_smi_msg = f" | nvidia-smi Mem {used_mb}/{total_mb} MiB | GPU Util {util_gpu}%"

                    print(
                        f"\n[Step {state.global_step}/{max_steps}] "
                        f"{step_time:.2f} s/it | "
                        f"Samples {processed_samples:,}/{total_sample_exposures:,} | "
                        f"ETA {format_eta(eta_seconds)} | "
                        f"GPU Mem Allocated: {mem_alloc:.2f} GB | Max: {mem_max:.2f} GB"
                        f"{nvidia_smi_msg}\n",
                        flush=True,
                    )
                
                self.last_log_step = state.global_step
                self.last_log_time = current_time

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=vlm_data_collator,
        callbacks=[GPUMemoryCallback()],
    )

    import shutil
    original_rmtree = shutil.rmtree
    def safe_rmtree(path, *args, **kwargs):
        try:
            original_rmtree(path, *args, **kwargs)
        except FileNotFoundError:
            pass
    shutil.rmtree = safe_rmtree

    # 5. 분산 학습 시작
    print("Starting training...")
    trainer.train()
    
    # 6. 학습 결과 로컬 저장 (FSDP 환경에서는 모든 워커가 통신에 참여하여 모델 가중치를 Gather 해야 하므로 if문을 사용하지 않습니다)
    trainer.save_model(args.output_path)
    if rank == 0:
        print(f"Model successfully saved to {args.output_path}")
        try:
            if processor is not None:
                processor.save_pretrained(args.output_path)
                print(f"Processor/tokenizer successfully saved to {args.output_path}")
        except Exception as e:
            print(
                f"[Warn] Model save succeeded but processor/tokenizer save failed at {args.output_path}: {e}",
                flush=True,
            )

    dist.barrier()

if __name__ == "__main__":
    main()

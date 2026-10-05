import contextlib
import math
import os
from functools import partial
from itertools import chain
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn

try:
    from einops import rearrange
    from timm.layers import LayerNorm, LayerNorm2d
    from timm.models.regnet import RegStage
except:
    print("packages needed for anyres are not imported")
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
)
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import no_init_weights

from .configuration_vlm import HCXVisionConfig

try:
    from .cosyvoice import (
        DEFAULT_SAMPLE_RATE,
        MIN_DISCRETE_AUDIO_CHUNK_SAMPLES,
        CosyvoiceEncoder,
    )
except:
    print("packages needed for discrete audio are not imported")
try:
    from .ta_tok import TextAlignedTokenizer
except:
    print("packages needed for discrete vision are not imported")

try:
    from .mambamia_videoaudio_compressor import (
        MambaMiaVideoAudioCompressor,
        MambaMiaVideoAudioCompressorConfig,
    )
except:
    print("packages needed for mambamia video-audio compressor are not imported")


def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def is_ampere_or_newer():
    if not torch.cuda.is_available():
        return False

    gpu_name = torch.cuda.get_device_name()

    ampere_keywords = [
        "RTX 30",
        "RTX 40",
        "A100",
        "H100",
        "A6000",
        "A5000",
        "A4000",
        "A3000",
        "A2000",
        "A1000",
    ]

    return any(keyword in gpu_name for keyword in ampere_keywords)


EOT = "<|endofturn|>"
IMG_LOC = "<|IMAGE_PAD|>"


class HCXVisionPreTrainedModel(PreTrainedModel):
    config_class = HCXVisionConfig
    base_model_prefix = "model"
    vision_model_name = "vision_model"
    _no_split_modules = [
        "CLIPAttention",
        "SiglipVisionModel",
    ]
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        if (
            isinstance(module, nn.Conv2d)
            or isinstance(module, nn.Embedding)
            or isinstance(module, nn.Linear)
        ):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Parameter):
            embed_std = 1 / torch.sqrt(
                torch.tensor(module.size(0), dtype=torch.float)
            ).to(module.dtype)
            module.data.normal_(mean=0.0, std=embed_std)


class HCXVisionModel(HCXVisionPreTrainedModel):
    def __init__(
        self,
        config: HCXVisionConfig,
        without_llm=False,
        **kwargs,
    ):
        super().__init__(config)

        self.flag_changed_max_position_embeddings = False
        self.without_llm = without_llm

        vision_model_type = config.vision_config.model_type

        self.is_qwen_visual = False
        if vision_model_type == "qwen2_5_vl_visual":
            self.is_qwen_visual = True

        self.freeze_before_sampler = kwargs.pop("freeze_before_sampler", False)

        vision_config = config.vision_config
        vision_config.anyres = config.anyres
        vision_config.max_num_grids = config.max_num_grids
        vision_config.update({"torch_dtype": config.torch_dtype})
        self.vision_config = vision_config

        if without_llm:
            vision_config.vison_pretrained_name_or_path = (
                config.vision_model_name_or_path
            )

        if self.is_qwen_visual and is_ampere_or_newer():
            vision_config._attn_implementation = "flash_attention_2"
        self.vision_model = AutoModel.from_config(vision_config, trust_remote_code=True)
        if not self.config.freeze_encoder:
            self.vision_model.gradient_checkpointing_enable()
        else:
            self.vision_model.eval()

        if vision_config.model_type == "qwen2_5_vl_visual":
            import torch.nn.functional as F

            def new_block_forward(
                self,
                hidden_states: torch.Tensor,
                cu_seqlens: torch.Tensor,
                rotary_pos_emb: Optional[torch.Tensor] = None,
                position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            ) -> torch.Tensor:
                hidden_states = hidden_states + self.attn(
                    self.norm1(hidden_states),
                    cu_seqlens=cu_seqlens,
                    rotary_pos_emb=rotary_pos_emb,
                    position_embeddings=position_embeddings,
                )
                if hidden_states.dtype == torch.float16:
                    org_type = hidden_states.dtype
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
                        hidden_states = hidden_states + self.mlp(
                            self.norm2(hidden_states)
                        )
                    hidden_states = hidden_states.to(org_type)
                return hidden_states

            def new_last_block_forward(
                self,
                hidden_states: torch.Tensor,
                cu_seqlens: torch.Tensor,
                rotary_pos_emb: Optional[torch.Tensor] = None,
                position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            ) -> torch.Tensor:
                hidden_states = hidden_states + self.attn(
                    self.norm1(hidden_states),
                    cu_seqlens=cu_seqlens,
                    rotary_pos_emb=rotary_pos_emb,
                    position_embeddings=position_embeddings,
                )
                if hidden_states.dtype == torch.float16:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
                        hidden_states = hidden_states + self.mlp(
                            self.norm2(hidden_states)
                        )
                    hidden_states = hidden_states.to(torch.float32)
                return hidden_states

            def new_merger_forward(self, x: torch.Tensor) -> torch.Tensor:
                if self.mlp[0].weight.dtype == torch.float16:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
                        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
                    x = x.to(self.mlp[0].weight.dtype)

                else:
                    x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
                return x

            import types

            for b_idx, blk in enumerate(self.vision_model.blocks):
                if b_idx == len(self.vision_model.blocks) - 1:
                    blk.forward = types.MethodType(
                        new_last_block_forward, self.vision_model.blocks[b_idx]
                    )
                elif b_idx in vision_config.fullatt_block_indexes:
                    blk.forward = types.MethodType(
                        new_block_forward, self.vision_model.blocks[b_idx]
                    )
            self.vision_model.merger.forward = types.MethodType(
                new_merger_forward, self.vision_model.merger
            )

        if config.mm_projector_type == "qwen_merger":

            import torch.nn.functional as F

            def new_forward(
                self, hidden_states: torch.Tensor, grid_thw: torch.Tensor
            ) -> torch.Tensor:
                """
                Args:
                    hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                        The final hidden states of the model.
                    grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                        The temporal, height and width of feature shape of each image in LLM.

                Returns:
                    `torch.Tensor`: hidden_states.
                """
                hidden_states = self.patch_embed(hidden_states)
                rotary_pos_emb = self.rot_pos_emb(grid_thw)
                window_index, cu_window_seqlens = self.get_window_index(grid_thw)
                cu_window_seqlens = torch.tensor(
                    cu_window_seqlens,
                    device=hidden_states.device,
                    dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
                )
                cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

                seq_len, _ = hidden_states.size()
                hidden_states = hidden_states.reshape(
                    seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
                )
                hidden_states = hidden_states[window_index, :, :]
                hidden_states = hidden_states.reshape(seq_len, -1)
                rotary_pos_emb = rotary_pos_emb.reshape(
                    seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
                )
                rotary_pos_emb = rotary_pos_emb[window_index, :, :]
                rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
                emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
                position_embeddings = (emb.cos(), emb.sin())

                cu_seqlens = torch.repeat_interleave(
                    grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
                ).cumsum(
                    dim=0,
                    dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
                )
                cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

                for layer_num, blk in enumerate(self.blocks):
                    if layer_num in self.fullatt_block_indexes:
                        cu_seqlens_now = cu_seqlens
                    else:
                        cu_seqlens_now = cu_window_seqlens
                    if self.gradient_checkpointing and self.training:
                        hidden_states = self._gradient_checkpointing_func(
                            blk.__call__,
                            hidden_states,
                            cu_seqlens_now,
                            None,
                            position_embeddings,
                        )
                    else:
                        hidden_states = blk(
                            hidden_states,
                            cu_seqlens=cu_seqlens_now,
                            position_embeddings=position_embeddings,
                        )

                return hidden_states, window_index

            import types

            self.vision_model.forward = types.MethodType(new_forward, self.vision_model)
            self.vision_model.merger = nn.Identity()

        self.discrete_vision_model = None
        discrete_vision_config = config.discrete_vision_config
        if discrete_vision_config is not None and isinstance(
            discrete_vision_config, dict
        ):
            discrete_vision_config.update({"torch_dtype": config.torch_dtype})
        self.discrete_vision_config = discrete_vision_config
        if (
            hasattr(config, "discrete_vision_config")
            and config.discrete_vision_config is not None
        ):
            try:
                assert (
                    discrete_vision_config["model_type"] == "ta_tok"
                ), "Only 'ta_tok' discrete vision model is supported currently."
                self.discrete_vision_model = TextAlignedTokenizer.from_checkpoint(
                    discrete_vision_config["model_name_or_path"],
                    load_teacher=False,
                    input_type="indices",
                )
                if self.discrete_vision_model is not None:
                    self.discrete_vision_model.eval()
            except Exception as e:
                print(f"Warning: Failed to initialize discrete vision model: {e}")
                self.discrete_vision_model = None

        self.audio_model = None
        audio_config = config.audio_config
        if audio_config is not None and isinstance(audio_config, dict):
            audio_config.update({"torch_dtype": config.torch_dtype})
        self.audio_config = audio_config
        if hasattr(config, "audio_config") and config.audio_config is not None:
            self.audio_model = (
                AutoModel.from_config(audio_config)
                if audio_config is not None
                else None
            )
            if self.audio_model is not None:
                self.audio_model.eval()

        self.discrete_audio_model = None
        discrete_audio_config = config.discrete_audio_config
        if discrete_audio_config is not None and isinstance(
            discrete_audio_config, dict
        ):
            discrete_audio_config.update({"torch_dtype": config.torch_dtype})
        self.discrete_audio_config = discrete_audio_config
        if (
            hasattr(config, "discrete_audio_config")
            and config.discrete_audio_config is not None
        ):
            try:
                assert (
                    discrete_audio_config["model_type"] == "cosyvoice2"
                ), "Only 'cosyvoice2' discrete audio model is supported currently."
                self.discrete_audio_model = CosyvoiceEncoder.from_pretrained(
                    discrete_audio_config["model_name_or_path"]
                )
                if self.discrete_audio_model is not None:
                    self.discrete_audio_model.eval()
            except Exception as e:
                print(f"Warning: Failed to initialize discrete audio model: {e}")
                self.discrete_audio_model = None

        if hasattr(config, "text_config") and config.text_config is not None:
            text_config = config.text_config
        else:
            raise ValueError("text_config is not defined")
        text_config.update({"torch_dtype": config.torch_dtype})
        if config.text_config.model_type in ["llama", "hyperclovax", "gpt2"]:
            text_config._attn_implementation = config._attn_implementation
        if text_config.model_type != "hyperclovax":
            text_config.logits_scaling = 1.0

        text_config.vocab_size = (
            text_config.padded_vocab_size
            if hasattr(text_config, "padded_vocab_size")
            else text_config.vocab_size
        )

        if not without_llm:
            with no_init_weights():
                self.language_model = AutoModelForCausalLM.from_config(
                    text_config, trust_remote_code=True
                )

            if config.text_config.model_type in ["llama", "hyperclovax", "gpt2"]:
                if not self.config.freeze_decoder:
                    self.language_model.gradient_checkpointing_enable()
            if (
                config.text_config.model_type == "hyperclovax"
                and hasattr(self, "use_liger")
                and self.use_liger
            ):
                self.language_model._get_apply_liger_kernel_converter()(
                    model=self.language_model
                )
                print("liger kernel for hcx 24b will be used")

        self.num_queries_vis_abstractor = config.num_queries_vis_abstractor

        input_hidden_size = vision_config.hidden_size
        if vision_config.model_type == "qwen2_5_vl_visual":
            input_hidden_size = vision_config.out_hidden_size
        if config.mm_projector_type == "linear":
            self.mm_projector = nn.Linear(input_hidden_size, text_config.hidden_size)

        elif config.mm_projector_type == "cabstractor":
            self.mm_projector = CAbstractor(
                num_queries=self.num_queries_vis_abstractor,
                num_input_tokens=(
                    self.vision_config.image_size // self.vision_config.patch_size
                )
                ** 2,
                encoder_hidden_size=input_hidden_size,
                hidden_size=input_hidden_size,
                output_hidden_size=text_config.hidden_size,
                pos_emb=config.proj_pos_emb,
                prenorm=config.proj_prenorm,
            )
            self.mm_projector.pos_emb.to(config.torch_dtype)
        elif config.mm_projector_type == "qwen_merger":
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                Qwen2_5_VLPatchMerger,
            )

            self.mm_projector = Qwen2_5_VLPatchMerger(
                dim=text_config.hidden_size, context_dim=input_hidden_size
            )

            def new_forward(self, inputs) -> torch.Tensor:
                x, window_index = inputs
                x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
                reverse_indices = torch.argsort(window_index)
                x = x[reverse_indices, :]
                return x

            self.mm_projector.forward = types.MethodType(new_forward, self.mm_projector)

        else:
            self.mm_projector = VLM_Mlp(
                config.mm_projector_type,
                input_hidden_size,
                hidden_features=input_hidden_size,
                out_features=text_config.hidden_size,
            )

        if audio_config is None:
            self.audio_projector = None
        else:
            if config.audio_projector_type == "linear":
                self.audio_projector = nn.Linear(
                    audio_config.d_model, text_config.hidden_size
                )
            else:
                self.audio_projector = VLM_Mlp(
                    config.audio_projector_type,
                    audio_config.d_model,
                    hidden_features=audio_config.d_model,
                    out_features=text_config.hidden_size,
                )

        self.video_audio_compressor = None
        if (
            audio_config is not None
            and config.video_audio_compressor_type == "mambamia"
        ):
            compressor_config = MambaMiaVideoAudioCompressorConfig(
                input_size=text_config.hidden_size,
                output_size=text_config.hidden_size,
                chunk_size=25,
                num_hidden_layers=1,
            )
            self.video_audio_compressor = MambaMiaVideoAudioCompressor(
                compressor_config
            )
            self.video_audio_compressor = self.video_audio_compressor.to(
                config.torch_dtype
            )

        self.use_nth_layer = config.use_nth_layer
        self.model_parallel = False
        self.device_map = None
        self.vision_model_use_no_grad = None

        self.text_config = text_config

        self.anyres = False
        self.unpad = config.unpad
        self.vision_input_chunk_size = kwargs.pop("vision_input_chunk_size", None)

        self.is_safetensor_save = kwargs.get("is_safetensor_save", True)
        self._backward_compatibility_gradient_checkpointing()
        self.mm_projector.to(config.torch_dtype)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[List[List[torch.FloatTensor]]] = None,
        discrete_pixel_values: Optional[List[List[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        image_sizes: Optional[List[List[List[int]]]] = None,
        mm_query_lengths: Optional[List[List[int]]] = None,
        non_mm_query_lengths: Optional[List[List[int]]] = None,
        img_start_ids_list: Optional[List[List[int]]] = None,
        num_queries_vis_abstractors: Optional[List[List[int]]] = None,
        num_queries_vis_abstractors_slow: Optional[List[List[int]]] = None,
        first_last_frames_slows: Optional[List[List[bool]]] = None,
        is_videos: Optional[List[List[bool]]] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        video_audio_values: Optional[torch.FloatTensor] = None,
        video_audio_masks: Optional[torch.FloatTensor] = None,
        audio_values: Optional[torch.FloatTensor] = None,
        discrete_audio_values: Optional[torch.LongTensor] = None,
        discrete_audio_value_num_per_sample: Optional[torch.LongTensor] = None,
        audio_masks: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        :param input_ids: torch.int64 : torch.size([batchsize, variable)]) : SystemPrompt with Question text token indices for tokenizer.
         In positions where images are inputted, the value is replaced by config.img_start_id, which is a vocabulary index used to indicate the start of image data.
        :param pixel_values: List of List of 4D tensor (torch.float32)
         Each outer list corresponds to a batch and contains inner lists, each holding tensors for images in a sample. The structure accounts for samples with multiple images.
        :param past_key_values: None
        :param inputs_embeds: None
        :param labels: Optional[torch.int64] : [batchsize, variable (input_ids.size(1)+ num visual tokens)] visual token 들은 모두 IGNORE_INDEX
        :param use_cache: None
        :param output_attentions: Optional[bool] : get attention weights of each layers of transformer network (true: 결과값에 포함, false: 결과값에 미포함)
        :param output_hidden_states: Optional[bool] : get hidden states of each layers of transformer network (true: 결과값에 포함, false: 결과값에 미포함)
        :param image_sizes: Stacked as a List of List, representing image sizes (width, height).
         In cases where a sample contains no images, a single dummy image is included.
        :param mm_query_lengths: A List of List that stores the lengths when each image is converted into visual tokens for LLM input.
         In cases where a sample does not contain any images, an empty list is included.
        :param non_mm_query_lengths: contains the lengths of text tokens (excluding visual tokens) for each sample in a batch.
        :img_start_ids_list: contains the indices of the img_start_id tokens for each sample.
        :num_queries_vis_abstractors: A List of List that contains the number of visual tokens for each image grid.
        :num_queries_vis_abstractors_slow: A List of List that contains the number of visual tokens for the slow part when applying the slowfast algorithm to video frames. If the slowfast algorithm is not applied, it will have a value of None.
        :first_last_frames_slows: A List of List that contains the only first and last frames slow mode for each sample in a batch.
        :is_videos: A List of List that contains the boolean value indicating whether each sample in a batch is a video.
        :image_grid_thw: A 3D tensor (torch.int64) for qwen2.5-vl visual encoder.
        :pixel_values_videos: A 2D tensor (torch.float32) for qwen2.5-vl visual encoder.
        :video_grid_thw: A 3D tensor (torch.int64) for qwen2.5-vl visual encoder.
        :audio_values: TBD
        :discrete_audio_values: TBD
        :audio_masks: TBD
        :return:
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.vision_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.vision_config.output_hidden_states
        )

        if inputs_embeds is None and past_key_values is None:
            inputs_embeds, labels = self.extract_inputs_embeds(
                input_ids=input_ids,
                labels=labels,
                pixel_values=pixel_values,
                discrete_pixel_values=discrete_pixel_values,
                past_key_values=past_key_values,
                image_sizes=image_sizes,
                mm_query_lengths=mm_query_lengths,
                non_mm_query_lengths=non_mm_query_lengths,
                img_start_ids_list=img_start_ids_list,
                num_queries_vis_abstractors=num_queries_vis_abstractors,
                num_queries_vis_abstractors_slow=num_queries_vis_abstractors_slow,
                first_last_frames_slows=first_last_frames_slows,
                is_videos=is_videos,
                image_grid_thw=image_grid_thw,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                video_audio_values=video_audio_values,
                video_audio_masks=video_audio_masks,
                audio_values=audio_values,
                discrete_audio_values=discrete_audio_values,
                discrete_audio_value_num_per_sample=discrete_audio_value_num_per_sample,
                audio_masks=audio_masks,
            )

        if inputs_embeds is not None:
            input_ids = None

        outputs = self.language_model.base_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return outputs

    def determine_non_mm_query_lengths(self, input_ids, pad_id, img_start_id):
        """non_mm_query_lengths 를 계산하는 함수
        input_ids 가 collate 될때, 오른쪽에 pad_id 가 채워지기 때문에 이 값을 찾는 방식을 통해 계산됨
        또한 img_start_id 는 visual token 이 들어서는 자리이기 때문에, 해당 indices 은 제거
        """
        non_mm_query_lengths = []
        batch_size, len_seq = input_ids.size(0), input_ids.size(1)

        for i in range(batch_size):
            temp_idx = (input_ids[i] == pad_id).nonzero()
            eos_idx = temp_idx[0, 0].item() if len(temp_idx) > 0 else len_seq
            num_imgs = (input_ids[i] == img_start_id).sum().item()
            non_mm_query_lengths.append(eos_idx - num_imgs)

        if all([pad_id in input_id for input_id in input_ids.tolist()]):
            non_mm_query_lengths = [
                non_mm_query_length + 1 for non_mm_query_length in non_mm_query_lengths
            ]

        return non_mm_query_lengths

    def determine_mm_query_lengths(self, image_features, image_cnts):
        """mm_query_lengths 를 계산하는 함수
        image_features tensor 의 shape 을 통해 계산된다.
        이미지가 1장도 없는 sample 의 경우 dummy image 1장이 들어가기 때문에, 따로 빈 list 처리 또한 추가
        """
        mm_query_lengths = [
            [image_feature.size(0) for image_feature in image_feature_list]
            for image_feature_list in image_features
        ]

        for i, image_cnt in enumerate(image_cnts):
            if image_cnt == 0:
                assert len(mm_query_lengths[i]) == 1
                mm_query_lengths[i] = []

        return mm_query_lengths

    def get_input_embeddings(self):
        if self.without_llm:
            return None
        else:
            return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        if self.without_llm:
            return None
        else:
            return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        if self.without_llm:
            return None
        else:
            return self.language_model.tie_weights()

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None
    ) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(
            new_num_tokens, pad_to_multiple_of
        )
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def extract_inputs_embeds(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[List[List[torch.FloatTensor]]] = None,
        discrete_pixel_values: Optional[List[List[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        image_sizes: Optional[List[List[List[int]]]] = None,
        mm_query_lengths: Optional[List[List[int]]] = None,
        non_mm_query_lengths: Optional[List[int]] = None,
        img_start_ids_list: Optional[List[List[int]]] = None,
        num_queries_vis_abstractors: Optional[List[List[int]]] = None,
        num_queries_vis_abstractors_slow: Optional[List[List[int]]] = None,
        first_last_frames_slows: Optional[List[List[bool]]] = None,
        is_videos: Optional[List[List[bool]]] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        video_audio_values: Optional[torch.FloatTensor] = None,
        video_audio_masks: Optional[torch.FloatTensor] = None,
        audio_values: Optional[torch.FloatTensor] = None,
        discrete_audio_values: Optional[torch.FloatTensor] = None,
        discrete_audio_value_num_per_sample: Optional[torch.LongTensor] = None,
        audio_masks: Optional[torch.FloatTensor] = None,
    ):
        """
        :param input_ids: torch.int64 : torch.size([batchsize, variable)]) : SystemPrompt with Question text token indices for tokenizer.
         In positions where images are inputted, the value is replaced by config.img_start_id, which is a vocabulary index used to indicate the start of image data.
         In cases where a sample contains no images, a single dummy image is included.
        :param pixel_values: List of List of 4D tensor (torch.float32)
         Each outer list corresponds to a batch and contains inner lists, each holding tensors for images in a sample. The structure accounts for samples with multiple images.
        :param past_key_values: None : (batch_size, num_heads, sequence_length - 1, embed_size_per_head): Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up
        :param image_sizes: Stacked as a List of List, representing image sizes (width, height).
         In cases where a sample contains no images, a single dummy image is included.
        :param mm_query_lengths: A List of List that stores the lengths when each image is converted into visual tokens for LLM input.
         In cases where a sample does not contain any images, an empty list is included.
        :param non_mm_query_lengths: contains the lengths of text tokens (excluding visual tokens) for each sample in a batch.
        :img_start_ids_list: contains the indices of the img_start_id tokens for each sample.
        :num_queries_vis_abstractors: A List of List that contains the number of visual tokens for each image grid.
        :num_queries_vis_abstractors_slow: A List of List that contains the number of visual tokens for the slow part when applying the slowfast algorithm to video frames. If the slowfast algorithm is not applied, it will have a value of None.
        :first_last_frames_slows: A List of bool that contains the information of whether the slowfast algorithm is applied to the first or last frames of the video.
        :is_videos: A List of List that contains the boolean value indicating whether each sample in a batch is a video.
        :image_grid_thw: A 3D tensor (torch.int64) for qwen2.5-vl visual encoder.
        :pixel_values_videos: A 2D tensor (torch.float32) for qwen2.5-vl visual encoder.
        :video_grid_thw: A 3D tensor (torch.int64) for qwen2.5-vl visual encoder.
        :audio_values: TBD
        :discrete_audio_values: TBD
        :audio_masks: TBD
        :return:
        """

        inputs_embeds = None
        if past_key_values:
            pass
        else:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            context_vision_model = (
                torch.no_grad()
                if self.config.freeze_encoder
                else contextlib.nullcontext()
            )

            if self.config.freeze_encoder:
                self.vision_model.eval()
                self.vision_model.gradient_checkpointing = False

            if (
                pixel_values is not None
                and len([i for i in pixel_values if i is not None]) > 0
            ):
                with context_vision_model:
                    image_features = self.vision_model(
                        pixel_values, grid_thw=image_grid_thw
                    )
                image_features = self.mm_projector(image_features)

                if img_start_ids_list is None:
                    image_cnts = (
                        (input_ids == self.config.img_start_id).sum(dim=1).tolist()
                    )
                else:
                    image_cnts = [
                        len(img_start_ids) for img_start_ids in img_start_ids_list
                    ]

                mask = input_ids.eq(self.config.img_start_id)
                positions = mask.nonzero(as_tuple=False)

                batch_idx = positions[:, 0]
                seq_idx = positions[:, 1]

                if sum(image_cnts) == 0:
                    image_features = image_features[0:0]
                inputs_embeds[batch_idx, seq_idx, :] = image_features

            if (
                pixel_values_videos is not None
                and len([i for i in pixel_values_videos if i is not None]) > 0
            ):
                with context_vision_model:
                    video_features = self.vision_model(
                        pixel_values_videos, grid_thw=video_grid_thw
                    )
                video_features = self.mm_projector(video_features)

                video_cnts = (
                    (input_ids == self.config.video_start_id).sum(dim=1).tolist()
                )
                mask = input_ids.eq(self.config.video_start_id)
                positions = mask.nonzero(as_tuple=False)

                batch_idx = positions[:, 0]
                seq_idx = positions[:, 1]

                if sum(video_cnts) == 0:
                    inputs_embeds[0, 0:0] = video_features[0:0]
                else:
                    inputs_embeds[batch_idx, seq_idx, :] = video_features

                del video_features, mask, positions, batch_idx, seq_idx

            if (
                discrete_pixel_values is not None
                and len([i for i in discrete_pixel_values if i is not None]) > 0
            ):
                assert (
                    self.config.discrete_image_start_id is not None
                ), "discrete_image_start_id 가 정의되어 있지 않습니다"
                discrete_image_cnt = (
                    (input_ids == self.config.discrete_image_start_id)
                    .sum(dim=1)
                    .tolist()
                )

                with torch.no_grad():
                    discrete_unit_lists = self.discrete_vision_model(
                        discrete_pixel_values
                    )["encoded"]
                discrete_unit_lists = discrete_unit_lists.detach().cpu()

                import os

                rank = int(os.environ.get("RANK", -1))

                if (discrete_unit_lists < 0).any() or (
                    discrete_unit_lists > 65535
                ).any():
                    min_val = discrete_unit_lists.min().item()
                    max_val = discrete_unit_lists.max().item()
                    print(f"[RANK {rank}] ❌ discrete_unit_lists has invalid values!")
                    print(f"  min={min_val}, max={max_val}")
                    print(
                        f"  Expected range: [0, 65535] for TA-Tok codebook (2^16 = 65536 codes)"
                    )
                    print(f"  Will map to tokens: <|vision00000|> ~ <|vision65535|>")
                    print(f"  Clamping to valid range...")
                    discrete_unit_lists = torch.clamp(discrete_unit_lists, 0, 65535)

                discrete_image_token_ids = (
                    discrete_unit_lists + self.config.discrete_image_unit_0_id
                )
                discrete_image_token_ids = discrete_image_token_ids.to(
                    device=input_ids.device
                ).to(dtype=input_ids.dtype)

                vocab_size = self.get_input_embeddings().num_embeddings
                if (discrete_image_token_ids >= vocab_size).any() or (
                    discrete_image_token_ids < 0
                ).any():
                    max_id = discrete_image_token_ids.max().item()
                    min_id = discrete_image_token_ids.min().item()
                    print(
                        f"[RANK {rank}] ⚠️ discrete_image_token_ids out of vocab range!"
                    )
                    print(f"  min={min_id}, max={max_id}, vocab_size={vocab_size}")
                    print(
                        f"  discrete_image_unit_0_id={self.config.discrete_image_unit_0_id}"
                    )
                    print(
                        f"  Expected token range: [{self.config.discrete_image_unit_0_id}, {self.config.discrete_image_unit_0_id + 65535}]"
                    )
                    print(f"  Clipping to valid vocab range [0, {vocab_size-1}]...")
                    discrete_image_token_ids = torch.clamp(
                        discrete_image_token_ids, 0, vocab_size - 1
                    )

                try:
                    discrete_image_embeddings = self.get_input_embeddings()(
                        discrete_image_token_ids
                    )
                except RuntimeError as e:
                    print(
                        f"[RANK {rank}] 🚨 FATAL: discrete_image embedding lookup failed!"
                    )
                    print(f"  Error: {e}")
                    print(
                        f"  discrete_image_token_ids shape: {discrete_image_token_ids.shape}"
                    )
                    print(
                        f"  discrete_image_token_ids min/max: {discrete_image_token_ids.min().item()}/{discrete_image_token_ids.max().item()}"
                    )
                    print(f"  vocab_size: {vocab_size}")
                    raise

                if sum(discrete_image_cnt) == 0:
                    inputs_embeds[0, 0:0] = discrete_image_embeddings[0, 0:0]
                else:
                    mask = input_ids.eq(self.config.discrete_image_start_id)
                    positions = mask.nonzero(as_tuple=False)
                    batch_idx = positions[:, 0]
                    seq_idx = positions[:, 1]
                    inputs_embeds[batch_idx, seq_idx, :] = (
                        discrete_image_embeddings.view(
                            -1, discrete_image_embeddings.shape[-1]
                        )
                    )

                    if labels is not None:
                        label_mask = labels.eq(self.config.discrete_image_start_id)
                        label_positions = label_mask.nonzero(as_tuple=False)

                        if label_positions.numel() > 0:

                            input_mask_flat = mask.view(-1)
                            label_mask_flat = label_mask.view(-1)

                            trainable_mask_in_input_positions = label_mask_flat[
                                input_mask_flat
                            ]

                            discrete_image_token_ids_flat = (
                                discrete_image_token_ids.view(-1)
                            )
                            trainable_token_ids = discrete_image_token_ids_flat[
                                trainable_mask_in_input_positions
                            ]

                            label_batch_idx = label_positions[:, 0]
                            label_seq_idx = label_positions[:, 1]
                            labels[label_batch_idx, label_seq_idx] = trainable_token_ids

                del (
                    discrete_unit_lists,
                    discrete_image_token_ids,
                    discrete_image_embeddings,
                )

            video_audio_features = None
            if (
                video_audio_values is not None
                and video_audio_masks is not None
                and self.audio_model is not None
                and self.audio_projector is not None
            ):
                is_dummy_data = isinstance(video_audio_values, torch.Tensor)

                if is_dummy_data:
                    dummy_chunk = video_audio_values[0]
                    dummy_mask = video_audio_masks[0]
                    video_audio_values = [[[dummy_chunk]]]
                    video_audio_masks = [[[dummy_mask]]]

                all_audio_chunks = []
                all_audio_masks = []
                chunks_per_video = []

                for sample_video_audios, sample_video_masks in zip(
                    video_audio_values, video_audio_masks
                ):
                    for video_audio_chunks, video_audio_chunk_masks in zip(
                        sample_video_audios, sample_video_masks
                    ):
                        if video_audio_chunks and len(video_audio_chunks) > 0:
                            chunks_per_video.append(len(video_audio_chunks))
                            all_audio_chunks.extend(video_audio_chunks)
                            all_audio_masks.extend(video_audio_chunk_masks)

                if len(all_audio_chunks) > 0:
                    audio_values_batch = torch.stack(all_audio_chunks, dim=0).to(
                        inputs_embeds.device
                    )
                    audio_masks_raw = torch.stack(all_audio_masks, dim=0).to(
                        inputs_embeds.device
                    )

                    num_chunks = audio_values_batch.shape[0]
                    max_mel_seq_len = 3000

                    audio_feat_lengths, audio_output_lengths = (
                        self.audio_model._get_feat_extract_output_lengths(
                            audio_masks_raw.sum(-1)
                        )
                    )
                    if isinstance(audio_feat_lengths, list):
                        audio_feat_lengths = torch.tensor(
                            audio_feat_lengths, device=inputs_embeds.device
                        )
                    if isinstance(audio_output_lengths, list):
                        audio_output_lengths = torch.tensor(
                            audio_output_lengths, device=inputs_embeds.device
                        )

                    max_seq_len = (max_mel_seq_len - 2) // 2 + 1

                    seq_range = (
                        torch.arange(
                            0,
                            max_seq_len,
                            dtype=audio_feat_lengths.dtype,
                            device=audio_feat_lengths.device,
                        )
                        .unsqueeze(0)
                        .expand(num_chunks, max_seq_len)
                    )
                    lengths_expand = audio_feat_lengths.unsqueeze(1).expand(
                        num_chunks, max_seq_len
                    )
                    padding_mask = seq_range >= lengths_expand
                    audio_mask_ = padding_mask.view(
                        num_chunks, 1, 1, max_seq_len
                    ).expand(num_chunks, 1, max_seq_len, max_seq_len)
                    video_audio_attn_masks = audio_mask_.to(
                        dtype=self.audio_model.conv1.weight.dtype,
                        device=self.audio_model.conv1.weight.device,
                    )
                    video_audio_attn_masks[audio_mask_] = float("-inf")

                    with torch.no_grad():
                        audio_chunk_features = self.audio_model(
                            audio_values_batch, attention_mask=video_audio_attn_masks
                        )

                    if getattr(self.config, "freeze_audio_projector", False):
                        with torch.no_grad():
                            audio_chunk_features = self.audio_projector(
                                audio_chunk_features.last_hidden_state
                            )
                    else:
                        audio_chunk_features = self.audio_projector(
                            audio_chunk_features.last_hidden_state
                        )

                    actual_seq_len = audio_chunk_features.shape[1]
                    audio_features_mask = torch.arange(
                        actual_seq_len, device=audio_output_lengths.device
                    )[None, :]
                    audio_features_mask = (
                        audio_features_mask < audio_output_lengths[:, None]
                    )

                    if self.video_audio_compressor is not None:

                        video_features_list = []
                        chunk_offset = 0

                        for num_chunks_in_video in chunks_per_video:
                            video_valid_features_list = []
                            for i in range(num_chunks_in_video):
                                chunk_idx = chunk_offset + i
                                chunk_mask = audio_features_mask[chunk_idx]
                                valid_features = audio_chunk_features[chunk_idx][
                                    chunk_mask
                                ]
                                video_valid_features_list.append(valid_features)

                            video_all_features = torch.cat(
                                video_valid_features_list, dim=0
                            )
                            video_features_list.append(video_all_features)
                            chunk_offset += num_chunks_in_video
                            del video_valid_features_list

                        mambamia_chunk_size = self.video_audio_compressor.chunk_size

                        seq_lens = [vf.shape[0] for vf in video_features_list]
                        num_queries_per_video = [
                            (sl + mambamia_chunk_size - 1) // mambamia_chunk_size
                            for sl in seq_lens
                        ]

                        max_seq_len = max(seq_lens)
                        hidden_dim = video_features_list[0].shape[1]
                        num_videos = len(video_features_list)

                        batched_features = torch.zeros(
                            num_videos,
                            max_seq_len,
                            hidden_dim,
                            dtype=video_features_list[0].dtype,
                            device=video_features_list[0].device,
                        )
                        for vid_idx, vf in enumerate(video_features_list):
                            batched_features[vid_idx, : vf.shape[0], :] = vf

                        del video_features_list

                        compressed_batch = self.video_audio_compressor(batched_features)
                        del batched_features

                        compressed_features_list = []
                        for vid_idx, nq in enumerate(num_queries_per_video):
                            valid_compressed = compressed_batch[vid_idx, :nq, :]
                            compressed_features_list.append(valid_compressed)

                        video_audio_features = torch.cat(
                            compressed_features_list, dim=0
                        )
                        del compressed_batch, compressed_features_list
                    else:
                        pooled_features_list = []
                        for chunk_idx in range(num_chunks):
                            chunk_mask = audio_features_mask[chunk_idx]
                            valid_features = audio_chunk_features[chunk_idx][chunk_mask]

                            valid_len = valid_features.shape[0]
                            pool_size = 25
                            num_pooled = (valid_len + pool_size - 1) // pool_size

                            for pool_idx in range(num_pooled):
                                start_idx = pool_idx * pool_size
                                end_idx = min(start_idx + pool_size, valid_len)
                                pooled_feat = valid_features[start_idx:end_idx].mean(
                                    dim=0
                                )
                                pooled_features_list.append(pooled_feat)

                        video_audio_features = torch.stack(pooled_features_list, dim=0)
                        del pooled_features_list

                    del (
                        audio_values_batch,
                        audio_masks_raw,
                        video_audio_attn_masks,
                        audio_chunk_features,
                    )
                    del (
                        seq_range,
                        lengths_expand,
                        padding_mask,
                        audio_mask_,
                        audio_features_mask,
                    )

                if video_audio_features is not None:
                    video_audio_token_id = self.config.video_audio_start_id
                    video_audio_cnts = (
                        (input_ids == video_audio_token_id).sum(dim=1).tolist()
                    )

                    if sum(video_audio_cnts) == 0:
                        inputs_embeds[0, 0:0] = video_audio_features[0:0]
                    else:
                        mask = input_ids.eq(video_audio_token_id)
                        positions = mask.nonzero(as_tuple=False)

                        batch_idx = positions[:, 0]
                        seq_idx = positions[:, 1]

                        num_placeholder_tokens = len(batch_idx)
                        num_actual_features = video_audio_features.shape[0]

                        if num_placeholder_tokens != num_actual_features:
                            rank = int(os.environ.get("RANK", -1))
                            print(
                                f"\n[RANK {rank}] ⚠️  VIDEO AUDIO SHAPE MISMATCH DETECTED!"
                            )
                            print(
                                f"  Placeholder tokens in input_ids: {num_placeholder_tokens}"
                            )
                            print(
                                f"  Actual audio features generated: {num_actual_features}"
                            )
                            print(
                                f"  Difference: {num_placeholder_tokens - num_actual_features}"
                            )
                            print(f"  video_audio_cnts per sample: {video_audio_cnts}")

                            if video_audio_values is not None and not isinstance(
                                video_audio_values, torch.Tensor
                            ):
                                print(f"  video_audio_values structure:")
                                for sample_idx, sample_audios in enumerate(
                                    video_audio_values
                                ):
                                    print(
                                        f"    Sample {sample_idx}: {len(sample_audios)} videos"
                                    )
                                    for vid_idx, vid_audios in enumerate(sample_audios):
                                        if vid_audios:
                                            print(
                                                f"      Video {vid_idx}: {len(vid_audios)} chunks, first chunk shape: {vid_audios[0].shape if len(vid_audios) > 0 else 'N/A'}"
                                            )
                                        else:
                                            print(
                                                f"      Video {vid_idx}: 0 chunks (no audio)"
                                            )
                            elif isinstance(video_audio_values, torch.Tensor):
                                print(
                                    f"  video_audio_values is dummy tensor (shape: {video_audio_values.shape})"
                                )

                            if (
                                video_audio_masks is not None
                                and audio_output_lengths is not None
                            ):
                                print(
                                    f"  video_audio_masks analysis (using actual audio_output_lengths):"
                                )
                                chunk_idx_global = 0
                                for sample_idx, sample_masks in enumerate(
                                    video_audio_masks
                                ):
                                    for vid_idx, vid_masks in enumerate(sample_masks):
                                        for chunk_idx_local, mask_tensor in enumerate(
                                            vid_masks
                                        ):
                                            mask_sum = int(mask_tensor.sum())
                                            actual_output_len = int(
                                                audio_output_lengths[chunk_idx_global]
                                            )
                                            pool_size = 25
                                            num_pooled = (
                                                actual_output_len + pool_size - 1
                                            ) // pool_size
                                            print(
                                                f"    Sample {sample_idx}, Video {vid_idx}, Chunk {chunk_idx_local}: mask_sum={mask_sum}, actual_output_len={actual_output_len}, pooled={num_pooled}"
                                            )
                                            chunk_idx_global += 1
                                print(f"  Total chunks processed: {chunk_idx_global}")

                            raise RuntimeError(
                                f"Video audio shape mismatch: {num_placeholder_tokens} placeholder tokens "
                                f"vs {num_actual_features} generated features. See debug info above."
                            )

                        inputs_embeds[batch_idx, seq_idx, :] = video_audio_features

                        del mask, positions, batch_idx, seq_idx

                    del video_audio_features

            if (
                audio_values is not None
                and len([i for i in audio_values if i is not None]) > 0
            ):
                assert audio_masks is not None
                batch_size, _, max_mel_seq_len = audio_values.size()
                assert (
                    max_mel_seq_len == 3000
                ), f"max_mel_seq_len should be 3000, but got {max_mel_seq_len}"
                audio_feat_lengths, audio_output_lengths = (
                    self.audio_model._get_feat_extract_output_lengths(
                        audio_masks.sum(-1)
                    )
                )
                max_seq_len = (max_mel_seq_len - 2) // 2 + 1
                seq_range = (
                    torch.arange(
                        0,
                        max_seq_len,
                        dtype=audio_feat_lengths.dtype,
                        device=audio_feat_lengths.device,
                    )
                    .unsqueeze(0)
                    .expand(batch_size, max_seq_len)
                )
                lengths_expand = audio_feat_lengths.unsqueeze(1).expand(
                    batch_size, max_seq_len
                )
                padding_mask = seq_range >= lengths_expand
                audio_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
                    batch_size, 1, max_seq_len, max_seq_len
                )
                audio_masks = audio_mask_.to(
                    dtype=self.audio_model.conv1.weight.dtype,
                    device=self.audio_model.conv1.weight.device,
                )
                audio_masks[audio_mask_] = float("-inf")

                with torch.no_grad():
                    audio_features = self.audio_model(
                        audio_values, attention_mask=audio_masks
                    )

                if getattr(self.config, "freeze_audio_projector", False):
                    with torch.no_grad():
                        audio_features = self.audio_projector(
                            audio_features.last_hidden_state
                        )
                else:
                    audio_features = self.audio_projector(
                        audio_features.last_hidden_state
                    )
                assert (
                    self.config.audio_start_id is not None
                ), "audio_start_id 가 정의되어 있지 않습니다"
                audio_cnts = (
                    (input_ids == self.config.audio_start_id).sum(dim=1).tolist()
                )
                mask = input_ids.eq(self.config.audio_start_id)
                positions = mask.nonzero(as_tuple=False)

                batch_idx = positions[:, 0]
                seq_idx = positions[:, 1]

                if sum(audio_cnts) == 0:
                    inputs_embeds[0, 0:0] = audio_features[0, 0:0]
                else:
                    num_audios, max_audio_tokens, embed_dim = audio_features.shape
                    audio_features_mask = torch.arange(
                        max_audio_tokens, device=audio_output_lengths.device
                    )[None, :]
                    audio_features_mask = (
                        audio_features_mask < audio_output_lengths[:, None]
                    )
                    audio_features = audio_features[audio_features_mask]
                    inputs_embeds[batch_idx, seq_idx, :] = audio_features
                    del audio_features_mask

                del seq_range, lengths_expand, padding_mask, audio_mask_, audio_features
                del (
                    mask,
                    positions,
                    batch_idx,
                    seq_idx,
                    audio_feat_lengths,
                    audio_output_lengths,
                )

            if (
                discrete_audio_values is not None
                and len([i for i in discrete_audio_values if i is not None]) > 0
            ):
                if (
                    not hasattr(self, "discrete_audio_model")
                    or self.discrete_audio_model is None
                ):
                    raise ValueError(
                        "[BUG] discrete_audio_model이 초기화되지 않았지만 discrete_audio_values가 제공되었습니다"
                    )
                assert (
                    self.config.discrete_audio_start_id is not None
                ), "discrete_audio_start_id 가 정의되어 있지 않습니다"
                discrete_audio_cnts = (
                    (input_ids == self.config.discrete_audio_start_id)
                    .sum(dim=1)
                    .tolist()
                )

                if sum(discrete_audio_cnts) == 0:
                    with torch.no_grad():
                        discrete_unit_list = self.discrete_audio_model.forward(
                            discrete_audio_values[0]
                        )
                    discrete_unit_list = discrete_unit_list.squeeze().detach().cpu()
                    discrete_audio_token_ids = (
                        discrete_unit_list + self.config.discrete_audio_unit_0_id
                    )
                    discrete_audio_token_ids = discrete_audio_token_ids.to(
                        device=input_ids.device
                    ).to(dtype=input_ids.dtype)
                    discrete_audio_embeddings = self.get_input_embeddings()(
                        discrete_audio_token_ids
                    )
                    inputs_embeds[0, 0:0] = discrete_audio_embeddings[0:0]
                    del (
                        discrete_unit_list,
                        discrete_audio_token_ids,
                        discrete_audio_embeddings,
                    )
                else:
                    discrete_audio_value_counter = 0
                    for b_idx in range(discrete_audio_value_num_per_sample.shape[0]):
                        if discrete_audio_value_num_per_sample[b_idx] > 0:
                            discrete_unit_list_parts = []
                            for _ in range(discrete_audio_value_num_per_sample[b_idx]):
                                wav = discrete_audio_values[
                                    discrete_audio_value_counter
                                ]
                                wav = wav[wav != -200.0]

                                if wav.shape[0] == 0:
                                    print(
                                        f"[RANK {rank}] ❌ wav length is 0 after padding removal! This should be caught by preprocessor."
                                    )
                                    wav = torch.zeros(
                                        400, device=wav.device, dtype=wav.dtype
                                    )

                                if wav.shape[0] < 400:
                                    print(
                                        f"[RANK {rank}] ⚠️  wav too short ({wav.shape[0]} < 400)! Padding to 400. Preprocessor should catch this."
                                    )
                                    wav = torch.nn.functional.pad(
                                        wav, (0, 400 - wav.shape[0])
                                    )

                                if torch.isnan(wav).any() or torch.isinf(wav).any():
                                    print(
                                        f"[RANK {rank}] ⚠️  wav contains NaN/Inf! Clamping. Preprocessor should catch this."
                                    )
                                    wav = torch.nan_to_num(
                                        wav, nan=0.0, posinf=1.0, neginf=-1.0
                                    )

                                with torch.no_grad():
                                    if wav.shape[0] > 80 * DEFAULT_SAMPLE_RATE:
                                        chunk_size = 80 * DEFAULT_SAMPLE_RATE
                                        min_chunk_size = (
                                            MIN_DISCRETE_AUDIO_CHUNK_SAMPLES
                                        )

                                        discrete_unit_list = None

                                        for start in range(0, wav.shape[0], chunk_size):
                                            end = start + chunk_size

                                            if (
                                                end < wav.shape[0]
                                                and wav.shape[0] - end < min_chunk_size
                                            ):
                                                end = wav.shape[0]

                                            chunk_wav = wav[start:end]

                                            if (
                                                chunk_wav.shape[0]
                                                < MIN_DISCRETE_AUDIO_CHUNK_SAMPLES
                                            ):
                                                raise RuntimeError(
                                                    f"[RANK {rank}] 🚨 CRITICAL BUG: chunk_wav too short "
                                                    f"({chunk_wav.shape[0]} < {MIN_DISCRETE_AUDIO_CHUNK_SAMPLES})!\n"
                                                    f"  This should NEVER happen - preprocessor asserts MIN_DISCRETE_AUDIO_CHUNK_SAMPLES.\n"
                                                    f"  start={start}, end={end}, wav.shape[0]={wav.shape[0]}\n"
                                                    f"  chunk_size={chunk_size}, min_chunk_size={min_chunk_size}\n"
                                                    f"  Check for: 1) collate_fn padding removal bug, 2) chunking logic mismatch, "
                                                    f"3) data corruption during transfer.\n"
                                                    f"  Cannot continue - skipping would cause shape mismatch downstream."
                                                )

                                            try:
                                                chunk_result = (
                                                    self.discrete_audio_model.forward(
                                                        chunk_wav
                                                    )
                                                )
                                                chunk_result = (
                                                    chunk_result.squeeze()
                                                    .detach()
                                                    .cpu()
                                                )
                                            except RuntimeError as e:
                                                print(
                                                    f"[RANK {rank}] ❌ discrete_audio_model.forward() FAILED!"
                                                )
                                                print(f"  RuntimeError: {e}")
                                                print(
                                                    f"  chunk_wav.shape: {chunk_wav.shape}"
                                                )
                                                print(
                                                    f"  chunk_wav.dtype: {chunk_wav.dtype}"
                                                )
                                                print(
                                                    f"  chunk_wav.device: {chunk_wav.device}"
                                                )
                                                print(
                                                    f"  chunk_wav range: [{chunk_wav.min():.4f}, {chunk_wav.max():.4f}]"
                                                )
                                                print(
                                                    f"  chunk_wav has NaN: {torch.isnan(chunk_wav).any().item()}"
                                                )
                                                print(
                                                    f"  chunk_wav has Inf: {torch.isinf(chunk_wav).any().item()}"
                                                )
                                                print(
                                                    f"  start={start}, end={end}, original_wav_len={wav.shape[0]}"
                                                )
                                                raise

                                            if chunk_result.dim() == 0:
                                                print(
                                                    f"[RANK {rank}] ⚠️  chunk_result is 0-dim scalar! Skipping this chunk."
                                                )
                                                print(
                                                    f"  chunk_wav.shape: {chunk_wav.shape}, start={start}, end={end}"
                                                )
                                                del chunk_result
                                                continue

                                            if chunk_result.numel() == 0:
                                                print(
                                                    f"[RANK {rank}] ⚠️  chunk_result is empty! Skipping this chunk."
                                                )
                                                print(
                                                    f"  chunk_wav.shape: {chunk_wav.shape}, start={start}, end={end}"
                                                )
                                                del chunk_result
                                                continue

                                            if discrete_unit_list is None:
                                                discrete_unit_list = chunk_result
                                            else:
                                                discrete_unit_list = torch.cat(
                                                    [discrete_unit_list, chunk_result],
                                                    dim=-1,
                                                )
                                            del chunk_result

                                            if end >= wav.shape[0]:
                                                break

                                        if discrete_unit_list is None:
                                            print(
                                                f"[RANK {rank}] ⚠️  All chunks failed for this audio! Using dummy token."
                                            )
                                            discrete_unit_list = torch.zeros(
                                                1, dtype=torch.int32
                                            )
                                    else:
                                        try:
                                            discrete_unit_list = (
                                                self.discrete_audio_model.forward(wav)
                                            )
                                            discrete_unit_list = (
                                                discrete_unit_list.squeeze()
                                                .detach()
                                                .cpu()
                                            )
                                        except RuntimeError as e:
                                            print(
                                                f"[RANK {rank}] ❌ discrete_audio_model.forward() FAILED!"
                                            )
                                            print(f"  RuntimeError: {e}")
                                            print(f"  wav.shape: {wav.shape}")
                                            print(f"  wav.dtype: {wav.dtype}")
                                            print(f"  wav.device: {wav.device}")
                                            print(
                                                f"  wav range: [{wav.min():.4f}, {wav.max():.4f}]"
                                            )
                                            print(
                                                f"  wav has NaN: {torch.isnan(wav).any().item()}"
                                            )
                                            print(
                                                f"  wav has Inf: {torch.isinf(wav).any().item()}"
                                            )
                                            raise

                                        if discrete_unit_list.dim() == 0:
                                            print(
                                                f"[RANK {rank}] ⚠️  discrete_unit_list is 0-dim scalar! Using dummy value."
                                            )
                                            print(f"  wav.shape: {wav.shape}")
                                            discrete_unit_list = (
                                                discrete_unit_list.unsqueeze(0)
                                            )

                                        if discrete_unit_list.numel() == 0:
                                            print(
                                                f"[RANK {rank}] ⚠️  discrete_unit_list is empty! Using dummy token."
                                            )
                                            print(f"  wav.shape: {wav.shape}")
                                            discrete_unit_list = torch.zeros(
                                                1, dtype=torch.int32
                                            )

                                    discrete_unit_list_parts.append(discrete_unit_list)
                                discrete_audio_value_counter += 1

                            if len(discrete_unit_list_parts) == 0:
                                print(
                                    f"[RANK {rank}] ⚠️  All discrete audio chunks were invalid/empty for batch {b_idx}!"
                                )
                                print(
                                    f"  discrete_audio_value_num_per_sample[{b_idx}] = {discrete_audio_value_num_per_sample[b_idx]}"
                                )
                                print(
                                    f"  Creating dummy discrete_unit_lists to match placeholder count."
                                )

                                discrete_audio_start_positions = (
                                    input_ids[b_idx]
                                    == self.config.discrete_audio_start_id
                                ).nonzero(as_tuple=True)[0]
                                num_placeholders = len(discrete_audio_start_positions)

                                discrete_unit_lists = torch.zeros(
                                    num_placeholders, dtype=torch.int32
                                )
                                print(
                                    f"[RANK {rank}]   Created {num_placeholders} dummy tokens (all zeros)."
                                )
                            else:
                                for idx, part in enumerate(discrete_unit_list_parts):
                                    if part.numel() == 0:
                                        print(
                                            f"[RANK {rank}] ⚠️  discrete_unit_list_parts[{idx}] is empty tensor!"
                                        )
                                    if (
                                        torch.isnan(part.float()).any()
                                        or torch.isinf(part.float()).any()
                                    ):
                                        print(
                                            f"[RANK {rank}] ⚠️  discrete_unit_list_parts[{idx}] contains NaN/Inf!"
                                        )
                                        part = torch.nan_to_num(
                                            part.float(),
                                            nan=0.0,
                                            posinf=6560.0,
                                            neginf=0.0,
                                        ).to(dtype=part.dtype)

                                discrete_unit_lists = torch.cat(
                                    discrete_unit_list_parts
                                )

                            del discrete_unit_list_parts

                            if (discrete_unit_lists < 0).any() or (
                                discrete_unit_lists > 6561
                            ).any():
                                print(
                                    f"[RANK {rank}] ❌ discrete_unit_lists has invalid values!"
                                )
                                print(
                                    f"  min={discrete_unit_lists.min().item()}, max={discrete_unit_lists.max().item()}"
                                )
                                print(
                                    f"  Expected range: [0, 6561] (tokenizer has audio0000~audio6561, 6562 tokens)"
                                )
                                print(
                                    f"  FSQ mathematically max is 6560, but allowing 6561 for tokenizer compatibility"
                                )
                                print(f"  Clamping to valid range...")
                                discrete_unit_lists = torch.clamp(
                                    discrete_unit_lists, 0, 6561
                                )

                            discrete_audio_token_ids = (
                                discrete_unit_lists
                                + self.config.discrete_audio_unit_0_id
                            )
                            discrete_audio_token_ids = discrete_audio_token_ids.to(
                                device=input_ids.device
                            ).to(dtype=input_ids.dtype)

                            discrete_audio_start_positions = (
                                input_ids[b_idx] == self.config.discrete_audio_start_id
                            ).nonzero(as_tuple=True)[0]

                            assert (
                                len(discrete_audio_start_positions) > 0
                            ), "discrete_audio_start_id가 input_ids에 존재하지 않습니다"

                            discrete_audio_embeddings = self.get_input_embeddings()(
                                discrete_audio_token_ids
                            )

                            if len(discrete_audio_start_positions) > 0:
                                inputs_embeds[
                                    b_idx, discrete_audio_start_positions, :
                                ] = discrete_audio_embeddings

                            if (
                                labels is not None
                                and len(discrete_audio_start_positions) > 0
                            ):
                                label_mask_audio = labels[b_idx].eq(
                                    self.config.discrete_audio_start_id
                                )

                                if label_mask_audio.any():
                                    input_mask_audio = (
                                        input_ids[b_idx]
                                        == self.config.discrete_audio_start_id
                                    )

                                    trainable_mask_in_input_positions_audio = (
                                        label_mask_audio[input_mask_audio]
                                    )

                                    trainable_audio_token_ids = (
                                        discrete_audio_token_ids[
                                            trainable_mask_in_input_positions_audio
                                        ]
                                    )

                                    label_positions_audio = label_mask_audio.nonzero(
                                        as_tuple=True
                                    )[0]
                                    labels[b_idx, label_positions_audio] = (
                                        trainable_audio_token_ids
                                    )

        return inputs_embeds, labels

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        **kwargs,
    ):
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            **kwargs,
        )

        model.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=True
        )
        return model

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        *args,
        **kwargs,
    ):
        super().register_for_auto_class("AutoModel")
        self.config.register_for_auto_class()
        super().save_pretrained(save_directory, *args, **kwargs)

    @torch.no_grad()
    def inference(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[
            Union[List[List[torch.FloatTensor]], torch.FloatTensor]
        ] = None,
        image_sizes: Optional[List[List[List[int]]]] = None,
        mm_query_lengths: Optional[List[List[int]]] = None,
        non_mm_query_lengths: Optional[List[int]] = None,
        num_queries_vis_abstractors: Optional[List[List[int]]] = None,
        num_queries_vis_abstractors_slow: Optional[List[List[int]]] = None,
        first_last_frames_slows: Optional[List[List[bool]]] = None,
        is_videos: Optional[List[List[bool]]] = None,
        img_start_ids_list: Optional[List[List[int]]] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        audio_values: Optional[torch.FloatTensor] = None,
        discrete_audio_values: Optional[torch.FloatTensor] = None,
        discrete_audio_value_num_per_sample: Optional[torch.LongTensor] = None,
        audio_masks: Optional[torch.LongTensor] = None,
        max_length: int = 196,
        min_length: int = 2,
        do_sample: bool = True,
        num_beams: int = 1,
        top_p: float = 0.6,
        top_k: int = 0,
        temperature: float = 0.5,
        repetition_penalty: float = 1.0,
        length_penalty: int = 1,
        early_stopping: Union[bool, str] = False,
        use_cache: bool = True,
        verbose: bool = False,
        **kwargs,
    ):
        """
        :param input_ids: torch.int64 : torch.size([batchsize, variable)]) : SystemPrompt with Question text token indices for tokenizer.
         In positions where images are inputted, the value is replaced by config.img_start_id, which is a vocabulary index used to indicate the start of image data.
         In cases where a sample contains no images, a single dummy image is included.
        :param pixel_values: List of List of 4D tensor (torch.float32)
         Each outer list corresponds to a batch and contains inner lists, each holding tensors for images in a sample. The structure accounts for samples with multiple images.
        :param attention_mask: not used
        :param max_length: int :  The maximum length the generated tokens can have. Corresponds to the length of the input prompt + max_new_tokens.
        :param min_length: int : The minimum length of the sequence to be generated. Corresponds to the length of the input prompt + min_new_tokens.
        :param num_beams: int :  Number of beams for beam search. 1 means no beam search.
        :param top_k: int : The number of highest probability vocabulary tokens to keep for top-k-filtering.
        :param temperature: float :  The value used to modulate the next token probabilities. ( scores / self.temperature )
        :param repetition_penalty: float : The parameter for repetition penalty.
        :param length_penalty: int :  It is applied as an exponent to the sequence length, which in turn is used to divide the score of the sequence.
        :param early_stopping: Union[bool, str] : True, where the generation stops as soon as there are num_beams complete candidates;
            False, where an heuristic is applied and the generation stops when is it very unlikely to find better candidates;
            "never", where the beam search procedure only stops when there cannot be better candidates (canonical beam search algorithm)
        :param use_cache: bool :  Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
        :param verbose: bool : print debug mention
        :param image_sizes: Stacked as a List of List, representing image sizes (width, height).
         In cases where a sample contains no images, a single dummy image is included.
        :param mm_query_lengths: A List of List that stores the lengths when each image is converted into visual tokens for LLM input.
         In cases where a sample does not contain any images, an empty list is included.
        :param non_mm_query_lengths: contains the lengths of text tokens (excluding visual tokens) for each sample in a batch.
        :param num_queries_vis_abstractors: A List of List that contains the number of visual tokens for each image grid.
        :param num_queries_vis_abstractors_slow: A List of List that contains the number of visual tokens for the slow part when applying the slowfast algorithm to video frames. If the slowfast algorithm is not applied, it will have a value of None.
        :param first_last_frames_slows: A List of List that stores the only first and last frames slow mode for each sample in a batch.
        :param is_videos: A List of List that stores the boolean value indicating whether each sample in a batch is a video.
        :image_grid_thw: A 3D tensor (torch.int64) for qwen2.5-vl visual encoder.
        :pixel_values_videos: A 2D tensor (torch.float32) for qwen2.5-vl visual encoder.
        :video_grid_thw: A 3D tensor (torch.int64) for qwen2.5-vl visual encoder.
        :param kwargs:
        :return:
        """
        inputs_embeds, _ = self.extract_inputs_embeds(
            input_ids=input_ids,
            pixel_values=self.to_vision_model_device(pixel_values),
            image_sizes=image_sizes,
            mm_query_lengths=mm_query_lengths,
            non_mm_query_lengths=non_mm_query_lengths,
            img_start_ids_list=img_start_ids_list,
            num_queries_vis_abstractors=num_queries_vis_abstractors,
            num_queries_vis_abstractors_slow=num_queries_vis_abstractors_slow,
            first_last_frames_slows=first_last_frames_slows,
            is_videos=is_videos,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            audio_values=audio_values,
            discrete_audio_values=discrete_audio_values,
            discrete_audio_value_num_per_sample=discrete_audio_value_num_per_sample,
            audio_masks=audio_masks,
        )

        if self.without_llm:
            inputs_embeds = (
                inputs_embeds.to(self.vision_model.device)
                if isinstance(inputs_embeds, torch.Tensor)
                else inputs_embeds
            )
            return inputs_embeds

        inputs_embeds = (
            inputs_embeds.to(self.base_model.device)
            if isinstance(inputs_embeds, torch.Tensor)
            else inputs_embeds
        )

        pred = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.encode("<|im_end|>")[0],
            bad_words_ids=[
                [
                    self.config.text_config.bos_token_id,
                ],
                [
                    self.config.text_config.eos_token_id,
                ],
            ],
            max_new_tokens=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=False if temperature == 0.0 else do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            early_stopping=False if num_beams <= 1 else True,
            use_cache=use_cache,
            no_repeat_ngram_size=20,
        )
        if verbose:
            llm_query = self.tokenizer.batch_decode(
                [
                    [
                        token_id
                        for token_id in input_ids_row
                        if token_id != self.tokenizer.pad_token_id
                    ]
                    for input_ids_row in input_ids.detach().cpu().tolist()
                ],
                skip_special_tokens=False,
            )[0]
            llm_pred = self.tokenizer.batch_decode(
                [
                    [
                        token_id
                        for token_id in pred_row
                        if token_id != self.tokenizer.pad_token_id
                    ]
                    for pred_row in pred.detach().cpu().tolist()
                ],
                skip_special_tokens=False,
            )[0]
            print(f"# [info] llm_query: {llm_query}")
            print(f"# [info] llm_pred: {llm_pred}")

        return pred

    def to_vision_model_device(self, input_tensor):
        if isinstance(input_tensor, list):
            return [self.to_vision_model_device(item) for item in input_tensor]
        elif isinstance(input_tensor, torch.Tensor):
            return input_tensor.to(self.vision_model.device)
        else:
            raise TypeError(
                "Unsupported data type. Only tensors and lists are allowed."
            )

    @classmethod
    def from_config(cls, config, vision_model_name_or_path):
        return cls(config, vision_model_name_or_path)

    def get_language_model(self):
        return self.language_model.base_model

    def get_vision_model(self):
        return self.vision_model

    def compute_adaptive_params(
        self,
        pixel_values: Optional[List[List[torch.FloatTensor]]] = None,
        num_queries_vis_abstractors: Optional[List[List[int]]] = None,
        num_queries_vis_abstractors_slow: Optional[List[List[int]]] = None,
        image_sizes: Optional[List[List[List[int]]]] = None,
        is_videos: Optional[List[List[bool]]] = None,
        first_last_frames_slows: Optional[List[List[bool]]] = None,
    ):
        assert all(
            all(isinstance(value, int) and value >= 0 for value in sublist)
            for sublist in num_queries_vis_abstractors
        ), "All values in num_queries_vis_abstractors must be integers >= 0."

        assert all(
            all(isinstance(value, int) and value >= 0 for value in sublist)
            for sublist in num_queries_vis_abstractors_slow
        ), "All values in num_queries_vis_abstractors_slow must be integers >= 0."

        assert is_videos is not None

        is_first_images = []
        is_last_images = []
        for is_video in is_videos:
            for idx, is_video_item in enumerate(is_video):
                if idx == 0:
                    is_first_images.append(True)
                else:
                    is_first_images.append(False)
                if idx == len(is_video) - 1:
                    is_last_images.append(True)
                else:
                    is_last_images.append(False)

        num_queries_vis_abstractors = list(chain(*num_queries_vis_abstractors))
        num_queries_vis_abstractors_slow = list(
            chain(*num_queries_vis_abstractors_slow)
        )
        image_sizes = list(chain(*image_sizes))
        is_videos = list(chain(*is_videos))
        first_last_frames_slows = list(chain(*first_last_frames_slows))

        use_slowfast = any(
            [num_query > 0 for num_query in num_queries_vis_abstractors_slow]
        )

        num_grids = [pixel_value.shape[0] for pixel_value in chain(*pixel_values)]
        num_grids = [0] + num_grids
        group_ids = []

        if use_slowfast:
            new_num_grids = [num_grids[0]]
            new_num_queries = []
            new_image_sizes = []
            new_is_videos = []

            for (
                num_query,
                num_query_slow,
                num_grid,
                image_size,
                is_video,
                first_last_frames_slow,
                is_first_image,
                is_last_image,
            ) in zip(
                num_queries_vis_abstractors,
                num_queries_vis_abstractors_slow,
                num_grids[1:],
                image_sizes,
                is_videos,
                first_last_frames_slows,
                is_first_images,
                is_last_images,
            ):

                if not first_last_frames_slow and num_query_slow > 0:
                    assert is_video is True

                    this_group_ids = [group_ids[-1][-1] + 1 if group_ids else 0]

                    new_num_grids.append(new_num_grids[-1] + 1)
                    new_num_queries.append(num_query_slow)
                    new_image_sizes.append(image_size)
                    new_is_videos.append(is_video)

                    if num_grid >= 2:
                        new_num_grids.append(new_num_grids[-1] + num_grid - 1)
                        new_num_queries.append(num_query)
                        new_image_sizes.append(image_size)
                        new_is_videos.append(is_video)
                        this_group_ids.append(this_group_ids[-1] + 1)

                    group_ids.append(this_group_ids)
                elif (
                    first_last_frames_slow
                    and num_query_slow > 0
                    and (is_first_image or is_last_image)
                ):
                    assert is_video is True

                    this_group_ids = [group_ids[-1][-1] + 1 if group_ids else 0]

                    if num_grid == 1:
                        new_num_grids.append(new_num_grids[-1] + 1)
                        new_num_queries.append(num_query_slow)
                        new_image_sizes.append(image_size)
                        new_is_videos.append(is_video)

                    if num_grid >= 2:
                        if is_first_image:
                            new_num_grids.append(new_num_grids[-1] + 1)
                            new_num_queries.append(num_query_slow)
                            new_image_sizes.append(image_size)
                            new_is_videos.append(is_video)
                            new_num_grids.append(new_num_grids[-1] + num_grid - 1)
                            new_num_queries.append(num_query)
                            new_image_sizes.append(image_size)
                            new_is_videos.append(is_video)
                            this_group_ids.append(this_group_ids[-1] + 1)
                        elif is_last_image:
                            new_num_grids.append(new_num_grids[-1] + num_grid - 1)
                            new_num_queries.append(num_query)
                            new_image_sizes.append(image_size)
                            new_is_videos.append(is_video)
                            new_num_grids.append(new_num_grids[-1] + 1)
                            new_num_queries.append(num_query_slow)
                            new_image_sizes.append(image_size)
                            new_is_videos.append(is_video)
                            this_group_ids.append(this_group_ids[-1] + 1)
                        else:
                            raise Exception("This case should not be reached.")
                    group_ids.append(this_group_ids)

                else:
                    new_num_grids.append(new_num_grids[-1] + num_grid)
                    new_num_queries.append(num_query)
                    new_image_sizes.append(image_size)
                    new_is_videos.append(is_video)

                    start_group_id = group_ids[-1][-1] + 1 if group_ids else 0
                    group_ids.append([start_group_id])

            num_grids = new_num_grids
            num_queries_vis_abstractors = new_num_queries
            image_sizes = new_image_sizes
            is_videos = new_is_videos
        else:
            num_grids = [sum(num_grids[:i]) for i in range(1, len(num_grids) + 1)]
            group_ids = [[group_id] for group_id in range(len(is_videos))]

        return num_queries_vis_abstractors, num_grids, image_sizes, is_videos, group_ids

    def split_adaptive_params(
        self, num_queries_vis_abstractors, num_grids, chunk_size: int, n_chunks: int
    ):
        """
        num_grids/num_queries 를 chunk_size 단위로 최대 n_chunks 만큼 자른다.
        실제 데이터가 부족하면 남은 chunk 는 더미([0,1]) 로 채운다.

        Returns
        -------
        chunk_qs   : List[List[int]]
        chunk_grids: List[List[int]]
            각 원소 길이는 동일하며, 전체 길이는 정확히 n_chunks.
        """
        total_len = num_grids[-1]
        chunk_qs, chunk_grids, is_splits = [], [], []

        slices = list(zip(num_grids[:-1], num_grids[1:], num_queries_vis_abstractors))
        slice_idx = 0

        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = start + chunk_size

            if start >= total_len:
                chunk_grids.append([0, 1])
                chunk_qs.append([num_queries_vis_abstractors[-1]])
                is_splits.append(False)
                continue

            grids_in_chunk = [0]
            qs_in_chunk = []

            while slice_idx < len(slices) and slices[slice_idx][1] <= start:
                slice_idx += 1

            is_split = False
            j = slice_idx
            while j < len(slices) and slices[j][0] < end:
                s, e, q = slices[j]

                left = max(s, start)
                right = min(e, end)
                off = right - start

                if off not in grids_in_chunk:
                    grids_in_chunk.append(off)
                    qs_in_chunk.append(q)
                    if right == end and e != end:
                        is_split = True

                if e > end:
                    break
                j += 1
            slice_idx = j

            final_off = min(end, total_len) - start
            if grids_in_chunk[-1] != final_off:
                grids_in_chunk.append(final_off)
                qs_in_chunk.append(
                    qs_in_chunk[-1] if qs_in_chunk else num_queries_vis_abstractors[-1]
                )
                is_split = True

            chunk_grids.append(grids_in_chunk)
            chunk_qs.append(qs_in_chunk)
            is_splits.append(is_split)

        return chunk_qs, chunk_grids, is_splits


class HCXVisionForCausalLM(HCXVisionPreTrainedModel, GenerationMixin):
    def __init__(
        self,
        config: HCXVisionConfig,
        without_llm=False,
        **kwargs,
    ):
        super().__init__(config, without_llm=without_llm, **kwargs)
        text_config = config.get_text_config()
        self.model = HCXVisionModel(config=config, **kwargs)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[List[List[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        image_sizes: Optional[List[List[List[int]]]] = None,
        vision_query_lengths: Optional[List[List[int]]] = None,
        non_vision_query_lengths: Optional[List[List[int]]] = None,
        img_start_ids_list: Optional[List[List[int]]] = None,
        num_queries_vis_abstractors: Optional[List[List[int]]] = None,
        num_queries_vis_abstractors_slow: Optional[List[List[int]]] = None,
        first_last_frames_slows: Optional[List[List[bool]]] = None,
        is_videos: Optional[List[List[bool]]] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        :param input_ids: torch.int64 : torch.size([batchsize, variable)]) : SystemPrompt with Question text token indices for tokenizer.
         In positions where images are inputted, the value is replaced by config.img_start_id, which is a vocabulary index used to indicate the start of image data.
        :param pixel_values: List of List of 4D tensor (torch.float32)
         Each outer list corresponds to a batch and contains inner lists, each holding tensors for images in a sample. The structure accounts for samples with multiple images.
        :param past_key_values: None
        :param inputs_embeds: None
        :param labels: Optional[torch.int64] : [batchsize, variable (input_ids.size(1)+ num visual tokens)] visual token 들은 모두 IGNORE_INDEX
        :param use_cache: None
        :param output_attentions: Optional[bool] : get attention weights of each layers of transformer network (true: 결과값에 포함, false: 결과값에 미포함)
        :param output_hidden_states: Optional[bool] : get hidden states of each layers of transformer network (true: 결과값에 포함, false: 결과값에 미포함)
        :param image_sizes: Stacked as a List of List, representing image sizes (width, height).
         In cases where a sample contains no images, a single dummy image is included.
        :param vision_query_lengths: A List of List that stores the lengths when each image is converted into visual tokens for LLM input.
         In cases where a sample does not contain any images, an empty list is included.
        :param non_vision_query_lengths: contains the lengths of text tokens (excluding visual tokens) for each sample in a batch.
        :img_start_ids_list: contains the indices of the img_start_id tokens for each sample.
        :num_queries_vis_abstractors: A List of List that contains the number of visual tokens for each image grid.
        :num_queries_vis_abstractors_slow: A List of List that contains the number of visual tokens for the slow part when applying the slowfast algorithm to video frames. If the slowfast algorithm is not applied, it will have a value of None.
        :first_last_frames_slows: A List of List that contains the only first and last frames slow mode for each sample in a batch.
        :is_videos: A List of List that contains the boolean value indicating whether each sample in a batch is a video.
        :image_grid_thw: A 3D tensor (torch.int64) for qwen2.5-vl visual encoder.
        :pixel_values_videos: A 2D tensor (torch.float32) for qwen2.5-vl visual encoder.
        :video_grid_thw: A 3D tensor (torch.int64) for qwen2.5-vl visual encoder.
        :return:
        """
        loss = None
        logits = None
        outputs = self.model.forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            image_sizes=image_sizes,
            vision_query_lengths=vision_query_lengths,
            non_vision_query_lengths=non_vision_query_lengths,
            img_start_ids_list=img_start_ids_list,
            num_queries_vis_abstractors=num_queries_vis_abstractors,
            num_queries_vis_abstractors_slow=num_queries_vis_abstractors_slow,
            first_last_frames_slows=first_last_frames_slows,
            is_videos=is_videos,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.model.language_model.lm_head(
            hidden_states[:, slice_indices, :]
        ) * getattr(self.config.text_config, "logits_scaling", 1)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                **kwargs,
            )
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def inference(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[
            Union[List[List[torch.FloatTensor]], torch.FloatTensor]
        ] = None,
        image_sizes: Optional[List[List[List[int]]]] = None,
        vision_query_lengths: Optional[List[List[int]]] = None,
        non_vision_query_lengths: Optional[List[int]] = None,
        num_queries_vis_abstractors: Optional[List[List[int]]] = None,
        num_queries_vis_abstractors_slow: Optional[List[List[int]]] = None,
        first_last_frames_slows: Optional[List[List[bool]]] = None,
        is_videos: Optional[List[List[bool]]] = None,
        img_start_ids_list: Optional[List[List[int]]] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        max_length: int = 196,
        min_length: int = 2,
        do_sample: bool = True,
        num_beams: int = 1,
        top_p: float = 0.6,
        top_k: int = 0,
        temperature: float = 0.5,
        repetition_penalty: float = 1.0,
        length_penalty: int = 1,
        early_stopping: Union[bool, str] = False,
        use_cache: bool = True,
        **kwargs,
    ):
        """
        :param input_ids: torch.int64 : torch.size([batchsize, variable)]) : SystemPrompt with Question text token indices for tokenizer.
         In positions where images are inputted, the value is replaced by config.img_start_id, which is a vocabulary index used to indicate the start of image data.
         In cases where a sample contains no images, a single dummy image is included.
        :param pixel_values: List of List of 4D tensor (torch.float32)
         Each outer list corresponds to a batch and contains inner lists, each holding tensors for images in a sample. The structure accounts for samples with multiple images.
        :param attention_mask: not used
        :param max_length: int :  The maximum length the generated tokens can have. Corresponds to the length of the input prompt + max_new_tokens.
        :param min_length: int : The minimum length of the sequence to be generated. Corresponds to the length of the input prompt + min_new_tokens.
        :param num_beams: int :  Number of beams for beam search. 1 means no beam search.
        :param top_k: int : The number of highest probability vocabulary tokens to keep for top-k-filtering.
        :param temperature: float :  The value used to modulate the next token probabilities. ( scores / self.temperature )
        :param repetition_penalty: float : The parameter for repetition penalty.
        :param length_penalty: int :  It is applied as an exponent to the sequence length, which in turn is used to divide the score of the sequence.
        :param early_stopping: Union[bool, str] : True, where the generation stops as soon as there are num_beams complete candidates;
            False, where an heuristic is applied and the generation stops when is it very unlikely to find better candidates;
            "never", where the beam search procedure only stops when there cannot be better candidates (canonical beam search algorithm)
        :param use_cache: bool :  Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
        :param verbose: bool : print debug mention
        :param image_sizes: Stacked as a List of List, representing image sizes (width, height).
         In cases where a sample contains no images, a single dummy image is included.
        :param vision_query_lengths: A List of List that stores the lengths when each image is converted into visual tokens for LLM input.
         In cases where a sample does not contain any images, an empty list is included.
        :param non_vision_query_lengths: contains the lengths of text tokens (excluding visual tokens) for each sample in a batch.
        :param num_queries_vis_abstractors: A List of List that contains the number of visual tokens for each image grid.
        :param num_queries_vis_abstractors_slow: A List of List that contains the number of visual tokens for the slow part when applying the slowfast algorithm to video frames. If the slowfast algorithm is not applied, it will have a value of None.
        :param first_last_frames_slows: A List of List that stores the only first and last frames slow mode for each sample in a batch.
        :param is_videos: A List of List that stores the boolean value indicating whether each sample in a batch is a video.
        :image_grid_thw: A 3D tensor (torch.int64) for qwen2.5-vl visual encoder.
        :pixel_values_videos: A 2D tensor (torch.float32) for qwen2.5-vl visual encoder.
        :video_grid_thw: A 3D tensor (torch.int64) for qwen2.5-vl visual encoder.
        :param kwargs:
        :return:
        """
        inputs_embeds, _ = self.model.extract_inputs_embeds(
            input_ids=input_ids,
            pixel_values=self.to_vision_model_device(pixel_values),
            image_sizes=image_sizes,
            vision_query_lengths=vision_query_lengths,
            non_vision_query_lengths=non_vision_query_lengths,
            img_start_ids_list=img_start_ids_list,
            num_queries_vis_abstractors=num_queries_vis_abstractors,
            num_queries_vis_abstractors_slow=num_queries_vis_abstractors_slow,
            first_last_frames_slows=first_last_frames_slows,
            is_videos=is_videos,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )

        if self.without_llm:
            inputs_embeds = (
                inputs_embeds.to(self.vision_model.device)
                if isinstance(inputs_embeds, torch.Tensor)
                else inputs_embeds
            )
            return inputs_embeds

        inputs_embeds = (
            inputs_embeds.to(self.base_model.device)
            if isinstance(inputs_embeds, torch.Tensor)
            else inputs_embeds
        )

        pred = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            pad_token_id=self.config.text_config.pad_token_id,
            eos_token_id=self.config.text_config.eos_token_id,
            bad_words_ids=[
                [
                    self.config.text_config.bos_token_id,
                ],
                [
                    self.config.text_config.eos_token_id,
                ],
            ],
            max_new_tokens=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=False if temperature == 0.0 else do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            early_stopping=False if num_beams <= 1 else True,
            use_cache=use_cache,
        )
        return pred

    def to_vision_model_device(self, input_tensor):
        if isinstance(input_tensor, list):
            return [self.to_vision_model_device(item) for item in input_tensor]
        elif isinstance(input_tensor, torch.Tensor):
            return input_tensor.to(self.vision_model.device)
        else:
            raise TypeError(
                "Unsupported data type. Only tensors and lists are allowed."
            )

    def get_input_embeddings(self):
        if self.without_llm:
            return None
        else:
            return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        if self.without_llm:
            return None
        else:
            return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        if self.without_llm:
            return None
        else:
            return self.language_model.tie_weights()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        **kwargs,
    ):
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            **kwargs,
        )

        model.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=True
        )
        return model

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        *args,
        **kwargs,
    ):
        super().register_for_auto_class("AutoModelForCausalLM")
        self.config.register_for_auto_class()
        super().save_pretrained(save_directory, *args, **kwargs)
        self.config.architectures = ["HCXVisionV2ForCausalLM"]
        self.config.auto_map["AutoModelForCausalLM"] = (
            "modeling_vlm.HCXVisionForCausalLM"
        )
        self.config.auto_map["AutoModelForSequenceClassification"] = (
            "modeling_vlm.HCXVisionForSequenceClassification"
        )
        self.config.save_pretrained(save_directory)

    @property
    def is_qwen_visual(self):
        return self.model.is_qwen_visual

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def vision_model(self):
        return self.model.vision_model

    @property
    def discrete_vision_model(self):
        return self.model.discrete_vision_model

    @property
    def audio_model(self):
        return self.model.audio_model

    @property
    def discrete_audio_model(self):
        return self.model.discrete_audio_model

    @property
    def text_config(self):
        return self.model.text_config

    @property
    def vision_config(self):
        return self.model.vision_config

    @property
    def audio_config(self):
        return self.model.audio_config

    @property
    def mm_projector(self):
        return self.model.mm_projector

    @property
    def audio_projector(self):
        return self.model.audio_projector

    @property
    def anyres(self):
        return self.model.anyres

    @property
    def is_safetensor_save(self):
        return self.model.is_safetensor_save

    @property
    def without_llm(self):
        return self.model.without_llm

    @property
    def image_newline(self):
        return self.model.image_newline


class HCXVisionForSequenceClassification(HCXVisionPreTrainedModel):
    """
    HCX Vision model for sequence classification tasks.
    """

    def __init__(self, config, **kwargs):
        super().__init__(config, without_llm=True, **kwargs)
        self.num_labels = config.num_labels if hasattr(config, "num_labels") else 2
        self.model = HCXVisionModel(config=config, **kwargs)
        self.score = nn.Linear(
            config.text_config.hidden_size, self.num_labels, bias=False
        )
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        image_sizes: Optional[List[List[List[int]]]] = None,
        vision_query_lengths: Optional[List[List[int]]] = None,
        non_vision_query_lengths: Optional[List[List[int]]] = None,
        img_start_ids_list: Optional[List[List[int]]] = None,
        num_queries_vis_abstractors: Optional[List[List[int]]] = None,
        num_queries_vis_abstractors_slow: Optional[List[List[int]]] = None,
        first_last_frames_slows: Optional[List[List[bool]]] = None,
        is_videos: Optional[List[List[bool]]] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
    ) -> SequenceClassifierOutputWithPast:
        """
        Forward pass for sequence classification.
        """
        transformer_outputs: BaseModelOutputWithPast = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            image_sizes=image_sizes,
            vision_query_lengths=vision_query_lengths,
            non_vision_query_lengths=non_vision_query_lengths,
            img_start_ids_list=img_start_ids_list,
            num_queries_vis_abstractors=num_queries_vis_abstractors,
            num_queries_vis_abstractors_slow=num_queries_vis_abstractors_slow,
            first_last_frames_slows=first_last_frames_slows,
            is_videos=is_videos,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )
        hidden_states = transformer_outputs.last_hidden_state
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            non_pad_mask = (input_ids != self.config.pad_token_id).to(
                logits.device, torch.int32
            )
            token_indices = torch.arange(
                input_ids.shape[-1], device=logits.device, dtype=torch.int32
            )
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1

        pooled_logits = logits[
            torch.arange(batch_size, device=logits.device), last_non_pad_token
        ]

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                pooled_logits=pooled_logits,
                config=self.config,
            )

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        *args,
        **kwargs,
    ):
        super().register_for_auto_class("AutoModelForSequenceClassification")
        self.config.register_for_auto_class()
        super().save_pretrained(save_directory, *args, **kwargs)


class VLM_Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        mm_projector_type,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mm_projector_type = mm_projector_type
        if self.mm_projector_type == "mlp":
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = act_layer()
            self.fc2 = nn.Linear(hidden_features, out_features)
        elif self.mm_projector_type == "inverted_mlp":
            self.fc1 = nn.Linear(in_features, 2 * hidden_features)
            self.act = act_layer()
            self.fc2 = nn.Linear(2 * hidden_features, out_features)
        else:
            raise NotImplementedError(
                "{} is not implemented".format(self.mm_projector_type)
            )

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Projector(nn.Module):
    """Base projector class"""

    def __init__(
        self,
        num_queries: int,
        num_input_tokens: int,
        encoder_hidden_size: int,
        hidden_size: int,
        output_hidden_size: int,
        pos_emb=True,
        prenorm=False,
    ):
        super().__init__()
        self.num_input_tokens = num_input_tokens
        self.output_hidden_size = output_hidden_size

        if pos_emb:
            self.pos_emb = torch.nn.Parameter(
                torch.zeros(1, num_input_tokens, encoder_hidden_size)
            )
            self.pos_emb.data.normal_(mean=0.0, std=0.02)
        else:
            self.pos_emb = None

        if prenorm:
            self.prenorm = LayerNorm(encoder_hidden_size)
        else:
            self.prenorm = None

        self.build_net(
            num_queries, encoder_hidden_size, hidden_size, output_hidden_size
        )

    def build_net(self):
        raise NotImplementedError()

    def _forward(
        self,
        x,
        num_queries_vis_abstractors: Optional[List[int]] = None,
        num_grids: Optional[List[int]] = None,
        freeze_before_sampler: bool = False,
    ):
        raise NotImplementedError()

    def forward(
        self,
        x: torch.Tensor,
        num_queries_vis_abstractors: Optional[List[int]] = None,
        num_grids: Optional[List[int]] = None,
        freeze_before_sampler: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, encoder_hidden_size) tensor from the visual backbone (CLIP visual encoder), including cls token.
        """
        if self.prenorm is not None:
            x = self.prenorm(x)

        if self.pos_emb is not None:
            x = x + self.pos_emb

        x = self._forward(
            x,
            num_queries_vis_abstractors=num_queries_vis_abstractors,
            num_grids=num_grids,
            freeze_before_sampler=freeze_before_sampler,
        )

        return x


class ConvProjector(Projector):
    def _forward(
        self,
        x,
        num_queries_vis_abstractors: Optional[List[int]] = None,
        num_grids: Optional[List[int]] = None,
        freeze_before_sampler: bool = False,
    ):
        hw = int(x.size(1) ** 0.5)
        x = rearrange(x, "b (h w) d -> b d h w", h=hw, w=hw)

        if num_queries_vis_abstractors is not None:
            assert num_grids is not None

            return self._forward_adaptive_num_query(
                x, num_queries_vis_abstractors, num_grids, freeze_before_sampler
            )

        if freeze_before_sampler:
            with torch.no_grad():
                x = self.net[0](x)
            x = self.net[1](x)
            x = self.net[2](x)
        else:
            x = self.net(x)
        x = rearrange(x, "b d h w -> b (h w) d")
        x = self.readout(x)

        return x

    def _forward_adaptive_num_query(
        self,
        x,
        num_queries_vis_abstractors: Optional[List[int]] = None,
        num_grids: Optional[List[int]] = None,
        freeze_before_sampler: bool = False,
    ):
        assert len(self.net) == 3

        if freeze_before_sampler:
            with torch.no_grad():
                x = self.net[0](x)
        else:
            x = self.net[0](x)

        new_x = []
        for i, num_queries in enumerate(num_queries_vis_abstractors):
            hw = int(num_queries**0.5)
            sampler = nn.AdaptiveAvgPool2d((hw, hw))
            out = sampler(x[num_grids[i] : num_grids[i + 1], :])
            out = self.net[2](out)

            out = rearrange(out, "b d h w -> b (h w) d")
            out = self.readout(out)

            new_x.append(out)

        return new_x


class CAbstractor(ConvProjector):
    """C-Abstractor"""

    def build_net(
        self,
        n_queries,
        encoder_hidden_size,
        hidden_size,
        output_hidden_size,
        depth=3,
        mlp_depth=2,
    ):
        assert (n_queries**0.5).is_integer(), "n_queries must be square number"
        hw = int(n_queries**0.5)

        RegBlock = partial(
            RegStage,
            stride=1,
            dilation=1,
            act_layer=nn.SiLU,
            norm_layer=LayerNorm2d,
        )

        s1 = RegBlock(
            depth,
            encoder_hidden_size,
            hidden_size,
        )
        sampler = nn.AdaptiveAvgPool2d((hw, hw))
        s2 = RegBlock(
            depth,
            hidden_size,
            hidden_size,
        )

        self.net = nn.Sequential(s1, sampler, s2)

        self.readout = self.build_mlp(mlp_depth, hidden_size, output_hidden_size)

    def build_mlp(self, depth, hidden_size, output_hidden_size):
        layers = [nn.Linear(hidden_size, output_hidden_size)]
        for _ in range(1, depth):
            layers.append(nn.SiLU())
            layers.append(nn.Linear(output_hidden_size, output_hidden_size))
        return nn.Sequential(*layers)


AutoConfig.register("vlm", HCXVisionConfig)
try:
    from .configuration_hyperclovax import HyperCLOVAXConfig
    from .modeling_hyperclovax import HyperCLOVAXForCausalLM

    AutoConfig.register("hyperclovax", HyperCLOVAXConfig)
    AutoModelForCausalLM.register(
        HyperCLOVAXConfig,
        HyperCLOVAXForCausalLM,
    )
except:
    pass

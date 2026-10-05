import copy
import math
import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers import Qwen2_5_VLProcessor
from transformers.image_processing_utils import (
    BaseImageProcessor,
    BatchFeature,
    get_size_dict,
)
from transformers.image_transforms import (
    convert_to_rgb,
    get_resize_output_image_size,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import (
    Qwen2_5_VLProcessorKwargs,
)
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import TensorType, logging
from transformers.video_utils import VideoInput
from typing_extensions import Unpack

logger = logging.get_logger(__name__)


def determine_possible_resolutions(
    anyres: bool, max_num_grids: int, grid_size: int, use_1x1_grid: bool = False
):
    """총 max_num_grids 이하의 possible resolution 조합을 찾아 반환합니다.
    max_num_grids 가 예를 들어 4인 경우, 총 가능한 grid 조합은 [1x1, 1x2, 1x3, 1x4, 2x1, 2x2, 3x1, 4x1] 이고, 따라서 아래와 같이 계산됩니다.
    >>> possible_resolutions = determine_possible_resolutions(anyres=True, max_num_grids=4, grid_size=336)
    >>> print(possible_resolutions)
    [[336, 336], [336, 672], [336, 1008], [336, 1344], [672, 336], [672, 672], [1008, 336], [1344, 336]]
    """
    possible_resolutions = []
    if anyres:
        assert max_num_grids > 0
        for i in range(1, max_num_grids + 1):
            for j in range(1, max_num_grids + 1):
                if i == 1 and j == 1 and not use_1x1_grid:
                    continue
                if i * j <= max_num_grids:
                    possible_resolutions.append([i, j])

        possible_resolutions = [
            [ys * grid_size, xs * grid_size] for ys, xs in possible_resolutions
        ]

    return possible_resolutions


def divide_to_grids(
    image: np.array, grid_size: int, input_data_format=None
) -> List[np.array]:
    """local image 를 (grid_size x grid_size) grid 로 divide"""
    grids = []
    height, width = get_image_size(image, channel_dim=input_data_format)
    for i in range(0, height, grid_size):
        for j in range(0, width, grid_size):
            if input_data_format == ChannelDimension.LAST:
                grid = image[i : i + grid_size, j : j + grid_size]
            else:
                grid = image[:, i : i + grid_size, j : j + grid_size]
            grids.append(grid)

    return grids


def pad(
    image: np.array,
    target_size: tuple,
    background_color=(127, 127, 127),
    input_data_format=None,
) -> np.array:
    """image 양옆, 좌우에 padding 을 하여 target_height, target_width 만큼 키움"""
    target_height, target_width = target_size
    height, width = get_image_size(image, channel_dim=input_data_format)

    result = np.empty((target_height, target_width, image.shape[2]), dtype=image.dtype)
    for i in range(image.shape[2]):
        result[..., i].fill(background_color[i])

    paste_x = (target_width - width) // 2
    paste_y = (target_height - height) // 2

    result[paste_y : paste_y + height, paste_x : paste_x + width, :] = image

    return result


def expand2square(
    image: np.array,
    bboxes_dict=None,
    background_color=(127, 127, 127),
    input_data_format=None,
) -> np.array:
    """
    새로운 canvas 를 만들어 두고, 거기에 이미지를 붙여넣는 방식으로 이미지를 정사각형으로 만드는 함수
    유의할 사항은, 이미지를 붙여 넣을 때 중앙으로 붙여넣는다는 점. 양옆 또는 위아래로 PADDING 이 들어가는 형태
    Args:
        pil_img: numpy array
        bboxes_dict: dict, {"ocr": NDArray shape (N, 4, 2), "html": NDArray shape (N, 4, 2), ... }
            `[[xtl, ytl], [xtr, ytr], [xbr, ybr], [xbl, ybl]]` 형태로 박스 형태는 통일. OCR, HTML 등 다양한 박스들을 한번에 처리 가능
        background_color: tuple, RGB
    # >>> _img = np.ones((80, 100), dtype=np.uint8) * 100
    # >>> _bboxes_dict = {"words": np.array([[[10, 10], [20, 10], [20, 20], [10, 20]],
    # ...                                    [[30, 30], [40, 30], [40, 40], [30, 40]]])}
    # >>> _img, _bboxes_dict = expand2square(_img, _bboxes_dict, (255, 255, 255))
    # >>> _img.shape
    # (100, 100)
    # >>> guessed_ocr_bboxes = np.array([[[20, 10], [30, 10], [30, 20], [20, 20]],
    # ...                                [[40, 30], [50, 30], [50, 40], [40, 40]]])
    # >>> np.testing.assert_array_almost_equal(_bboxes_dict["words"], guessed_ocr_bboxes) is None
    # True
    """
    height, width = get_image_size(image, channel_dim=input_data_format)
    if width == height:
        return image, bboxes_dict
    elif width > height:
        result = np.empty((width, width, image.shape[2]), dtype=image.dtype)
        for i in range(image.shape[2]):
            result[..., i].fill(background_color[i])

        result[(width - height) // 2 : (width - height) // 2 + height, :] = image
        if bboxes_dict is not None:
            for key in bboxes_dict:
                bboxes_dict[key][:, :, 1] += (width - height) // 2
        return result, bboxes_dict
    else:
        result = np.empty((height, height, image.shape[2]), dtype=image.dtype)
        for i in range(image.shape[2]):
            result[..., i].fill(background_color[i])

        result[:, (height - width) // 2 : (height - width) // 2 + width] = image
        if bboxes_dict is not None:
            for key in bboxes_dict:
                bboxes_dict[key][:, :, 0] += (height - width) // 2
        return result, bboxes_dict


def resize_longside(
    image: np.array,
    size: int,
    resample: PILImageResampling = PILImageResampling.BICUBIC,
    data_format: Optional[Union[str, ChannelDimension]] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
):
    """
    장축 길이를 size 에 맞게 resize
    """
    height, width = get_image_size(image, channel_dim=input_data_format)

    if width == height:
        target_height, target_width = size, size
    elif width > height:
        target_width = size
        target_height = math.ceil(height / width * size)
    else:
        target_width = math.ceil(width / height * size)
        target_height = size

    return resize(
        image,
        size=(target_height, target_width),
        resample=resample,
        data_format=data_format,
        input_data_format=input_data_format,
    )


def select_best_resolution(original_size: tuple, possible_resolutions: list) -> tuple:
    """From LLaVA-Next (https://github.com/huggingface/transformers/blob/v4.40.2/src/transformers/models/llava_next/image_processing_llava_next.py)
    Selects the best resolution from a list of possible resolutions based on the original size.
    This is done by calculating the effective and wasted resolution for each possible resolution.
    The best fit resolution is the one that maximizes the effective resolution and minimizes the wasted resolution.

    Args:
        original_size (tuple):
            The original size of the image in the format (height, width).
        possible_resolutions (list):
            A list of possible resolutions in the format [(height1, width1), (height2, width2), ...].

    Returns:
        tuple: The best fit resolution in the format (height, width).
    """
    original_height, original_width = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for height, width in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(
            original_height * scale
        )
        effective_resolution = min(
            downscaled_width * downscaled_height, original_width * original_height
        )
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution
            and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (height, width)

    return best_fit


def _get_local_grids_output_size(
    image: np.array, target_resolution: tuple, input_data_format=None
):
    original_height, original_width = get_image_size(
        image, channel_dim=input_data_format
    )
    target_height, target_width = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    return new_height, new_width


def determine_anyres_num_vision_patches(
    num_grids,
    image_size,
    grid_size,
    patch_size,
    possible_resolutions,
    anyres=False,
    unpad=True,
    num_queries_vis_abstractor=0,
    num_queries_vis_abstractor_slow=0,
    video=False,
    first_last_frames_slow=False,
    is_first_or_last_frames=False,
):
    """visual tokens 수를 계산해주는 함수"""
    if not anyres:
        return (
            num_queries_vis_abstractor
            if num_queries_vis_abstractor > 0
            else (grid_size // patch_size) ** 2
        )

    if num_queries_vis_abstractor > 0:
        num_patch_per_grid = int(num_queries_vis_abstractor**0.5)
    else:
        num_patch_per_grid = grid_size // patch_size

    num_global_per_grid = num_patch_per_grid

    height, width = select_best_resolution(image_size, possible_resolutions)

    num_patch_height = (height // grid_size) * num_patch_per_grid
    num_patch_width = (width // grid_size) * num_patch_per_grid

    if unpad:
        original_height, original_width = image_size

        original_aspect_ratio = original_width / original_height
        current_aspect_ratio = num_patch_width / num_patch_height

        if original_aspect_ratio > current_aspect_ratio:
            scale_factor = num_patch_width / original_width
            new_height = int(original_height * scale_factor)
            padding = (num_patch_height - new_height) // 2
            num_patch_height = num_patch_height - padding * 2
        else:
            scale_factor = num_patch_height / original_height
            new_width = int(original_width * scale_factor)
            padding = (num_patch_width - new_width) // 2
            num_patch_width = num_patch_width - padding * 2

        num_patches = num_patch_width * num_patch_height + num_patch_height
    else:
        num_patches = num_patch_width * num_patch_height

    if num_queries_vis_abstractor_slow > 0:
        if first_last_frames_slow:
            if is_first_or_last_frames:
                num_patches += (
                    num_queries_vis_abstractor_slow - num_queries_vis_abstractor
                )
        else:
            num_patches += num_queries_vis_abstractor_slow - num_queries_vis_abstractor
        assert unpad is False

    if not video:
        num_patches += num_global_per_grid**2

    return num_patches


class HCXVisionImageProcessor(BaseImageProcessor):
    r"""
    Constructs a VLM image processor. Based on [`CLIPImageProcessor`] with incorporation of additional techniques for processing high resolution images.

    Args:
        anyres: (bool) anyres 기능을 사용할지 안할지
        unpad: (bool) anyres 사용시, unpad 기능 (순수 pad 영역에 해당하는 visual tokens 은 LLM input 에서 제거) 을 사용할지 안할지
        num_queries_vis_abstractor: (int) 각 grid 에 대해서 resampler 를 사용하는 경우, visual query 수
        possible_resolutions: (List) anyres 기능 사용시, 가능한 resolution 조합, 예: [[336, 336], [336, 672], [672, 336]]
        patch_size: (int) ViT patch size
        pad_to_square: (bool) 정사각형으로 padding 을 수행할지, 안할지를 결정. False 이면 정사각형이 아니기 때문에 center crop 을 거쳐 ViT 의 입력으로 들어감
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        anyres: bool = False,
        unpad: bool = False,
        num_queries_vis_abstractor: int = 0,
        possible_resolutions: List = [],
        patch_size: int = 14,
        pad_to_square: bool = True,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_center_crop: bool = True,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"shortest_edge": 336}
        size = get_size_dict(size, default_to_square=False)
        crop_size = (
            crop_size if crop_size is not None else {"height": 336, "width": 336}
        )
        crop_size = get_size_dict(
            crop_size, default_to_square=True, param_name="crop_size"
        )

        self.do_resize = do_resize
        self.size = size
        self.anyres = anyres
        self.unpad = unpad
        self.num_queries_vis_abstractor = num_queries_vis_abstractor
        self.possible_resolutions = [
            _resolution for _resolution in possible_resolutions
        ]
        self.patch_size = patch_size
        self.pad_to_square = pad_to_square
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.do_convert_rgb = do_convert_rgb

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        default_to_square = True
        if "shortest_edge" in size:
            size = size["shortest_edge"]
            default_to_square = False
        elif "height" in size and "width" in size:
            size = (size["height"], size["width"])
        else:
            raise ValueError(
                "Size must contain either 'shortest_edge' or 'height' and 'width'."
            )

        output_size = get_resize_output_image_size(
            image,
            size=size,
            default_to_square=default_to_square,
            input_data_format=input_data_format,
        )

        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def _preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_center_crop: bool = None,
        crop_size: int = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> Image.Image:
        images = make_list_of_images(images)

        if do_resize:
            images = [
                self.resize(
                    image=image,
                    size=size,
                    resample=resample,
                    input_data_format=input_data_format,
                )
                for image in images
            ]

        if do_center_crop:
            images = [
                self.center_crop(
                    image=image, size=crop_size, input_data_format=input_data_format
                )
                for image in images
            ]

        if do_rescale:
            images = [
                self.rescale(
                    image=image,
                    scale=rescale_factor,
                    input_data_format=input_data_format,
                )
                for image in images
            ]

        if do_normalize:
            images = [
                self.normalize(
                    image=image,
                    mean=image_mean,
                    std=image_std,
                    input_data_format=input_data_format,
                )
                for image in images
            ]

        images = [
            to_channel_dimension_format(
                image, data_format, input_channel_dim=input_data_format
            )
            for image in images
        ]

        return images

    def _resize_for_local_grids(
        self,
        image: np.array,
        target_resolution: tuple,
        resample,
        input_data_format: ChannelDimension,
    ) -> np.array:
        new_height, new_width = _get_local_grids_output_size(
            image, target_resolution, input_data_format
        )

        resized_image = resize(
            image,
            (new_height, new_width),
            resample=resample,
            input_data_format=input_data_format,
        )

        return resized_image

    def _pad_for_patching(
        self,
        image: np.array,
        target_resolution: tuple,
        input_data_format: ChannelDimension,
    ) -> np.array:
        """
        Pad an image to a target resolution while maintaining aspect ratio.
        """
        target_height, target_width = target_resolution

        background_color = tuple(int(x * 255) for x in self.image_mean)
        padded_image = pad(
            image,
            target_size=(target_height, target_width),
            background_color=background_color,
            input_data_format=input_data_format,
        )

        return padded_image

    def get_image_grids(
        self,
        image: np.array,
        possible_resolutions,
        grid_size: int,
        resample: PILImageResampling,
        data_format: ChannelDimension,
        input_data_format: ChannelDimension,
    ) -> List[np.array]:
        if not isinstance(possible_resolutions, list):
            raise ValueError(
                "possible_resolutions must be a list of possible resolutions."
            )

        image_size = get_image_size(image, channel_dim=input_data_format)
        best_resolution = select_best_resolution(image_size, possible_resolutions)
        resized_image = self._resize_for_local_grids(
            image,
            best_resolution,
            resample=resample,
            input_data_format=input_data_format,
        )
        padded_image = self._pad_for_patching(
            resized_image, best_resolution, input_data_format=input_data_format
        )
        local_grids = divide_to_grids(
            padded_image, grid_size=grid_size, input_data_format=input_data_format
        )

        local_grids = [
            to_channel_dimension_format(
                grid, channel_dim=data_format, input_channel_dim=input_data_format
            )
            for grid in local_grids
        ]

        return local_grids

    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        anyres: bool = None,
        unpad: bool = None,
        video: bool = None,
        num_queries_vis_abstractor: int = None,
        possible_resolutions: List = None,
        patch_size: int = None,
        pad_to_square: bool = None,
        resample: PILImageResampling = None,
        do_center_crop: bool = None,
        crop_size: int = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        return_dummy_image: bool = False,
        num_queries_vis_abstractor_slow: int = 0,
        first_last_frames_slow: bool = False,
        is_first_or_last_frames: bool = False,
    ):
        """
        HCXVisionImageProcessor 로 image tensor, original image size (width, height), visual tokens

        :return pixel_values: List of 4D tensor 로 image tensor
        :return image_sizes: List of Dict 로 image width, height [{"width": image 1 의 width, "height": image 1 의 height}, {"width": image 2 의 width, "height": image 2 의 height}, ...]
        :return vision_query_lengths: List of int 로 각 image 가 LLM 입력으로 전달될때 변환되는 visual token 수
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        size = get_size_dict(size, param_name="size", default_to_square=False)
        anyres = anyres if anyres is not None else self.anyres
        unpad = unpad if unpad is not None else self.unpad
        if video:
            unpad = False
        num_queries_vis_abstractor = (
            num_queries_vis_abstractor
            if num_queries_vis_abstractor is not None
            else self.num_queries_vis_abstractor
        )
        possible_resolutions = (
            possible_resolutions
            if possible_resolutions is not None
            else self.possible_resolutions
        )
        patch_size = patch_size if patch_size is not None else self.patch_size
        pad_to_square = (
            pad_to_square if pad_to_square is not None else self.pad_to_square
        )
        resample = resample if resample is not None else self.resample
        do_center_crop = (
            do_center_crop if do_center_crop is not None else self.do_center_crop
        )
        crop_size = crop_size if crop_size is not None else self.crop_size
        crop_size = get_size_dict(
            crop_size, param_name="crop_size", default_to_square=True
        )
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = (
            rescale_factor if rescale_factor is not None else self.rescale_factor
        )
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = (
            do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        )

        if return_dummy_image:
            images = Image.new("RGB", (224, 224), (0, 0, 0))

        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        images = [to_numpy_array(image) for image in images]

        if is_scaled_image(images[0]) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(images[0])

        new_images = []
        image_sizes = [
            get_image_size(image, channel_dim=input_data_format) for image in images
        ]
        vision_query_lengths = []

        assert crop_size["height"] == crop_size["width"]

        if anyres:
            anyres_global_images = copy.deepcopy(images)
            if pad_to_square:
                background_color = tuple(int(x * 255) for x in self.image_mean)
                anyres_global_images = [
                    resize_longside(
                        copy.deepcopy(image),
                        size["shortest_edge"],
                        resample,
                        input_data_format,
                    )
                    for image in anyres_global_images
                ]
                anyres_global_images = [
                    expand2square(
                        image,
                        background_color=background_color,
                        input_data_format=input_data_format,
                    )[0]
                    for image in anyres_global_images
                ]
            else:
                anyres_global_images = [
                    self.resize(
                        image=image,
                        size={
                            "height": size["shortest_edge"],
                            "width": size["shortest_edge"],
                        },
                        resample=resample,
                        input_data_format=input_data_format,
                    )
                    for image in anyres_global_images
                ]
        else:
            anyres_global_images = [None for _ in range(len(images))]
            if pad_to_square:
                background_color = tuple(int(x * 255) for x in self.image_mean)
                images = [
                    resize_longside(
                        image, size["shortest_edge"], resample, input_data_format
                    )
                    for image in images
                ]
                images = [
                    expand2square(
                        image,
                        background_color=background_color,
                        input_data_format=input_data_format,
                    )[0]
                    for image in images
                ]

        for image, anyres_global_image, image_size in zip(
            images, anyres_global_images, image_sizes
        ):
            if anyres:
                image_grids = self.get_image_grids(
                    image,
                    possible_resolutions,
                    grid_size=crop_size["height"],
                    resample=resample,
                    data_format=input_data_format,
                    input_data_format=input_data_format,
                )
                if not video:
                    image_grids = [anyres_global_image] + image_grids
            else:
                image_grids = [image]

            pixel_values = self._preprocess(
                image_grids,
                do_resize=do_resize,
                size=size,
                resample=resample,
                do_center_crop=do_center_crop,
                crop_size=crop_size,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
                data_format=data_format,
                input_data_format=input_data_format,
            )

            pixel_values = np.array(pixel_values)
            new_images.append(pixel_values)

            num_grids = pixel_values.shape[0]

            vision_query_length = determine_anyres_num_vision_patches(
                num_grids=num_grids,
                image_size=image_size,
                grid_size=crop_size["height"],
                patch_size=patch_size,
                possible_resolutions=possible_resolutions,
                anyres=anyres,
                unpad=unpad,
                num_queries_vis_abstractor=num_queries_vis_abstractor,
                num_queries_vis_abstractor_slow=num_queries_vis_abstractor_slow,
                video=video,
                first_last_frames_slow=first_last_frames_slow,
                is_first_or_last_frames=is_first_or_last_frames,
            )

            vision_query_lengths.append(vision_query_length)

        if return_dummy_image:
            vision_query_lengths = []

        data = {
            "pixel_values": [torch.tensor(new_image) for new_image in new_images],
            "image_sizes": [
                {"width": image_size[1], "height": image_size[0]}
                for image_size in image_sizes
            ],
            "vision_query_lengths": vision_query_lengths,
        }

        return BatchFeature(data=data)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        *args,
        **kwargs,
    ):
        self.register_for_auto_class()
        super().save_pretrained(save_directory, *args, **kwargs)


class HCXVisionV2Processor(Qwen2_5_VLProcessor):
    attributes = ["image_processor", "tokenizer", "video_processor"]
    image_processor_class = "AutoImageProcessor"
    video_processor_class = "AutoVideoProcessor"
    tokenizer_class = (
        "GPT2Tokenizer",
        "GPT2TokenizerFast",
        "PreTrainedTokenizer",
        "PreTrainedTokenizerFast",
    )

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        super().__init__(
            image_processor,
            tokenizer,
            video_processor,
            chat_template=self.tokenizer.chat_template,
        )

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        *args,
        **kwargs,
    ):
        self.register_for_auto_class()
        super().save_pretrained(save_directory, *args, **kwargs)

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[
            TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]
        ] = None,
        videos: VideoInput = None,
        **kwargs: Unpack[Qwen2_5_VLProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the vision inputs, this method forwards the `vision_infos` and `kwrags` arguments to
        Qwen2VLImageProcessor's [`~Qwen2VLImageProcessor.__call__`] if `vision_infos` is not `None`.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `list[str]`, `list[list[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            videos (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The image or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch
                tensor, or a nested list of 3D frames. Both channels-first and channels-last formats are supported.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **pixel_values_videos** -- Pixel values of videos to be fed to a model. Returned when `videos` is not `None`.
            - **image_grid_thw** -- List of image 3D grid in LLM. Returned when `images` is not `None`.
            - **video_grid_thw** -- List of video 3D grid in LLM. Returned when `videos` is not `None`.
        """
        output_kwargs = self._merge_kwargs(
            Qwen2_5_VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        image_inputs = videos_inputs = {}
        if images is not None:
            image_inputs = self.image_processor(
                images=images, **output_kwargs["images_kwargs"]
            )
            image_grid_thw = image_inputs["image_grid_thw"]

        if videos is not None:
            videos_inputs = self.video_processor(
                videos=videos, **output_kwargs["videos_kwargs"]
            )
            video_grid_thw = videos_inputs["video_grid_thw"]

        if not isinstance(text, list):
            text = [text]

        text = text.copy()

        if images is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(
                        self.image_token, "<|placeholder|>" * num_image_tokens, 1
                    )
                    text[i] = text[i].replace(
                        '{"resolution": [w, h]}',
                        '{"resolution": ' + str(list(images[i].size)) + "}",
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if videos is not None:
            merge_length = self.video_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    num_video_tokens = video_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(
                        self.video_token, "<|placeholder|>" * num_video_tokens, 1
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop(
            "return_mm_token_type_ids", False
        )
        text_inputs = self.tokenizer(
            text, **output_kwargs["text_kwargs"], return_tensors=None
        )
        self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(
            data={**text_inputs, **image_inputs, **videos_inputs},
            tensor_type=return_tensors,
        )

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
from dataclasses import dataclass
from typing import Any, List, Optional
import io
import json
import math
import cv2
import numpy as np

import torch
import torch.nn.functional as F
from einops import rearrange
from megatron.energon import DefaultTaskEncoder, Sample, SkipSample
from megatron.energon.task_encoder.base import stateless
from megatron.energon.task_encoder.cooking import Cooker, basic_sample_keys
from torchvision import transforms

from nemo.lightning.io.mixin import IOMixin
from nemo.utils.sequence_packing_utils import first_fit_decreasing
from PIL import Image, ImageOps, ImageDraw


@dataclass
class DiffusionSample(Sample):
    """
    Data class representing a sample for diffusion tasks.

    Attributes:
        video (torch.Tensor): Video latents (C T H W).
        t5_text_embeddings (torch.Tensor): Text embeddings (S D).
        t5_text_mask (torch.Tensor): Mask for text embeddings.
        loss_mask (torch.Tensor): Mask indicating valid positions for loss computation.
        image_size (Optional[torch.Tensor]): Tensor containing image dimensions.
        fps (Optional[torch.Tensor]): Frame rate of the video.
        num_frames (Optional[torch.Tensor]): Number of frames in the video.
        padding_mask (Optional[torch.Tensor]): Mask indicating padding positions.
        seq_len_q (Optional[torch.Tensor]): Sequence length for query embeddings.
        seq_len_kv (Optional[torch.Tensor]): Sequence length for key/value embeddings.
        pos_ids (Optional[torch.Tensor]): Positional IDs.
        latent_shape (Optional[torch.Tensor]): Shape of the latent tensor.
    """

    video: torch.Tensor  # video latents (C T H W)
    t5_text_embeddings: torch.Tensor  # (S D)
    t5_text_mask: torch.Tensor  # 1
    loss_mask: torch.Tensor
    image_size: Optional[torch.Tensor] = None
    fps: Optional[torch.Tensor] = None
    num_frames: Optional[torch.Tensor] = None
    padding_mask: Optional[torch.Tensor] = None
    seq_len_q: Optional[torch.Tensor] = None
    seq_len_kv: Optional[torch.Tensor] = None
    pos_ids: Optional[torch.Tensor] = None
    latent_shape: Optional[torch.Tensor] = None

    def to_dict(self) -> dict:
        """Converts the sample to a dictionary."""
        return dict(
            video=self.video,
            t5_text_embeddings=self.t5_text_embeddings,
            t5_text_mask=self.t5_text_mask,
            loss_mask=self.loss_mask,
            image_size=self.image_size,
            fps=self.fps,
            num_frames=self.num_frames,
            padding_mask=self.padding_mask,
            seq_len_q=self.seq_len_q,
            seq_len_kv=self.seq_len_kv,
            pos_ids=self.pos_ids,
            latent_shape=self.latent_shape,
        )

    def __add__(self, other: Any) -> int:
        """Adds the sequence length of this sample with another sample or integer."""
        if isinstance(other, DiffusionSample):
            # Combine the values of the two instances
            return self.seq_len_q.item() + other.seq_len_q.item()
        elif isinstance(other, int):
            # Add an integer to the value
            return self.seq_len_q.item() + other
        raise NotImplementedError

    def __radd__(self, other: Any) -> int:
        """Handles reverse addition for summing with integers."""
        # This is called if sum or other operations start with a non-DiffusionSample object.
        # e.g., sum([DiffusionSample(1), DiffusionSample(2)]) -> the 0 + DiffusionSample(1) calls __radd__.
        if isinstance(other, int):
            return self.seq_len_q.item() + other
        raise NotImplementedError

    def __lt__(self, other: Any) -> bool:
        """Compares this sample's sequence length with another sample or integer."""
        if isinstance(other, DiffusionSample):
            return self.seq_len_q.item() < other.seq_len_q.item()
        elif isinstance(other, int):
            return self.seq_len_q.item() < other
        raise NotImplementedError


def cook(sample: dict) -> dict:
    """
    Processes a raw sample dictionary from energon dataset and returns a new dictionary with specific keys.

    Args:
        sample (dict): The input dictionary containing the raw sample data.

    Returns:
        dict: A new dictionary containing the processed sample data with the following keys:
            - All keys from the result of `basic_sample_keys(sample)`
            - 'json': The contains meta data like resolution, aspect ratio, fps, etc.
            - 'pth': contains video latent tensor
            - 'pickle': contains text embeddings
    """
    return dict(
        **basic_sample_keys(sample),
        json=sample['.json'],
        pth=sample['.pth'],
        pickle=sample['.pickle'],
    )


class BasicDiffusionTaskEncoder(DefaultTaskEncoder, IOMixin):
    """
    BasicDiffusionTaskEncoder is a class that encodes image/video samples for diffusion tasks.
    Attributes:
        cookers (list): A list of Cooker objects used for processing.
        max_frames (int, optional): The maximum number of frames to consider from the video. Defaults to None.
        text_embedding_padding_size (int): The padding size for text embeddings. Defaults to 512.
    Methods:
        __init__(*args, max_frames=None, text_embedding_padding_size=512, **kwargs):
            Initializes the BasicDiffusionTaskEncoder with optional maximum frames and text embedding padding size.
        encode_sample(sample: dict) -> dict:
            Encodes a given sample dictionary containing video and text data.
            Args:
                sample (dict): A dictionary containing 'pth' for video latent and 'json' for additional info.
            Returns:
                dict: A dictionary containing encoded video, text embeddings, text mask, and loss mask.
            Raises:
                SkipSample: If the video latent contains NaNs, Infs, or is not divisible by the tensor parallel size.
    """

    cookers = [
        Cooker(cook),
    ]

    def __init__(
        self,
        *args,
        max_frames: int = None,
        text_embedding_padding_size: int = 512,
        seq_length: int = None,
        max_seq_length: int = None,
        patch_spatial: int = 2,
        patch_temporal: int = 1,
        aesthetic_score: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_frames = max_frames
        self.text_embedding_padding_size = text_embedding_padding_size
        self.seq_length = seq_length
        self.max_seq_length = max_seq_length
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.aesthetic_score = aesthetic_score

    @stateless(restore_seeds=True)
    def encode_sample(self, sample: dict) -> dict:
        """
        Encodes video / text sample.
        """
        video_latent = sample['pth']

        if torch.isnan(video_latent).any() or torch.isinf(video_latent).any():
            raise SkipSample()
        if torch.max(torch.abs(video_latent)) > 1e3:
            raise SkipSample()

        info = sample['json']
        if info['aesthetic_score'] < self.aesthetic_score:
            raise SkipSample()

        C, T, H, W = video_latent.shape
        seq_len = (
            video_latent.shape[-1]
            * video_latent.shape[-2]
            * video_latent.shape[-3]
            // self.patch_spatial**2
            // self.patch_temporal
        )
        is_image = T == 1

        if self.seq_length is not None and seq_len > self.seq_length:
            raise SkipSample()
        if self.max_seq_length is not None and seq_len > self.max_seq_length:
            raise SkipSample()

        if self.max_frames is not None:
            video_latent = video_latent[:, : self.max_frames, :, :]

        video_latent = rearrange(
            video_latent,
            'C (T pt) (H ph) (W pw) -> (T H W) (ph pw pt C)',
            ph=self.patch_spatial,
            pw=self.patch_spatial,
            pt=self.patch_temporal,
        )

        if is_image:
            t5_text_embeddings = torch.from_numpy(sample['pickle']).to(torch.bfloat16)
        else:
            t5_text_embeddings = torch.from_numpy(sample['pickle'][0]).to(torch.bfloat16)
        t5_text_embeddings_seq_length = t5_text_embeddings.shape[0]

        if t5_text_embeddings_seq_length > self.text_embedding_padding_size:
            t5_text_embeddings = t5_text_embeddings[: self.text_embedding_padding_size]
        else:
            t5_text_embeddings = F.pad(
                t5_text_embeddings,
                (
                    0,
                    0,
                    0,
                    self.text_embedding_padding_size - t5_text_embeddings_seq_length,
                ),
            )
        t5_text_mask = torch.ones(t5_text_embeddings_seq_length, dtype=torch.bfloat16)

        if is_image:
            h, w = info['image_height'], info['image_width']
            fps = torch.tensor([30] * 1, dtype=torch.bfloat16)
            num_frames = torch.tensor([1] * 1, dtype=torch.bfloat16)
        else:
            h, w = info['height'], info['width']
            fps = torch.tensor([info['framerate']] * 1, dtype=torch.bfloat16)
            num_frames = torch.tensor([info['num_frames']] * 1, dtype=torch.bfloat16)
        image_size = torch.tensor([[h, w, h, w]] * 1, dtype=torch.bfloat16)

        pos_ids = rearrange(
            pos_id_3d.get_pos_id_3d(t=T // self.patch_temporal, h=H // self.patch_spatial, w=W // self.patch_spatial),
            'T H W d -> (T H W) d',
        )

        if self.seq_length is not None and self.max_seq_length is None:
            pos_ids = F.pad(pos_ids, (0, 0, 0, self.seq_length - seq_len))
            loss_mask = torch.zeros(self.seq_length, dtype=torch.bfloat16)
            loss_mask[:seq_len] = 1
            video_latent = F.pad(video_latent, (0, 0, 0, self.seq_length - seq_len))
        else:
            loss_mask = torch.ones(seq_len, dtype=torch.bfloat16)

        return DiffusionSample(
            __key__=sample['__key__'],
            __restore_key__=sample['__restore_key__'],
            __subflavor__=None,
            __subflavors__=sample['__subflavors__'],
            video=video_latent,
            t5_text_embeddings=t5_text_embeddings,
            t5_text_mask=t5_text_mask,
            image_size=image_size,
            fps=fps,
            num_frames=num_frames,
            loss_mask=loss_mask,
            seq_len_q=torch.tensor(seq_len, dtype=torch.int32),
            seq_len_kv=torch.tensor(self.text_embedding_padding_size, dtype=torch.int32),
            pos_ids=pos_ids,
            latent_shape=torch.tensor([C, T, H, W], dtype=torch.int32),
        )

    def select_samples_to_pack(self, samples: List[DiffusionSample]) -> List[List[DiffusionSample]]:
        """
        Selects sequences to pack for mixed image-video training.
        """
        results = first_fit_decreasing(samples, self.max_seq_length)
        random.shuffle(results)
        return results

    @stateless
    def pack_selected_samples(self, samples: List[DiffusionSample]) -> DiffusionSample:
        """Construct a new Diffusion sample by concatenating the sequences."""

        def stack(attr):
            return torch.stack([getattr(sample, attr) for sample in samples], dim=0)

        def cat(attr):
            return torch.cat([getattr(sample, attr) for sample in samples], dim=0)

        video = concat_pad([i.video for i in samples], self.max_seq_length)
        loss_mask = concat_pad([i.loss_mask for i in samples], self.max_seq_length)
        pos_ids = concat_pad([i.pos_ids for i in samples], self.max_seq_length)

        return DiffusionSample(
            __key__=",".join([s.__key__ for s in samples]),
            __restore_key__=(),  # Will be set by energon based on `samples`
            __subflavor__=None,
            __subflavors__=samples[0].__subflavors__,
            video=video,
            t5_text_embeddings=cat('t5_text_embeddings'),
            t5_text_mask=cat('t5_text_mask'),
            # image_size=stack('image_size'),
            # fps=stack('fps'),
            # num_frames=stack('num_frames'),
            loss_mask=loss_mask,
            seq_len_q=stack('seq_len_q'),
            seq_len_kv=stack('seq_len_kv'),
            pos_ids=pos_ids,
            latent_shape=stack('latent_shape'),
        )

    @stateless
    def batch(self, samples: List[DiffusionSample]) -> dict:
        """Return dictionary with data for batch."""
        if self.max_seq_length is None:
            # no packing
            return super().batch(samples).to_dict()

        # packing
        sample = samples[0]
        return dict(
            video=sample.video.unsqueeze_(0),
            t5_text_embeddings=sample.t5_text_embeddings.unsqueeze_(0),
            t5_text_mask=sample.t5_text_mask.unsqueeze_(0),
            loss_mask=sample.loss_mask.unsqueeze_(0),
            # image_size=sample.image_size,
            # fps=sample.fps,
            # num_frames=sample.num_frames,
            # padding_mask=sample.padding_mask.unsqueeze_(0),
            seq_len_q=sample.seq_len_q,
            seq_len_kv=sample.seq_len_kv,
            pos_ids=sample.pos_ids.unsqueeze_(0),
            latent_shape=sample.latent_shape,
        )


class PosID3D:
    """
    Generates 3D positional IDs for video data.

    Attributes:
        max_t (int): Maximum number of time frames.
        max_h (int): Maximum height dimension.
        max_w (int): Maximum width dimension.
    """

    def __init__(self, *, max_t=32, max_h=128, max_w=128):
        self.max_t = max_t
        self.max_h = max_h
        self.max_w = max_w
        self.generate_pos_id()

    def generate_pos_id(self):
        """Generates a grid of positional IDs based on max_t, max_h, and max_w."""
        self.grid = torch.stack(
            torch.meshgrid(
                torch.arange(self.max_t, device='cpu'),
                torch.arange(self.max_h, device='cpu'),
                torch.arange(self.max_w, device='cpu'),
            ),
            dim=-1,
        )

    def get_pos_id_3d(self, *, t, h, w):
        """Retrieves positional IDs for specified dimensions."""
        if t > self.max_t or h > self.max_h or w > self.max_w:
            self.max_t = max(self.max_t, t)
            self.max_h = max(self.max_h, h)
            self.max_w = max(self.max_w, w)
            self.generate_pos_id()
        return self.grid[:t, :h, :w]


def pad_divisible(x, padding_value=0):
    """
    Pads the input tensor to make its size divisible by a specified value.

    Args:
        x (torch.Tensor): Input tensor.
        padding_value (int): The value to make the tensor size divisible by.

    Returns:
        torch.Tensor: Padded tensor.
    """
    if padding_value == 0:
        return x
    # Get the size of the first dimension
    n = x.size(0)

    # Compute the padding needed to make the first dimension divisible by 16
    padding_needed = (padding_value - n % padding_value) % padding_value

    if padding_needed <= 0:
        return x

    # Create a new shape with the padded first dimension
    new_shape = list(x.shape)
    new_shape[0] += padding_needed

    # Create a new tensor filled with zeros
    x_padded = torch.zeros(new_shape, dtype=x.dtype, device=x.device)

    # Assign the original tensor to the beginning of the new tensor
    x_padded[:n] = x
    return x_padded


def concat_pad(tensor_list, max_seq_length):
    """
    Efficiently concatenates a list of tensors along the first dimension and pads with zeros
    to reach max_seq_length.

    Args:
        tensor_list (list of torch.Tensor): List of tensors to concatenate and pad.
        max_seq_length (int): The desired size of the first dimension of the output tensor.

    Returns:
        torch.Tensor: A tensor of shape [max_seq_length, ...], where ... represents the remaining dimensions.
    """
    import torch

    # Get common properties from the first tensor
    other_shape = tensor_list[0].shape[1:]
    dtype = tensor_list[0].dtype
    device = tensor_list[0].device

    # Initialize the result tensor with zeros
    result = torch.zeros((max_seq_length, *other_shape), dtype=dtype, device=device)

    current_index = 0
    for tensor in tensor_list:
        length = tensor.shape[0]
        # Directly assign the tensor to the result tensor without checks
        result[current_index : current_index + length] = tensor
        current_index += length

    return result


pos_id_3d = PosID3D()


def cook_raw_iamges(sample: dict) -> dict:
    """
    Processes a raw sample dictionary from energon dataset and returns a new dictionary with specific keys.

    Args:
        sample (dict): The input dictionary containing the raw sample data.

    Returns:
        dict: A new dictionary containing the processed sample data with the following keys:
            - All keys from the result of `basic_sample_keys(sample)`
            - 'jpg': original images
            - 'png': contains control images
            - 'txt': contains raw text
    """
    return dict(
        **basic_sample_keys(sample),
        images=sample['jpg'],
        hint=sample['png'],
        txt=sample['txt'],
    )


class RawImageDiffusionTaskEncoder(DefaultTaskEncoder, IOMixin):
    '''
    Dummy task encoder takes raw image input on CrudeDataset.
    '''

    cookers = [
        # Cooker(cook),
        Cooker(cook_raw_iamges),
    ]


def cook_image_masks_with_precached_captions(sample: dict) -> dict:

    return dict(
        **basic_sample_keys(sample),
        image=sample['jpg'],
        mask=sample['single_mask'],
        caption=sample['caption'],
        text_ids=sample['text_ids'],
        prompt_embeds=sample['prompt_embeds'],
        pooled_prompt_embeds=sample['pooled_prompt_embeds'],
        new_caption=sample['new_caption'],
        aesthetic_score=sample['aesthetic_score'],
    )


class PrecachedCaptionWithImageMaskTaskEncoder(DefaultTaskEncoder, IOMixin):
    '''
    Dummy task encoder takes raw image input on CrudeDataset.
    '''

    cookers = [
        # Cooker(cook),
        Cooker(cook_image_masks_with_precached_captions),
    ]

    def __init__(
        self,
        do_cropping: bool = True,
        target_resolutions: list[tuple[int, int]] | None = None,
        p_outpainting_mask: float = 0.5,
        p_empty_prompts: float = 0.15,
        seed: int = 42,
    ):
        super().__init__()
        self.do_cropping = do_cropping
        
        self.target_resolutions = target_resolutions # (w, h)
        if self.target_resolutions is None:
            self.target_resolutions = [
                (672, 1568),
                (688, 1504),
                (720, 1456),
                (752, 1392),
                (800, 1328),
                (832, 1248),
                (880, 1184),
                (944, 1104),
                (1024, 1024),
                (1104, 944),
                (1184, 880),
                (1248, 832),
                (1328, 800),
                (1392, 752),
                (1456, 720),
                (1504, 688),
                (1568, 672),
            ]
        
        self.aspect_ratios = [bucket[0] / bucket[1] for bucket in self.target_resolutions]
        self.p_outpainting_mask = p_outpainting_mask
        self.p_empty_prompts = p_empty_prompts

        random.seed(seed)
        np.random.seed(seed)

    @stateless(restore_seeds=True)
    def encode_sample(self, sample: dict) -> dict:
        image = sample['image']
        text_ids = torch.load(io.BytesIO(sample['text_ids']), map_location=torch.device('cpu'))
        pooled_prompt_embeds = torch.load(io.BytesIO(sample['pooled_prompt_embeds']), map_location=torch.device('cpu'))
        prompt_embeds = torch.load(io.BytesIO(sample['prompt_embeds']), map_location=torch.device('cpu'))
        new_caption = sample['new_caption'].decode("utf-8")
        caption = sample['caption'].decode("utf-8")

        width, height = image.size

        if random.random() < self.p_outpainting_mask:
            mask = self.create_outpainting_mask(height, width)
        else:
            mask = self.random_brush_gen(height, width)

        mask = Image.fromarray(mask)

        # Find the index of the closest aspect ratio to the aspect ratio of the current image
        asp_ratio = width / height
        closest_idx = min(range(len(self.aspect_ratios)), key=lambda i: abs(self.aspect_ratios[i] - asp_ratio))
        
        target_aspect = self.aspect_ratios[closest_idx]
        target_w, target_h = self.target_resolutions[closest_idx]

        # Resize such that the image is greater than or equal to the bucket in both dimensions
        if asp_ratio > target_aspect:
            new_h = target_h
            new_w = int(math.ceil(asp_ratio * new_h))
        else:
            new_w = target_w
            new_h = int(math.ceil(new_w / asp_ratio))

        image = image.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)
        mask = mask.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)

        if self.do_cropping:
            image, mask = self.random_crop_with_mask(image, mask, (target_h, target_w))

        mask = ImageOps.invert(mask).convert('RGB')

        to_tensor = transforms.ToTensor()
        image = transforms.Normalize([0.5], [0.5])(to_tensor(image))
        mask = to_tensor(mask)

        if random.random() < self.p_empty_prompts:
            prompt_embeds.zero_()
            pooled_prompt_embeds.zero_()

        return dict(
            images=image,
            hint=mask,
            text_ids=text_ids,
            pooled_prompt_embeds=pooled_prompt_embeds,
            prompt_embeds=prompt_embeds,
            txt=new_caption,
            caption=caption)

    def decode_mask(self, mask:list, height: int, width: int):
        mask = np.array(mask)
        starts, lengths = [np.asarray(x, dtype=int) for x in (mask[0:][::2], mask[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(height*width, dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return np.clip(img.reshape((height, width), order='F') * 255, 0, 255)

    def create_outpainting_mask(
        self,
        height: int,
        width: int,
        min_percentage: float = 0.1,
        max_percentage: float = 0.6,
    ) -> np.ndarray:
        """
        Generates a random outpainting mask for a given image size.

        The mask is black in the center and has white paddings 
        on two opposite sides (either top/bottom or left/right).
        The direction and size of the padding are randomized based on the
        provided percentage range.

        Args:
            width (int): The width of the mask.
            height (int): The height of the mask.
            min_percentage (float): The minimum percentage (0.0 to 1.0) of the
                                    relevant dimension to use for padding.
            max_percentage (float): The maximum percentage (0.0 to 1.0) of the
                                    relevant dimension to use for padding.

        Returns:
            np.ndarray: A numpy array of shape (height, width) and dtype uint8,
                        representing the outpainting mask.
        """
        # Ensure percentages are within a valid range
        min_percentage = max(0.0, min_percentage)
        max_percentage = min(1.0, max_percentage)

        # Create a black mask (value 0) with the target dimensions.
        mask = np.zeros((height, width), dtype=np.uint8)

        # Randomly choose the outpainting direction
        direction = random.choice(['horizontal', 'vertical'])

        # Randomly sample the padding percentage from the given range
        padding_percentage = random.uniform(min_percentage, max_percentage)

        if direction == 'horizontal':
            # Outpaint on the left and right sides
            total_padding_width = int(width * padding_percentage)
            
            padding_left = total_padding_width // 2
            padding_right = total_padding_width - padding_left

            if padding_left > 0:
                mask[:, :padding_left] = 255
            if padding_right > 0:
                mask[:, -padding_right:] = 255
        
        else: # direction == 'vertical'
            # Outpaint on the top and bottom sides
            total_padding_height = int(height * padding_percentage)

            padding_top = total_padding_height // 2
            padding_bottom = total_padding_height - padding_top

            if padding_top > 0:
                mask[:padding_top, :] = 255
            if padding_bottom > 0:
                mask[-padding_bottom:, :] = 255

        return mask

    def random_brush_gen(
        self,
        h,
        w,
        max_starting_points = 3,
        min_num_vertex = 1,
        max_num_vertex = 6,
        mean_angle = 5,
        angle_range = 15,
        min_width = 80,
        max_width = 250,
        radius_divider = 8,
    ):
        mean_angle = 2 * math.pi / mean_angle
        angle_range = 2 * math.pi / angle_range
        H, W = h, w
        average_radius = math.sqrt(H*H+W*W) / radius_divider
        mask = Image.new('L', (W, H), 0)
        num_starting_points = np.random.randint(1, max_starting_points + 1)
        for _ in range(num_starting_points):
            num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
            angle_min = mean_angle - np.random.uniform(0, angle_range)
            angle_max = mean_angle + np.random.uniform(0, angle_range)
            angles = []
            vertex = []
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
                else:
                    angles.append(np.random.uniform(angle_min, angle_max))

            w, h = mask.size
            vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
            for i in range(num_vertex):
                r = np.clip(
                    np.random.normal(loc=average_radius, scale=average_radius//2),
                    0, 2*average_radius)
                new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
                vertex.append((int(new_x), int(new_y)))

            draw = ImageDraw.Draw(mask)
            width = int(np.random.uniform(min_width, max_width))
            draw.line(vertex, fill=1, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width//2,
                            v[1] - width//2,
                            v[0] + width//2,
                            v[1] + width//2),
                            fill=1)
            if np.random.random() > 0.5:
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if np.random.random() > 0.5:
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        mask = np.asarray(mask, np.uint8)

        return np.clip((mask * 255).astype(np.uint8), 0, 255)
    
    def random_crop_with_mask(self, image: Image.Image, mask: Image.Image, crop_size: tuple[int, int]):
        """
        Randomly crops an image and mask pair such that the object in the mask is entirely within the cropped region.

        Args:
            image (PIL.Image.Image): The input image to be cropped.
            mask (PIL.Image.Image): The binary mask corresponding to the image.
            crop_size (tuple[int, int]): The desired crop size (height, width).

        Returns:
            cropped_image (PIL.Image.Image): The cropped image.
            cropped_mask (PIL.Image.Image): The cropped mask.
        """
        image = np.array(image)
        mask = np.array(mask)

        # Ensure the crop size is not larger than the input size
        assert image.shape[:2] >= crop_size, "Crop size must be smaller than the image size."

        crop_h, crop_w = crop_size
        img_h, img_w = image.shape[:2]

        # Find the bounding box of the object in the mask
        x_min, y_min, obj_width, obj_height = cv2.boundingRect(mask)
        x_max = x_min + obj_width - 1
        y_max = y_min + obj_height - 1

        # Ensure the bounding box fits within the crop
        crop_y_min = max(0, y_max - crop_h + 1)
        crop_x_min = max(0, x_max - crop_w + 1)
        crop_y_max = min(y_min, img_h - crop_h)
        crop_x_max = min(x_min, img_w - crop_w)
        
        if not (crop_y_max >= crop_y_min and crop_x_max >= crop_x_min):
            resized_image = Image.fromarray(image).resize((crop_w, crop_h), resample=Image.Resampling.BICUBIC)
            resized_mask = Image.fromarray(mask).resize((crop_w, crop_h), resample=Image.Resampling.BICUBIC)

            return resized_image, resized_mask

        # Randomly select a valid crop position
        top = np.random.randint(crop_y_min, crop_y_max + 1)
        left = np.random.randint(crop_x_min, crop_x_max + 1)

        # Perform the cropping
        cropped_image = image[top:top + crop_h, left:left + crop_w]
        cropped_mask = mask[top:top + crop_h, left:left + crop_w]

        cropped_image = Image.fromarray(cropped_image)
        cropped_mask = Image.fromarray(cropped_mask)

        return cropped_image, cropped_mask

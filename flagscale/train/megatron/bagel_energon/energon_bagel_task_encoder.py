import os
import sys
import traceback
import random
from PIL import Image
from typing import Dict, List, Union, Iterable, Tuple, Optional, Protocol, Any
from functools import lru_cache

import torch

from megatron.training import get_args, get_tokenizer
from megatron.core import parallel_state, tensor_parallel
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_rank,
)

from megatron.energon import TaskEncoder, stateless
from megatron.energon.task_encoder.cooking import Cooker
from megatron.energon import (
    LimitDataset,
    RepeatDataset,
    WorkerConfig,
    get_loader,
    get_savable_loader,
    get_train_dataset,
    get_val_datasets,
)

from flagscale.train.megatron.bagel_energon.data_utils import pil_img2rgb, get_flattened_position_ids_extrapolate, get_flattened_position_ids_interpolate
from flagscale.train.megatron.bagel_energon.transforms import ImageTransform, AudioTransform
from flagscale.train.megatron.bagel_energon.video_utils import FrameSampler

from .packing import make_sequence_status, pack_sequence, to_tensor
from .sample_types import BagelPackedBatch, BagelSample
from .transforms import ImageTransform
from .task_handlers import TASK_REGISTRY
from .cooker import video_cooker, image_cooker


class BagelDataConfig:
    """Configuration for Bagel data processing."""

    def __init__(
        self,
        text_cond_dropout_prob=0.0,
        vit_cond_dropout_prob=0.0,
        vae_cond_dropout_prob=0.0,
        vae_image_downsample=16,
        max_latent_size=64,
        vit_patch_size=14,
        max_num_patch_per_side=70,
        max_num_tokens=32768,
        expected_num_tokens=31000,
        max_num_tokens_per_sample=16384,
    ):
        self.text_cond_dropout_prob = text_cond_dropout_prob
        self.vit_cond_dropout_prob = vit_cond_dropout_prob
        self.vae_cond_dropout_prob = vae_cond_dropout_prob
        self.vae_image_downsample = vae_image_downsample
        self.vit_patch_size = vit_patch_size
        self.max_latent_size = max_latent_size
        self.max_num_patch_per_side = max_num_patch_per_side
        self.max_num_tokens = max_num_tokens
        self.expected_num_tokens = expected_num_tokens
        self.max_num_tokens_per_sample = max_num_tokens_per_sample


@lru_cache(maxsize=16)
def _get_image_transform(
    max_image_size,
    min_image_size,
    image_stride,
    max_pixels=14*14*9*1024,
    image_mean=[0.5, 0.5, 0.5],
    image_std=[0.5, 0.5, 0.5],
):
    return ImageTransform(
        max_image_size=max_image_size,
        min_image_size=min_image_size,
        image_stride=image_stride,
        max_pixels=max_pixels,
        image_mean=image_mean,
        image_std=image_std,
    )


@lru_cache(maxsize=16)
def _get_frame_sampler(
    max_num_frames=-1,
    min_num_frames=8,
    sample='rand',
):
    return FrameSampler(
        max_num_frames=max_num_frames,
        min_num_frames=min_num_frames,
        sample=sample,
    )


class BagelTaskEncoder(
    TaskEncoder[
        Dict[str, Any],
        dict,
        dict,
        dict
    ]
):
    """Energon TaskEncoder for Bagel multimodal model.

    Pipeline:
      1. encode_sample: raw dict → BagelSample (transform + tokenize + build sequence_plan)
      2. select_samples_to_pack: buffer of BagelSamples → groups to pack together
      3. pack_selected_samples: group of BagelSamples → BagelPackedBatch
    """

    cookers = [
        video_cooker,  # if subflavors task in ('video_sft', 'video_qa', 'vlm_video')
        image_cooker,  # fallback
    ]

    def __init__(
        self,
        data_config: BagelDataConfig,
        interpolate_pos: bool = False,
        max_seq_length: Optional[int] = None,
    ):
        super().__init__()
        self.tokenizer = get_tokenizer()
        self.special_tokens = self.tokenizer.new_special_token_ids
        print(f"{self.special_tokens=}")
        self.data_config = data_config
        self.group_size = max_seq_length if max_seq_length is not None else 4096

        self.handlers = {
            name: cls(self.tokenizer, self.special_tokens, self.data_config)
            for name, cls in TASK_REGISTRY.items()
        }

        if interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

        # Read parallelism settings directly from training args (these live in TransformerConfig).
        _args = get_args()
        self._cp_size = getattr(_args, 'context_parallel_size', 1)
        self._tp_size = getattr(_args, 'tensor_model_parallel_size', 1)
        self._sequence_parallel = getattr(_args, 'sequence_parallel', False)

    def _build_transforms(self, subflavors):
        transforms_item = {}
        image_args = subflavors.get('image_transform_args')
        if image_args:
            transforms_item["transform"] = _get_image_transform(
                max_image_size=image_args['max_image_size'],
                min_image_size=image_args['min_image_size'],
                image_stride=image_args['image_stride'],
                max_pixels=image_args.get('max_pixels', 14*14*9*1024),
            )
        vit_args = subflavors.get('vit_image_transform_args')
        if vit_args:
            transforms_item["vit_transform"] = _get_image_transform(
                max_image_size=vit_args['max_image_size'],
                min_image_size=vit_args['min_image_size'],
                image_stride=vit_args['image_stride'],
                max_pixels=vit_args.get('max_pixels', 14*14*9*1024),
            )
        frame_args = subflavors.get('frame_sampler_args')
        if frame_args:
            transforms_item["frame_sampler"] = _get_frame_sampler(
                max_num_frames=frame_args.get('max_num_frames', -1),
                min_num_frames=frame_args.get('min_num_frames', 8),
                sample=frame_args.get('sample', 'rand'),
            )

        return transforms_item

    def select_samples_to_pack(self, samples: List[BagelSample]) -> List[List[Dict[str, torch.Tensor]]]:
        """Select samples from buffer to form packs.

        Implements Bagel's packing strategy (mirrors dataset_base.py __iter__):
        - Each pack starts by consuming all mandatory samples first
        - Then greedily fills with non-mandatory samples
        - Uses expected_num_tokens as the "pack is ready" threshold
        - Uses max_num_tokens as the hard upper limit
        - Samples exceeding max_num_tokens_per_sample are skipped
        - Overflow samples go into a buffer for priority use in next pack

        Selects which samples will be packed together.

        This function receives a list of samples (size according to the selected packing_buffer_size), 
        and partitions those samples into groups that shall be packed together.

        Args:
            samples (List[Dict[str, torch.Tensor]]): List of samples from the buffer, each containing
                tokenized data with keys like 'input_ids', 'labels', 'loss_mask', etc.

        Returns:
            List[List[Dict[str, torch.Tensor]]]: List of groups, where each group is a list of samples
                that should be packed together. Each group's total length will not exceed group_size.

        NOTE: Energon dataloader calls this method internally if packing is used.
        Please see https://nvidia.github.io/Megatron-Energon/advanced/packing.html
        """
        max_tokens = self.data_config.max_num_tokens
        max_per_sample = self.data_config.max_num_tokens_per_sample
        expected = self.data_config.expected_num_tokens

        # Filter out oversized samples
        valid_samples = [s for s in samples if s.num_tokens <= max_per_sample]
        print(f"{len(valid_samples)=}, {valid_samples=}")
        if not valid_samples:
            return []

        # # Separate mandatory and non-mandatory
        # mandatory = [s for s in valid_samples if s.is_mandatory]
        # non_mandatory = [s for s in valid_samples if not s.is_mandatory]

        packs = []
        current_pack = []
        current_tokens = 0

        # # Start each pack with a mandatory sample if available
        # if mandatory:
        #     m_sample = mandatory.pop(0)
        #     current_pack.append(m_sample)
        #     current_tokens = m_sample.num_tokens + 2 * len(m_sample.sequence_plan)

        # # Fill with remaining samples (sorted by size for better packing)
        # remaining = mandatory + non_mandatory
        # random.shuffle(remaining)

        for sample in samples:
            sample_tokens = sample.num_tokens + 2 * len(sample.sequence_plan)
            if current_tokens + sample_tokens <= max_tokens:
                current_pack.append(sample)
                current_tokens += sample_tokens
            elif current_pack:
                # Current pack is full, start a new one
                packs.append(current_pack)
                current_pack = [sample]
                current_tokens = sample_tokens

        if current_pack:
            packs.append(current_pack)

        print(f"{len(packs)=}, {packs=}")
        return packs

    @stateless
    def pack_selected_samples(self, samples: List[BagelSample]) -> BagelPackedBatch:
        """Pack a group of BagelSamples into a single BagelPackedBatch.

        This is the core packing logic from Bagel's PackedDataset.
        """
        sequence_status = make_sequence_status()

        for sample in samples:
            sequence_status = pack_sequence(
                sample=sample,
                sequence_status=sequence_status,
                special_tokens=self.special_tokens,
                data_config=self.data_config,
                get_flattened_position_ids=self.get_flattened_position_ids,
            )

        return to_tensor(sequence_status, self.data_config.max_num_tokens)

    @stateless
    def encode_sample(self, sample: Dict[str, Any]):
        """encode raw dict sample.

        Dispatches based on subflavors['task']:
          - 'vlm': VLM SFT data (jsonl conversations + images)
          - 't2i': Text-to-image data (parquet image + caption)
        """
        print(f"{sample=}")
        subflavors = sample.get('__subflavors__', {})
        task = subflavors.get('task', None)
        print(f"{subflavors=}")

        kwargs = self._build_transforms(subflavors)
        handler = self.handlers.get(task)

        if handler is None:
            raise ValueError(f"Unknown task: {task}. Available: {list(self.handlers.keys())}")
        return handler.encode(sample, **kwargs)

    def batch(self, samples: List[BagelPackedBatch]):
        """batch_size=1 for Bagel, so just return the single packed batch."""
        if not samples:
            return {}

        assert len(samples) == 1, "Bagel uses batch_size=1 with packing"

        if self._cp_size > 1:
            print("")

        return samples[0].to_dict()

    def encode_batch(self, batch: BagelPackedBatch) -> BagelPackedBatch:
        """No additional batch encoding needed."""
        return batch


def bagel_vlm_dataloader_provider(train_val_test_num_samples, max_seq_length: Optional[int] = None):
    args = get_args()

    bagel_config = BagelDataConfig(
        text_cond_dropout_prob=getattr(args, 'text_cond_dropout_prob', 0.1),
        vit_cond_dropout_prob=getattr(args, 'vit_cond_dropout_prob', 0.4),
        vae_cond_dropout_prob=getattr(args, 'vae_cond_dropout_prob', 0.1),
        vae_image_downsample=getattr(args, 'vae_image_downsample', 16),
        vit_patch_size=getattr(args, 'vit_patch_size', 14),
        max_latent_size=getattr(args, 'max_latent_size', 64),
        max_num_patch_per_side=getattr(args, 'max_num_patch_per_side', 70),
        max_num_tokens=getattr(args, 'max_num_tokens', 16384), # 36864
        expected_num_tokens=getattr(args, 'expected_num_tokens', 16384), # 32768
        max_num_tokens_per_sample=getattr(args, 'max_num_tokens_per_sample', 16384),
    )

    return train_valid_test_dataloaders_provider(
        train_val_test_num_samples,
        task_encoder=BagelTaskEncoder(
            data_config=bagel_config,
            interpolate_pos=args.interpolate_pos,
            max_seq_length=max_seq_length,
        )
    )


def is_first_or_last_stage(pp_size):
    """Check if the current pipeline parallel stage is the first or last stage."""
    if pp_size == 1:    # No pipeline parallelism.
        return True

    # With no separate pipeline stage for the vision model (epp=0), 
    # run the dataloader on the first and last pipeline stage.
    pp_rank = get_pipeline_model_parallel_rank()
    is_valid_rank = pp_rank in (0, pp_size-1)

    return is_valid_rank


def is_dataloader_rank():
    """Check if we should have the dataloader on this tensor and pipeline parallel rank."""
    # Run dataloader only on the first tensor parallel rank (will be broadcasted to others).
    is_first_rank = get_tensor_model_parallel_rank() == 0

    pp_size = get_pipeline_model_parallel_world_size()
    is_first_rank = is_first_rank and is_first_or_last_stage(pp_size)

    return is_first_rank


def print_error_handler(exc: Exception, key: Optional[str]):
    print(
        f"The following exception occurred in the dataloader for sample {key} and is skipped",
        file=sys.stderr,
    )
    traceback.print_exc()


def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x


class EnergonDataloader:
    """A wrapper to use Megatron Energon dataloader with the Megatron-LM training loop."""
    def __init__(self, dataloader):
        self._dataloader = dataloader
        self._iter = iter(cyclic_iter(dataloader))

    def __next__(self):
        return self._iter.__next__()

    def __iter__(self):
        return self._iter.__iter__()

    def save_state(self):
        return self._dataloader.save_state_rank()


def train_valid_test_dataloaders_provider(train_val_test_num_samples, task_encoder=None):

    args = get_args()

    # Dataloader is only on specific ranks.
    if not is_dataloader_rank():
        return None, None, None

    tokenizer = get_tokenizer()
    worker_debug_path = None
    worker_log_level = 0

    rank = parallel_state.get_data_parallel_rank()
    world_size = parallel_state.get_data_parallel_world_size()
    data_parallel_group = parallel_state.get_data_parallel_group()

    worker_config = WorkerConfig(
        rank=rank,
        world_size=world_size,
        num_workers=args.num_workers,
        data_parallel_group=data_parallel_group,
        worker_debug_path=worker_debug_path,
        worker_log_level=worker_log_level,
    )

    # task_encoder = BagelTaskEncoder(
    #     tokenizer=tokenizer,
    #     special_tokens=tokenizer.new_special_token_ids,
    #     data_config=bagel_config,
    # )
    # task_encoder=BagelTaskEncoder(
    #     data_config=bagel_config,
    #     interpolate_pos=args.interpolate_pos,
    #     max_seq_length=max_seq_length,
    # )

    # # Build dataset paths with weights
    # dataset_configs = getattr(args, 'datasets', [])
    # if not dataset_configs:
    #     raise ValueError("data_config must contain 'datasets' list")
    dname = args.data_path[0] if type(args.data_path) is list else args.data_path

    # For single dataset
    # if len(dataset_configs) == 1:
    dataset = get_train_dataset(
        dname,
        batch_size=args.micro_batch_size,
        task_encoder=task_encoder,
        virtual_epoch_length=1000,
        max_samples_per_sequence=getattr(args, 'max_samples_per_sequence', None),
        shuffle_buffer_size=getattr(args, 'shuffle_buffer_size', 1000),
        worker_config=worker_config,
        packing_buffer_size=getattr(args, 'packing_buffer_size', 12),
        handler=print_error_handler,
        image_decode="pil",
    )
    # else:
    #     # For blended datasets
    #     blend_config = []
    #     for ds_cfg in dataset_configs:
    #         blend_config.append({
    #             'path': ds_cfg['path'],
    #             'weight': ds_cfg.get('weight', 1.0),
    #             'subflavors': ds_cfg.get('subflavors', {}),
    #         })
    #     dataset = get_train_dataset(
    #         blend_config,
    #         worker_config=worker_config,
    #         batch_size=args.micro_batch_size,
    #         task_encoder=task_encoder,
    #         max_samples_per_sequence=getattr(args, 'max_samples_per_sequence', None),
    #         shuffle_buffer_size=getattr(args, 'shuffle_buffer_size', 1000),
    #     )

    # Build savable dataloader
    dataloader = get_savable_loader(
        dataset,
        worker_config=worker_config,
    )

    return EnergonDataloader(dataloader), None, None

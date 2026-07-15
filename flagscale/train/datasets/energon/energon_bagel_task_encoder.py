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

from flagscale.train.datasets.energon.data_utils import pil_img2rgb, get_flattened_position_ids_extrapolate, get_flattened_position_ids_interpolate
from flagscale.train.datasets.energon.transforms import ImageTransform, AudioTransform
from flagscale.train.datasets.energon.video_utils import FrameSampler
from flagscale.train.datasets.energon.packing import make_sequence_status, pack_sequence, to_tensor
from flagscale.train.datasets.energon.sample_types import BagelPackedBatch, BagelSample
from flagscale.train.datasets.energon.transforms import ImageTransform
from flagscale.train.datasets.energon.task_handlers import TASK_REGISTRY
from flagscale.train.datasets.energon.cooker import video_cooker, image_cooker


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
        max_num_tokens=36864,
        expected_num_tokens=32768,
        max_num_tokens_per_sample=16384,
        max_buffer_size=50,
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
        self.max_buffer_size = max_buffer_size


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
    ):
        super().__init__()
        self.tokenizer = get_tokenizer()
        self.special_tokens = self.tokenizer.new_special_token_ids
        print(f"{self.special_tokens=}")
        self.data_config = data_config

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

        # Overflow buffer: samples that didn't fit in the previous pack are held
        # here and prepended to the next select_samples_to_pack call so that no
        # sample is wasted and every yielded pack meets expected_num_tokens.
        self._overflow_buffer: List = []

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

    def select_samples_to_pack(self, samples: List[BagelSample]) -> List[List[BagelSample]]:
        """Select samples from buffer to form packs with overflow management.

        Mirrors the original Bagel PackedDataset.__iter__ packing strategy:
        - Prepend overflow samples from the previous call (like the original buffer)
        - Uses expected_num_tokens as the soft threshold to yield a pack
        - Uses max_num_tokens as the hard upper limit per pack
        - Samples exceeding max_num_tokens_per_sample are skipped
        - If the last pack doesn't reach expected_num_tokens, its samples are
          held in the overflow buffer for the next call (no short batches)

        Args:
            samples: List of samples from Energon's reading buffer.

        Returns:
            List of groups where each group is a list of samples to pack together.
            Every returned pack is guaranteed to have >= expected_num_tokens
            (except when the data source is exhausted).

        NOTE: Energon dataloader calls this method internally if packing is used.
        Please see https://nvidia.github.io/Megatron-Energon/advanced/packing.html
        """
        # --- Step 1: Load packing configuration ---
        # max_tokens: hard ceiling for a single pack (prevents OOM)
        # max_per_sample: discard any sample longer than this
        # expected: soft target — once a pack reaches this, emit it
        # max_buffer_size: cap on how many "doesn't fit" samples we hold locally
        max_tokens = self.data_config.max_num_tokens
        max_per_sample = self.data_config.max_num_tokens_per_sample
        expected = self.data_config.expected_num_tokens
        max_buffer_size = self.data_config.max_buffer_size

        # --- Step 2: Merge overflow from previous call with new samples ---
        # Overflow samples are placed first so they get priority (equivalent to
        # the original code's "prefer_buffer_before" behavior where buffered
        # samples are consumed before drawing new ones from the data stream).
        all_samples = self._overflow_buffer + list(samples)
        self._overflow_buffer = []

        # --- Step 3: Filter out oversized samples ---
        # Samples exceeding max_num_tokens_per_sample are permanently discarded,
        # matching the original "skip a sample with length ..." behavior.
        # Token count includes +2 per segment in sequence_plan (bos/eos overhead).
        valid_samples = []
        for s in all_samples:
            token_count = s.num_tokens + 2 * len(s.sequence_plan)
            if token_count <= max_per_sample:
                valid_samples.append((s, token_count))

        if not valid_samples:
            return []

        # --- Step 4: Greedy bin-packing with overflow buffer ---
        # Walk through valid_samples one by one, trying to fit each into
        # current_pack. Three outcomes per sample:
        #   (a) Fits and pack not yet full → append to current_pack
        #   (b) Fits and pack reaches expected → emit pack, start fresh
        #   (c) Doesn't fit → stash in pending_candidates for later
        packs = []
        current_pack = []
        current_tokens = 0
        pending_candidates = []

        for sample, token_count in valid_samples:
            if current_tokens + token_count <= max_tokens:
                # Case (a)/(b): sample fits within the hard limit
                current_pack.append(sample)
                current_tokens += token_count

                if current_tokens >= expected:
                    # Pack reached the soft target — emit it
                    packs.append(current_pack)
                    current_pack = []
                    current_tokens = 0

                    # Drain pending_candidates into the fresh pack immediately.
                    # This gives previously-buffered (typically large) samples a
                    # chance to be placed while the new pack is still empty.
                    for pend_sample, pend_tokens in pending_candidates:
                        if current_tokens + pend_tokens <= max_tokens:
                            current_pack.append(pend_sample)
                            current_tokens += pend_tokens
                            if current_tokens >= expected:
                                packs.append(current_pack)
                                current_pack = []
                                current_tokens = 0
                        else:
                            # Still doesn't fit — persist to cross-call overflow
                            self._overflow_buffer.append(pend_sample)
                    pending_candidates = []
            else:
                # Case (c): sample would exceed the hard limit for current pack
                if len(pending_candidates) < max_buffer_size:
                    # Stash it; we'll try again after the current pack is emitted
                    pending_candidates.append((sample, token_count))
                else:
                    # Overflow buffer is full — force-emit the current pack even
                    # though it may not have reached `expected`. This matches the
                    # original behavior: "buffer full + can't fit → yield batch".
                    if current_pack:
                        packs.append(current_pack)
                    # Start a new pack with the sample that triggered the flush
                    current_pack = [sample]
                    current_tokens = token_count
                    # Try to fit pending_candidates into the new pack
                    for pend_sample, pend_tokens in pending_candidates:
                        if current_tokens + pend_tokens <= max_tokens:
                            current_pack.append(pend_sample)
                            current_tokens += pend_tokens
                        else:
                            self._overflow_buffer.append(pend_sample)
                    pending_candidates = []

        # --- Step 5: Handle leftover samples at the end of this call ---
        # current_pack is within max_tokens, but pending_candidates may push
        # the total over. Only emit if both conditions are met: total tokens
        # >= expected AND <= max_tokens. Otherwise, hold everything for next call.
        remaining = current_pack + [s for s, _ in pending_candidates]
        if remaining:
            remaining_tokens = sum(
                s.num_tokens + 2 * len(s.sequence_plan) for s in remaining
            )
            if remaining_tokens >= expected and remaining_tokens <= max_tokens:
                packs.append(remaining)
            else:
                self._overflow_buffer.extend(remaining)

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
        # print(f"{sample=}")
        subflavors = sample.get('__subflavors__', {})
        task = subflavors.get('task', None)
        # print(f"{subflavors=}")

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


def bagel_vlm_dataloader_provider(train_val_test_num_samples):
    args = get_args()

    bagel_config = BagelDataConfig(
        text_cond_dropout_prob=getattr(args, 'text_cond_dropout_prob', 0.1),
        vit_cond_dropout_prob=getattr(args, 'vit_cond_dropout_prob', 0.4),
        vae_cond_dropout_prob=getattr(args, 'vae_cond_dropout_prob', 0.1),
        vae_image_downsample=getattr(args, 'vae_image_downsample', 16),
        vit_patch_size=getattr(args, 'vit_patch_size', 14),
        max_latent_size=getattr(args, 'max_latent_size', 64),
        max_num_patch_per_side=getattr(args, 'max_num_patch_per_side', 70),
        max_num_tokens=getattr(args, 'max_num_tokens', 36864),
        expected_num_tokens=getattr(args, 'expected_num_tokens', 32768),
        max_num_tokens_per_sample=getattr(args, 'max_num_tokens_per_sample', 16384),
        max_buffer_size=getattr(args, 'max_buffer_size', 50),
    )

    return train_valid_test_dataloaders_provider(
        train_val_test_num_samples,
        task_encoder=BagelTaskEncoder(
            data_config=bagel_config,
            interpolate_pos=args.interpolate_pos,
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


def train_valid_test_dataloaders_provider(train_val_test_num_samples, task_encoder):

    args = get_args()

    # Dataloader is only on specific ranks.
    if not is_dataloader_rank():
        return None, None, None

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

    # Build dataset paths with weights
    assert (isinstance(args.data_path, list) and len(args.data_path) == 1) or \
        isinstance(args.data_path, str)
    dname = args.data_path[0] if isinstance(args.data_path, list) else args.data_path

    # For single dataset
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

    # Build savable dataloader
    dataloader = get_savable_loader(
        dataset,
        worker_config=worker_config,
    )

    return EnergonDataloader(dataloader), None, None

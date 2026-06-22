# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain BAGEL multimodal model for unified image understanding and generation."""
import sys
import warnings
import logging
import traceback
from copy import deepcopy
from functools import partial
from typing import Optional

import torch
from torch.nn.attention.flex_attention import create_block_mask

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.models.multimodal import context_parallel
from megatron.core.models.vision.clip_vit_model import get_num_image_embeddings
from megatron.core.models.vision.vit_layer_specs import (
    get_vit_layer_with_transformer_engine_spec,
)
from megatron.core.utils import log_single_rank
from megatron.training import (
    get_args,
    get_timers,
    get_tokenizer,
    pretrain,
    print_rank_0,
)
from megatron.training.arguments import core_transformer_config_from_args
from flagscale.models.megatron.bagel.bagel_model import BagelModel
from flagscale.models.megatron.bagel.bagel_spec import get_bagel_layer_with_transformer_engine_spec, get_mlp_module_spec
from flagscale.train.megatron.training.multimodal_args import add_multimodal_extra_args
from flagscale.models.megatron.bagel.config import get_vision_model_config, get_language_model_config
from flagscale.train.datasets.energon.energon_bagel_task_encoder import bagel_vlm_dataloader_provider


_DATASET_PROVIDERS = {
    "vlm": bagel_vlm_dataloader_provider,
}


def model_provider(
    pre_process=True, post_process=True, add_encoder=True, add_decoder=True, parallel_output=True,
    vp_stage=None, config=None, pg_collection=None
) -> BagelModel:
    """Build the BAGEL multimodal model.

    Args:
        pre_process (bool): Include the embedding layer in the gpt decoder (used with pipeline parallelism). Defaults to True.
        post_process (bool): Include an output layer and a layernorm in the gpt decoder (used with pipeline parallelism). Defaults to True.
        add_encoder (bool): Construct the encoder module (used with pipeline parallelism). Defaults to True. When we use pipelining, the encoder
            will live on only a subset of the pipeline stages (specifically, only the first stage).
        add_decoder (bool): Construct the decoder module (used with pipeline parallelism). Defaults to True. When we use pipelining, the decoder
            will live on only a subset of the pipeline stages (specifically, every stage after the first one).
        parallel_output (bool): Enable model parallel output.

    Returns:
        model (megatron.core.models.multimodal.bagel_model.BagelModel): A multimodal model
    """
    args = get_args()
    args.use_te = args.transformer_impl == "transformer_engine"

    assert args.use_te
    assert args.ckpt_format == 'torch', "Only ckpt-format torch is supported for VLM training currently."
    assert not (args.context_parallel_size > 1 and args.pipeline_model_parallel_size > 1), "PP+CP is not yet supported by this script. \
    Current mock dataset does not support natively packed sequence dataset required for correct PP comm shapes."

    num_image_embeddings = get_num_image_embeddings(
        args.image_size,
        args.image_size,
        args.patch_dim,
        args.vision_model_type,
        args.disable_vision_class_token,
        1,
        args.pixel_shuffle,
        args.use_tile_tags,
        args.max_num_tiles,
        args.tokenizer_prompt_format
    )
    old_seq_length = args.seq_length
    args.decoder_seq_length = args.max_position_embeddings
    # seq_length and encoder_seq_length denote the vision model sequence length. Override if the user provided something else.
    args.seq_length = args.encoder_seq_length = num_image_embeddings
    if torch.distributed.get_rank() == 0 and old_seq_length != args.seq_length:
        log_single_rank(
            logging.getLogger(__name__),
            logging.WARNING,
            f"Changed seq_length and encoder_seq_length (vision model sequence length) from {old_seq_length} to num_image_tokens ({num_image_embeddings})"
        )

    print_rank_0('building Bagel: a multimodal model ...')
    print_rank_0(f'{num_image_embeddings=}')

    # --- Language model layer spec (MoT) ---
    base_config = core_transformer_config_from_args(get_args())
    base_config.language_model_type = args.language_model_type
    base_config.vision_model_type = args.vision_model_type
    base_config.calculate_per_token_loss = True
    base_config.variable_seq_lengths = True

    if args.decoder_num_layers is not None:
        base_config.num_layers = args.decoder_num_layers
    else:
        base_config.num_layers = args.num_layers

    language_config = deepcopy(base_config)
    language_config = get_language_model_config(language_config)
    print(f"{language_config=}")

    language_transformer_layer_spec = get_bagel_layer_with_transformer_engine_spec(
        qk_layernorm=args.qk_layernorm
    )

    # --- Vision model config ---
    vision_config = deepcopy(base_config)
    vision_config = get_vision_model_config(
        vision_config, args.apply_query_key_layer_scaling
    )

    vision_config.first_pipeline_num_layers = None
    vision_config.last_pipeline_num_layers = None
    vision_config.context_parallel_size = 1 # Force CP=1 for Vision Transformer
    if vision_config.sequence_parallel:
        print_rank_0("> Disabling Sequence parallelism in Vision Transformer. Not yet supported")
        vision_config.sequence_parallel = False
    if vision_config.tp_comm_overlap:
        print_rank_0("> Disabling TP Comm overlap in Vision Transformer. Not yet supported")
        vision_config.tp_comm_overlap = False
    # Vision Encoder should live on PP rank0
    vision_config.pipeline_model_parallel_size = 1
    print(f"{vision_config=}")

    # --- Vision layer spec ---
    vision_transformer_layer_spec = get_vit_layer_with_transformer_engine_spec()

    # --- Vision projection config ---
    vision_projection_type = "mlp"
    vision_projection_config = deepcopy(base_config)
    vision_projection_config.context_parallel_size = 1
    if vision_projection_config.sequence_parallel:
        print_rank_0("> Disabling Sequence parallelism in Vision Projection. Not yet supported")
        vision_projection_config.sequence_parallel = False
    if vision_projection_config.tp_comm_overlap:
        print_rank_0("> Disabling TP Comm overlap in Vision Projection. Not yet supported")
        vision_projection_config.tp_comm_overlap = False
    # Projection should live on PP rank0
    vision_projection_config.pipeline_model_parallel_size = 1
    vision_projection_layer_spec = get_mlp_module_spec(use_te=args.use_te).submodules

    assert args.context_parallel_size == 1
    # language_max_sequence_length = args.decoder_seq_length
    # if args.context_parallel_size > 1:
    #     if args.use_packed_sequence or mp_padding_needed > 0:
    #         # Use THD data format
    #         language_max_sequence_length = args.decoder_seq_length * args.micro_batch_size

    print(f"{pre_process=}, {post_process=}, {add_encoder=}, {add_decoder=}")
    # --- Build model ---
    model = BagelModel(
        language_transformer_config=language_config,
        language_transformer_layer_spec=language_transformer_layer_spec,
        language_vocab_size=args.padded_vocab_size,
        language_max_sequence_length=args.decoder_seq_length,
        vision_transformer_config=vision_config,
        vision_transformer_layer_spec=vision_transformer_layer_spec,
        vision_projection_config=vision_projection_config,
        vision_projection_layer_spec=vision_projection_layer_spec,
        vision_projection_type=vision_projection_type,
        parallel_output=parallel_output,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        language_position_embedding_type=args.position_embedding_type,
        language_rotary_percent=args.rotary_percent,
        visual_gen=getattr(args, 'visual_gen', True),
        visual_und=getattr(args, 'visual_und', True),
        latent_patch_size=getattr(args, 'latent_patch_size', 2),
        max_latent_size=getattr(args, 'max_latent_size', 32),
        vit_max_num_patch_per_side=getattr(args, 'vit_max_num_patch_per_side', 70),
        latent_channel=getattr(args, 'latent_channel', 16),
        vae_downsample=getattr(args, 'vae_downsample', 16),
        timestep_shift=getattr(args, 'timestep_shift', 1.0),
        pre_process=pre_process,
        post_process=post_process,
        add_encoder=add_encoder,
        add_decoder=add_decoder,
        image_size=args.image_size,
        patch_dim=args.patch_dim,
        language_rotary_base=args.rotary_base,
        language_rope_scaling=args.use_rope_scaling,
    )
    print(f"{model=}")

    # --- Freeze components ---
    model.freeze(
        freeze_language_model=getattr(args, 'freeze_LM', False),
        freeze_vision_model=getattr(args, 'freeze_ViT', False),
        freeze_vision_projection=getattr(args, 'freeze_vision_projection', False),
    )

    return model


def get_batch(data_iterator):
    """Generate a batch from data iterator.

    The dataloader yields a dict from BagelPackedBatch.to_dict() with fields:
      - Always present: sequence_length, sample_lens, packed_text_ids,
        packed_text_indexes, packed_position_ids, split_lens, attn_modes
      - VAE (optional): padded_images, patchified_vae_latent_shapes,
        packed_latent_position_ids, packed_vae_token_indexes
      - ViT (optional): packed_vit_tokens, packed_vit_position_ids,
        packed_vit_token_indexes, vit_token_seqlens
      - Diffusion (optional): packed_timesteps, mse_loss_indexes
      - CE loss (optional): packed_label_ids, ce_loss_indexes, ce_loss_weights

    Returns:
        Dictionary with all batch tensors needed for BAGEL forward.
    """
    args = get_args()

    # Currently only support TP=1, PP=1 — no cross-rank broadcast needed.
    assert (getattr(args, 'pipeline_model_parallel_size', 1) == 1)
    assert (getattr(args, 'tensor_model_parallel_size', 1) == 1)

    data = next(data_iterator)
    print(f"{data=}")

    # --- Assemble batch dict ---
    batch = {
        # Core text fields (always present)
        'packed_text_ids': data['packed_text_ids'].cuda(non_blocking=True),
        'packed_text_indexes': data['packed_text_indexes'].cuda(non_blocking=True),
        'packed_position_ids': data['packed_position_ids'].cuda(non_blocking=True),
        # Non-tensor metadata (needed for FlexAttention mask construction)
        'sequence_length': data['sequence_length'],
        'sample_lens': data['sample_lens'],
        'split_lens': data['split_lens'],
        'attn_modes': data['attn_modes'],
        'attention_mask': None,
    }

    # --- CE loss fields ---
    if 'packed_label_ids' in data:
        batch['packed_label_ids'] = data['packed_label_ids'].cuda(non_blocking=True)
    if 'ce_loss_indexes' in data:
        batch['ce_loss_indexes'] = data['ce_loss_indexes'].cuda(non_blocking=True)
    if 'ce_loss_weights' in data:
        batch['ce_loss_weights'] = data['ce_loss_weights'].cuda(non_blocking=True)

    # --- VAE image generation fields ---
    if 'padded_images' in data:
        batch['padded_images'] = data['padded_images'].cuda(non_blocking=True)
    if 'patchified_vae_latent_shapes' in data:
        batch['patchified_vae_latent_shapes'] = data['patchified_vae_latent_shapes']
    if 'packed_latent_position_ids' in data:
        batch['packed_latent_position_ids'] = data['packed_latent_position_ids'].cuda(non_blocking=True)
    if 'packed_vae_token_indexes' in data:
        batch['packed_vae_token_indexes'] = data['packed_vae_token_indexes'].cuda(non_blocking=True)

    # --- ViT image understanding fields ---
    if 'packed_vit_tokens' in data:
        batch['packed_vit_tokens'] = data['packed_vit_tokens'].cuda(non_blocking=True)
    if 'packed_vit_position_ids' in data:
        batch['packed_vit_position_ids'] = data['packed_vit_position_ids'].cuda(non_blocking=True)
    if 'packed_vit_token_indexes' in data:
        batch['packed_vit_token_indexes'] = data['packed_vit_token_indexes'].cuda(non_blocking=True)
    if 'vit_token_seqlens' in data:
        batch['vit_token_seqlens'] = data['vit_token_seqlens'].cuda(non_blocking=True)

    # --- Diffusion timesteps ---
    if 'packed_timesteps' in data:
        batch['packed_timesteps'] = data['packed_timesteps'].cuda(non_blocking=True)
    if 'mse_loss_indexes' in data:
        batch['mse_loss_indexes'] = data['mse_loss_indexes'].cuda(non_blocking=True)

    print(f"{batch=}")
    return batch


def bagel_loss_func(loss_mask, output_tensor, model=None):
    """Loss function for BAGEL that combines CE and MSE losses.

    Args:
        loss_mask: Loss mask tensor.
        output_tensor: Combined loss tensor from forward_step.
        model: The model (unused).

    Returns:
        Tuple of (loss, num_tokens, report_dict).
    """
    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()

    loss = torch.sum(losses * loss_mask)
    num_tokens = loss_mask.sum().clone().detach().to(torch.int)

    report = {'lm loss': torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])}

    return loss, num_tokens, report


def forward_step(data_iterator, model: BagelModel):
    """Forward training step.

    Args:
        data_iterator: Iterable dataset.
        model: BagelModel instance.

    Returns:
        output_tensor: Combined loss tensor.
        loss_func: Partial loss function with loss mask.
    """
    args = get_args()
    timers = get_timers()

    # Get the batch
    timers('batch-generator', log_level=2).start()
    batch = get_batch(data_iterator)
    timers('batch-generator').stop()

    ce_weight = getattr(args, 'ce_weight', 1.0)
    mse_weight = getattr(args, 'mse_weight', 1.0)

    # Forward pass — batch keys match model forward signature directly
    output = model(
        packed_text_ids=batch['packed_text_ids'],
        packed_text_indexes=batch['packed_text_indexes'],
        packed_position_ids=batch['packed_position_ids'],
        attention_mask=batch.get('attention_mask'),
        sequence_length=batch.get('sequence_length'),
        sample_lens=batch.get('sample_lens'),
        split_lens=batch.get('split_lens'),
        attn_modes=batch.get('attn_modes'),
        # CE loss
        packed_label_ids=batch.get('packed_label_ids'),
        ce_loss_indexes=batch.get('ce_loss_indexes'),
        ce_loss_weights=batch.get('ce_loss_weights'),
        # ViT understanding
        packed_vit_tokens=batch.get('packed_vit_tokens'),
        packed_vit_token_indexes=batch.get('packed_vit_token_indexes'),
        packed_vit_position_ids=batch.get('packed_vit_position_ids'),
        vit_token_seqlens=batch.get('vit_token_seqlens'),
        # VAE generation
        padded_images=batch.get('padded_images'),
        patchified_vae_latent_shapes=batch.get('patchified_vae_latent_shapes'),
        packed_latent_position_ids=batch.get('packed_latent_position_ids'),
        packed_vae_token_indexes=batch.get('packed_vae_token_indexes'),
        packed_timesteps=batch.get('packed_timesteps'),
        mse_loss_indexes=batch.get('mse_loss_indexes'),
    )

    # Combine CE and MSE losses into a single tensor
    loss_parts = []
    if output.get('ce') is not None:
        loss_parts.append(ce_weight * output['ce'].mean())
    if output.get('mse') is not None:
        loss_parts.append(mse_weight * output['mse'].mean())

    if loss_parts:
        combined_loss = sum(loss_parts)
    else:
        combined_loss = torch.tensor(0.0, device=batch['packed_text_ids'].device, requires_grad=True)

    # Use ce_loss_weights for the standard loss reduction path
    loss_mask = batch.get('ce_loss_weights', torch.ones_like(combined_loss))

    return combined_loss.unsqueeze(0), partial(bagel_loss_func, loss_mask)


def add_bagel_extra_args(parser):
    """Add BAGEL-specific command-line arguments."""
    parser = add_multimodal_extra_args(parser)
    group = parser.add_argument_group(title='BAGEL model arguments')

    # Modality switches
    group.add_argument(
        '--visual-gen', action='store_true',
        help='Enable visual generation path',
    )
    group.add_argument(
        '--visual-und', action='store_true',
        help='Enable visual understanding path',
    )
    group.add_argument(
        '--interpolate-pos', action='store_true',
        help="Interpolate positional embeddings when image resolution differs from pre-training."
    )

    # VAE latent configuration
    group.add_argument(
        '--latent-patch-size', type=int, default=2,
        help='Spatial size (in VAE pixels) covered by each latent patch.',
    )
    group.add_argument(
        '--max-latent-size', type=int, default=32,
        help='Maximum latent grid size (patches per side) for the VAE latent tensor.',
    )
    group.add_argument(
        '--latent-channel', type=int, default=16,
        help='Number of VAE latent channels',
    )
    group.add_argument(
        '--vae-downsample', type=int, default=16,
        help='VAE spatial downsampling factor',
    )
    group.add_argument(
        '--timestep-shift', type=float, default=1.0,
        help='Flow matching timestep shift parameter',
    )

    # ViT configuration
    group.add_argument(
        '--image-size', type=int, default=70,
        help='Maximum Image size for ViTs position embedding',
    )
    group.add_argument(
        '--vit-max-num-patch-per-side', type=int, default=70,
        help='Maximum ViT patches per side for position embedding',
    )

    # Loss weights
    group.add_argument(
        '--ce-weight', type=float, default=1.0,
        help='Weight for cross-entropy (understanding) loss',
    )
    group.add_argument(
        '--mse-weight', type=float, default=1.0,
        help='Weight for MSE (generation) loss',
    )

    # Freeze controls
    group.add_argument(
        '--freeze-vision-projection', action='store_true', default=False,
        help='Freeze vision projection weights',
    )

    # for dataset
    group.add_argument(
        '--vit-patch-size', type=int, default=14,
        help='Patch size (pixels) for the Vision Transformer encoder.'
    )

    return parser


def train_valid_test_datasets_provider(train_val_test_num_samples):
    args = get_args()
    kwargs = {}
    dataset_provider = _DATASET_PROVIDERS["vlm"]
    # max_seq_length = args.total_seq_length
    # kwargs['max_seq_length'] = max_seq_length
    # print_rank_0(f"Bagel Training: Using max_seq_length = {max_seq_length} "
    #             f"(total_seq_length: {args.total_seq_length})")
    kwargs['max_seq_length'] = 32078
    return dataset_provider(train_val_test_num_samples, **kwargs)


if __name__ == "__main__":
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider=model_provider,
        model_type=ModelType.encoder_or_decoder,
        forward_step_func=forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_bagel_extra_args,
    )

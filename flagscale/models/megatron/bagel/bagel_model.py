# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""BAGEL multimodal model for unified image understanding and generation.

BAGEL uses a Mixture-of-Transformers (MoT) architecture where understanding
(text + ViT) and generation (VAE latent) tokens have separate projections
but share the core attention computation.
"""

import math
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.activations import ACT2FN

from megatron.core import tensor_parallel
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.gpt.gpt_model import GPTModel
# from megatron.core.models.vision.clip_vit_model import CLIPViTModel
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.packed_seq_params import PackedSeqParams, MoTPackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import deprecate_inference_params, is_te_min_version, log_single_rank

from .autoencoder import AutoEncoderParams, AutoEncoder
from .siglip_model import SiglipVisionModel


def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    """Generate 2D sine-cosine positional embeddings (from DiT)."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])

    half = embed_dim // 2
    embs = []
    for g in [grid[0], grid[1]]:
        omega = np.arange(half // 2, dtype=np.float64)
        omega /= half / 2.0
        omega = 1.0 / 10000 ** omega
        pos = g.reshape(-1)
        out = np.einsum('m,d->md', pos, omega)
        embs.append(np.concatenate([np.sin(out), np.cos(out)], axis=1))
    return np.concatenate(embs, axis=1)


class TimestepEmbedder(MegatronModule):
    """Embeds scalar timesteps into vector representations (from DiT).

    Uses sinusoidal frequency encoding followed by a 2-layer MLP.
    """

    def __init__(self, config: TransformerConfig, frequency_embedding_size: int = 256):
        super().__init__(config=config)
        hidden_size = config.hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: Tensor) -> Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class PositionEmbedding(MegatronModule):
    """Frozen 2D sin-cos position embeddings for VAE latents and ViT tokens."""

    def __init__(self, config: TransformerConfig, max_num_patch_per_side: int):
        super().__init__(config=config)
        hidden_size = config.hidden_size
        self.max_num_patch_per_side = max_num_patch_per_side
        self.pos_embed = nn.Parameter(
            torch.zeros(max_num_patch_per_side ** 2, hidden_size),
            requires_grad=False,
        )
        pos_embed = _get_2d_sincos_pos_embed(hidden_size, max_num_patch_per_side)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

    def forward(self, position_ids: Tensor) -> Tensor:
        return self.pos_embed[position_ids]


class BagelModel(MegatronModule):
    """BAGEL multimodal model for unified image understanding and generation.

    Follows the LLaVA model pattern with added generation capabilities through
    Mixture-of-Transformers (MoT) layers and VAE latent processing.

    Args:
        language_transformer_config: Config for the language model backbone.
        language_transformer_layer_spec: Layer spec for decoder layers (MoT spec).
        language_vocab_size: Vocabulary size for the language model.
        language_max_sequence_length: Maximum sequence length.
        vision_transformer_config: Config for the vision encoder (SigLIP ViT).
        vision_transformer_layer_spec: Layer spec for vision encoder layers.
        vision_projection_config: Config for the vision projection MLP.
        vision_projection_layer_spec: Submodule spec for the vision projection.
        visual_gen: Enable visual generation path.
        visual_und: Enable visual understanding path.
        latent_patch_size: Patch size for patchifying VAE latents.
        max_latent_size: Maximum latent grid size (one side).
        vit_max_num_patch_per_side: Max ViT patches per side for position embedding.
        latent_channel: Number of VAE latent channels.
        vae_downsample: VAE spatial downsampling factor.
        timestep_shift: Flow matching timestep shift parameter.
        pre_process: Include embedding layer (pipeline parallel).
        post_process: Include output layer (pipeline parallel).
        add_encoder: Include vision encoder (pipeline parallel).
        add_decoder: Include decoder (pipeline parallel).
        parallel_output: Keep outputs split across TP ranks.
    """

    def __init__(
        self,
        language_transformer_config: TransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        language_vocab_size: int,
        language_max_sequence_length: int,
        vision_transformer_config: TransformerConfig,
        vision_transformer_layer_spec: ModuleSpec,
        vision_projection_config: TransformerConfig,
        vision_projection_layer_spec: ModuleSpec,
        vision_projection_type: str = "mlp",
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        language_position_embedding_type: str = 'learned_absolute',
        language_rotary_percent: float = 1.0,
        visual_gen: bool = True,
        visual_und: bool = True,
        latent_patch_size: int = 2,
        max_latent_size: int = 32,
        vit_max_num_patch_per_side: int = 70,
        latent_channel: int = 16,
        vae_downsample: int = 16,
        timestep_shift: float = 1.0,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        patch_dim: int = 14,
        image_size: int = 980,
        language_rotary_base: int = 10000,
        language_rope_scaling: bool = False,
        language_rope_scaling_factor: float = 8.0,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
        use_vision_backbone_fp8_arch: bool = False,
    ) -> None:
        super().__init__(config=language_transformer_config)

        if has_config_logger_enabled(language_transformer_config):
            log_config_to_disk(language_transformer_config, locals(), prefix=type(self).__name__)

        log_single_rank(
            logging.getLogger(__name__),
            logging.WARNING,
            "LLaVA is work in progress. Features are missing and methods can change.",
        )

        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder
        self.vp_stage = vp_stage

        self.encoder_hidden_state = None
        self.vision_model = None
        self.vision_projection = None
        self.language_model = None

        self.visual_gen = visual_gen
        self.visual_und = visual_und
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection

        # --- Language Model (GPTModel with MoT layers) ---
        if self.add_decoder:
            self.language_model = GPTModel(
                config=language_transformer_config,
                transformer_layer_spec=language_transformer_layer_spec,
                vocab_size=language_vocab_size,
                max_sequence_length=language_max_sequence_length,
                parallel_output=parallel_output,
                position_embedding_type=language_position_embedding_type,
                rotary_percent=language_rotary_percent,
                pre_process=self.pre_process,
                post_process=False,  # We handle post-processing ourselves
                rotary_base=language_rotary_base,
                rope_scaling=language_rope_scaling,
                rope_scaling_factor=language_rope_scaling_factor,
                scatter_embedding_sequence_parallel=False,
                share_embeddings_and_output_weights=share_embeddings_and_output_weights,
                pg_collection=self.pg_collection,
                vp_stage=vp_stage,
            )
            if hasattr(language_transformer_config, 'position_embedding_type'):
                self.position_embedding_type = language_transformer_config.position_embedding_type
            else:
                self.position_embedding_type = language_position_embedding_type

        # --- Vision Encoder (SigLIP ViT) ---
        if self.add_encoder and self.visual_und:
            assert vit_max_num_patch_per_side == image_size // patch_dim
            self.vision_model = SiglipVisionModel(
                vision_transformer_config,
                vision_transformer_layer_spec,
                patch_dim=patch_dim,
                image_size=image_size,
                pg_collection=self.pg_collection,
                vp_stage=vp_stage,
            )
            self.vision_model.convert_conv2d_to_linear()

            # Vision projection: ViT hidden -> LLM hidden
            self.vision_projection = MultimodalProjector(
                vision_projection_config,
                vision_projection_layer_spec,
                vision_projection_type,
                vision_transformer_config.hidden_size,
                tp_group=self.pg_collection.tp,
            )

            # 2D sincos position embedding for ViT tokens
            self.vit_pos_embed = PositionEmbedding(
                config=language_transformer_config,
                max_num_patch_per_side=vit_max_num_patch_per_side,
            )

        # --- Visual Generation components ---
        if self.visual_gen:
            vae_params = AutoEncoderParams(
                resolution=256,
                in_channels=3,
                downsample=8,
                ch=128,
                out_ch=3,
                ch_mult=[1, 2, 4, 4],
                num_res_blocks=2,
                z_channels=16,
                scale_factor=0.3611,
                shift_factor=0.1159,
            )

            # Loading the autoencoder
            self.vae = AutoEncoder(vae_params)

            self.latent_patch_size = latent_patch_size
            self.max_latent_size = max_latent_size
            self.latent_channel = latent_channel
            self.vae_downsample = vae_downsample
            self.timestep_shift = timestep_shift
            self.patch_latent_dim = latent_patch_size ** 2 * latent_channel

            # VAE latent <-> LLM space projections
            self.vae2llm = nn.Linear(self.patch_latent_dim, language_transformer_config.hidden_size)
            self.llm2vae = nn.Linear(language_transformer_config.hidden_size, self.patch_latent_dim)

            # Timestep and position embeddings for generation
            self.time_embedder = TimestepEmbedder(
                config=language_transformer_config,
            )
            self.latent_pos_embed = PositionEmbedding(
                config=language_transformer_config,
                max_num_patch_per_side=max_latent_size,
            )

            # Initialize llm2vae to zero (model starts with no generation output)
            nn.init.constant_(self.llm2vae.weight, 0)
            nn.init.constant_(self.llm2vae.bias, 0)

        # --- LM head for text prediction ---
        if self.post_process:
            self.output_layer = tensor_parallel.ColumnParallelLinear(
                language_transformer_config.hidden_size,
                language_vocab_size,
                config=language_transformer_config,
                init_method=language_transformer_config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=not parallel_output,
                skip_weight_param_allocation=self.pre_process
                and self.share_embeddings_and_output_weights,
                embedding_activation_buffer=None,
                grad_output_buffer=None,
                tp_group=self.pg_collection.tp,
            )
            self.output_layer.weight.is_embedding_or_output_parameter = True

    def shared_embedding_or_output_weight(self):
        """This is a convenience method to surface the language model's word embeddings, which is
        necessary for `finalize_model_grads._allreduce_word_embedding_grads`."""
        if self.add_decoder:
            return self.language_model.shared_embedding_or_output_weight()
        return None

    def set_input_tensor(self, input_tensor) -> None:
        """Set model chunk input tensor."""
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for BagelModel'

        if self.add_encoder and self.add_decoder:
            self.vision_model.set_input_tensor(input_tensor[0])
        elif self.add_encoder:
            self.vision_model.set_input_tensor(input_tensor[0])
        elif self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])

    def freeze(
        self,
        freeze_language_model: bool = False,
        freeze_vision_model: bool = False,
        freeze_vision_projection: bool = False,
    ):
        """Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_vision_model (bool): Freeze the vision model module.
            freeze_vision_projection (bool): Freeze the vision projection module.
        """
        modules = []
        if freeze_language_model and hasattr(self, 'language_model'):
            modules.append(self.language_model)
        if freeze_vision_model and hasattr(self, 'vision_model'):
            modules.append(self.vision_model)
        if freeze_vision_projection and hasattr(self, 'vision_projection'):
            modules.append(self.vision_projection)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def forward(
        self,
        # Core packed text fields (always present)
        packed_text_ids: torch.Tensor,
        packed_text_indexes: torch.Tensor,
        packed_position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        # Sequence packing metadata
        sequence_length: Optional[int] = None,
        sample_lens: Optional[List[int]] = None,
        split_lens: Optional[List[int]] = None,
        attn_modes: Optional[List[str]] = None,
        # CE loss fields
        packed_label_ids: Optional[Tensor] = None,
        ce_loss_indexes: Optional[Tensor] = None,
        ce_loss_weights: Optional[Tensor] = None,
        # ViT understanding inputs
        packed_vit_tokens: Optional[Tensor] = None,
        packed_vit_token_indexes: Optional[Tensor] = None,
        packed_vit_position_ids: Optional[Tensor] = None,
        vit_token_seqlens: Optional[Tensor] = None,
        # VAE generation inputs
        padded_images: Optional[Tensor] = None,
        patchified_vae_latent_shapes: Optional[List] = None,
        packed_latent_position_ids: Optional[Tensor] = None,
        packed_vae_token_indexes: Optional[Tensor] = None,
        packed_timesteps: Optional[Tensor] = None,
        mse_loss_indexes: Optional[Tensor] = None,
        # Standard args
        inference_context: BaseInferenceContext = None,
        packed_seq_params: Optional[MoTPackedSeqParams] = None,
        extra_block_kwargs: Optional[dict] = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ) -> Dict[str, Optional[Tensor]]:
        """Forward pass for BAGEL model.

        Returns:
            Dictionary with 'ce' and 'mse' loss tensors (or None).
        """

        # --- Step 1: Get text embeddings from language model embedding ---
        if self.pre_process and self.add_decoder:
            # Use language model's embedding to get text token embeddings
            # GPTModel.embedding gives us [seq, batch, hidden]
            text_embeddings = self.language_model.embedding(
                input_ids=packed_text_ids, position_ids=packed_position_ids,
            )
            print(f"{packed_text_ids.shape=}")

            # Create the combined sequence tensor
            # For packed sequences: [total_seq, 1, hidden]
            seq_len = text_embeddings.shape[0]
            combined_embeddings = text_embeddings.new_zeros(size=(sequence_length, self.language_model.config.hidden_size))

            # --- Step 2: Process ViT tokens (understanding path) ---
            if self.visual_und and self.add_encoder and packed_vit_tokens is not None:
                # Get ViT embeddings
                cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
                cu_seqlens = cu_seqlens.to(torch.int32)
                max_seqlen = torch.max(vit_token_seqlens).item()
                # [num_patches, 1, vit_hidden]
                print(f"{packed_vit_tokens.shape=}, {packed_vit_tokens.shape=}")
                dtype=next(self.vision_model.parameters()).dtype
                packed_vit_tokens = packed_vit_tokens.to(dtype)
                image_embeddings = self.vision_model(
                    packed_pixel_values=packed_vit_tokens,
                    packed_flattened_position_ids=packed_vit_position_ids,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                )
                # [num_patches, 1, hidden]
                image_embeddings = self.vision_projection(image_embeddings)
                print(f"{image_embeddings.shape=}")

                # Add ViT position embeddings
                vit_pos_emb = self.vit_pos_embed(packed_vit_position_ids)
                # vit_pos_emb: [num_patches, hidden] -> [num_patches, 1, hidden]
                if vit_pos_emb.dim() == 2:
                    vit_pos_emb = vit_pos_emb.unsqueeze(1)
                image_embeddings = image_embeddings + vit_pos_emb

                # Place ViT embeddings into combined sequence
                if packed_vit_token_indexes is not None:
                    if image_embeddings.dtype != combined_embeddings.dtype:
                        image_embeddings = image_embeddings.to(combined_embeddings.dtype)
                    combined_embeddings[packed_vit_token_indexes] = image_embeddings.squeeze(1)

            # --- Step 3: Process VAE latents (generation path) ---
            if self.visual_gen and padded_images is not None and packed_vae_token_indexes is not None:
                # Encode images to latent space via VAE
                packed_latent = self.vae.encode(padded_images)
                p = self.latent_patch_size
                packed_latent = []
                for latent, (h, w) in zip(packed_latent, patchified_vae_latent_shapes):
                    latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
                    latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
                    packed_latent.append(latent)
                packed_latent_clean = torch.cat(packed_latent, dim=0)

                # Add noise via flow matching schedule
                noise = torch.randn_like(packed_latent_clean)
                t = torch.sigmoid(packed_timesteps)
                t = self.timestep_shift * t / (1 + (self.timestep_shift - 1) * t)
                noisy_latent = (1 - t[:, None]) * packed_latent + t[:, None] * noise

                # Project to LLM space and add timestep/position embeddings
                latent_embed = self.vae2llm(noisy_latent)
                timestep_embed = self.time_embedder(t)
                latent_pos_emb = self.latent_pos_embed(packed_latent_position_ids)
                latent_embed = latent_embed + timestep_embed + latent_pos_emb

                # Place into combined sequence
                if latent_embed.dtype != combined_embeddings.dtype:
                    latent_embed = latent_embed.to(combined_embeddings.dtype)
                # Handle shape: latent_embed is [num_latent_tokens, hidden]
                # combined_embeddings is [seq, batch, hidden]
                if latent_embed.dim() == 2 and combined_embeddings.dim() == 3:
                    combined_embeddings[packed_vae_token_indexes, 0] = latent_embed
                else:
                    combined_embeddings[packed_vae_token_indexes] = latent_embed

                # Store clean latent and noise for loss computation
                self._packed_latent_clean = packed_latent_clean
                self._noise = noise
                self._timesteps = t
            else:
                self._packed_latent_clean = None
                self._noise = None
                self._timesteps = None

            decoder_input = combined_embeddings

        else:
            decoder_input = None

        assert self.position_embedding_type == 'rope'
        assert not self.language_model.config.multi_latent_attention

        max_pos = int(packed_position_ids.max().item()) + 1
        rotary_pos_emb_full = self.language_model.rotary_pos_emb(
            max_pos,
            packed_seq=False,
            cp_group=None,
        )  # [max_pos, 1, 1, rot_dim]
        rotary_pos_emb = rotary_pos_emb_full[packed_position_ids]  # [total_tokens, 1, 1, rot_dim]
        rotary_pos_emb = (rotary_pos_emb, rotary_pos_emb)  # (q_pos_emb, k_pos_emb)

        # --- Step 4: Run through language model decoder ---
        # Build MoTPackedSeqParams for MoT layers
        print(f"{self.language_model.decoder=}")
        print(f"{packed_text_indexes.shape=}, {packed_vit_token_indexes.shape=}")

        packed_und_token_indexes = packed_text_indexes
        if packed_vit_token_indexes is not None:
            packed_und_token_indexes=torch.cat(
                [packed_text_indexes, packed_vit_token_indexes],
                dim=0
            )

        assert packed_und_token_indexes is not None
        if packed_vae_token_indexes is None:
            packed_vae_token_indexes = packed_und_token_indexes.new_ones(size=[0])

        print(f"{packed_und_token_indexes.shape=}")
        packed_seq_params = MoTPackedSeqParams()
        packed_seq_params.packed_und_token_indexes = packed_und_token_indexes
        packed_seq_params.packed_gen_token_indexes = packed_vae_token_indexes
        packed_seq_params.sample_lens = sample_lens
        packed_seq_params.split_lens = split_lens
        packed_seq_params.attn_modes = attn_modes

        # Run through GPTModel with decoder_input (skips embedding)
        hidden_states = self.language_model.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=None,
            rotary_pos_sin=None,
            rotary_pos_cos_sin=None,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=None,
            **(extra_block_kwargs or {}),
        )

        # --- Step 5: Compute losses ---
        result = {'ce': None, 'mse': None}

        if self.post_process and hidden_states is not None:
            # CE loss for text prediction (understanding)
            if ce_loss_indexes is not None and packed_label_ids is not None:
                print(f"{hidden_states.shape=}")
                ce_hidden = hidden_states[ce_loss_indexes]
                print(f"{ce_hidden.shape=}")
                packed_ce_logits, _ = self.output_layer(hidden_states[ce_loss_indexes])
                # ce_labels = packed_label_ids[ce_loss_indexes] if packed_label_ids.dim() == 1 else packed_label_ids
                ce_loss = F.cross_entropy(packed_ce_logits, packed_label_ids, reduction='none')
                # Apply per-token ce_loss_weights if provided
                if ce_loss_weights is not None:
                    ce_loss = ce_loss * ce_loss_weights
                result['ce'] = ce_loss

            # MSE loss for latent prediction (generation)
            if (
                self.visual_gen
                and mse_loss_indexes is not None
                and self._packed_latent_clean is not None
            ):
                mse_preds = self.llm2vae(hidden_states[mse_loss_indexes])
                target = self._noise - self._packed_latent_clean  # v_t = dx_t/dt = x_1 - x_0
                has_mse = self._timesteps > 0
                result['mse'] = (mse_preds - target[has_mse]) ** 2

        return result

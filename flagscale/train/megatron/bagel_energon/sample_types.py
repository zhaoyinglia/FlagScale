# Copyright (c) 2025, BAAI. All rights reserved.
# Sample type definitions for Bagel-Energon integration.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class BagelSample:
    """Intermediate sample representation after encode_sample, before packing.

    This is what goes into the PackingDataset buffer.
    """
    image_tensor_list: List[torch.Tensor]  # ViT/VAE preprocessed images
    text_ids_list: List[List[int]]         # Tokenized text segments
    sequence_plan: List[Dict[str, Any]]    # Packing instructions per segment
    num_tokens: int                        # Total token count for this sample
    is_mandatory: bool = False             # Whether this sample must appear in every pack
    subflavor: str = ""                    # Task type: "t2i" / "vlm"
    # Energon metadata
    __key__: str = ""
    __restore_key__: tuple = ()


@dataclass
class BagelPackedBatch:
    """Output of pack_selected_samples — a fully packed sequence ready for the model."""
    sequence_length: int
    sample_lens: List[int]
    packed_text_ids: torch.Tensor
    packed_text_indexes: torch.Tensor
    packed_position_ids: torch.Tensor
    # FlexAttention fields
    split_lens: List[int] = field(default_factory=list)
    attn_modes: List[str] = field(default_factory=list)
    # VAE image generation (optional)
    padded_images: Optional[torch.Tensor] = None
    patchified_vae_latent_shapes: Optional[List] = None
    packed_latent_position_ids: Optional[torch.Tensor] = None
    packed_vae_token_indexes: Optional[torch.Tensor] = None
    # ViT image understanding (optional)
    packed_vit_tokens: Optional[torch.Tensor] = None
    packed_vit_position_ids: Optional[torch.Tensor] = None
    packed_vit_token_indexes: Optional[torch.Tensor] = None
    vit_token_seqlens: Optional[torch.Tensor] = None
    # Diffusion timesteps (optional)
    packed_timesteps: Optional[torch.Tensor] = None
    mse_loss_indexes: Optional[torch.Tensor] = None
    # CE loss (optional)
    packed_label_ids: Optional[torch.Tensor] = None
    ce_loss_indexes: Optional[torch.Tensor] = None
    ce_loss_weights: Optional[torch.Tensor] = None

    def to_dict(self):
        """Convert to dict for get_batch compatibility."""
        data = {
            'sequence_length': self.sequence_length,
            'sample_lens': self.sample_lens,
            'packed_text_ids': self.packed_text_ids,
            'packed_text_indexes': self.packed_text_indexes,
            'packed_position_ids': self.packed_position_ids,
            'split_lens': self.split_lens,
            'attn_modes': self.attn_modes,
        }
        if self.padded_images is not None:
            data['padded_images'] = self.padded_images
            data['patchified_vae_latent_shapes'] = self.patchified_vae_latent_shapes
            data['packed_latent_position_ids'] = self.packed_latent_position_ids
            data['packed_vae_token_indexes'] = self.packed_vae_token_indexes
        if self.packed_vit_tokens is not None:
            data['packed_vit_tokens'] = self.packed_vit_tokens
            data['packed_vit_position_ids'] = self.packed_vit_position_ids
            data['packed_vit_token_indexes'] = self.packed_vit_token_indexes
            data['vit_token_seqlens'] = self.vit_token_seqlens
        if self.packed_timesteps is not None:
            data['packed_timesteps'] = self.packed_timesteps
            data['mse_loss_indexes'] = self.mse_loss_indexes
        if self.packed_label_ids is not None:
            data['packed_label_ids'] = self.packed_label_ids
            data['ce_loss_indexes'] = self.ce_loss_indexes
            data['ce_loss_weights'] = self.ce_loss_weights
        return data


@dataclass
class T2ISample:
    #: The input image tensor in the shape (C, H, W)
    image: torch.Tensor
    #: The captions of the image
    captions: str


class VLMSample:
    #: The input image tensor in the shape (C, H, W)
    image: torch.Tensor

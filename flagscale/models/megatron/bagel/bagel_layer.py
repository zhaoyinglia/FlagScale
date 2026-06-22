# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Mixture-of-Transformers (MoT) layer for BAGEL multimodal model.

Implements dual-branch attention and MLP where understanding (text + ViT)
and generation (VAE) tokens use separate projections but share the core
attention computation.  Uses PyTorch ``flex_attention`` with ``BlockMask``
for efficient block-sparse attention (causal + per-sample boundaries).
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType, CudaGraphScope
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.module import GraphableMegatronModule, MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    BaseTransformerLayer,
    get_transformer_layer_offset,
)
from megatron.core.utils import (
    get_pg_rank,
    log_single_rank,
    make_viewless_tensor,
)

logger = logging.getLogger(__name__)


@dataclass
class MoTTransformerLayerSubmodules:
    """Submodule specifications for the MoT transformer layer.

    Extends TransformerLayerSubmodules with generation-branch variants.
    """

    # Understanding branch (standard)
    input_layernorm: Union[ModuleSpec, type] = IdentityOp
    self_attention: Union[ModuleSpec, type] = IdentityOp
    self_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp
    pre_mlp_layernorm: Union[ModuleSpec, type] = IdentityOp
    mlp: Union[ModuleSpec, type] = IdentityOp
    mlp_bda: Union[ModuleSpec, type] = IdentityFuncOp

    # Generation branch (dual)
    input_layernorm_gen: Union[ModuleSpec, type] = IdentityOp
    pre_mlp_layernorm_gen: Union[ModuleSpec, type] = IdentityOp
    mlp_gen: Union[ModuleSpec, type] = IdentityOp
    mlp_bda_gen: Union[ModuleSpec, type] = IdentityFuncOp

    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


class MoTTransformerLayer(GraphableMegatronModule, BaseTransformerLayer):
    """A single Mixture-of-Transformers layer for BAGEL.

    Implements dual-branch processing where understanding and generation tokens
    have separate layernorms, QKV projections, output projections, and MLPs,
    but share the core attention computation.

    Input/Output shape: [seq, batch, hidden].
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MoTTransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
        is_mtp_layer: bool = False,
        add_layer_offset: bool = True,
        pp_layer_offset: Optional[int] = None,
    ):
        self.submodules_config = submodules
        super().__init__(config=config, vp_stage=vp_stage)

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp

        assert is_mtp_layer is False

        self.layer_number = layer_number + get_transformer_layer_offset(
            self.config, vp_stage, get_pg_rank(pg_collection.pp)
        )
        self.hidden_dropout = config.hidden_dropout if hidden_dropout is None else hidden_dropout
        self.is_mtp_layer = is_mtp_layer

        # --- Understanding branch layernorms ---
        self.input_layernorm = submodules.input_layernorm(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.pre_mlp_layernorm = submodules.pre_mlp_layernorm(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        # --- Generation branch layernorms ---
        self.input_layernorm_gen = submodules.input_layernorm_gen(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.pre_mlp_layernorm_gen = submodules.pre_mlp_layernorm_gen(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        # --- MoT Attention (handles dual QKV internally) ---
        attention_optional_kwargs = {"pg_collection": pg_collection}
        if config.context_parallel_size > 1 and config.cp_comm_type is not None:
            if isinstance(config.cp_comm_type, list):
                # layer_number is 1-indexed, so we need to subtract 1 to get the correct index
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type[
                    self.layer_number - 1
                ]
            else:
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type

        attention_optional_kwargs["pg_collection"] = pg_collection
        if pp_layer_offset is not None:
            attention_optional_kwargs["pp_layer_offset"] = pp_layer_offset

        self.self_attention = build_module(
            submodules.self_attention,
            config=self.config,
            layer_number=self.layer_number,
            **attention_optional_kwargs,
        )

        # --- Bias-Dropout-Add for attention ---
        self.self_attn_bda = build_module(submodules.self_attn_bda)

        # MLP block, import here to avoid circular import
        from megatron.core.extensions.transformer_engine import TEFusedMLP
        from megatron.core.transformer.moe.experts import SequentialMLP, TEGroupedMLP
        from megatron.core.transformer.moe.moe_layer import MoELayer

        # MLP expects tp_group but MoELayer expects pg_collection to be passed in.
        # We can change MLP to accept pg_collection but it makes the logic implicit
        # The conditional below is to make the logic explicit
        # if submodules.mlp is not a ModuleSpec,we dont have to handle passing additional kwargs
        # --- Understanding branch MLP ---
        additional_mlp_kwargs = {}
        if isinstance(submodules.mlp, ModuleSpec):
            if submodules.mlp.module in (MoELayer, TEGroupedMLP, SequentialMLP):
                additional_mlp_kwargs["pg_collection"] = pg_collection
                # Pass is_mtp_layer flag to MoELayer to distinguish MTP MoE layers.
                if submodules.mlp.module == MoELayer:
                    additional_mlp_kwargs["is_mtp_layer"] = self.is_mtp_layer
            elif submodules.mlp.module == MLP:
                assert hasattr(
                    pg_collection, 'tp'
                ), 'TP process group is required for MLP in TransformerLayer'
                additional_mlp_kwargs["tp_group"] = pg_collection.tp
            elif TEFusedMLP is not None and submodules.mlp.module == TEFusedMLP:
                assert hasattr(
                    pg_collection, 'tp'
                ), 'TP process group is required for TEFusedMLP in TransformerLayer'
                additional_mlp_kwargs["tp_group"] = pg_collection.tp
            else:
                log_single_rank(
                    logger,
                    logging.WARNING,
                    f"Unknown MLP type: {type(submodules.mlp)}. Using default kwargs.",
                )
        self.mlp = build_module(submodules.mlp, config=self.config, **additional_mlp_kwargs)
        if hasattr(self.mlp, 'set_layer_number'):
            self.mlp.set_layer_number(self.layer_number)
        # BiasDropoutFusion
        self.mlp_bda = build_module(submodules.mlp_bda)

        # --- Generation branch MLP ---
        additional_mlp_gen_kwargs = {}
        if isinstance(submodules.mlp_gen, ModuleSpec):
            if submodules.mlp_gen.module in (MoELayer, TEGroupedMLP, SequentialMLP):
                additional_mlp_gen_kwargs["pg_collection"] = pg_collection
                # Pass is_mtp_layer flag to MoELayer to distinguish MTP MoE layers.
                if submodules.mlp_gen.module == MoELayer:
                    additional_mlp_gen_kwargs["is_mtp_layer"] = self.is_mtp_layer
            elif submodules.mlp_gen.module == MLP:
                assert hasattr(
                    pg_collection, 'tp'
                ), 'TP process group is required for MLP in TransformerLayer'
                additional_mlp_kwargs["tp_group"] = pg_collection.tp
            elif TEFusedMLP is not None and submodules.mlp_gen.module == TEFusedMLP:
                assert hasattr(
                    pg_collection, 'tp'
                ), 'TP process group is required for TEFusedMLP in TransformerLayer'
                additional_mlp_kwargs["tp_group"] = pg_collection.tp
            else:
                log_single_rank(
                    logger,
                    logging.WARNING,
                    f"Unknown MLP type: {type(submodules.mlp_gen)}. Using default kwargs.",
                )
        self.mlp_gen = build_module(submodules.mlp_gen, config=self.config, **additional_mlp_gen_kwargs)
        if hasattr(self.mlp, 'set_layer_number'):
            self.mlp_gen.set_layer_number(self.layer_number)
        # BiasDropoutFusion
        self.mlp_bda_gen = build_module(submodules.mlp_bda_gen)

        self.is_moe_layer = isinstance(self.mlp, MoELayer)

        # TODO(zhaoyinglia): recompute

        # @jcasper how should we handle nvfuser?
        # Set bias+dropout+add fusion grad_enable execution handler.
        # TORCH_MAJOR = int(torch.__version__.split('.')[0])
        # TORCH_MINOR = int(torch.__version__.split('.')[1])
        # use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        # self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad
        self.bias_dropout_add_exec_handler = torch.enable_grad

    def create_mcore_cudagraph_manager(self, config):
        """Register the transformer layer for cudagraphs."""

        from megatron.core.transformer.cuda_graphs import CudaGraphManager

        # If full scope, just cudagraph the entire layer
        if not self.config.cuda_graph_scope:
            self.cudagraph_manager = CudaGraphManager(config)
        elif (
            CudaGraphScope.attn in self.config.cuda_graph_scope
            and self.submodules_config.self_attention != IdentityOp
        ):
            self.cudagraph_manager = CudaGraphManager(config)
        elif (
            CudaGraphScope.mlp in self.config.cuda_graph_scope
            and self.submodules_config.mlp != IdentityOp
        ):
            # Cudagraphing MoE layers are supposed handled by MoeTransforerLayer
            assert not self.is_moe_layer
            self.cudagraph_manager = CudaGraphManager(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask=None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[Any] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
    ):
        """Forward pass through the MoT transformer layer.

        Args:
            hidden_states: ``[S, B=1, H]``.
            attention_mask: ``BlockMask`` for ``flex_attention``, or ``None``.
            packed_und_token_indexes: 1-D ``LongTensor`` — positions of
                understanding tokens in the packed sequence.
            packed_gen_token_indexes: 1-D ``LongTensor`` — positions of
                generation tokens.

        Returns:
            ``(output, context)``
        """
        packed_und_token_indexes = packed_seq_params.packed_und_token_indexes
        packed_gen_token_indexes = packed_seq_params.packed_gen_token_indexes

        has_mot = (
            packed_und_token_indexes is not None
            and packed_gen_token_indexes is not None
        )

        # ==============================================================
        # Dual Input LayerNorm
        # ==============================================================
        residual = hidden_states

        if has_mot:
            normed_und = self.input_layernorm(hidden_states)
            normed_gen = self.input_layernorm_gen(hidden_states)

            input_layernorm_output = torch.zeros_like(hidden_states)
            input_layernorm_output[packed_und_token_indexes] = normed_und[
                packed_und_token_indexes
            ]
            input_layernorm_output[packed_gen_token_indexes] = normed_gen[
                packed_gen_token_indexes
            ]
        else:
            input_layernorm_output = self.input_layernorm(hidden_states)

        # ==============================================================
        # MoT Self-Attention (flex_attention inside)
        # ==============================================================
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(
                self.training, self.config.bias_dropout_fusion
            )(attention_output_with_bias, residual, self.hidden_dropout)

        # ==============================================================
        # Dual Pre-MLP LayerNorm + Dual MLPs
        # ==============================================================
        residual = hidden_states

        if has_mot:
            normed_und = self.pre_mlp_layernorm(hidden_states)
            mlp_output_und, mlp_bias_und = self.mlp(normed_und)

            normed_gen = self.pre_mlp_layernorm_gen(hidden_states)
            mlp_output_gen, mlp_bias_gen = self.mlp_gen(normed_gen)

            # Merge by index scatter
            mlp_output = torch.zeros_like(mlp_output_und)
            mlp_output[packed_und_token_indexes] = mlp_output_und[
                packed_und_token_indexes
            ]
            mlp_output[packed_gen_token_indexes] = mlp_output_gen[
                packed_gen_token_indexes
            ]

            if mlp_bias_und is not None:
                mlp_bias = torch.zeros_like(mlp_bias_und)
                mlp_bias[packed_und_token_indexes] = mlp_bias_und[
                    packed_und_token_indexes
                ]
                mlp_bias[packed_gen_token_indexes] = mlp_bias_gen[
                    packed_gen_token_indexes
                ]
            else:
                mlp_bias = None

            mlp_output_with_bias = (mlp_output, mlp_bias)
        else:
            pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)
            mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(
                self.training, self.config.bias_dropout_fusion
            )(mlp_output_with_bias, residual, self.hidden_dropout)

        output = make_viewless_tensor(
            inp=hidden_states,
            requires_grad=hidden_states.requires_grad,
            keep_graph=True,
        )

        return output, context

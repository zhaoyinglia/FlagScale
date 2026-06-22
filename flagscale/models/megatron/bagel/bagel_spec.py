# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Layer specifications for the BAGEL MoT (Mixture-of-Transformers) decoder."""
import warnings
from typing import Optional, Union

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.gpt.gpt_layer_specs import get_mlp_module_spec
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.typed_torch import not_none

try:
    import transformer_engine as te  # type: ignore[import-untyped]  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine import TEFusedMLP, TENorm
    from megatron.core.extensions.transformer_engine import (
        TEDotProductAttention,
        TELayerNormColumnParallelLinear,
        TENorm,
        TERowParallelLinear,
        TEColumnParallelLinear,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    import apex  # pylint: disable=unused-import

    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:
    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    LNImpl = WrappedTorchNorm
    HAVE_APEX = False

from .bagel_attention import PackedAttentionMoT, MoTSelfAttentionSubmodules
from .bagel_layer import MoTTransformerLayer, MoTTransformerLayerSubmodules


def get_bagel_layer_with_transformer_engine_spec(
    qk_layernorm: bool = False
) -> ModuleSpec:
    """ Bagel decoder TE spec"""

    return ModuleSpec(
        module=MoTTransformerLayer,
        submodules=MoTTransformerLayerSubmodules(
            # Understanding branch
            input_layernorm=TENorm,
            self_attention=ModuleSpec(
                module=PackedAttentionMoT,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=MoTSelfAttentionSubmodules(
                    # Understanding QKV
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention, # use flex attention actually
                    linear_proj=TERowParallelLinear,
                    q_layernorm=TENorm if qk_layernorm else IdentityOp,
                    k_layernorm=TENorm if qk_layernorm else IdentityOp,
                    # Generation QKV
                    linear_qkv_gen=TELayerNormColumnParallelLinear,
                    linear_proj_gen=TERowParallelLinear,
                    q_layernorm_gen=TENorm if qk_layernorm else IdentityOp,
                    k_layernorm_gen=TENorm if qk_layernorm else IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=TENorm,
            mlp=get_mlp_module_spec(use_te=True),
            mlp_bda=get_bias_dropout_add,
            # Generation branch
            input_layernorm_gen=TENorm,
            pre_mlp_layernorm_gen=TENorm,
            mlp_gen=get_mlp_module_spec(use_te=True), # The generation branch uses the same MLP architecture
            mlp_bda_gen=get_bias_dropout_add,
        ),
    )


def get_mlp_module_spec(use_te: bool = True) -> ModuleSpec:
    # Dense MLP w/ or w/o TE modules.
    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=not_none(TEColumnParallelLinear) if use_te else ColumnParallelLinear,
            linear_fc2=not_none(TERowParallelLinear) if use_te else RowParallelLinear,
        ),
    )

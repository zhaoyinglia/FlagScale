import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import (
    BlockMask,
    flex_attention,
    or_masks,
    and_masks,
    create_block_mask,
)

from megatron.core.parallel_state import (
    get_data_parallel_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.packed_seq_params import MoTPackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.attention import (
    CoreAttentionBuilder,
    LinearQkvBuilder,
    Attention,
)
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.tensor_parallel.mappings import all_gather_last_dim_from_tensor_parallel_region
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.enums import AttnMaskType, CudaGraphScope
from megatron.core.typed_torch import apply_module, not_none
from megatron.core.utils import (
    deprecate_inference_params,
    divide,
    get_pg_rank,
    get_pg_size,
    is_fa_min_version,
    is_te_min_version,
    is_using_quantization_scales,
    nvtx_range_pop,
    nvtx_range_push,
)

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import (
        SplitAlongDim,
        TELinear,
        set_save_original_input,
    )
else:
    SplitAlongDim, TELinear, set_save_original_input = None, None, None

# Increase ``torch.compile`` cache limits for flex_attention's generated kernels.
torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.accumulated_cache_size_limit = 4096
_compiled_flex_attention = torch.compile(flex_attention)


def _pad_to_length(tensor: Tensor, target_len: int, dim: int = 1) -> Tensor:
    """Pad *tensor* with zeros along *dim* so that ``tensor.size(dim) == target_len``."""
    pad_size = target_len - tensor.size(dim)
    if pad_size <= 0:
        return tensor
    shape = list(tensor.shape)
    shape[dim] = pad_size
    return torch.cat([tensor, tensor.new_zeros(shape)], dim=dim)


def create_sparse_mask(document_lens, split_lens, attn_modes, device):
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def full_and_noise_mask(b, h, q_idx, kv_idx):
        return (full_and_noise_seq_id[q_idx] == full_and_noise_seq_id[kv_idx]) & (full_and_noise_seq_id[q_idx] >= 0)

    def remove_noise_mask(b, h, q_idx, kv_idx):
        return (~((noise_seq_id[kv_idx] >= 0) & (noise_seq_id[q_idx] != noise_seq_id[kv_idx])))

    def sample_mask(b, h, q_idx, kv_idx):
        return document_id[q_idx] == document_id[kv_idx]

    full_and_noise_tmp = []
    noise_tmp = []

    for i, (length, model) in enumerate(zip(split_lens, attn_modes)):
        value = i if model in ['full', 'noise'] else -1
        full_and_noise_tmp.extend([value] * length)
        value_noise = i if model == 'noise' else -1
        noise_tmp.extend([value_noise] * length)

    full_and_noise_seq_id = torch.Tensor(full_and_noise_tmp).to(device)
    noise_seq_id = torch.Tensor(noise_tmp).to(device)

    document_id = torch.cat([torch.full((l,), i) for i, l in enumerate(document_lens, start=1)]).to(device)

    return and_masks(or_masks(causal_mask, full_and_noise_mask), remove_noise_mask, sample_mask)


@dataclass
class MoTSelfAttentionSubmodules:
    """Submodule specs for MoT self-attention with dual QKV projections."""

    # Understanding branch
    linear_qkv: LinearQkvBuilder = None
    core_attention: CoreAttentionBuilder = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None

    # Generation branch
    linear_qkv_gen: LinearQkvBuilder = None
    linear_proj_gen: Union[ModuleSpec, type] = None
    q_layernorm_gen: Union[ModuleSpec, type] = None
    k_layernorm_gen: Union[ModuleSpec, type] = None


class PackedAttentionMoT(Attention):
    """Mixture-of-Transformers Self-Attention using PyTorch ``flex_attention``.

    Each token type (understanding / generation) has its own QKV and output
    projections while the core attention kernel is shared.  The shared
    attention is computed via ``flex_attention`` which accepts a compiled
    ``BlockMask`` for efficient block-sparse masking (causal + per-sample
    document boundaries).

    Data layout (training, packed 1D)::

        Input:   [total_seq, 1, hidden]   (Megatron convention: [s, b, h])
        QKV:     [total_seq, num_heads, head_dim]
        flex_attention expects [batch=1, num_heads, total_seq, head_dim]

    Token routing uses **index tensors** (``packed_und_token_indexes`` and
    ``packed_gen_token_indexes``) for scatter / gather — matching the original
    BAGEL implementation.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MoTSelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.causal,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        pp_layer_offset: Optional[int] = None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            pp_layer_offset=pp_layer_offset,
        )

        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.config.hidden_size // self.num_heads
        self.linear_qkv_out_dim = self.query_projection_size + 2 * self.kv_projection_size
        self.num_kv_heads = self.config.num_query_groups
        if self.config.attention_output_gate:
            self.linear_qkv_out_dim += self.config.kv_channels * self.config.num_attention_heads

        # ---- Understanding branch (reuse SelfAttention for QKV + layernorms) ----
        self.linear_qkv = submodules.linear_qkv(
            self.config.hidden_size,
            self.linear_qkv_out_dim,
            config=self.config,
            init_method=not_none(self.config.init_method),
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='qkv',
            tp_group=self.pg_collection.tp,
        )

        self.linear_qkv_gen = submodules.linear_qkv_gen(
            self.config.hidden_size,
            self.linear_qkv_out_dim,
            config=self.config,
            init_method=not_none(self.config.init_method),
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='qkv_gen',
            tp_group=self.pg_collection.tp,
        )

        tp_world_size = get_tensor_model_parallel_world_size()
        assert tp_world_size <= 1, "TP world size must be less than 1 for qk_layernorm_hidden_dim"
        self.q_layernorm = submodules.q_layernorm(
            hidden_size=self.hidden_size_per_attention_head,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )
        self.q_layernorm_gen = submodules.q_layernorm_gen(
            hidden_size=self.hidden_size_per_attention_head,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )
        self.k_layernorm = submodules.k_layernorm(
            hidden_size=self.hidden_size_per_attention_head,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )
        self.k_layernorm_gen = submodules.k_layernorm_gen(
            hidden_size=self.hidden_size_per_attention_head,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

        self.flex_attention = _compiled_flex_attention

        self.linear_proj = build_module(
            submodules.linear_proj,
            self.query_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name='proj',
            tp_group=self.pg_collection.tp,
        )

        self.linear_proj_gen = build_module(
            submodules.linear_proj_gen,
            self.query_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name='proj_gen',
            tp_group=self.pg_collection.tp,
        )

    # def _split_gen_qkv(self, mixed_qkv_gen: Tensor):
    #     """Split fused gen QKV into (query, key, value), apply layernorms."""
    #     num_qh_per_group = self.num_heads // self.num_kv_heads
    #     new_shape = mixed_qkv_gen.size()[:-1] + (
    #         self.num_kv_heads,
    #         (num_qh_per_group + 2) * self.head_dim,
    #     )
    #     mixed_qkv_gen = mixed_qkv_gen.view(*new_shape)

    #     split_sizes = [
    #         num_qh_per_group * self.head_dim,
    #         self.head_dim,
    #         self.head_dim,
    #     ]
    #     query, key, value = torch.split(mixed_qkv_gen, split_sizes, dim=-1)
    #     # [*, ng, np/ng * hn] -> [*, np, hn]
    #     query = query.reshape(*query.shape[:-2], -1, self.head_dim)

    #     if self.q_layernorm_gen is not None:
    #         query = self.q_layernorm_gen(query)
    #     if self.k_layernorm_gen is not None:
    #         key = self.k_layernorm_gen(key)
    #     return query, key, value

    def get_query_key_value_tensors(
        self,
        hidden_states: Tensor,
        key_value_states: Tensor | None = None,
        output_gate: bool = False,
        split_qkv: bool = True,
        mode: str = "und"
    ) -> (
        tuple[Tensor, Tensor, Tensor, Tensor]
        | tuple[Tensor, Tensor, Tensor]
        | tuple[Tensor, list[int]]
    ):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        If `output_gate` is True, then also derives `gate` tensor.
        If `split_qkv=False`, then the unsplit mixed_qkv tensor is returned.
        """
        # If no output gate: Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        # If have output gate: Attention heads [sq, b, h] --> [sq, b, ng * (2 * np/ng + 2) * hn)]
        linear_qkv = self.linear_qkv if mode == "und" else self.linear_proj_gen
        mixed_qkv, _ = apply_module(linear_qkv)(hidden_states)
        print(f"{mixed_qkv.shape=}")
        num_query_heads_per_group = (
            self.num_attention_heads_per_partition // self.num_query_groups_per_partition
        )
        num_qkv_heads_per_group = num_query_heads_per_group + 2
        if output_gate:
            num_qkv_heads_per_group += num_query_heads_per_group

        assert self.config.num_query_groups is not None
        if self.config.num_query_groups < self.world_size:
            # Note that weights are interleaved in the following manner:
            # q1 q2 k1 v1 | q3 q4 k2 v2 | q5 q6 k3 v3 | ...
            # When tp_size > num_kv_heads, we split "q1 q2 k1 v1" over multiple
            # ranks, so a rank does not have a clean partitioning of just the q_heads
            # it needs. Instead, we perform the following steps:
            # 1. Assemble the full "q1 q2 k1 v1 | q3 q4 k2 v2 | q5 q6 k3 v3 | ..."
            #    through an AG.
            # 2. Pull out the right slice (e.g., "q1 q2 k1 v1" or "q3 q4 k2 v2").
            # 3. Split q_heads (e.g., q1, q2), k_heads (e.g., k1), v_heads (e.g., v1).
            # 4. Further index into query to get only the q_heads that this rank is
            #    responsible for (e.g., q1).
            # The block of code below performs steps 1 and 2.
            mixed_qkv = all_gather_last_dim_from_tensor_parallel_region(
                mixed_qkv, group=self.pg_collection.tp
            )
            idx = get_pg_rank(self.pg_collection.tp) // (
                self.world_size // self.config.num_query_groups
            )
            size = mixed_qkv.size()[-1] // self.config.num_query_groups
            mixed_qkv = mixed_qkv[:, :, idx * size : (idx + 1) * size]

        # If no output gate: [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        # If have output gate: [sq, b, hp] --> [sq, b, ng, (2 * np/ng + 2) * hn]
        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            num_qkv_heads_per_group * self.hidden_size_per_attention_head,
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        # Split the tensor into query, gate, key, and value.
        if output_gate:
            if not split_qkv:
                raise ValueError("split_qkv not supported for gated attention yet.")
            # If have output gate: [sq, b, ng, (2 * np/ng + 2) * hn]
            # --> [sq, b, ng, np/ng * hn], [sq, b, ng, np/ng * hn],
            # [sq, b, ng, hn], [sq, b, ng, hn]
            split_arg_list = [
                num_query_heads_per_group * self.hidden_size_per_attention_head,
                num_query_heads_per_group * self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head,
            ]

            if SplitAlongDim is not None:
                (query, gate, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list)
            else:
                (query, gate, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)
        else:
            # If no output gate: [sq, b, ng, (np/ng + 2) * hn]
            # --> [sq, b, ng, np/ng * hn], None, [sq, b, ng, hn], [sq, b, ng, hn]
            split_arg_list = [
                num_query_heads_per_group * self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head,
            ]

            # Return unsplit mixed_qkv and split_arg_list
            if not split_qkv:
                return mixed_qkv, split_arg_list

            print(f"{mixed_qkv.shape=}, {split_arg_list=}")
            if SplitAlongDim is not None:
                (query, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list)
            else:
                (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)

        # Query [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(query.size(0), query.size(1), self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)

        if self.config.num_query_groups < self.world_size:
            # query above corresponds to (num_q_heads / num_kv_heads) q_heads.
            # Index appropriately into query to get (num_q_heads / tp_size) q_heads.
            # This is step 4 in the list of steps above.
            idx = get_pg_rank(self.pg_collection.tp) % (
                self.world_size // self.config.num_query_groups
            )
            size = self.num_attention_heads_per_partition // (
                self.world_size // self.config.num_query_groups
            )
            query = query[:, :, idx * size : (idx + 1) * size, :]

        q_layernorm = self.q_layernorm if mode == "und" else self.q_layernorm_gen
        k_layernorm = self.k_layernorm if mode == "und" else self.k_layernorm_gen

        if q_layernorm is not None:
            ######### FlagScale Begin #########
            if not self.config.qk_layernorm_hidden_dim:
                query = apply_module(q_layernorm)(query)
            else:
                query_shape = list(query.shape)
                query = query.reshape(query.size(0), query.size(1), 1, -1)
                query = apply_module(q_layernorm)(query)
                query = query.reshape(*query_shape)
            ######### FlagScale End #########

        if k_layernorm is not None:
            ######### FlagScale Begin #########
            if not self.config.qk_layernorm_hidden_dim:
                key = apply_module(k_layernorm)(key)
            else:
                key_shape = list(key.shape)
                key = key.reshape(key.size(0), key.size(1), 1, -1)
                key = apply_module(k_layernorm)(key)
                key = key.reshape(*key_shape)
            ######### FlagScale End #########

        if self.config.test_mode:
            self.run_realtime_tests()

        if output_gate:
            # Gate [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
            gate = gate.reshape(*gate.shape[:2], -1, self.hidden_size_per_attention_head)
            if self.config.num_query_groups < self.world_size:
                idx = get_tensor_model_parallel_rank() % (
                    self.world_size // self.config.num_query_groups
                )
                size = self.num_attention_heads_per_partition // (
                    self.world_size // self.config.num_query_groups
                )
                gate = gate[:, :, idx * size : (idx + 1) * size, :]
            return query, key, value, gate

        return query, key, value

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        packed_und_token_indexes: Optional[Tensor] = None,
        packed_gen_token_indexes: Optional[Tensor] = None,
        inference_context: Optional[Any] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[MoTPackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
    ):
        """Forward with ``flex_attention`` for shared MoT attention.

        Args:
            hidden_states: ``[S, B=1, H]`` packed sequence.
            attention_mask: A ``BlockMask`` from ``create_block_mask`` (handles
                causal + document boundaries), **or** ``None`` for fallback to
                the und-branch ``SelfAttention``.
            packed_und_token_indexes: 1-D ``LongTensor`` — positions of
                *understanding* tokens in the packed sequence.
            packed_gen_token_indexes: 1-D ``LongTensor`` — positions of
                *generation* tokens.
            sample_lens: Per-sample lengths.  When
                ``sum(sample_lens) > seq_len`` the QKV tensors are zero-padded
                to match the ``BlockMask`` length before calling
                ``flex_attention`` and trimmed afterwards.

        Returns:
            ``(output, bias)`` with ``output.shape == hidden_states.shape``.
        """
        # # ---- fallback: no MoT routing → standard und attention ----
        # if packed_und_token_indexes is None or packed_gen_token_indexes is None:
        #     return self.attention_und(
        #         hidden_states,
        #         attention_mask=attention_mask,
        #         inference_context=inference_context,
        #         rotary_pos_emb=rotary_pos_emb,
        #         rotary_pos_cos=rotary_pos_cos,
        #         rotary_pos_sin=rotary_pos_sin,
        #         rotary_pos_cos_sin=rotary_pos_cos_sin,
        #         attention_bias=attention_bias,
        #         packed_seq_params=packed_seq_params,
        #         sequence_len_offset=sequence_len_offset,
        #     )
        sample_lens = packed_seq_params.sample_lens
        if attention_mask is None:
            print(f"{packed_seq_params=}, {packed_seq_params.sample_lens=}, {packed_seq_params.split_lens=}, {packed_seq_params.attn_modes=}")
            sparse_mask = create_sparse_mask(
                sample_lens,
                packed_seq_params.split_lens,
                packed_seq_params.attn_modes,
                hidden_states.device
            )
            seqlen = sum(sample_lens)
            block_mask = create_block_mask(
                sparse_mask, B=1, H=self.num_heads, Q_LEN=seqlen, KV_LEN=seqlen, 
                device=hidden_states.device, BLOCK_SIZE=128, _compile=True
            )
            attention_mask = block_mask
            print(f"{attention_mask=}")

        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        print(f"{hidden_states.shape=}")
        # ---- squeeze batch dim: [S, 1, H] -> [S, H] ----
        squeeze = hidden_states.dim() == 3 and hidden_states.size(1) == 1
        h2d = hidden_states.squeeze(1) if squeeze else hidden_states
        S = h2d.size(0)

        # ==============================================================
        # 1. Dual QKV projections — index-based scatter
        # ==============================================================
        q_buf = h2d.new_zeros(S, self.num_heads * self.head_dim)
        k_buf = h2d.new_zeros(S, self.num_kv_heads * self.head_dim)
        v_buf = h2d.new_zeros(S, self.num_kv_heads * self.head_dim)

        # --- und branch ---
        print(f"{h2d.shape=}, {packed_und_token_indexes.shape=}")
        und_h = h2d[packed_und_token_indexes]
        print(f"{und_h.shape=}")
        q_und, k_und, v_und = self.get_query_key_value_tensors(
            und_h.unsqueeze(1), mode="und"
        )
        # shapes: [und, 1, nH/nKV, D] -> [und, nH/nKV * D]
        q_buf[packed_und_token_indexes] = q_und.squeeze(1).reshape(q_und.size(0), -1)
        k_buf[packed_und_token_indexes] = k_und.squeeze(1).reshape(k_und.size(0), -1)
        v_buf[packed_und_token_indexes] = v_und.squeeze(1).reshape(v_und.size(0), -1)

        # --- gen branch ---
        if torch.sum(packed_gen_token_indexes) > 0:
            gen_h = h2d[packed_gen_token_indexes]
            q_gen, k_gen, v_gen = self.get_query_key_value_tensors(
                gen_h.unsqueeze(1), mode="und"
            )
            q_buf[packed_gen_token_indexes] = q_gen.reshape(q_gen.size(0), -1)
            k_buf[packed_gen_token_indexes] = k_gen.reshape(k_gen.size(0), -1)
            v_buf[packed_gen_token_indexes] = v_gen.reshape(v_gen.size(0), -1)

        # [S, nH, D]
        query = q_buf.view(S, self.num_heads, self.head_dim)
        key = k_buf.view(S, self.num_kv_heads, self.head_dim)
        value = v_buf.view(S, self.num_kv_heads, self.head_dim)

        torch.cuda.empty_cache()
        # ==============================================================
        # 2. RoPE
        # ==============================================================
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            # # --- Original: 直接调用，但输入缺 batch dim 会 broadcast 出错 ---
            # print(f"{query.shape=}, {q_pos_emb.shape=}")
            # query = apply_rotary_pos_emb(query, q_pos_emb, config=self.config)
            # key = apply_rotary_pos_emb(key, k_pos_emb, config=self.config)

            # --- 方案A: freqs 已按 position_ids 索引好，走 bshd 路径 ---
            # query: [S, nH, D] → unsqueeze(1) → [S, 1, nH, D] 补 batch dim
            # q_pos_emb: [S, 1, 1, rot_dim] 可正常 broadcast
            query = apply_rotary_pos_emb(
                query.unsqueeze(1), q_pos_emb, config=self.config, cu_seqlens=None
            ).squeeze(1)
            key = apply_rotary_pos_emb(
                key.unsqueeze(1), k_pos_emb, config=self.config, cu_seqlens=None
            ).squeeze(1)

        # ==============================================================
        # 3. flex_attention  (expects [B, nH, S, D])
        # ==============================================================
        q_fa = query.permute(1, 0, 2).unsqueeze(0)   # [1, nH, S, D]
        k_fa = key.permute(1, 0, 2).unsqueeze(0)     # [1, nKV, S, D]
        v_fa = value.permute(1, 0, 2).unsqueeze(0)   # [1, nKV, S, D]

        # Pad if the BlockMask was created for a padded length.
        pad_size = (sum(sample_lens) - S) if sample_lens is not None else 0
        if pad_size > 0:
            L = S + pad_size
            q_fa = _pad_to_length(q_fa, L, dim=2)
            k_fa = _pad_to_length(k_fa, L, dim=2)
            v_fa = _pad_to_length(v_fa, L, dim=2)

        use_gqa = self.num_kv_heads != self.num_heads

        if isinstance(attention_mask, BlockMask):
            attn_out = self.flex_attention(
                q_fa, k_fa, v_fa,
                block_mask=attention_mask,
                enable_gqa=use_gqa,
            )
        else:
            attn_out = self.flex_attention(
                q_fa, k_fa, v_fa,
                enable_gqa=use_gqa,
            )

        # Remove padding → [1, nH, S, D] -> [S, nH*D]
        if pad_size > 0:
            attn_out = attn_out[:, :, :S, :]
        attn_out = attn_out.squeeze(0).permute(1, 0, 2).reshape(S, -1)

        # ==============================================================
        # 4. Separate output projections — index-based scatter
        # ==============================================================
        out_buf = attn_out.new_zeros(S, attn_out.size(-1))

        und_attn = attn_out[packed_und_token_indexes].unsqueeze(1)
        proj_und, bias_und = apply_module(self.linear_proj)(und_attn)
        out_buf[packed_und_token_indexes] = proj_und.squeeze(1)

        gen_attn = attn_out[packed_gen_token_indexes].unsqueeze(1)
        proj_gen, bias_gen = apply_module(self.linear_proj_gen)(gen_attn)
        out_buf[packed_gen_token_indexes] = proj_gen.squeeze(1)

        # Restore [S, 1, H]
        if squeeze:
            out_buf = out_buf.unsqueeze(1)

        # Combine biases (typically ``None`` for RowParallelLinear w/o bias)
        if bias_und is not None and bias_und.numel() > 0:
            bias_buf = torch.zeros_like(out_buf)
            # biases from linear_proj are [und/gen_len, 1, H] or [H]
            b_und = bias_und.squeeze(1) if bias_und.dim() > 1 else bias_und
            b_gen = bias_gen.squeeze(1) if bias_gen.dim() > 1 else bias_gen
            if squeeze:
                bias_buf_2d = bias_buf.squeeze(1)  # [S, H]
                bias_buf_2d[packed_und_token_indexes] = b_und
                bias_buf_2d[packed_gen_token_indexes] = b_gen
            else:
                bias_buf[packed_und_token_indexes] = b_und
                bias_buf[packed_gen_token_indexes] = b_gen
        else:
            bias_buf = None

        return out_buf, bias_buf

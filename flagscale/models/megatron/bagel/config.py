import torch
from megatron.core.activations import fast_gelu, quick_gelu, squared_relu, gelu_tanh


def get_language_model_config(config):
    assert config.language_model_type == "qwen2.5_7B"
    config.activation_func = torch.nn.functional.silu
    config.add_bias_linear = False
    config.add_qkv_bias = True
    config.bias_activation_fusion = False
    config.gated_linear_unit = True
    config.apply_query_key_layer_scaling = False
    config.layernorm_zero_centered_gamma = (
        False  # Zero centered gamma not supported for RMSNorm
    )
    config.bias_dropout_fusion = False
    config.apply_rope_fusion = False
    config.attention_softmax_in_fp32 = True
    config.ffn_hidden_size = 18944
    return config


def get_vision_model_config(config, apply_query_key_layer_scaling):
    """Get the vision model config."""
    assert config.vision_model_type == "siglip"
    # # Select the output of the penultimate layer, not the last.
    # # So the num layer is the raw number 27 - 1
    # config.num_layers = 26
    # config.num_attention_heads = 16
    # config.add_bias_linear = True
    # config.add_qkv_bias = True
    # config.hidden_size = 1152
    # config.hidden_dropout = 0.0
    # config.attention_dropout = 0.0
    # config.ffn_hidden_size = 4304
    # config.gated_linear_unit = False
    # config.activation_func = fast_gelu
    # config.kv_channels = 72
    # config.num_attention_heads = 16
    # config.num_query_groups = 16
    # config.layernorm_zero_centered_gamma = False
    # config.apply_query_key_layer_scaling = apply_query_key_layer_scaling
    # config.bias_activation_fusion = False
    # config.bias_dropout_fusion = False
    # config.attention_softmax_in_fp32 = True
    # config.normalization = "LayerNorm"
    # config.apply_rope_fusion = False
    # config.layernorm_epsilon = 1e-6
    # # This is the temporary setting of recompute for the siglip model
    # config.recompute_method = None
    # config.recompute_granularity = None
    # config.recompute_num_layers = None

    config.num_layers = 26
    config.num_attention_heads = 16
    config.add_bias_linear = True
    config.add_qkv_bias = True
    config.hidden_size = 1152
    config.hidden_dropout = 0.0
    config.attention_dropout = 0.0
    config.ffn_hidden_size = 4304
    config.gated_linear_unit = False
    config.activation_func = gelu_tanh
    config.kv_channels = 72
    config.num_query_groups = 16
    config.layernorm_zero_centered_gamma = False
    config.apply_query_key_layer_scaling = apply_query_key_layer_scaling
    config.bias_activation_fusion = False
    config.bias_dropout_fusion = False
    config.attention_softmax_in_fp32 = True
    config.normalization = 'LayerNorm'
    config.apply_rope_fusion = False
    config.qk_layernorm = False
    config.layernorm_epsilon = 1e-6

    return config


def get_vision_projection_config(config, hidden_size):
    config.first_last_layers_bf16 = False
    config.num_layers_at_start_in_bf16 = 0
    config.num_layers_at_end_in_bf16 = 0

    config.gated_linear_unit = False
    config.bias_activation_fusion = False
    config.add_bias_linear = False
    config.hidden_size = hidden_size  # Used as the vision projection output size, i.e., the input to the language model.

    assert config.language_model_type == "qwen2.5_7B"
    config.ffn_hidden_size = 3584
    config.activation_func = gelu_tanh

    return config

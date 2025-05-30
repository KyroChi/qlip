import torch

config_template = { 
        "_attn_implementation_autoset": True,
        "attention_dropout": 0.0,
        "hidden_act": "quick_gelu",
        "hidden_size": 1024,
        "image_size": 336,
        "initializer_factor": 1.0,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "layer_norm_eps": 1e-05,
        "model_type": "clip_vision_model",
        "num_attention_heads": 16,
        "num_channels": 3,
        "num_hidden_layers": 24,
        "patch_size": 14,
        "projection_dim": 768,
        "transformers_version": "4.46.1",
        "vocab_size": 32000
}

mlp_default_config = {
        "in_features": 2,
        "out_features": 1024,
        "hidden_features": [1024, 1024],
        "activation": torch.nn.Tanh(),
        "num_fourier_features": 16
}
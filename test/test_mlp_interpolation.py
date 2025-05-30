# Trained MLP should have low error on the grid
import pytest
import torch

from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from transformers.models.clip.modeling_clip import CLIPVisionEmbeddings
from transformers.models.clip.configuration_clip import CLIPVisionConfig

from anyres_clip.clip_models import AnyresCLIPVisionEmbeddings
from anyres_clip.quadtree import TensorQuadtree, TensorQuadtreeNode, _get_interpolated_cell_indices

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

mlp_weights_file = "/nas04/krchicke/projects/anyres-clip-2/training/imagenette_trial/external/winter-energy_39.pt"

real_image_path = './assets/dog_fetch.jpg'

ATOL = 5e-5

def test_interpolation_loss():
    # Verification that we are training on the correct signal!
    clip_weights = "/nas04/krchicke/projects/anyres-clip-2/clip_emb_weights.pt"

    image_path = real_image_path
    image = Image.open(image_path)
    transform = Compose([
        Resize((336, 336)),
        ToTensor()
    ])
    image = transform(image).unsqueeze(0)
    image = 0.0001 * torch.rand_like(image)
    image = torch.zeros_like(image)

    config_base = CLIPVisionConfig(**config_template)

    model_qt = AnyresCLIPVisionEmbeddings(
        config_base, alpha=0.0, interpolation_mode="mlp", selection_mode="random",
    )

    model_interpolated = model_qt(image)
    model_qt.mode = "zero"
    embeddings_only = model_qt(image)

    interpolated_pos_enc = (model_interpolated - embeddings_only)[:, 1:, ...]

    x_coords = torch.linspace(-1, 1, 24)
    y_coords = torch.linspace(-1, 1, 24)

    grid = torch.stack(torch.meshgrid(y_coords, x_coords), dim=-1)

    predicted_pos_enc = model_qt.mlp(grid.flatten(0, 1)).reshape(24, 24, -1).flatten(0, 1)

    assert torch.allclose(predicted_pos_enc, interpolated_pos_enc, atol=ATOL), f"Predicted and interpolated position encodings differ: {torch.max(torch.abs(predicted_pos_enc - interpolated_pos_enc))} > {ATOL}"

def test_interpolate_bicubic():
    clip_weights = "/nas04/krchicke/projects/anyres-clip-2/clip_emb_weights.pt"
    image_path = real_image_path
    image = Image.open(image_path)
    transform = Compose([
        Resize((336, 336)),
        ToTensor()
    ])
    image = transform(image).unsqueeze(0)
    image = 0.0001 * torch.randn(1, 3, 336, 336)
    # image = torch.zeros_like(image)

    config_base = CLIPVisionConfig(**config_template)

    model_base = CLIPVisionEmbeddings(config_base)
    # model_base.load_state_dict(torch.load(clip_weights, map_location="cpu"), strict=False)

    model_qt = AnyresCLIPVisionEmbeddings(
        config_base, alpha=0.0, interpolation_mode="bicubic", zero_mode=False
    )

    model_qt.patch_embedding.weight = model_base.patch_embedding.weight
    model_qt.class_embedding = model_base.class_embedding
    model_qt.position_embedding.weight = model_base.position_embedding.weight

    embeddings_base = model_base(image)
    embeddings_qt = model_qt(image)

    assert embeddings_base.shape == embeddings_qt.shape
    print(embeddings_base[:, :10, :5])
    print(embeddings_qt[:, :10, :5])
    assert torch.nn.functional.mse_loss(embeddings_base, embeddings_qt) < ATOL, f"Base and QT embeddings differ L2: {torch.nn.functional.mse_loss(embeddings_base, embeddings_qt)} > {ATOL}"
    assert torch.allclose(embeddings_base, embeddings_qt, atol=ATOL**(1/2)), f"Base and QT embeddings differ Linf: {torch.max(torch.abs(embeddings_base - embeddings_qt))} > {ATOL**(1/2)}"

def test_interpolate_mlp_zero():
    clip_weights = "/nas04/krchicke/projects/anyres-clip-2/clip_emb_weights.pt"

    image_path = real_image_path
    image = Image.open(image_path)
    transform = Compose([
        Resize((336, 336)),
        ToTensor()
    ])
    image = transform(image).unsqueeze(0)
    image = 0.0001 * torch.rand_like(image)
    image = torch.zeros_like(image)

    config_base = CLIPVisionConfig(**config_template)

    model_base = CLIPVisionEmbeddings(config_base)
    model_base.load_state_dict(torch.load(clip_weights, map_location="cpu"), strict=False)

    model_base.position_embedding.weight = torch.nn.Parameter(torch.zeros_like(model_base.position_embedding.weight))

    model_qt = AnyresCLIPVisionEmbeddings(
        config_base, alpha=0.0, mlp_weights=mlp_weights_file, interpolation_mode="zero", selection_mode="random",
        mlp_depth=4, num_fourier_features=48
    )

    model_qt.patch_embedding.weight = model_base.patch_embedding.weight
    model_qt.class_embedding = model_base.class_embedding
    model_qt.position_embedding.weight = model_base.position_embedding.weight

    embeddings_base = model_base(image)
    embeddings_qt = model_qt(image)

    assert embeddings_base.shape == embeddings_qt.shape
    print(embeddings_base[:, :10, :5])
    print(embeddings_qt[:, :10, :5])
    assert torch.nn.functional.mse_loss(embeddings_base, embeddings_qt) < ATOL, f"Base and QT embeddings differ L2: {torch.nn.functional.mse_loss(embeddings_base, embeddings_qt)} > {ATOL}"
    assert torch.allclose(embeddings_base, embeddings_qt, atol=ATOL**(1/2)), f"Base and QT embeddings differ Linf: {torch.max(torch.abs(embeddings_base - embeddings_qt))} > {ATOL**(1/2)}"

def test_interpolate_mlp():
    clip_weights = "/nas04/krchicke/projects/anyres-clip-2/clip_emb_weights.pt"

    image_path = real_image_path
    image = Image.open(image_path)
    transform = Compose([
        Resize((336, 336)),
        ToTensor()
    ])
    image = transform(image).unsqueeze(0)
    image = 0.0001 * torch.rand_like(image)
    image = torch.zeros_like(image)

    config_base = CLIPVisionConfig(**config_template)

    model_base = CLIPVisionEmbeddings(config_base)
    model_base.load_state_dict(torch.load(clip_weights, map_location="cpu"), strict=False)

    model_qt = AnyresCLIPVisionEmbeddings(
        config_base, alpha=0.0, mlp_weights=mlp_weights_file, interpolation_mode="mlp", selection_mode="random",
        mlp_depth=4, num_fourier_features=48
    )

    model_qt.patch_embedding.weight = model_base.patch_embedding.weight
    model_qt.class_embedding = model_base.class_embedding
    model_qt.position_embedding.weight = model_base.position_embedding.weight

    embeddings_base = model_base(image)
    embeddings_qt = model_qt(image)

    assert embeddings_base.shape == embeddings_qt.shape
    print(embeddings_base[:, :10, :5])
    print(embeddings_qt[:, :10, :5])
    assert torch.nn.functional.mse_loss(embeddings_base, embeddings_qt) < ATOL, f"Base and QT embeddings differ L2: {torch.nn.functional.mse_loss(embeddings_base, embeddings_qt)} > {ATOL}"
    assert torch.allclose(embeddings_base, embeddings_qt, atol=ATOL**(1/2)), f"Base and QT embeddings differ Linf: {torch.max(torch.abs(embeddings_base - embeddings_qt))} > {ATOL**(1/2)}"

# def test_interpolate_mlp_zero_2():
#     mlp_weights_file = "/nas04/krchicke/projects/anyres-clip-2/training/imagenette_trial/fresh-paper-205/_299.pt" # "/nas04/krchicke/projects/anyres-clip/model_weights/frosty-durian-5_epoch_24.pt" #"/nas04/krchicke/projects/anyres-clip-2/training/imagenette_trial/april_9/skilled_oath_76_epoch_25.pt"
#     clip_weights = "/nas04/krchicke/projects/anyres-clip-2/clip_emb_weights.pt"

#     image_path = real_image_path
#     image = Image.open(image_path)
#     transform = Compose([
#         Resize((336, 336)),
#         ToTensor()
#     ])
#     image = transform(image).unsqueeze(0)
#     image = 0.0001 * torch.rand_like(image)

#     config_base = CLIPVisionConfig(**config_template)

#     model_base = CLIPVisionEmbeddings(config_base)
#     model_base.load_state_dict(torch.load(clip_weights, map_location="cpu"), strict=False)

#     model_qt = AnyresCLIPVisionEmbeddings(
#         config_base, alpha=0.0, mlp_weights=mlp_weights_file, interpolation_mode="mlp"
#     )

#     model_qt.patch_embedding.weight = model_base.patch_embedding.weight
#     model_qt.class_embedding = model_base.class_embedding
#     model_qt.position_embedding.weight = model_base.position_embedding.weight

#     embeddings_base = model_base(image)
#     embeddings_qt = model_qt(image)

#     assert embeddings_base.shape == embeddings_qt.shape
#     # print(embeddings_base[:, :10, :5])
#     # print(embeddings_qt[:, :10, :5])
#     assert torch.nn.functional.mse_loss(embeddings_base, embeddings_qt) < ATOL, f"Base and QT embeddings differ: {torch.nn.functional.mse_loss(embeddings_base, embeddings_qt)} > {ATOL}"
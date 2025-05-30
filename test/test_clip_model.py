import pytest
import torch

from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from transformers.models.clip.modeling_clip import CLIPVisionEmbeddings
from transformers.models.clip.configuration_clip import CLIPVisionConfig

from qlip.clip_models import QLIPCLIPVisionEmbeddings
from qlip.quadtree import TensorQuadtree, TensorQuadtreeNode, _get_interpolated_cell_indices

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

real_image_path = './assets/dog_fetch.jpg'

ATOL = 1e-5

def test_batched_model():
    """Tests that the quadtree model and base clip model have the same
    output shapes"""
    image = torch.rand(10, 3, 336, 672)
    mask = torch.ones(10, 1, 336, 672)

    batch_with_mask = torch.cat([image, mask], dim=1)

    config_base = CLIPVisionConfig(**config_template)
    model_qt = QLIPCLIPVisionEmbeddings(config_base)

    batched_embeddings = model_qt(batch_with_mask)

    single_embeddings = torch.cat(
        [model_qt(image[i].unsqueeze(0)) for i in range(10)],
        dim=0
    )

    assert single_embeddings.shape == batched_embeddings.shape
    assert torch.allclose(single_embeddings, batched_embeddings, atol=ATOL), f"Failed batched model test. Diff max: {torch.abs(single_embeddings - batched_embeddings).max()}"

    images = []
    for i in range(10):
        image = torch.rand(3, 224, 224 + i * 14)
        images.append(image)

    image_tensor = torch.zeros((10, 3, 224, 224 + 10 * 14))
    mask_tensor = torch.zeros((10, 1, 224, 224 + 10 * 14))

    for i, image in enumerate(images):
        w, h = image.shape[-2], image.shape[-1]
        image_tensor[i, :, :w, :h] = image
        mask_tensor[i, :, :w, :h] = 1

    batch_with_mask = torch.cat([image_tensor, mask_tensor], dim=1)

    batched_embeddings = model_qt(batch_with_mask)

    single_embeddings = []
    for i in range(10):
        image = images[i].unsqueeze(0)
        single_embeddings.append(model_qt(image))

    for i in range(10):
        assert single_embeddings[i].shape == batched_embeddings[i].shape
        assert torch.allclose(single_embeddings[i], batched_embeddings[i], atol=ATOL), f"Failed batched model test. Diff max: {torch.abs(single_embeddings[i] - batched_embeddings[i]).max()}"

def test_qt_vs_clip_shape():
    """Tests that the quadtree model and base clip model have the same
    output shapes"""
    image_path = real_image_path
    image = Image.open(image_path)
    transform = Compose([
        Resize((336, 336)),
        ToTensor()
    ])
    image = transform(image).unsqueeze(0)

    config_base = CLIPVisionConfig(**config_template)

    model_base = CLIPVisionEmbeddings(config_base)
    model_qt = QLIPCLIPVisionEmbeddings(config_base)

    embeddings_base = model_base(image)
    embeddings_qt = model_qt(image)

    assert embeddings_base.shape == embeddings_qt.shape

    model_qt = QLIPCLIPVisionEmbeddings(config=config_base, alpha=2 * 336**2 * 1000)

    embeddings_qt = model_qt(image)

    assert embeddings_qt.shape[1] == 9 + 1 # 9 channels from patch embeddings and 1 channel for class embedding

def test_qt_vs_clip_no_interpolation():
    image_path = real_image_path
    image = Image.open(image_path)
    transform = Compose([
        Resize((336, 336)),
        ToTensor()
    ])
    image = torch.ones(1, 3, 336, 336) + 1e-5 * torch.rand(1, 3, 336, 336)
    # image = transform(image).unsqueeze(0)

    config_base = CLIPVisionConfig(**config_template)

    model_base = CLIPVisionEmbeddings(config_base)
    model_qt = QLIPCLIPVisionEmbeddings(config_base, alpha=0.0, interpolation_mode="zero")

    # Ensure that the weights are the same
    model_qt.patch_embedding.weight = model_base.patch_embedding.weight
    model_qt.class_embedding = model_base.class_embedding
    model_qt.position_embedding.weight = model_base.position_embedding.weight

    with torch.no_grad():
      embeddings_base = model_base(image)
      embeddings_qt = model_qt(image)[:, 1:, :]
      
    # subtract off CLIP positional embedding
    embeddings_base = embeddings_base - model_base.position_embedding(torch.arange(577).expand((1, -1)))
    embeddings_base = embeddings_base[:, 1:, :]

    print(embeddings_base[:, 10:30, 125])
    print(embeddings_qt[:, 10:30, 125])

    assert torch.allclose(embeddings_qt, embeddings_base, atol=ATOL), f"Failed comparison test. Diff max: {torch.abs(embeddings_qt - embeddings_base).max()}"

def test_qt_vs_clip_zero():
    """Outputs should be identical when alpha is zero"""
    image = 0.01 * torch.randn(1, 3, 336, 336)
    cell = torch.ones(3, 14, 14)

    image[:, :, 14:28, 14:28] = cell

    config_base = CLIPVisionConfig(**config_template)

    model_base = CLIPVisionEmbeddings(config_base)
    model_qt = QLIPCLIPVisionEmbeddings(
        config_base, 
        interpolation_mode="bilinear",
        alpha=0.0,
    )

    # Ensure that the weights are the same
    model_qt.patch_embedding.weight = model_base.patch_embedding.weight
    model_qt.class_embedding = model_base.class_embedding
    model_qt.position_embedding.weight = model_base.position_embedding.weight

    with torch.no_grad():
      embeddings_base = model_base(image)
      embeddings_qt, qt = model_qt(
            image, return_qt=True
        )

    assert embeddings_base.shape == embeddings_qt.shape

    # Reshape the quadtree output to have the same order that CLIP does
    reshaped_base = embeddings_base[:, 1:, :]
    reshaped_qt = embeddings_qt[:, 1:, :] #embeddings_qt[:, 1:, :].transpose(1, 2).reshape(1, 1024, 24, 24).flatten(2).transpose(1, 2)

    diff = torch.abs(reshaped_qt - reshaped_base)
    assert torch.allclose(reshaped_qt, reshaped_base, atol=ATOL), f"Failed simple comparison test. Diff max: {diff.max()}"
    
    image_path = real_image_path
    image = Image.open(image_path)
    transform = Compose([
        Resize((336, 336)),
        ToTensor()
    ])
    image = transform(image).unsqueeze(0)

    config_base = CLIPVisionConfig(**config_template)

    model_base = CLIPVisionEmbeddings(config_base)
    model_qt = QLIPCLIPVisionEmbeddings(config_base, alpha=0.0, interpolation_mode="bicubic")

    # Ensure that the weights are the same
    model_qt.patch_embedding.weight = model_base.patch_embedding.weight
    model_qt.class_embedding = model_base.class_embedding
    model_qt.position_embedding.weight = model_base.position_embedding.weight

    with torch.no_grad():
      embeddings_base = model_base(image)
      embeddings_qt = model_qt(image)

    # Reshape the quadtree output to have the same order that CLIP does
    reshaped_base = embeddings_base[:, 1:, :]
    reshaped_qt = embeddings_qt[:, 1:, :]

    diff = torch.abs(reshaped_qt - reshaped_base)
    assert torch.allclose(reshaped_qt, reshaped_base, atol=ATOL), f"Failed real image comparison test, diff mean: {diff.mean()}"

    # Stochastic testing
    # for i in range(10):
    #     image = torch.randn(10, 3, 336, 336)

    #     config_base = CLIPVisionConfig(**config_template)

    #     model_base = CLIPVisionEmbeddings(config_base)
    #     model_qt = QLIPCLIPVisionEmbeddings(config_base, alpha=0.0, interpolation_mode="bilinear")

    #     # Ensure that the weights are the same
    #     model_qt.patch_embedding.weight = model_base.patch_embedding.weight
    #     model_qt.class_embedding = model_base.class_embedding
    #     model_qt.position_embedding.weight = model_base.position_embedding.weight

    #     with torch.no_grad():
    #         embeddings_base = model_base(image)
    #         embeddings_qt = model_qt(image)

    #     # Reshape the quadtree output to have the same order that CLIP does
    #     reshaped_base = embeddings_base[:, 1:, :]
    #     reshaped_qt = embeddings_qt[:, 1:, :]

    #     print(reshaped_base[3, 10:30, 125])
    #     print(reshaped_qt[3, 10:30, 125])

    #     diff = torch.abs(reshaped_qt - reshaped_base)
    #     assert torch.allclose(reshaped_qt, reshaped_base, atol=ATOL), f"Failed stochastic image comparison test {i}, diff mean: {diff.mean()}"

def test__get_interpolated_cell_indices():
    config_qt = CLIPVisionConfig(**config_template)
    model_qt = QLIPCLIPVisionEmbeddings(config_qt, alpha=0.0)

    image = torch.randn((1, 3, 336, 336))

    dummy_qt = TensorQuadtree(image, patch_size=14, alpha=0)
    dummy_qt.leaf_nodes = [[]]
    for ii in range(24):
        for jj in range(24):
            dummy_qt.leaf_nodes[0].append(TensorQuadtreeNode(0, ii * 14, jj * 14, 14, 14))
            assert _get_interpolated_cell_indices(model_qt.patch_size, image, ii * 14, jj * 14, 14, 14) == [ii * 24 + jj + 1]

    n_patchs = 8
    image = torch.randn((1, 3, 14 * n_patchs, 14 * n_patchs))
    dummy_qt = TensorQuadtree(image, patch_size=14, alpha=0)

    dummy_qt.leaf_nodes = [
        [TensorQuadtreeNode(0, 0, 0, 14 * n_patchs, 14 * n_patchs)] for _ in range(3)
    ]

    assert _get_interpolated_cell_indices(dummy_qt.patch_size, image, 0, 0, 14 * n_patchs, 14 * n_patchs) == list(range(1, n_patchs**2 + 1))

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_interpolate_pos_encoding(dtype):
    image_path = real_image_path
    image = Image.open(image_path)
    transform = Compose([
        Resize((336, 336)),
        ToTensor()
    ])
    image = transform(image).unsqueeze(0).to(dtype)

    config_base = CLIPVisionConfig(**config_template)
    model_base = CLIPVisionEmbeddings(config_base).to(dtype)

    model_qt = QLIPCLIPVisionEmbeddings(config_base).to(dtype)

    model_qt.patch_embedding.weight = model_base.patch_embedding.weight
    model_qt.class_embedding = model_base.class_embedding
    model_qt.position_embedding.weight = model_base.position_embedding.weight

    out = model_qt(image)

    # Test that the positional encoding is the same if we feed the patches into 
    # the model in the same order as we do for CLIP
    dummy_qt = TensorQuadtree(image, patch_size=14, alpha=0)
    dummy_qt.leaf_nodes = [[]]
    for ii in range(24):
        for jj in range(24):
            dummy_qt.leaf_nodes[0].append(TensorQuadtreeNode(0, ii * 14, jj * 14, 14, 14))
            assert _get_interpolated_cell_indices(dummy_qt.patch_size, dummy_qt.input_tensor, ii * 14, jj * 14, 14, 14) == [ii * 24 + jj + 1]

    qt_pos_seq = model_qt.interpolate_pos_encoding(out, dummy_qt)
    base_pos_seq = model_base.position_embedding(model_base.position_ids)

    assert qt_pos_seq.shape == base_pos_seq.shape
    assert qt_pos_seq.dtype == base_pos_seq.dtype, f"qt_pos_seq.dtype={qt_pos_seq.dtype}, base_pos_seq.dtype={base_pos_seq.dtype}"
    assert qt_pos_seq.dtype == dtype, f"qt_pos_seq.dtype={qt_pos_seq.dtype}, dtype={dtype}"

    diff = torch.abs(qt_pos_seq - base_pos_seq)
    assert torch.allclose(diff, torch.zeros_like(diff), atol=ATOL), f"Failed sequence comparison test. Max diff: {diff.max()}. dtype={dtype}."

    # Test that we can "sort" the quadtreeified patches into the correct order
    out, qt = model_qt(image, debug=True)

    quadtree_patch_order = [0]
    for node in qt.leaf_nodes[0]:
        patch_idxs = _get_interpolated_cell_indices(
            qt.patch_size,
            qt.input_tensor, 
            node.x, node.y, node.height, node.width
        )
        quadtree_patch_order.append(patch_idxs[0]) # already adds +1 for the class token

    qt_pos_seq = model_qt.interpolate_pos_encoding(out, qt)
    base_pos_seq = model_base.position_embedding(model_base.position_ids)

    order_tensor = torch.Tensor(quadtree_patch_order).to(torch.long)
    sorted_indices = torch.argsort(order_tensor)
    sorted_qt_pos_seq = qt_pos_seq[:, sorted_indices, :]

    diff = torch.abs(base_pos_seq - sorted_qt_pos_seq)
    print(diff.shape)
    print(torch.mean(diff, dim=(-1)))
    assert torch.allclose(base_pos_seq, sorted_qt_pos_seq, atol=ATOL), f"Failed real pos encoding test. Diff mean: {diff}"

def test_pos_enc_double():
    image_path = real_image_path
    image = Image.open(image_path)
    transform_1 = Compose([
        Resize((336, 336)),
        ToTensor()
    ])
    image_1 = transform_1(image).unsqueeze(0)

    transform_2 = Compose([
        Resize((336 * 2, 336 * 2)),
        ToTensor()
    ])
    image_2 = transform_2(image).unsqueeze(0)

    assert image_1.shape == (1, 3, 336, 336)
    assert image_2.shape == (1, 3, 672, 672)

    config_qt_1 = CLIPVisionConfig(**config_template)
    model_qt_1 = QLIPCLIPVisionEmbeddings(config_qt_1, alpha=0.0)

    config_template_mod = config_template.copy()

    config_template_mod['image_size'] = 336 * 2
    config_template_mod['patch_size'] = 28
    config_qt_2 = CLIPVisionConfig(**config_template_mod)
    model_qt_2 = QLIPCLIPVisionEmbeddings(config_qt_2, alpha=0.0)

    assert model_qt_1.class_embedding.shape == model_qt_2.class_embedding.shape
    model_qt_1.class_embedding = model_qt_2.class_embedding

    assert model_qt_1.position_embedding.weight.shape == model_qt_2.position_embedding.weight.shape
    model_qt_1.position_embedding.weight = model_qt_2.position_embedding.weight

    out_1 = model_qt_1(image_1)
    out_2 = model_qt_2(image_2)

    # Test that the positional encoding is the same if we feed the patches into 
    # the model in the same order as we do for CLIP
    dummy_qt_1 = TensorQuadtree(image_1, patch_size=14, alpha=0)
    dummy_qt_1.leaf_nodes = [[]]
    for ii in range(24):
        for jj in range(24):
            dummy_qt_1.leaf_nodes[0].append(TensorQuadtreeNode(0, ii * 14, jj * 14, 14, 14))
            assert _get_interpolated_cell_indices(dummy_qt_1.patch_size, dummy_qt_1.input_tensor, ii * 14, jj * 14, 14, 14) == [ii * 24 + jj + 1]

    qt_pos_seq_1 = model_qt_1.interpolate_pos_encoding(out_1, dummy_qt_1)

    dummy_qt_2 = TensorQuadtree(image_2, patch_size=28, alpha=0)
    dummy_qt_2.leaf_nodes = [[]]
    for ii in range(24):
        for jj in range(24):
            dummy_qt_2.leaf_nodes[0].append(TensorQuadtreeNode(0, ii * 28, jj * 28, 28, 28))
            assert _get_interpolated_cell_indices(dummy_qt_2.patch_size, dummy_qt_2.input_tensor, ii * 28, jj * 28, 28, 28) == [ii * 24 + jj + 1]


    qt_pos_seq_2 = model_qt_2.interpolate_pos_encoding(out_2, dummy_qt_2)

    assert qt_pos_seq_1.shape == qt_pos_seq_2.shape, f"Failed shape test. qt_pos_seq_1.shape={qt_pos_seq_1.shape}, qt_pos_seq_2.shape={qt_pos_seq_2.shape}"

    diff = torch.abs(qt_pos_seq_1 - qt_pos_seq_2)
    assert torch.allclose(diff, torch.zeros_like(diff), atol=ATOL), f"Failed sequence comparison test. Max diff: {diff.max()}"

def test_qt_shape():
    image = 30 * torch.rand(3, 3, 336, 336)
    config_qt = CLIPVisionConfig(**config_template)
    model_qt = QLIPCLIPVisionEmbeddings(config_qt, alpha=0.0)

    out = model_qt(image)

    assert out.shape == (3, 577, 1024)

def test_prepare_image():
    pass

def test_zero_mode():
    """Test that zero mode is identical to the original model"""
    image = 0.01 * torch.randn(1, 3, 336, 336)

    config_base = CLIPVisionConfig(**config_template)

    model_base = CLIPVisionEmbeddings(config_base)
    model_qt = QLIPCLIPVisionEmbeddings(
        config_base, 
        interpolation_mode="bilinear",
        alpha=0.0,
        zero_mode=True,
    )

    # Ensure that the weights are the same
    model_qt.patch_embedding.weight = model_base.patch_embedding.weight
    model_qt.class_embedding = model_base.class_embedding
    model_qt.position_embedding.weight = model_base.position_embedding.weight

    with torch.no_grad():
      embeddings_base = model_base(image)
      embeddings_qt = model_qt(image)

    assert embeddings_base.shape == embeddings_qt.shape

    # Reshape the quadtree output to have the same order that CLIP does
    reshaped_base = embeddings_base[:, 1:, :]
    reshaped_qt = embeddings_qt[:, 1:, :] #embeddings_qt[:, 1:, :].transpose(1, 2).reshape(1, 1024, 24, 24).flatten(2).transpose(1, 2)

    diff = torch.abs(reshaped_qt - reshaped_base)
    assert torch.allclose(reshaped_qt, reshaped_base, atol=ATOL), f"Failed simple comparison test. Diff max: {diff.max()}"
    
    image_path = real_image_path
    image = Image.open(image_path)
    transform = Compose([
        Resize((336, 336)),
        ToTensor()
    ])
    image = transform(image).unsqueeze(0)

    config_base = CLIPVisionConfig(**config_template)

    model_base = CLIPVisionEmbeddings(config_base)
    model_qt = QLIPCLIPVisionEmbeddings(config_base, alpha=0.0, interpolation_mode="bicubic", zero_mode=True)

    # Ensure that the weights are the same
    model_qt.patch_embedding.weight = model_base.patch_embedding.weight
    model_qt.class_embedding = model_base.class_embedding
    model_qt.position_embedding.weight = model_base.position_embedding.weight

    with torch.no_grad():
      embeddings_base = model_base(image)
      embeddings_qt = model_qt(image)

    # Reshape the quadtree output to have the same order that CLIP does
    reshaped_base = embeddings_base[:, 1:, :]
    reshaped_qt = embeddings_qt[:, 1:, :]

    diff = torch.abs(reshaped_qt - reshaped_base)
    assert torch.allclose(reshaped_qt, reshaped_base, atol=ATOL), f"Failed real image comparison test, diff mean: {diff.mean()}"

def test_embeddings_align():
    """Test that zero mode is identical to the original model"""
    image = 0.01 * torch.randn(1, 3, 336, 336)

    config_base = CLIPVisionConfig(**config_template)

    model_base = CLIPVisionEmbeddings(config_base)
    model_qt = QLIPCLIPVisionEmbeddings(
        config_base, 
        interpolation_mode="bilinear",
        alpha=0.0,
        zero_mode=False,
    )

    # Ensure that the weights are the same
    model_qt.patch_embedding.weight = model_base.patch_embedding.weight
    model_qt.class_embedding = model_base.class_embedding
    model_qt.position_embedding.weight = model_base.position_embedding.weight

    with torch.no_grad():
      embeddings_base = model_base(image)
      embeddings_qt = model_qt(image)

    assert embeddings_base.shape == embeddings_qt.shape

    # Reshape the quadtree output to have the same order that CLIP does
    reshaped_base = embeddings_base
    reshaped_qt = embeddings_qt

    diff = torch.abs(reshaped_qt - reshaped_base)
    assert torch.allclose(reshaped_qt, reshaped_base, atol=ATOL), f"Failed simple comparison test. Diff max: {diff.max()}"
    
    image_path = real_image_path
    image = Image.open(image_path)
    transform = Compose([
        Resize((336, 336)),
        ToTensor()
    ])
    image = transform(image).unsqueeze(0)

    config_base = CLIPVisionConfig(**config_template)

    model_base = CLIPVisionEmbeddings(config_base)
    model_qt = QLIPCLIPVisionEmbeddings(config_base, alpha=0.0, interpolation_mode="bicubic", zero_mode=True)

    # Ensure that the weights are the same
    model_qt.patch_embedding.weight = model_base.patch_embedding.weight
    model_qt.class_embedding = model_base.class_embedding
    model_qt.position_embedding.weight = model_base.position_embedding.weight

    with torch.no_grad():
      embeddings_base = model_base(image)
      embeddings_qt = model_qt(image)

    # Reshape the quadtree output to have the same order that CLIP does
    reshaped_base = embeddings_base[:, 1:, :]
    reshaped_qt = embeddings_qt[:, 1:, :]

    diff = torch.abs(reshaped_qt - reshaped_base)
    assert torch.allclose(reshaped_qt, reshaped_base, atol=ATOL), f"Failed real image comparison test, diff mean: {diff.mean()}"
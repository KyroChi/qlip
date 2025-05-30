import pytest
import torch

from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor

from anyres_clip.quadtree import TensorQuadtree
from anyres_clip.utils import get_max_depth

real_image_path = './assets/dog_fetch.jpg'

def test_quadtree():
    test_image = torch.zeros(1, 3, 28, 28)
    cell = torch.ones(3, 14, 14)

    test_image[0, :, 0:14, 0:14] = cell
    test_image[0, :, 14:28, 14:28] = cell

    qt = TensorQuadtree(test_image, patch_size=14, alpha=0.0)

    assert len(qt.leaf_nodes) == 1 # For batch size 1
    assert len(qt.leaf_nodes[0]) == 4 # zero alpha should maximally split

    qt_tensor = qt.as_tensor()

    assert torch.allclose(test_image, qt_tensor)

def test_batched_quadtree():
    batched_test_images = torch.zeros(3, 3, 28, 28)

    cell = torch.ones(3, 14, 14)

    batched_test_images[0, :, 0:14, 0:14] = cell
    batched_test_images[0, :, 14:28, 14:28] = cell

    batched_test_images[2, :, 0:14, 14:28] = cell
    batched_test_images[2, :, 14:28, 0:14] = cell

    qt = TensorQuadtree(batched_test_images, patch_size=14, alpha=0.0)

    assert len(qt.leaf_nodes) == 3
    assert len(qt.leaf_nodes[0]) == 4
    assert len(qt.leaf_nodes[1]) == 1
    assert len(qt.leaf_nodes[2]) == 4

    qt_tensor = qt.as_tensor()

    assert torch.allclose(batched_test_images, qt_tensor)

def test_real_image():
    image_path = real_image_path
    image = Image.open(image_path)
    transform = Compose([
        Resize((336, 336)),
        ToTensor()
    ])
    image = transform(image).unsqueeze(0)

    # Max splits
    qt = TensorQuadtree(image, patch_size=14, alpha=0.0)
    assert len(qt.leaf_nodes) == 1
    assert len(qt.leaf_nodes[0]) == 576

    # Minimum splits
    qt = TensorQuadtree(image, patch_size=14, alpha=2 * 336**2 * image.var())
    assert len(qt.leaf_nodes) == 1
    assert len(qt.leaf_nodes[0]) == 9

def test_nonzero_alpha():
    batched_test_images = torch.zeros(3, 3, 28, 28)

    cell = torch.ones(3, 14, 14)

    batched_test_images[0, :, 0:14, 0:14] = 0.1 * cell
    batched_test_images[0, :, 14:28, 14:28] = 0.1 * cell

    batched_test_images[1, :, 0:14, 14:28] = 0.2 * cell
    batched_test_images[1, :, 14:28, 0:14] = 0.2 * cell

    batched_test_images[2, :, 0:14, 0:14] = 0.3 * cell
    batched_test_images[2, :, 14:28, 14:28] = 0.3 * cell

    alpha_1 = torch.var(batched_test_images[0, ...]) * 28**2
    alpha_2 = torch.var(batched_test_images[1, ...]) * 28**2
    alpha_3 = torch.var(batched_test_images[2, ...]) * 28**2

    assert alpha_1 < alpha_2 and alpha_2 < alpha_3

    # qt1 should split for channels 1 and 2, but not for 0
    # qt2 should split for channel 2, but not for 0 and 1
    # qt3 will not do any splits

    qt1 = TensorQuadtree(batched_test_images, patch_size=14, alpha=alpha_1)
    qt2 = TensorQuadtree(batched_test_images, patch_size=14, alpha=alpha_2)
    qt3 = TensorQuadtree(batched_test_images, patch_size=14, alpha=alpha_3)

    assert len(qt1.leaf_nodes[0]) == 1
    assert len(qt1.leaf_nodes[1]) == 4
    assert len(qt1.leaf_nodes[2]) == 4

    assert len(qt2.leaf_nodes[0]) == 1
    assert len(qt2.leaf_nodes[1]) == 1
    assert len(qt2.leaf_nodes[2]) == 4

    assert len(qt3.leaf_nodes[0]) == 1
    assert len(qt3.leaf_nodes[1]) == 1
    assert len(qt3.leaf_nodes[2]) == 1

def test_non_square_image():
    image_path = real_image_path
    image = Image.open(image_path)
    transform = Compose([
        ToTensor()
    ])
    image = transform(image).unsqueeze(0)

    # Max splits
    qt = TensorQuadtree(image, patch_size=10, alpha=0.0)
    assert len(qt.leaf_nodes) == 1
    assert len(qt.leaf_nodes[0]) == 32 * 48
    assert qt.max_size[0] == 32
    assert qt.max_size[1] == 48

    print(image.shape)
    # Minimum splits
    qt = TensorQuadtree(image, patch_size=32, alpha=1000 * 2 * 400 * 600 * image.var())
    print(qt.max_size)
    assert len(qt.leaf_nodes) == 1
    assert len(qt.leaf_nodes[0]) == 3 * 4
    assert qt.max_size[0] == 12
    assert qt.max_size[1] == 16
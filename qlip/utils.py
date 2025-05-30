import math
import numpy as np
import os
import random
import torch

def seed_everything(seed: int = 42) -> None:
    # Copied from StackOverflow user cookiemonster:
    # https://stackoverflow.com/a/57417097
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _get_max_depth_one_dim(dim: int, patch_size: int):
    assert dim >= patch_size, f"Max depth computation failed, dim={dim} is smaller than patch_size={patch_size}."

    divisions = dim / patch_size
    return int(math.log2(divisions))

def get_max_depth(tensor: torch.Tensor, patch_size: int):
   return min(
        _get_max_depth_one_dim(tensor.shape[-1], patch_size),
        _get_max_depth_one_dim(tensor.shape[-2], patch_size)
   )

def pad_for_patchify(tensor: torch.Tensor, patch_size: int):
    assert tensor.dim() == 3, f"pad_for_patchify failed, tensor.dim()={tensor.dim()} is not 3."
    assert tensor.shape[1] >= patch_size, f"pad_for_patchify failed, tensor.shape[1]={tensor.shape[1]} is smaller than patch_size={patch_size}."
    assert tensor.shape[2] >= patch_size, f"pad_for_patchify failed, tensor.shape[2]={tensor.shape[2]} is smaller than patch_size={patch_size}."

    width = tensor.shape[1]
    height = tensor.shape[2]
    
    new_width = width + 2 * patch_size - width % patch_size
    new_height = height + 2 * patch_size - height % patch_size

    new_tensor = torch.zeros(tensor.shape[0], new_width, new_height)

    new_tensor[:, patch_size:patch_size + width, patch_size:patch_size + height] = tensor

    return new_tensor 

def center_crop_to_patch_size(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    _, _, h, w = image.shape

    new_height = h - h % patch_size
    new_width = w - w % patch_size

    offset_h = (h - new_height) // 2
    offset_w = (w - new_width) // 2

    return image[:, :, offset_h:offset_h + new_height, offset_w:offset_w + new_width]

def center_crop_to_quadtree_size(image: torch.Tensor, patch_size: int, crop_info: bool = False) -> torch.Tensor:
    """
    Center crops the input image tensor such that 
    H and W are of the form 2^n * patch_size * m, where n is the largest possible
    integer and m is strictly less than 2^n.
    """
    _, _, h, w = image.shape

    max_h_depth = _get_max_depth_one_dim(h, patch_size)
    max_w_depth = _get_max_depth_one_dim(w, patch_size)

    if max_h_depth != 0:
        M = int( h / patch_size / 2**(max_h_depth - 1) )
        new_height = int(M * patch_size * 2**(max_h_depth - 1))
    else:
        new_height = patch_size

    if max_w_depth != 0:
        N = int( w / patch_size / 2**(max_w_depth - 1) )
        new_width = int(N * patch_size * 2**(max_w_depth - 1))
    else:
        new_width = patch_size

    offset_h = (h - new_height) // 2
    offset_w = (w - new_width) // 2

    if crop_info:
        return image[:, :, offset_h:offset_h + new_height, offset_w:offset_w + new_width], (offset_h, offset_w, new_height, new_width)
    else:
        return image[:, :, offset_h:offset_h + new_height, offset_w:offset_w + new_width]
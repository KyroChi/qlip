import pytest
import torch

import anyres_clip.utils as utils

def test__get_max_depth_one_dim():
    func = utils._get_max_depth_one_dim

    # dimension, patch_size, max_depth
    test_cases = [
        (2, 2, 0),
        (4, 2, 1),
        (14, 2, 2),
    ]

    for dim, ps, md in test_cases:
        assert( func(dim, ps) == md )


def test_get_max_depth():
    func = utils.get_max_depth

    # height, width, patch_size, max_depth
    test_cases = [
        (2, 2, 2, 0),
        (2, 4, 2, 0),
        (4, 4, 2, 1),
        (14, 32, 2, 2),
    ]

    for h, w, ps, md in test_cases:
        tensor = torch.zeros(3, h, w)
        assert( func(tensor, ps) == md )

def test_pad_for_patchify():
    func = utils.pad_for_patchify

    # height, width, patch_size, new_height, new_width
    test_cases = [
        (3, 3, 2, 6, 6),
        (5, 5, 4, 12, 12),
        (9, 5, 4, 16, 12)
    ]

    for h, w, ps, nh, nw in test_cases:
        tensor = torch.zeros(3, h, w)
        new_tensor = func(tensor, ps)
        assert( new_tensor.shape[1] == nh )
        assert( new_tensor.shape[2] == nw )

def test_center_crop_to_patch_size():
    func = utils.center_crop_to_patch_size

    # height, width, patch_size, new_height, new_width
    test_cases = [
        (3, 3, 2, 2, 2),
        (13, 13, 4, 12, 12),
        (13, 9, 4, 12, 8),
        (336, 336, 14, 336, 336),
        (340, 349, 14, 336, 336),
        (400, 620, 45, 360, 585)
    ]

    for h, w, ps, nh, nw in test_cases:
        tensor = torch.zeros(3, h, w).unsqueeze(0)
        new_tensor = func(tensor, ps)
        assert( new_tensor.shape[2] == nh )
        assert( new_tensor.shape[3] == nw )
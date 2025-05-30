import matplotlib.pyplot as plt
import torch
import torchvision.transforms as tf

from matplotlib.lines import Line2D
from typing import List

from qlip.utils import (
    center_crop_to_patch_size, center_crop_to_quadtree_size, 
    _get_max_depth_one_dim
)

def _get_interpolated_cell_indices(
        patch_size,
        pixel_values: torch.Tensor, 
        x:int, 
        y:int, 
        h:int, 
        w:int
    ) -> List[int]:
        assert h == w
        number_of_patches = h // patch_size

        total_w_patches = pixel_values.shape[-1] // patch_size

        patch_idxs = []
        coords = []

        for ii in range(number_of_patches):
            for jj in range(number_of_patches):
                x_start = x + ii * patch_size
                y_start = y + jj * patch_size

                idx = x_start // patch_size * total_w_patches + y_start // patch_size
                patch_idxs.append(idx)
                coords.append((y_start // patch_size, y_start // patch_size))

        return patch_idxs, torch.tensor(coords)

class TensorQuadtree():
    def __init__(
        self, 
        input_tensor: torch.Tensor,
        patch_size: int, 
        alpha: float,
        selection_mode: str="var",
        zero_mode: bool = False,
    ):
        assert input_tensor is not None
        self.patch_size = patch_size
        self.input_tensor = input_tensor
        # self.input_tensor = center_crop_to_patch_size(input_tensor, self.patch_size)
        self.input_tensor = center_crop_to_quadtree_size(
            self.input_tensor, self.patch_size
        )

        assert len(input_tensor.shape) == 4, "Quadtree only accepts batched 4D tensors."
        self.batch_size = input_tensor.shape[0]

        self.alpha = alpha
        assert selection_mode in ["var", "deriv", "random", "new_deriv"]
        self.selection_mode = selection_mode

        self.resample = tf.Resize((self.patch_size, self.patch_size))

        # (i, w, j, h) pairs
        self.leaf_nodes = [
            [] for _ in range(self.batch_size)
        ]

        self.max_size = (0, 0) # Max height and width, measured in patches
        self.quadtree = self._build_quadtree()

        self.sort_order = None
        self.zero_mode = zero_mode

    def _build_quadtree(self):
        roots = []

        max_h_depth = _get_max_depth_one_dim(self.input_tensor.shape[-2], self.patch_size)
        max_w_depth = _get_max_depth_one_dim(self.input_tensor.shape[-1], self.patch_size)

        max_depth = min(max_h_depth, max_w_depth)

        bottom_patch_dim_h = int(self.input_tensor.shape[-2] / self.patch_size / 2**(max_depth - 1))
        bottom_patch_dim_w = int(self.input_tensor.shape[-1] / self.patch_size / 2**(max_depth - 1))
        bottom_patch_size_h = int(self.input_tensor.shape[-2] / bottom_patch_dim_h)
        bottom_patch_size_w = int(self.input_tensor.shape[-1] / bottom_patch_dim_w)

        assert bottom_patch_size_h == bottom_patch_size_w, "The bottom patch size must be square."

        check_h = int(bottom_patch_size_h / self.patch_size)
        check_w = int(bottom_patch_size_w / self.patch_size)
        assert check_h & (check_h - 1) == 0 and check_w & (check_w - 1) == 0, "The top level patches must be able to be divided in half until we bottom out with cells of size patch size."

        for b in range(self.batch_size):
            input_tensor = self.input_tensor[b, ...]
            root = TensorQuadtreeNode(0, 0, 0, int(input_tensor.shape[-2]), int(input_tensor.shape[-1]))

            if bottom_patch_dim_h == 2 and bottom_patch_dim_w == 2:
                self._build_quadtree_recursive(root, b)
            else:
                # Split the input tensor into a bottom_patch_dim_h x bottom_patch_dim_w grid
                for ii in range(bottom_patch_dim_h):
                    for jj in range(bottom_patch_dim_w):
                        x = ii * bottom_patch_size_h
                        y = jj * bottom_patch_size_w
                        root.child_nodes.append(
                            TensorQuadtreeNode(1, x, y, bottom_patch_size_h, bottom_patch_size_w)
                        )

                for child in root.child_nodes:
                    self._build_quadtree_recursive(child, b)

            self.leaf_nodes[b] = root.get_leaf_nodes()
            roots.append(root)

        self.max_size = (bottom_patch_dim_h * 2**(max_depth - 1), bottom_patch_dim_w * 2**(max_depth - 1))

        return roots
    
    def _build_quadtree_recursive(self, node, b):
        if node.height < 2 * self.patch_size or node.width < 2 * self.patch_size:
            return

        split_criteria = None
        if self.selection_mode == "var":
            split_criteria = node.get_scaled_var(self.input_tensor[b, ...])
        elif self.selection_mode == "deriv":
            split_criteria = node.get_max_deriv(self.input_tensor[b, ...])
        elif self.selection_mode == "new_deriv":
            split_criteria = node.get_new_max_deriv(self.input_tensor[b, ...])
        elif self.selection_mode == "random":
            split_criteria = torch.rand(1).item()
        else:
            raise ValueError(f"Unknown selection mode: {self.selection_mode}")
        
        if split_criteria > self.alpha:
            node.split()

            for child in node.child_nodes:
                self._build_quadtree_recursive(child, b)

    def as_tensor(self):
        out_tensor = torch.zeros_like(self.input_tensor)
        for b in range(self.batch_size):
            for node in self.leaf_nodes[b]:
                in_data = self.input_tensor[b, :, node.x:node.x + node.width, node.y:node.y + node.height]
                resampled_data = self.resample(in_data)
                upsample_data = tf.Resize((node.width, node.height), interpolation=tf.InterpolationMode.NEAREST)(resampled_data)
                out_tensor[b, :, node.x:node.x + node.width, node.y:node.y + node.height] = upsample_data

        return out_tensor
    
    def leaf_borders(self):
        borders = [[] for _ in range(self.batch_size)]
        for b in range(self.batch_size):
            for node in self.leaf_nodes[b]:
                borders[b].append((node.x, node.y, node.width, node.height))
        
        return borders
    
    def as_sequence(self, raster_sort: bool = True):
        """Quadtree as a patchified sequence, represented by a (b, n, c, h, w) tensor.
        b is batch size, n is number of patches, c number of channels, h and w are the patch_size."""
        if self.zero_mode:
            return self.sequence
        
        longest_seq = max([len(seq) for seq in self.leaf_nodes])

        device = self.input_tensor.device
        dtype = self.input_tensor.dtype

        sequence = torch.zeros(
            (self.batch_size, 
             longest_seq, 
             self.input_tensor.shape[1], 
             self.patch_size, self.patch_size)
        ).to(device).to(dtype)

        for b in range(self.batch_size):
            for i, node in enumerate(self.leaf_nodes[b]):
                data = self.input_tensor[b, :, node.x:node.x + node.width, node.y:node.y + node.height]
                resampled_data = self.resample(data)
                sequence[b, i, ...] = resampled_data

        if raster_sort:
            sort_order, _ = self._get_raster_sort_order()
            return sequence[:, sort_order, ...].contiguous()
        else:
            return sequence
        
    def _get_raster_sort_order(self, force_cache_update: bool = False):
        """
            Caches the result
        """
        if self.sort_order is not None and not force_cache_update:
            return self.sort_order, None
        
        if self.zero_mode:
            return self.sort_order, None

        qt_patch_list = []
        max_patches = 0
        for node in self.leaf_nodes[0]:
            patch_idxs, _ = _get_interpolated_cell_indices(
                self.patch_size, self.input_tensor, node.x, node.y, node.width, node.height
            )
            qt_patch_list.append(patch_idxs)
            max_patches = max(max_patches, len(patch_idxs))
            
        cell_tensor = torch.ones((len(qt_patch_list), max_patches), dtype=torch.long) * len(qt_patch_list)
        for i, patch_idxs in enumerate(qt_patch_list):
            cell_tensor[i, :len(patch_idxs)] = torch.tensor(patch_idxs)

        # Sort so that the columns go in descending order
        cell_tensor.sort(dim=1, descending=False)

        sort_order = cell_tensor[:, 0].argsort()

        self.sort_order = sort_order

        return sort_order, qt_patch_list
    
    def save_as_image(self, path, linewidth=1, alpha=1.0):
        qt_tensor = self.as_tensor()
        leaf_borders = self.leaf_borders()

        print(f"leaf nodes: {self.leaf_nodes}")
        print(f"len leaf nodes: {len(self.leaf_nodes)}")

        _, axs = plt.subplots(1, self.batch_size)
        if self.batch_size == 1:
            axs = [axs]
        
        for b in range(self.batch_size):
            axs[b].imshow(qt_tensor[b, ...].permute(1, 2, 0))
            print(leaf_borders[b])
            for x, y, w, h in leaf_borders[b]:
                lines = [
                    Line2D([y, y + w], [x, x], color="red", linewidth=linewidth, alpha=alpha),
                    Line2D([y, y + w], [x + h, x + h], color="red", linewidth=linewidth, alpha=alpha),
                    Line2D([y, y], [x, x + h], color="red", linewidth=linewidth, alpha=alpha),
                    Line2D([y + w, y + w], [x, x + h], color="red", linewidth=linewidth, alpha=alpha)
                ]
                for line in lines:
                    axs[b].add_line(line)

        plt.savefig(path)

class TensorQuadtreeNode():
    """
    This represents the region of the image 
    I[:, x:x + width, y:y + height]
    """
    def __init__(
        self, 
        depth: int, 
        x: int, 
        y: int, 
        width: int,
        height: int
    ):
        self.depth = depth
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self.child_nodes = []

    # Inefficient implementation
    def get_scaled_var(self, tensor: torch.Tensor):
        return torch.max(self.width * self.height * torch.var(tensor[:, self.x:self.x + self.width, self.y:self.y + self.height], dim=[1, 2]))
        # return self.width * self.height * torch.var(tensor[:, self.x:self.x + self.width, self.y:self.y + self.height])
    
    def get_new_max_deriv(self, image_tensor: torch.Tensor):

        # Interior pixels
        grad_x = (image_tensor[:, :, 1:] - image_tensor[:, :, :-1]) / 2
        grad_y = (image_tensor[:, 1:, :] - image_tensor[:, :-1, :]) / 2

        grad_norm = torch.max(
            torch.abs(grad_x[:, 1:, :]) + torch.abs(grad_y[:, :, 1:]),
        )

        return grad_norm
    
    def get_max_deriv(self, tensor: torch.Tensor):
        deriv = tensor[:, self.x + 1:self.x + self.width, self.y + 1:self.y + self.height] - tensor[:, self.x:self.x + self.width - 1, self.y:self.y + self.height - 1]
        return torch.max(deriv)
    
    def _get_split_coords(self):
        h = self.height // 2
        w = self.width // 2

        return [
            (self.x, self.y, h, w),
            (self.x + w, self.y, h, w),
            (self.x, self.y + h, h, w),
            (self.x + w, self.y + h, h, w)
        ]

    def split(self):
        for x, y, h, w in self._get_split_coords():
            self.child_nodes.append(
                TensorQuadtreeNode(
                    self.depth + 1,
                    x, y, h, w
                )
            )

    def get_leaf_nodes(self):
        if self.is_leaf():
            return [self]
        
        leaf_nodes = []
        for child in self.child_nodes:
            leaf_nodes += child.get_leaf_nodes()
        
        return leaf_nodes

    def is_leaf(self):
        return len(self.child_nodes) == 0
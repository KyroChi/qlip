import torch
import torchvision.transforms as tf

from qlip.quadtree import TensorQuadtree, TensorQuadtreeNode
from qlip.utils import _get_max_depth_one_dim, center_crop_to_patch_size

def chunk_image_quadtree(
    image: torch.Tensor,
    patch_size: int,
    align: str = 'center'
):
    assert image.shape[-1] % patch_size == 0 and image.shape[-2] % patch_size == 0, f"Image dimensions must be divisible by patch_size={patch_size}."
    _, _, h, w = image.shape

    max_h_depth = _get_max_depth_one_dim(h, patch_size)
    max_w_depth = _get_max_depth_one_dim(w, patch_size)

    max_depth = min(max_h_depth, max_w_depth)

    if max_h_depth != 0:
        M = int( h / patch_size / 2**(max_depth - 1) )
        new_chunk_height = int(patch_size * 2**(max_depth - 1))
    else:
        M = 1
        new_chunk_height = patch_size

    if max_w_depth != 0:
        N = int ( w / patch_size / 2**(max_depth - 1) )
        new_chunk_width = int(patch_size * 2**(max_depth - 1))
    else:
        N = 1
        new_chunk_width = patch_size

    chunks = []
    if align == 'center':
        # offset_h and offset_w must be multiples of patch_size
        offset_h = (h - M * new_chunk_height) // 14
        offset_w = (w - N * new_chunk_width) // 14

        offset_h = (offset_h // 2) * 14
        offset_w = (offset_w // 2) * 14
    
        for i in range(M):
            for j in range(N):
                chunks.append((i * new_chunk_height + offset_h, j * new_chunk_width + offset_w, new_chunk_height, new_chunk_width))

    return chunks

def _gather_chunk_raster_idx(
    image: torch.Tensor,
    offset_h: int, offset_w: int, chunk_height: int, chunk_width: int, patch_size: int
): 
    raster_idxs = []
    n_patches = chunk_height // patch_size
    total_w_patches = image.shape[-1] // patch_size

    for ii in range(n_patches):
        for jj in range(n_patches):
            x_start = offset_h + ii * patch_size
            y_start = offset_w + jj * patch_size

            idx = x_start // patch_size * total_w_patches + y_start // patch_size
            raster_idxs.append(idx)

    return raster_idxs
    

def gather_chunk_raster_idx(
        image: torch.Tensor, 
        chunks: list[tuple[int, int, int, int]],
        patch_size: int
):
    """
        Gather the raster ids of all patches contained in the chunks.
    """
    _, _, h, w = image.shape

    chunk_raster_idx = []
    for chunk in chunks:
        offset_h, offset_w, new_chunk_height, new_chunk_width = chunk
        chunk_raster_idx += _gather_chunk_raster_idx(image, offset_h, offset_w, new_chunk_height, new_chunk_width, patch_size)

    return chunk_raster_idx


class MultiTensorQuadtree(TensorQuadtree):
    def __init__(
        self,
        input_tensor: torch.Tensor,
        patch_size: int,
        alpha: float,
        selection_mode: str="var",
        align: str="center",
        zero_mode: bool = False,
    ):
        """
            align: "center" or "boundary". 
        """
        self.input_tensor = center_crop_to_patch_size(input_tensor, patch_size)
        self.patch_size = patch_size
        assert align in ["center", "boundary"], f"align must be either 'center' or 'boundary', got {align}."
        assert align == "center"
        self.align = align
        self.alpha = alpha
        self.zero_mode = zero_mode

        self.batch_size = input_tensor.shape[0]

        # if self.batch_size > 1:
        #     raise NotImplementedError("Batched MultiTensorQuadtree not supported yet.")

        self.resample = tf.Resize((self.patch_size, self.patch_size))

        assert len(input_tensor.shape) == 4, f"Input tensor must have 4 dimensions, got {len(input_tensor.shape)}."

        input_tensor = center_crop_to_patch_size(input_tensor, patch_size)

        chunks = chunk_image_quadtree(input_tensor, patch_size, align=self.align)
        chunk_raster_idxs = torch.tensor(gather_chunk_raster_idx(input_tensor, chunks, patch_size)).to(torch.long)

        global_patchification = self.input_tensor[0, ...].reshape(1, -1, self.input_tensor.shape[1], self.patch_size, self.patch_size)
        mask = torch.zeros(global_patchification.shape[1])
        mask[chunk_raster_idxs] = 1

        non_chunk_raster_idxs = torch.where(mask == 0)[0]

        if not self.zero_mode:
            self.quadtrees = []
            for chunk in chunks:
                qt = TensorQuadtree(
                    input_tensor=self.input_tensor[:, :, chunk[0]:chunk[0] + chunk[2], chunk[1]:chunk[1] + chunk[3]],
                    patch_size=self.patch_size,
                    alpha=self.alpha,
                    selection_mode=selection_mode
                ) 
                self.quadtrees.append(qt)

            # amend chunk locations
            for quadtree, chunk in zip(self.quadtrees, chunks):
                for node in quadtree.leaf_nodes[0]:
                    node.x += chunk[0]
                    node.y += chunk[1]

            self.leaf_nodes = []
            for b in range(self.batch_size):
                self.leaf_nodes.append([])
                for qt in self.quadtrees:
                    self.leaf_nodes[b] += qt.leaf_nodes[b]

            for b in range(self.batch_size):
                for idx in non_chunk_raster_idxs:
                    self.leaf_nodes[b].append(
                        TensorQuadtreeNode(
                            -1,
                            (idx // (self.input_tensor.shape[-1] // self.patch_size) * self.patch_size).item(),
                            (idx % (self.input_tensor.shape[-1] // self.patch_size) * self.patch_size).item(),
                            self.patch_size,
                            self.patch_size
                        )
                    )

            self.sort_order = None
        else:
            bs = self.input_tensor.shape[0]
            c = self.input_tensor.shape[1]
            h = self.input_tensor.shape[2]
            w = self.input_tensor.shape[3]
            self.sequence = input_tensor.view(bs, c, h // patch_size, patch_size, w//patch_size, patch_size).permute(0, 1, 2, 4, 3, 5).reshape(bs, c, h // patch_size * w // patch_size, patch_size, patch_size).permute(0, 2, 1, 3, 4)
            self.sort_order = torch.arange(self.sequence.shape[1])
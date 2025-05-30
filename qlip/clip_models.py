import math
import torch

from torchvision.transforms import Resize
from transformers.models.clip.configuration_clip import CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPVisionEmbeddings, CLIPVisionTransformer
from transformers.modeling_outputs import BaseModelOutputWithPooling
from typing import List, Optional, Union, Tuple

from qlip.mlp import InterpolationMLP
from qlip.quadtree import TensorQuadtree
from qlip.multi_quadtree import MultiTensorQuadtree

def resize_too_small(image, min_image_size: int = 14):
    width, height = image.shape[-2:]
    if width < min_image_size:
        new_width = min_image_size
        new_height = int((min_image_size / width) * height)
    else:
        new_width = width
        new_height = height

    if new_height < min_image_size:
        new_height = min_image_size
        new_width = int((min_image_size / new_height) * new_width)

    transform = Resize((new_height, new_width))
    image = transform(image)
    return image

def resize_too_big(image, max_image_size):
    width, height = image.shape[-2:]
    if width > max_image_size:
        new_width = max_image_size
        new_height = int((max_image_size / width) * height)
    else:
        new_width = width
        new_height = height

    if new_height > max_image_size:
        new_height = max_image_size
        new_width = int((max_image_size / new_height) * new_width)
    
    transform = Resize((new_height, new_width))
    image = transform(image)

    return image

class QLIPCLIPVisionEmbeddings(CLIPVisionEmbeddings):
    def __init__(
        self,
        config: CLIPVisionConfig,
        mlp_depth: int = 2,
        interpolation_mode: str = "mlp",
        alpha: float = 0.0,
        selection_mode: str = "var",
        flops_file: str = None,
        alpha_is_max_alpha: bool = False,
        debug: bool = False,
        mlp_weights: str = None,
        save_pos_enc: bool = False,
        original_embeddings=None,
        zero_mode: bool = False,
        num_fourier_features: bool = 16,
    ):
        super().__init__(config)

        assert interpolation_mode in ["mlp", "bilinear", "bicubic", "zero"], f"Unsupported mode {interpolation_mode}."
        self.mode = interpolation_mode

        assert config.patch_size % 2 == 0, "As implemented, the patch size must be even."
        self.config = config

        assert alpha >= 0.0, "Alpha must be non-negative."
        self.alpha = alpha

        assert selection_mode in ["var", "deriv", "random", "original", "new_deriv"], f"Unsupported selection mode {selection_mode}."
        self.selection_mode = selection_mode

        self.flops_file = flops_file
        self.alpha_is_max_alpha = alpha_is_max_alpha
        self.debug = debug

        if mlp_weights is not None:
            loaded_weights = torch.load(mlp_weights)
            mlp_depth = ( len(loaded_weights) - 2 ) // 2
            num_fourier_features = ( loaded_weights['layers.0.weight'].shape[1] - 2 ) // 4

        if self.debug:
            print(f"Inferred MLP depth: {mlp_depth}")
            print(f"Inferred number of Fourier features: {num_fourier_features}")

        self.mlp = InterpolationMLP(
            in_features=2,
            out_features=config.hidden_size,
            hidden_features=[1024] * mlp_depth,
            activation=torch.nn.Tanh(),
            num_fourier_features=num_fourier_features
        )

        if mlp_weights is not None:
            self.mlp.load_state_dict(torch.load(mlp_weights))

        self.last_pos_enc = None
        self.last_emb = None
        self.save_pos_enc = save_pos_enc

        self.original_embeddings = original_embeddings
        self.zero_mode = zero_mode

    def interpolate_pos_encoding(
        self,
        embeddings: torch.Tensor,
        quadtree: TensorQuadtree # In practice this is a MultiTensorQuadtree
    ) -> torch.Tensor:
        device = embeddings.device
        dtype = embeddings.dtype

        # Sampling points are pixel coordinates at the center of the patches
        sampling_points = []
        if not self.zero_mode:
            for node in quadtree.leaf_nodes[0]:
                x, y, h, w = node.x, node.y, node.height, node.width

                x /= self.patch_size
                y /= self.patch_size
                h /= self.patch_size
                w /= self.patch_size

                sampling_points.append((x + w // 2, y + h // 2))
        else:
            image_width = quadtree.input_tensor.shape[-1]
            image_height = quadtree.input_tensor.shape[-2]
            patch_size = self.config.patch_size

            for i in range(0, image_height, patch_size):
                for j in range(0, image_width, patch_size):
                    sampling_points.append((j + patch_size // 2, i + patch_size // 2))

        sampling_points = torch.tensor(sampling_points).to(
            device=device, dtype=torch.float32
        ).reshape(-1, 2)

        if self.mode == "mlp":
            pos_enc = self._interpolate_mlp(sampling_points, device, dtype)
            sort_order, _ = quadtree._get_raster_sort_order()
            sort_order = sort_order
            pos_enc[:, ...] = pos_enc[:, sort_order, ...].contiguous()
        elif self.mode == "bilinear" or self.mode == "bicubic":
            pos_enc = self._interpolate_torch_grid(quadtree, sampling_points, device, dtype)
        elif self.mode == "zero":
            pos_enc = torch.zeros_like(embeddings[:, 1:, :], device=device, dtype=dtype)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        zero = torch.zeros(1, 1, device=device, dtype=torch.long)
        
        pos_enc = torch.cat([
            self.position_embedding(zero), pos_enc
        ], dim=1)

        return pos_enc.to(device=device, dtype=dtype)

    def _interpolate_mlp(
        self,
        sampling_points: torch.Tensor,
        device: str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        self.mlp.to(device=device, dtype=dtype)

        sampling_points[:, 0] = 2 * (sampling_points[:, 0] / sampling_points[:, 0].max()) - 1
        sampling_points[:, 1] = 2 * (sampling_points[:, 1] / sampling_points[:, 1].max()) - 1

        # Needs the first dimension as well
        pos_enc = self.mlp(sampling_points).unsqueeze(0).to(torch.float16 if dtype == torch.float16 else dtype)

        return pos_enc

    def _interpolate_torch_grid(
        self,
        quadtree: TensorQuadtree,
        sampling_points: torch.Tensor,
        device: str,
        dtype: torch.dtype,
    ):
        sampling_points[:, 0] = 2 * (sampling_points[:, 0] / sampling_points[:, 0].max()) - 1
        sampling_points[:, 1] = 2 * (sampling_points[:, 1] / sampling_points[:, 1].max()) - 1
        
        image_height = quadtree.input_tensor.shape[-2]
        image_width = quadtree.input_tensor.shape[-1]
        patch_size = self.config.patch_size

        height_patches = image_height // patch_size
        width_patches = image_width // patch_size
        x = torch.linspace(-1, 1, height_patches)
        y = torch.linspace(-1, 1, width_patches)
        x, y = torch.meshgrid(y, x, indexing='ij')
        sampling_grid = torch.stack((y, x), dim=-1).unsqueeze(0).to(device).to(torch.float32)

        if self.position_embedding.weight.shape[0] == 577:
            input_pos_enc = self.position_embedding.weight[1:].reshape(1, 1024, 24, 24).transpose(-1, -2)
        elif self.position_embedding.weight.shape[0] == 257:
            input_pos_enc = self.position_embedding.weight[1:].reshape(1, 1024, 16, 16).transpose(-1, -2)
        else:
            raise ValueError(f"Unsupported positional encoding size. Got {self.position_embedding.weight.shape[0]}")

        input_pos_enc = input_pos_enc.to(torch.float32)
        resampled_pos_enc = torch.nn.functional.grid_sample(
            input_pos_enc,
            sampling_grid,
            mode=self.mode,
            align_corners=True,
        ).to(dtype)

        resampled_pos_enc = resampled_pos_enc.transpose(-1, -2).reshape(1, -1, 1024)

        return resampled_pos_enc

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        return_qt: bool = False,
        **kwargs
    ) -> torch.FloatTensor:
        batch_size = pixel_values.shape[0]

        assert batch_size == 1, "Batch size > 1 not implemented yet."

        if pixel_values.shape[-1] < 28 or pixel_values.shape[-2] < 28 or \
            pixel_values.shape[-1] > 896 or pixel_values.shape[-2] > 896:
            original_shape = pixel_values.shape
            try:
                pixel_values = resize_too_small(pixel_values, 28)
                pixel_values = resize_too_big(pixel_values, 896)
            except Exception as e:
                pixel_values = torch.zeros((1, 3, 28, 28), device=pixel_values.device, dtype=pixel_values.dtype)

            if self.debug:
                print(f"Resized image from {original_shape} to {pixel_values.shape}")


        if pixel_values.shape[-1] < 28 or pixel_values.shape[-2] < 28:
            if self.debug:
                print(f"Image size too small: {pixel_values.shape}")
            pixel_values = torch.zeros((1, 3, 28, 28), device=pixel_values.device, dtype=pixel_values.dtype)

        dtype = self.patch_embedding.weight.dtype
        target_dtype = self.patch_embedding.weight.dtype

        base_alpha = self.alpha

        if self.debug:
            print(f"Base alpha: {base_alpha}")
            print(f"Pixel values shape: {pixel_values.shape}")

        if self.selection_mode == "var":
            alpha = base_alpha * pixel_values.shape[-2] * pixel_values.shape[-1] / 1000
        else:
            alpha = base_alpha

        if self.alpha_is_max_alpha:
            alpha = torch.rand(1) * alpha
            alpha = alpha.to(device=pixel_values.device, dtype=target_dtype)
        else:
            alpha = alpha

        if self.debug:
            print(f"Alpha: {alpha}")
            print(f"Selection mode: {self.selection_mode}")
            print(f"Interpolation mode: {self.mode}")
        
        qt = MultiTensorQuadtree(
            pixel_values, 
            patch_size=self.patch_size, 
            alpha=alpha,
            selection_mode=self.selection_mode,
            zero_mode=self.zero_mode,
        )

        if self.debug:
            if self.zero_mode:
                print(f"Number of patches: {qt.sequence.shape[1]}")
            else:
                print(f"Number of patches: {len(qt.leaf_nodes[0])}")

        if self.flops_file is not None:
            with open(self.flops_file, "a") as f:
                max_tokens = pixel_values.shape[-2] * pixel_values.shape[-1] // (self.patch_size ** 2)
                f.write(f"{max_tokens} {len(qt.leaf_nodes[0])}\n")

        pre_embed = qt.as_sequence()
        pre_embed = torch.cat([pre_embed[:, i, ...] for i in range(pre_embed.shape[1])], dim=-2).to(pixel_values.device)

        patch_embeds = self.patch_embedding(pre_embed.to(dtype=target_dtype)).squeeze(-1)
        patch_embeds = patch_embeds.transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)

        pos_enc = self.interpolate_pos_encoding(embeddings, qt)

        if self.save_pos_enc:
            pos_enc.requires_grad_(True)
            pos_enc.retain_grad()

            self.last_pos_enc = pos_enc
            self.last_emb = embeddings

        embeddings = embeddings + pos_enc    

        if return_qt:
            return embeddings, qt
        
        return embeddings


def qlip_embedding_from_embedding(
        embeddings, 
        base_config,
        mode: str,
        interpolation_mode: str,
        flops_file: str,
        alpha: float,
        alpha_is_max_alpha: bool,
        debug: bool,
        save_pos_enc: bool = False,
        zero_mode: bool = False,
        mlp_depth: int = 2,
        num_fourier_features: int = 16,
        mlp_weights: str = None,
):
    candidate_embeddings = QLIPCLIPVisionEmbeddings(
        config=base_config,
        interpolation_mode=interpolation_mode,
        selection_mode=mode,
        flops_file=flops_file,
        alpha=alpha,
        alpha_is_max_alpha=alpha_is_max_alpha,
        debug=debug,
        save_pos_enc=save_pos_enc,
        zero_mode=zero_mode,
        mlp_depth=mlp_depth,
        num_fourier_features=num_fourier_features,
        mlp_weights=mlp_weights,
    )

    candidate_embeddings.class_embedding = embeddings.class_embedding
    candidate_embeddings.patch_embedding.weight = embeddings.patch_embedding.weight
    candidate_embeddings.position_embedding.weight = embeddings.position_embedding.weight

    return candidate_embeddings

def convert_to_qlip_CLIP(
        model: torch.nn.Module,
        mode: str,
        interpolation_mode: str,
        flops_file: str,
        alpha: float,
        alpha_is_max_alpha: bool,
        debug: bool,
        mlp_weights: str = None,
        save_pos_enc: bool = False,
        zero_mode: bool = False,
        mlp_depth: int = 2,
        num_fourier_features: int = 16,
):
    if hasattr(model, "vision_model"):
        existing_embeddings = model.vision_model.embeddings
    elif hasattr(model, "vision_tower"):
        existing_embeddings = model.vision_tower.vision_model.embeddings
    else:
        raise ValueError("Model does not have an embeddings attribute.")
    base_config = existing_embeddings.config

    candidate_embeddings = qlip_embedding_from_embedding(
        embeddings=existing_embeddings, 
        base_config=base_config,
        mode=mode,
        interpolation_mode=interpolation_mode,
        flops_file=flops_file,
        alpha=alpha,
        alpha_is_max_alpha=alpha_is_max_alpha,
        debug=debug,
        save_pos_enc=save_pos_enc,
        zero_mode=zero_mode,
        mlp_depth=mlp_depth,
        num_fourier_features=num_fourier_features,
        mlp_weights=mlp_weights,
    )

    if hasattr(model, "vision_model"):
        model.vision_model.embeddings = candidate_embeddings
    elif hasattr(model, "vision_tower"):
        model.vision_tower.vision_model.embeddings = candidate_embeddings

    return model

class QLIPVisionTransformer(CLIPVisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_CLIPVisionTransformer(
        cls,
        existing_model: CLIPVisionTransformer,
    ):
        qvt = cls(existing_model.config)
        qvt.embeddings = existing_model.embeddings
        qvt.pre_layrnorm = existing_model.pre_layrnorm
        qvt.encoder = existing_model.encoder
        qvt.post_layernorm = existing_model.post_layernorm

        return qvt

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        
        if pixel_values.shape[1] == 4:
            mask = pixel_values[:, -1, ...].unsqueeze(1)
            pixel_values = pixel_values[:, :-1, ...]
        else:
            mask = None

        dtype = pixel_values.dtype
        
        batched_hidden_states = []
        for i in range(pixel_values.shape[0]):
            if mask is not None:
                row_idx = mask[i, 0, ...].nonzero(as_tuple=True)[0]
                col_idx = mask[i, 0, ...].nonzero(as_tuple=True)[0]

                h = row_idx.max() + 1
                w = col_idx.max() + 1
            else:
                h = pixel_values.shape[-2]
                w = pixel_values.shape[-1]

            hidden_states = self.embeddings(pixel_values[i:i + 1, :, :w, :h])
            hidden_states = self.pre_layrnorm(hidden_states)

            batched_hidden_states.append(hidden_states)

        max_seq_len = max([hidden_states.shape[1] for hidden_states in batched_hidden_states])
        hidden_states = torch.zeros((len(batched_hidden_states), max_seq_len, hidden_states.shape[-1]), device=hidden_states.device, dtype=hidden_states.dtype)
        attention_mask = torch.zeros((len(batched_hidden_states), max_seq_len), device=hidden_states.device, dtype=torch.bool)

        for i, hs in enumerate(batched_hidden_states):
            attention_mask[i, :hs.shape[1]] = 1.
            hidden_states[i, :hs.shape[1], ...] = hs

        attention_mask = torch.einsum("ij,ik->ijk", attention_mask, attention_mask).unsqueeze(1)
        attention_mask = ~attention_mask
        attention_mask = torch.zeros_like(attention_mask, dtype=dtype).masked_fill(attention_mask, -torch.finfo(dtype).max)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
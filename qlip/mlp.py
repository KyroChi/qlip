import math
import torch

from typing import List

def generate_fourier_features(x: torch.Tensor, num_features: int) -> torch.Tensor:
    """
    Generate Fourier features for the input tensor.
    """
    batch_size, _ = x.shape
    freqs = torch.arange(num_features, dtype=torch.float32, device=x.device) * math.pi
    x = torch.einsum("bi,j->bij", x, freqs)
    fourier_features = torch.cat([torch.sin(x * freqs), torch.cos(x * freqs)], dim=-1)
    fourier_features = fourier_features.flatten(1, 2)
    return fourier_features.view(batch_size, -1)


class InterpolationMLP(torch.nn.Module):
    def __init__(
            self, 
            in_features: int, 
            out_features: int, 
            hidden_features: List[int], 
            activation: torch.nn.Module, 
            num_fourier_features: int = 0
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.activation = activation
        self.num_fourier_features = num_fourier_features
        
        self.layers = torch.nn.ModuleList()

        prev_features = in_features + 2 * num_fourier_features * in_features

        for hidden_feature in hidden_features:
            self.layers.append(torch.nn.Linear(prev_features, hidden_feature))
            prev_features = hidden_feature

        self.layers.append(torch.nn.Linear(prev_features, out_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_fourier_features > 0:
            fourier_features = generate_fourier_features(x, self.num_fourier_features).to(x.device, x.dtype)
            x = torch.cat([x, fourier_features], dim=-1)

        for layer in self.layers[:-1]:
            # assert layer.weight.dtype == x.dtype, f"Layer weight type: {layer.weight.dtype}, x type: {x.dtype}"
            x = x.to(device=layer.weight.device, dtype=layer.weight.dtype)
            # print(f"x type: {x.dtype}")
            x = self.activation(layer(x))

        return self.layers[-1](x)

    def interpolate(
        self,
        device: str,
        dtype: torch.dtype,
        image_height: int,
        image_width: int,
        patch_size: int, 
    ):
        height_patches = image_height // patch_size
        width_patches = image_width // patch_size

        assert height_patches > 1 and width_patches > 1, "Image resolution is too low for the patch size."

        # Base grid is 24 x 24, (x, y) coordinates range from (-1, -1) to (1, 1)
        # We interpolate the grid to the size of the number of patches
        x_coords = torch.linspace(-1, 1, width_patches)
        y_coords = torch.linspace(-1, 1, height_patches)

        grid = torch.stack(torch.meshgrid(y_coords, x_coords), dim=-1).to(device, torch.bfloat16 if dtype == torch.float16 else dtype)

        # Return MLP values M(x, y) in order of how the convolution generates patches
        pos_encodings = self.forward(grid.flatten(0, 1)).view(height_patches, width_patches, -1)

        return pos_encodings
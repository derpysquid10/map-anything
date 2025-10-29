# encoders.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class PoseEncoder(nn.Module):
    def __init__(self, in_dim=12, embed_dim=1024):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 4*embed_dim),
            nn.GELU(),
            nn.Linear(4*embed_dim, embed_dim)
        )

    def forward(self, x, **kw):
        return self.encoder(x)


class PoseEncoder6D(nn.Module):
    def __init__(self, in_dim=9, embed_dim=1024):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 4*embed_dim),
            nn.GELU(),
            nn.Linear(4*embed_dim, embed_dim)
        )

    def forward(self, x, **kw):
        return self.encoder(x)


class PoseEncoderQuaternion(nn.Module):
    def __init__(self, in_dim=7, embed_dim=1024):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 4*embed_dim),
            nn.GELU(),
            nn.Linear(4*embed_dim, embed_dim)
        )

    def forward(self, x, **kw):
        return self.encoder(x)


class RayEncoder(nn.Module):
    def __init__(self,  patch_size=14, in_chans=6, embed_dim=1024, norm_layer=None, flatten=True, position_getter=None):
        super().__init__()
        
        # Store parameters
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten = flatten
        self.position_getter = position_getter
        
        # Initialize normalization layer
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()
        
        # Create the projection module (combining PixelUnshuffle, Permute, and MLP)
        self.proj = nn.Sequential(
            PixelUnshuffle(patch_size), 
            Permute((0, 2, 3, 1)),
            nn.Linear(in_chans * patch_size**2, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),            
            Permute((0, 3, 1, 2)),
        )

    def forward(self, x, **kw):
        B, C, H, W = x.shape
        
        # Validate input dimensions
        assert H % self.patch_size[0] == 0, f"Input image height ({H}) is not a multiple of patch size ({self.patch_size[0]})."
        assert W % self.patch_size[1] == 0, f"Input image width ({W}) is not a multiple of patch size ({self.patch_size[1]})."
        
        # Apply projection
        x = self.proj(x)
        
        # Get position embeddings if position_getter is provided
        pos = None
        if self.position_getter is not None:
            pos = self.position_getter(B, x.size(2), x.size(3), x.device)
        
        # Flatten if required
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        
        # Apply normalization
        x = self.norm(x)
        
        return x, pos

        
class DepthEncoder(nn.Module):
    def __init__(self,  patch_size=14, in_chans=1, embed_dim=1024, norm_layer=None, flatten=True, position_getter=None):
        super().__init__()
        
        # Store parameters
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten = flatten
        self.position_getter = position_getter
        
        # Initialize normalization layer
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()
        
        # Create the projection module (combining PixelUnshuffle, Permute, and MLP)
        self.proj = nn.Sequential(
            PixelUnshuffle(patch_size), 
            Permute((0, 2, 3, 1)),
            nn.Linear(in_chans * patch_size**2, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),            
            Permute((0, 3, 1, 2)),
        )
    
    def forward(self, x, **kw):
        B, C, H, W = x.shape
        
        # Validate input dimensions
        assert H % self.patch_size[0] == 0, f"Input image height ({H}) is not a multiple of patch size ({self.patch_size[0]})."
        assert W % self.patch_size[1] == 0, f"Input image width ({W}) is not a multiple of patch size ({self.patch_size[1]})."
        
        # Apply projection
        x = self.proj(x)
        
        # Get position embeddings if position_getter is provided
        pos = None
        if self.position_getter is not None:
            pos = self.position_getter(B, x.size(2), x.size(3), x.device)
        
        # Flatten if required
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        
        # Apply normalization
        x = self.norm(x)
        
        return x, pos


class PixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor):
        super().__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        if input.numel() == 0:
            # this is not in the original torch implementation
            C, H, W = input.shape[-3:]
            assert H and W and H % self.downscale_factor == W % self.downscale_factor == 0
            return input.view(*input.shape[:-3], C*self.downscale_factor**2, H//self.downscale_factor, W//self.downscale_factor)
        else:
            return F.pixel_unshuffle(input, self.downscale_factor)


class Permute(torch.nn.Module):
    def __init__(self, dims: tuple[int, ...]) -> None:
        super().__init__()
        self.dims = tuple(dims)

    def __repr__(self):
        return f"Permute{self.dims}"

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.permute(*self.dims)

if __name__ == "__main__": # Tests and Sanity Checks
    # # parameters
    # BATCH_SIZE   = 2          # any positive integer
    # C, H, W      = 3, 294, 518
    # PATCH_SIZE   = 14
    # EMBED_DIM    = 1024

    # # dummy ray map
    # x = torch.randn(BATCH_SIZE, C, H, W)

    # # build the encoder (uses default flatten=True)
    # encoder = RayEncoder(patch_size=PATCH_SIZE,
    #                      in_chans=C,
    #                      embed_dim=EMBED_DIM)

    # # forward pass
    # tokens, pos = encoder(x)

    # # expected numbers
    # h_patches = H // PATCH_SIZE     # 294 // 14 = 21
    # w_patches = W // PATCH_SIZE     # 518 // 14 = 37
    # expected_len = h_patches * w_patches  # 777

    # # results
    # print(f"Output   shape: {tokens.shape}")
    # print(f"Position shape: {None if pos is None else pos.shape}")
    # assert tokens.shape == (BATCH_SIZE, expected_len, EMBED_DIM), \
    #     "Shape mismatch! Something is wrong."

    # print("✓ RayEncoder forward-shape test passed.")

    my_tensor = torch.zeros((78, 12))
    pose_encoder = PoseEncoder()
    pose_encodings = pose_encoder(my_tensor)
    pose_encodings = pose_encodings.unsqueeze(1)
    print(f"Shape of pose encodings: {pose_encodings.shape}")


class RaymapReprojectionLayer(nn.Module):
    """
    A simple MLP that projects concatenated patch and ray embeddings
    from dimension 2048 to 1024.
    """
    def __init__(self, in_dim=2048, embed_dim=1024):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B*S, P, 2*embed_dim) - concatenated patch and ray embeddings
        Returns:
            Tensor of shape (B*S, P, embed_dim) - projected embeddings
        """
        return self.mlp(x)


class ScaleEncoder(nn.Module):
    def __init__(self, in_dim=1, embed_dim=1024):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 4*embed_dim),
            nn.GELU(),
            nn.Linear(4*embed_dim, embed_dim)
        )

    def forward(self, x, **kw):
        return self.encoder(x)
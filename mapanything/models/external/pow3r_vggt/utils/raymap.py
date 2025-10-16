import torch

def generate_raymap(
    images: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Compute per-pixel camera-space ray directions (unit-length) for batched images.
    
    Parameters
    ----------
    images : torch.Tensor
        Shape **(B, N, C, H, W)** – a batch of B×N images with B groups.
        Only H and W are used; the pixel values are ignored.
    intrinsics : torch.Tensor
        Shape **(B, N, 3, 3)** – batch of B×N intrinsics matrices, one per image.
        Camera intrinsics format::
            [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]
    device, dtype : optional
        If provided, the output will be moved / cast accordingly.
        (Defaults to images.device / images.dtype.)
    
    Returns
    -------
    torch.Tensor
        Shape **(B, N, 3, H, W)** – ray directions for each image.
    """
    # Validate input shapes
    if images.dim() != 5:
        raise ValueError(f"images must have shape (B,N,C,H,W), got {images.shape}")
    B, N, _, H, W = images.shape
    
    if intrinsics.shape != (B, N, 3, 3):
        raise ValueError(f"intrinsics must have shape (B,N,3,3) where B={B}, N={N}, got {intrinsics.shape}")
    
    # ------------------------------------------------------------------
    # choose device / dtype
    # ------------------------------------------------------------------
    if device is None:
        device = images.device
    if dtype is None:
        dtype = images.dtype
    
    K = intrinsics.to(device=device, dtype=dtype)
    
    # Extract intrinsics parameters for all images
    fx = K[:, :, 0, 0]  # (B, N)
    fy = K[:, :, 1, 1]  # (B, N)
    cx = K[:, :, 0, 2]  # (B, N)
    cy = K[:, :, 1, 2]  # (B, N)
    
    # ------------------------------------------------------------------
    # pixel grid in image coordinates
    # ------------------------------------------------------------------
    u = torch.arange(W, device=device, dtype=dtype)  # cols (x-axis)
    v = torch.arange(H, device=device, dtype=dtype)  # rows (y-axis)
    u, v = torch.meshgrid(u, v, indexing="xy")  # shapes (H,W)
    
    # Expand u, v to match batch dimensions
    u = u.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    v = v.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    # ------------------------------------------------------------------
    # Compute rays for each image using its own intrinsics
    # ------------------------------------------------------------------
    # Reshape intrinsics parameters for broadcasting
    fx = fx.unsqueeze(-1).unsqueeze(-1)  # (B, N, 1, 1)
    fy = fy.unsqueeze(-1).unsqueeze(-1)  # (B, N, 1, 1)
    cx = cx.unsqueeze(-1).unsqueeze(-1)  # (B, N, 1, 1)
    cy = cy.unsqueeze(-1).unsqueeze(-1)  # (B, N, 1, 1)
    
    # Compute normalized coordinates
    x = (u - cx) / fx  # (B, N, H, W)
    y = (v - cy) / fy  # (B, N, H, W)
    z = torch.ones_like(x)  # (B, N, H, W)
    
    # Stack to form ray directions
    rays = torch.stack((x, y, z), dim=2)  # (B, N, 3, H, W)
    
    # Normalize rays to unit length
    rays = rays / torch.linalg.norm(rays, dim=2, keepdim=True)
    
    return rays.to(device=device, dtype=dtype)
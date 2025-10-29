import torch

def generate_raymap(
    images: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    normalize: bool = True,
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
    normalize : bool, optional
        If True, normalize ray directions to unit length. Default is True.
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
    
    # Normalize rays to unit length if requested
    if normalize:
        rays = rays / torch.linalg.norm(rays, dim=2, keepdim=True)
    
    return rays.to(device=device, dtype=dtype)

def generate_origin_raymaps(
    images: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor | None = None,
    *,
    normalize: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Compute per-pixel ray origins and directions in world coordinates for batched images.
    
    Parameters
    ----------
    images : torch.Tensor
        Shape **(B, N, C, H, W)** – a batch of B×N images with B groups.
        Only H and W are used; the pixel values are ignored.
    intrinsics : torch.Tensor
        Shape **(B, N, 3, 3)** – batch of B×N intrinsics matrices, one per image.
    extrinsics : torch.Tensor | None, optional
        Shape **(B, S, 3, 4)** – batch of B×S extrinsics matrices [R|t] where S >= N.
        Maps from world coordinates to camera coordinates. If None, origins are set to zero
        and directions remain in camera space.
    normalize : bool, optional
        If True, normalize ray directions to unit length. Default is True.
    device, dtype : optional
        If provided, the output will be moved / cast accordingly.
        
    Returns
    -------
    torch.Tensor
        Shape **(B, N, 6, H, W)** – [ray_origins, ray_directions] in world coordinates for each pixel.
        First 3 channels are origins, last 3 channels are directions.
    """
    # Validate input shapes
    if images.dim() != 5:
        raise ValueError(f"images must have shape (B,N,C,H,W), got {images.shape}")
    B, N, _, H, W = images.shape
    
    if intrinsics.shape != (B, N, 3, 3):
        raise ValueError(f"intrinsics must have shape (B,N,3,3) where B={B}, N={N}, got {intrinsics.shape}")
    
    if extrinsics is not None:
        if extrinsics.dim() != 4 or extrinsics.shape[0] != B or extrinsics.shape[2:] != (3, 4):
            raise ValueError(f"extrinsics must have shape (B,S,3,4) where B={B}, got {extrinsics.shape}")
        
        S = extrinsics.shape[1]
        if S < N:
            raise ValueError(f"extrinsics has {S} cameras but need at least {N} for {N} images")
    
    # Choose device / dtype
    if device is None:
        device = images.device
    if dtype is None:
        dtype = images.dtype
    
    intrinsics = intrinsics.to(device=device, dtype=dtype)
    
    # Get camera-space ray directions using existing function
    ray_dirs_cam = generate_raymap(images, intrinsics, normalize=normalize, device=device, dtype=dtype)  # (B, N, 3, H, W)
    
    if extrinsics is None:
        # Set origins to zero and keep directions in camera space
        origins = torch.zeros(B, N, 3, H, W, device=device, dtype=dtype)
        ray_dirs_world = ray_dirs_cam
    else:
        extrinsics = extrinsics.to(device=device, dtype=dtype)
        
        # Extract extrinsics for first N cameras
        R = extrinsics[:, :N, :3, :3]  # (B, N, 3, 3)
        t = extrinsics[:, :N, :3, 3]   # (B, N, 3)
        
        # Transform ray directions to world coordinates: d_world = R^T @ d_cam
        R_T = R.transpose(-2, -1)  # (B, N, 3, 3)
        ray_dirs_world = torch.einsum('bnij,bnjhw->bnihw', R_T, ray_dirs_cam)  # (B, N, 3, H, W)
        
        # Compute camera centers in world coordinates: o = -R^T @ t
        camera_centers = -torch.bmm(R_T.view(B*N, 3, 3), t.view(B*N, 3, 1)).view(B, N, 3)  # (B, N, 3)
        
        # Expand camera centers to match pixel dimensions
        origins = camera_centers.unsqueeze(-1).unsqueeze(-1).expand(B, N, 3, H, W)  # (B, N, 3, H, W)
    
    # Concatenate origins and directions
    rays = torch.cat([origins, ray_dirs_world], dim=2)  # (B, N, 6, H, W)
    
    return rays.to(device=device, dtype=dtype)

def generate_plucker_origin_raymaps(
    images: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor | None = None,
    *,
    normalize: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Compute per-pixel Plucker coordinates for rays in world coordinates.
    
    Plucker coordinates represent a 3D line with 6 components: (d, o×d)
    where d is the direction vector and o is a point on the line.
    
    Parameters
    ----------
    images : torch.Tensor
        Shape **(B, N, C, H, W)** – a batch of B×N images with B groups.
        Only H and W are used; the pixel values are ignored.
    intrinsics : torch.Tensor
        Shape **(B, N, 3, 3)** – batch of B×N intrinsics matrices, one per image.
    extrinsics : torch.Tensor | None, optional
        Shape **(B, S, 3, 4)** – batch of B×S extrinsics matrices [R|t] where S >= N.
        Maps from world coordinates to camera coordinates. If None, origins are set to zero
        and directions remain in camera space.
    normalize : bool, optional
        If True, normalize ray directions to unit length. Default is True.
    device, dtype : optional
        If provided, the output will be moved / cast accordingly.
        
    Returns
    -------
    torch.Tensor
        Shape **(B, N, 6, H, W)** – Plucker coordinates [d, o×d] for each pixel ray.
    """
    # Validate input shapes
    if images.dim() != 5:
        raise ValueError(f"images must have shape (B,N,C,H,W), got {images.shape}")
    B, N, _, H, W = images.shape
    
    if intrinsics.shape != (B, N, 3, 3):
        raise ValueError(f"intrinsics must have shape (B,N,3,3) where B={B}, N={N}, got {intrinsics.shape}")
    
    if extrinsics is not None:
        if extrinsics.dim() != 4 or extrinsics.shape[0] != B or extrinsics.shape[2:] != (3, 4):
            raise ValueError(f"extrinsics must have shape (B,S,3,4) where B={B}, got {extrinsics.shape}")
        
        S = extrinsics.shape[1]
        if S < N:
            raise ValueError(f"extrinsics has {S} cameras but need at least {N} for {N} images")
    
    # Choose device / dtype
    if device is None:
        device = images.device
    if dtype is None:
        dtype = images.dtype
    
    intrinsics = intrinsics.to(device=device, dtype=dtype)
    
    # Get camera-space ray directions using existing function
    ray_dirs_cam = generate_raymap(images, intrinsics, normalize=normalize, device=device, dtype=dtype)  # (B, N, 3, H, W)
    
    if extrinsics is None:
        # Set origins to zero and keep directions in camera space
        origins = torch.zeros(B, N, 3, H, W, device=device, dtype=dtype)
        ray_dirs_world = ray_dirs_cam
    else:
        extrinsics = extrinsics.to(device=device, dtype=dtype)
        
        # Extract extrinsics for first N cameras
        R = extrinsics[:, :N, :3, :3]  # (B, N, 3, 3)
        t = extrinsics[:, :N, :3, 3]   # (B, N, 3)
        
        # Transform ray directions to world coordinates: d_world = R^T @ d_cam
        R_T = R.transpose(-2, -1)  # (B, N, 3, 3)
        ray_dirs_world = torch.einsum('bnij,bnjhw->bnihw', R_T, ray_dirs_cam)  # (B, N, 3, H, W)
        
        # Compute camera centers in world coordinates: o = -R^T @ t
        camera_centers = -torch.bmm(R_T.view(B*N, 3, 3), t.view(B*N, 3, 1)).view(B, N, 3)  # (B, N, 3)
        
        # Expand camera centers to match pixel dimensions
        origins = camera_centers.unsqueeze(-1).unsqueeze(-1).expand(B, N, 3, H, W)  # (B, N, 3, H, W)
    
    # Compute Plucker coordinates: [d, o×d]
    # Cross product: o × d
    cross_product = torch.cross(origins, ray_dirs_world, dim=2)  # (B, N, 3, H, W)
    
    # Concatenate direction and cross product to form Plucker coordinates
    plucker_coords = torch.cat([ray_dirs_world, cross_product], dim=2)  # (B, N, 6, H, W)
    
    return plucker_coords.to(device=device, dtype=dtype)

def generate_unified_raymap(
    images: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor | None = None,
    raymap_format: str | None = None,
    *,
    normalize: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Unified function to generate ray maps in different formats.
    
    Parameters
    ----------
    images : torch.Tensor
        Shape **(B, N, C, H, W)** – a batch of B×N images with B groups.
        Only H and W are used; the pixel values are ignored.
    intrinsics : torch.Tensor
        Shape **(B, N, 3, 3)** – batch of B×N intrinsics matrices, one per image.
    extrinsics : torch.Tensor | None, optional
        Shape **(B, S, 3, 4)** – batch of B×S extrinsics matrices [R|t] where S >= N.
        If None, origins are set to zero and directions remain in camera space.
    raymap_format : str | None
        Format of the output raymap:
        - "no_origin": Returns 6D tensor [zero_origins, ray_directions] 
        - "origin": Returns 6D tensor [ray_origins, ray_directions]
        - "plucker": Returns 6D tensor [ray_directions, origins×directions] (Plucker coordinates)
    normalize : bool, optional
        If True, normalize ray directions to unit length. Default is True.
    device, dtype : optional
        If provided, the output will be moved / cast accordingly.
        
    Returns
    -------
    torch.Tensor
        Shape **(B, N, 6, H, W)** – ray data in the specified format.
        
    Notes
    -----
    When extrinsics is None and raymap_format is "origin" or "plucker":
    - Origins are set to zero (camera space assumption)
    - Ray directions remain in camera space
    - A warning message is printed
    """
    # Validate raymap_format
    valid_formats = ["no_origin", "origin", "plucker"]
    if raymap_format not in valid_formats:
        raise ValueError(f"raymap_format must be one of {valid_formats}, got {raymap_format}")
    
    # Print warning if extrinsics is None for origin/plucker formats
    if extrinsics is None and raymap_format in ["origin", "plucker"]:
        print(f"Warning: extrinsics is None with raymap_format='{raymap_format}'. "
              f"Origins will be set to zero and directions will remain in camera space.")
    
    # Choose appropriate function based on format
    if raymap_format == "no_origin":
        # Generate 6D tensor with zero origins and ray directions
        if images.dim() != 5:
            raise ValueError(f"images must have shape (B,N,C,H,W), got {images.shape}")
        B, N, _, H, W = images.shape
        
        # Choose device / dtype
        if device is None:
            device = images.device
        if dtype is None:
            dtype = images.dtype
        
        # Get camera-space ray directions
        ray_dirs = generate_raymap(images, intrinsics, normalize=normalize, device=device, dtype=dtype)  # (B, N, 3, H, W)
        
        # Create zero origins
        zero_origins = torch.zeros(B, N, 3, H, W, device=device, dtype=dtype)
        
        # Concatenate zero origins and directions
        result = torch.cat([zero_origins, ray_dirs], dim=2)  # (B, N, 6, H, W)
        
        return result
    
    elif raymap_format == "origin":
        print("using origin raymaps")
        return generate_origin_raymaps(images, intrinsics, extrinsics, normalize=normalize, device=device, dtype=dtype)
    
    elif raymap_format == "plucker":
        return generate_plucker_origin_raymaps(images, intrinsics, extrinsics, normalize=normalize, device=device, dtype=dtype) 
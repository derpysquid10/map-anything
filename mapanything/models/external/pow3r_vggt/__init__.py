# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Inference wrapper for Pow3r-VGGT
"""

# import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from mapanything.models.external.pow3r_vggt.heads.scale_head import *
from mapanything.models.external.pow3r_vggt.models.pow3r_vggt import Pow3rVGGT
from mapanything.models.external.pow3r_vggt.utils.geometry import (
    closed_form_inverse_se3,
)
from mapanything.models.external.pow3r_vggt.utils.pose_enc import (
    pose_encoding_to_extri_intri,
)
from mapanything.models.external.pow3r_vggt.utils.rotation import (
    mat_to_quat,
    quat_to_mat,
)
from mapanything.utils.geometry import (
    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap,
    convert_z_depth_to_depth_along_ray,
    depthmap_to_camera_frame,
    get_rays_in_camera_frame,
    normalize_multiple_pointclouds,
)


def apply_depth_sparsification(depths, sparsification_factor=0.1):
    """
    Apply sparsification to depth maps by removing outliers and randomly sampling.

    Args:
        depths (torch.Tensor): Depth tensor of shape (B, V, H, W)
        sparsification_factor (float): Fraction of valid depths to keep (default: 0.1)

    Returns:
        torch.Tensor: Sparsified depth tensor of the same shape
    """
    if depths is None or sparsification_factor <= 0 or sparsification_factor >= 1:
        return depths

    # Create mask for depth sparsification
    # Step 1: Remove top 5% and bottom 5% of depths (outlier removal)
    valid_mask = depths > 0  # Only consider non-zero depths

    # Calculate percentiles for each view separately - vectorized version
    mask = torch.zeros_like(depths, dtype=torch.bool)

    # Vectorized processing across all views
    B, V, H, W = depths.shape
    depths_flat = depths.view(B * V, -1)  # (B*V, H*W)
    valid_mask_flat = valid_mask.view(B * V, -1)  # (B*V, H*W)

    for i in range(B * V):
        valid_depths = depths_flat[i][valid_mask_flat[i]]

        if valid_depths.numel() > 0:
            # Calculate 5th and 95th percentiles on GPU
            p5 = torch.quantile(valid_depths, 0.05)
            p95 = torch.quantile(valid_depths, 0.95)

            # Create mask for middle 90% of depth values
            depth_view = depths_flat[i].view(H, W)
            valid_view = valid_mask_flat[i].view(H, W)
            middle_90_mask = (depth_view >= p5) & (depth_view <= p95) & valid_view

            # Step 2: From remaining valid pixels, keep only target fraction
            if middle_90_mask.sum() > 0:
                target_fraction = sparsification_factor / 0.9  # 0.1111...
                num_to_keep = int(middle_90_mask.sum().float() * target_fraction)

                # Get indices of valid pixels in middle 90%
                valid_indices = torch.where(middle_90_mask)
                if len(valid_indices[0]) > 0:
                    # Randomly select indices to keep
                    perm = torch.randperm(len(valid_indices[0]), device=depths.device)
                    keep_indices = perm[:num_to_keep]

                    # Create final mask
                    final_mask = torch.zeros_like(depth_view, dtype=torch.bool)
                    final_mask[
                        valid_indices[0][keep_indices],
                        valid_indices[1][keep_indices],
                    ] = True

                    # Convert back to batch/view indexing
                    b_idx = i // V
                    v_idx = i % V
                    mask[b_idx, v_idx] = final_mask

    return depths * mask.float()


def scale_depths_and_poses(
    depths_z, poses, intrinsics, depths_along_ray=None, cam_points=None
):
    """
    Scale depths and pose translations based on world point distances.

    Args:
        depths_z (torch.Tensor): Z-depths of shape (B, V, H, W)
        poses (torch.Tensor): Camera poses of shape (B, V, 3, 4)
        intrinsics (torch.Tensor): Camera intrinsics of shape (B, V, 3, 3)
        depths_along_ray (torch.Tensor, optional): Depth along ray of shape (B, V, H, W)
        cam_points (torch.Tensor, optional): Camera points (unused but kept for compatibility)

    Returns:
        tuple: (scaled_depths_z, scaled_poses, scaled_depths_along_ray, world_points, B_scales)
    """
    if depths_z is None or poses is None or intrinsics is None:
        return depths_z, poses, depths_along_ray, None, None

    # Ensure all inputs are on the same device (prefer CUDA if available)
    device = (
        poses.device
        if poses.device.type == "cuda"
        else (
            depths_z.device
            if depths_z.device.type == "cuda"
            else intrinsics.device
            if intrinsics.device.type == "cuda"
            else poses.device
        )
    )

    depths_z = depths_z.to(device)
    poses = poses.to(device)
    intrinsics = intrinsics.to(device)
    if depths_along_ray is not None:
        depths_along_ray = depths_along_ray.to(device)

    B, V, H, W = depths_z.shape

    # Create point mask for valid depths
    point_masks = depths_z > 0  # (B, V, H, W)

    # Calculate world points for all views in batch - vectorized version
    # Reshape for batch processing: (B*V, H, W)
    depths_flat = depths_z.view(B * V, H, W)
    intrinsics_flat = intrinsics.view(B * V, 3, 3)
    poses_flat = poses.view(B * V, 3, 4)

    # Convert all depths to camera frame points in one batch
    pts3d_cam_batch, _ = depthmap_to_camera_frame(
        depths_flat, intrinsics_flat
    )  # (B*V, H, W, 3)

    # Ensure pts3d_cam is on the same device and dtype as poses
    pts3d_cam_batch = pts3d_cam_batch.to(device=device, dtype=poses_flat.dtype)

    # Convert all poses to 4x4 matrices for transformation
    pose_4x4_batch = torch.zeros(
        B * V, 4, 4, device=poses_flat.device, dtype=poses_flat.dtype
    )
    pose_4x4_batch[:, :3, :] = poses_flat
    pose_4x4_batch[:, 3, 3] = 1.0

    # Transform all to world coordinates in batch
    pts3d_cam_homo = torch.cat(
        [
            pts3d_cam_batch,
            torch.ones(*pts3d_cam_batch.shape[:-1], 1, device=pts3d_cam_batch.device),
        ],
        dim=-1,
    )  # (B*V, H, W, 4)
    pts3d_world_batch = torch.matmul(
        pts3d_cam_homo.unsqueeze(-2), pose_4x4_batch.unsqueeze(1).unsqueeze(1)
    ).squeeze(-2)  # (B*V, H, W, 4)
    pts3d_world_batch = pts3d_world_batch[..., :3]  # (B*V, H, W, 3)

    # Reshape back to (B, V, H, W, 3)
    new_world_points = pts3d_world_batch.view(B, V, H, W, 3)

    # Clone inputs for scaling
    new_depths = depths_z.clone()
    new_extrinsics = poses.clone()

    # Convert poses to 4x4 format for scaling
    new_extrinsics_4x4 = torch.zeros(B, V, 4, 4, device=poses.device, dtype=poses.dtype)
    new_extrinsics_4x4[:, :, :3, :] = new_extrinsics
    new_extrinsics_4x4[:, :, 3, 3] = 1.0

    # Calculate distance from origin for each point
    dist = new_world_points.norm(dim=-1)  # (B, V, H, W)

    dist_sum = (dist * point_masks.float()).sum(dim=[1, 2, 3])  # (B,)
    valid_count = point_masks.sum(dim=[1, 2, 3]).float()  # (B,)
    avg_scale = (dist_sum / (valid_count + 1e-3)).clamp(min=1e-6, max=1e6)  # (B,)

    # Apply scaling
    new_world_points = new_world_points / avg_scale.view(-1, 1, 1, 1, 1)
    new_extrinsics_4x4[:, :, :3, 3] = new_extrinsics_4x4[:, :, :3, 3] / avg_scale.view(
        -1, 1, 1
    )
    new_depths = new_depths / avg_scale.view(-1, 1, 1, 1)

    # Update poses back to 3x4 format
    scaled_poses = new_extrinsics_4x4[:, :, :3, :]

    # Scale depths_along_ray by the same factor
    scaled_depths_along_ray = None
    if depths_along_ray is not None:
        scaled_depths_along_ray = depths_along_ray / avg_scale.view(-1, 1, 1, 1)

    # print(avg_scale)

    return (
        new_depths,
        scaled_poses,
        scaled_depths_along_ray,
        new_world_points,
        avg_scale,
    )


class ModelArgs:
    def __init__(self):
        self.skip_connections = None

        self.pose_encoder_type = None
        self.intrinsics_encoder_type = None
        self.special_attention_skip_connections = None
        self.use_vggt_rope = True

        self.enable_zero_conv_inject = False
        self.enable_concat_inject = False
        self.enable_cross_attention = False

        self.encode_pose_in_register = False


class Pow3rVGGTWrapper(torch.nn.Module):
    def __init__(
        self,
        name,
        torch_hub_force_reload,
        load_pretrained_weights=True,
        depth=24,
        num_heads=16,
        intermediate_layer_idx=[4, 11, 17, 23],
        load_custom_ckpt=False,
        custom_ckpt_path=None,
        enable_camera=True,
        enable_depth=True,
        enable_point=True,
        enable_track=False,
        enable_scale=True,
        enable_ray=False,
        scale_head_type="ScaleHead_MLP_LCP",
        input_priors=None,
        geometric_input_config=None,
        ablation=None,
    ):
        super().__init__()
        self.name = name
        self.torch_hub_force_reload = torch_hub_force_reload
        self.load_custom_ckpt = load_custom_ckpt
        self.custom_ckpt_path = custom_ckpt_path

        # Store input priors configuration (legacy support)
        if input_priors is None:
            self.input_priors = []
        else:
            self.input_priors = (
                input_priors if isinstance(input_priors, list) else [input_priors]
            )

        # Store geometric input configuration for probabilistic conditioning
        if geometric_input_config is not None:
            self.geometric_input_config = geometric_input_config
        else:
            # Default config for backward compatibility
            self.geometric_input_config = {
                "ray_dirs_prob": 1.0,
                "cam_prob": 1.0,
                "overall_prob": 1.0,
                "dropout_prob": 0.0,
                "depth_prob": 1.0,
                "depth_sparsification": 0.1,
            }

        # Determine dtype based on GPU capability (same pattern as VGGT)
        if torch.cuda.is_available():
            if torch.cuda.get_device_capability()[0] >= 8:
                # Ampere or newer GPUs support bfloat16
                self.dtype = torch.bfloat16
                print("Using bfloat16 for inference (Ampere+ GPU detected)")
            else:
                # Older GPUs use float16
                self.dtype = torch.float16
                print("Using float16 for inference (pre-Ampere GPU detected)")
        else:
            self.dtype = torch.float32
            print("Using float32 for inference (CPU)")

        # Load custom checkpoint if requested
        if self.load_custom_ckpt:
            print(f"Loading checkpoint from {self.custom_ckpt_path} ...")
            assert self.custom_ckpt_path is not None, (
                "custom_ckpt_path must be provided if load_custom_ckpt is set to True"
            )
            os_ckpt_path = Path(self.custom_ckpt_path)
            yaml_file = next(os_ckpt_path.parent.glob("*.yaml"))
            with open(yaml_file, "r") as f:
                print(f"opening config file at path: {yaml_file}")
                model_config = OmegaConf.load(f)

            # Use ablation from checkpoint if available, otherwise use provided ablation parameter
            checkpoint_ablation = model_config.model.ablation if hasattr(model_config.model, 'ablation') else ablation
            final_ablation = checkpoint_ablation if checkpoint_ablation is not None else ablation

            # Convert dict/DictConfig to ModelArgs
            if final_ablation is not None and not isinstance(final_ablation, ModelArgs):
                # Create ModelArgs and populate from dict
                model_args = ModelArgs()
                for key, value in final_ablation.items():
                    setattr(model_args, key, value)
                final_ablation = model_args

            self.model = Pow3rVGGT(
                ablation=final_ablation,
                enable_camera=enable_camera,
                enable_depth=enable_depth,
                enable_point=enable_point,
                enable_track=enable_track,
                enable_scale=enable_scale,
                enable_ray=enable_ray,
                scale_head_type=scale_head_type,
            )

            custom_ckpt = torch.load(self.custom_ckpt_path, weights_only=False)
            # Handle different checkpoint formats
            if "model" in custom_ckpt:
                state_dict = custom_ckpt["model"]
                print("Loading from checkpoint['model'] (training checkpoint format)")
            elif "state_dict" in custom_ckpt:
                state_dict = custom_ckpt["state_dict"]
                print("Loading from checkpoint['state_dict']")
            else:
                state_dict = custom_ckpt
                print("Loading checkpoint directly")

            load_result = self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded checkpoint: {load_result}")

            # Check for scale_head loading issues
            if load_result.missing_keys:
                scale_head_missing = [k for k in load_result.missing_keys if 'scale_head' in k]
                if scale_head_missing:
                    print(f"WARNING: Scale head keys missing from checkpoint: {scale_head_missing}")
            if load_result.unexpected_keys:
                scale_head_unexpected = [k for k in load_result.unexpected_keys if 'scale_head' in k]
                if scale_head_unexpected:
                    print(f"WARNING: Unexpected scale head keys in checkpoint: {scale_head_unexpected}")

            del custom_ckpt, state_dict  # in case it occupies memory

        else:
            # Use provided ablation config, or default ModelArgs if not provided
            if ablation is None:
                final_ablation = ModelArgs()
            else:
                # Convert dict/DictConfig to ModelArgs
                if not isinstance(ablation, ModelArgs):
                    model_args = ModelArgs()
                    for key, value in ablation.items():
                        setattr(model_args, key, value)
                    final_ablation = model_args
                else:
                    final_ablation = ablation

            self.model = Pow3rVGGT(
                ablation=final_ablation,
                enable_camera=enable_camera,
                enable_depth=enable_depth,
                enable_point=enable_point,
                enable_track=enable_track,
                enable_scale=enable_scale,
                enable_ray=enable_ray,
                scale_head_type=scale_head_type,
            )
            self.default_path = "/work/weights/vggt/model.pt"
            custom_ckpt = torch.load(self.default_path, weights_only=False)

            # Handle different checkpoint formats
            if "model" in custom_ckpt:
                state_dict = custom_ckpt["model"]
                print("Loading from checkpoint['model'] (training checkpoint format)")
            elif "state_dict" in custom_ckpt:
                state_dict = custom_ckpt["state_dict"]
                print("Loading from checkpoint['state_dict']")
            else:
                state_dict = custom_ckpt
                print("Loading checkpoint directly")

            load_result = self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded checkpoint: {load_result}")

            # Check for scale_head loading issues
            if load_result.missing_keys:
                scale_head_missing = [k for k in load_result.missing_keys if 'scale_head' in k]
                if scale_head_missing:
                    print(f"WARNING: Scale head keys missing from checkpoint: {scale_head_missing}")
            if load_result.unexpected_keys:
                scale_head_unexpected = [k for k in load_result.unexpected_keys if 'scale_head' in k]
                if scale_head_unexpected:
                    print(f"WARNING: Unexpected scale head keys in checkpoint: {scale_head_unexpected}")

            del custom_ckpt, state_dict  # in case it occupies memory

    def forward(self, views):
        """
        Forward pass wrapper for Pow3r-VGGT

        Assumption:
        - All the input views have the same image shape.

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.
                                Each dictionary should contain the following keys:
                                    "img" (tensor): Image tensor of shape (B, C, H, W).
                                    "data_norm_type" (list): ["identity"]
                                    Optional conditioning inputs:
                                        "ray_directions_cam" (tensor): Ray directions in camera frame (B, H, W, 3)
                                        "camera_pose_trans" (tensor): Camera translation (B, 3)
                                        "camera_pose_quats" (tensor): Camera rotation quaternions (B, 4)

        Returns:
            List[dict]: A list containing the final outputs for all N views, with each dict containing:
                        "pts3d": 3D points in world frame (B, H, W, 3)
                        "pts3d_cam": 3D points in camera frame (B, H, W, 3)
                        "ray_directions": Ray directions in camera frame (B, H, W, 3)
                        "depth_along_ray": Depth along ray (B, H, W, 1)
                        "cam_trans": Camera translation (B, 3)
                        "cam_quats": Camera rotation quaternions (B, 4)
                        "conf": Confidence scores (B, H, W)
        """
        # Get input shape of the images, number of views, and batch size per view
        batch_size_per_view, _, height, width = views[0]["img"].shape
        num_views = len(views)

        # Check the data norm type
        # Pow3r-VGGT expects a normalized image but without the DINOv2 mean and std applied ("identity")
        data_norm_type = views[0]["data_norm_type"][0]
        assert data_norm_type == "identity", (
            "Pow3r-VGGT expects a normalized image but without the DINOv2 mean and std applied (data_norm_type='identity')"
        )

        # Concatenate the images to create a single (B, V, C, H, W) tensor
        img_list = [view["img"] for view in views]
        images = torch.stack(img_list, dim=1)

        # Extract optional conditioning inputs if provided by RMVD adapter
        intrinsics = None
        poses = None
        depths_z = None
        depths_along_ray = None

        # Check if intrinsics are provided (RMVD adapter now stores both rays and intrinsics)
        if "camera_intrinsics" in views[0]:
            # RMVD adapter provides the intrinsic matrix directly
            intrinsics_list = []
            for view in views:
                K = view["camera_intrinsics"]  # (B, 3, 3)
                intrinsics_list.append(K)
            intrinsics = torch.stack(intrinsics_list, dim=1)  # (B, V, 3, 3)

        # Initialize poses as None
        poses = None

        # Debug: Print available keys in first view

        # Check if poses are provided (RMVD provides trans + quats)
        if "camera_pose_trans" in views[0] and "camera_pose_quats" in views[0]:
            # Convert from translation + quaternion to 3x4 matrix format
            # RMVD provides poses in the format expected by MapAnything models

            pose_matrices = []
            for view in views:
                trans = view["camera_pose_trans"]  # (B, 3)
                quats = view["camera_pose_quats"]  # (B, 4) in XYZW format
                rot_mat = quat_to_mat(quats)  # (B, 3, 3)
                # Combine into 3x4 matrix [R | t]
                pose_3x4 = torch.cat(
                    [rot_mat, trans.unsqueeze(-1)], dim=-1
                )  # (B, 3, 4)
                pose_matrices.append(pose_3x4)
            poses = torch.stack(pose_matrices, dim=1)  # (B, V, 3, 4)

            # normalize poses such that they are with respect to the first view in each batch
            # Convert 3x4 to 4x4 for easier matrix operations
            B, V = poses.shape[:2]
            poses_4x4 = torch.zeros(B, V, 4, 4, device=poses.device, dtype=poses.dtype)
            poses_4x4[:, :, :3, :] = poses
            poses_4x4[:, :, 3, 3] = 1.0  # Set bottom-right to 1

            # # convert poses from cam2world to world2cam
            # poses_4x4 = torch.inverse(poses_4x4)

            # Get inverse of first pose for each batch
            first_pose_inv = torch.inverse(poses_4x4[:, 0])  # (B, 4, 4)
            first_pose_inv = first_pose_inv.unsqueeze(1)  # (B, 1, 4, 4)

            # Apply inverse transformation to all poses
            poses_4x4_normalized = torch.matmul(first_pose_inv, poses_4x4)

            # Convert back to 3x4 format
            poses = poses_4x4_normalized[:, :, :3, :]

        # Extract depth information if available
        if "depth_z" in views[0]:
            # Extract z-depths from views
            depth_z_list = []
            for view in views:
                depth_z = view["depth_z"]  # (B, H, W, 1)
                depth_z = depth_z.squeeze(-1)  # (B, H, W)
                depth_z_list.append(depth_z)
            depths_z = torch.stack(depth_z_list, dim=1)  # (B, V, H, W)

            # Apply sparsification to z-depths
            sparsification_factor = self.geometric_input_config.get(
                "depth_sparsification", 0.1
            )
            depths_z = apply_depth_sparsification(depths_z, sparsification_factor)

        if "depth_along_ray" in views[0]:
            # Extract depth-along-ray from views
            depth_along_ray_list = []
            for view in views:
                depth_along_ray = view["depth_along_ray"]  # (B, H, W, 1)
                depth_along_ray = depth_along_ray.squeeze(-1)  # (B, H, W)
                depth_along_ray_list.append(depth_along_ray)
            depths_along_ray = torch.stack(depth_along_ray_list, dim=1)  # (B, V, H, W)

            # Apply sparsification to depth-along-ray
            sparsification_factor = self.geometric_input_config.get(
                "depth_sparsification", 0.1
            )
            depths_along_ray = apply_depth_sparsification(
                depths_along_ray, sparsification_factor
            )

        # Debug I/O operations removed for performance - all data transfers to CPU eliminated

        depths_z, poses, depths_along_ray, world_points, B_scales = (
            scale_depths_and_poses(depths_z, poses, intrinsics, depths_along_ray)
        )
        # Debug prints removed for performance

        # import pdb
        # pdb.set_trace()
        # if depths_z is not None:
        #     # Expand (B, S, H, W) to (B, S, H, W, 1) for log normalization
        #     depths_z = depths_z.unsqueeze(-1)  # (B, S, H, W, 1)
        #     # Apply log normalization: norm -> normalize -> scale by log(1+norm)
        #     org_d = depths_z.norm(dim=-1, keepdim=True)  # (B, S, H, W, 1)
        #     depths_z = depths_z / org_d.clip(min=1e-8)  # normalize
        #     depths_z = depths_z * torch.log1p(org_d)  # scale by log(1+norm)
        #     # Squeeze back to (B, S, H, W)
        #     depths_z = depths_z.squeeze(-1)  # (B, S, H, W)

        # if depths_along_ray is not None:
        #     # Expand (B, S, H, W) to (B, S, H, W, 1) for log normalization
        #     depths_along_ray = depths_along_ray.unsqueeze(-1)  # (B, S, H, W, 1)
        #     # Apply log normalization: norm -> normalize -> scale by log(1+norm)
        #     org_d = depths_along_ray.norm(dim=-1, keepdim=True)  # (B, S, H, W, 1)
        #     depths_along_ray = depths_along_ray / org_d.clip(min=1e-8)  # normalize
        #     depths_along_ray = depths_along_ray * torch.log1p(org_d)  # scale by log(1+norm)
        #     # Squeeze back to (B, S, H, W)
        #     depths_along_ray = depths_along_ray.squeeze(-1)  # (B, S, H, W)

        # Convert input format to what the model expects
        # Apply probabilistic dropout to geometric inputs during training
        model_intrinsics = None
        model_poses = None
        model_depths = None

        # Get probabilities from config
        overall_prob = self.geometric_input_config.get("overall_prob", 1.0)
        ray_dirs_prob = self.geometric_input_config.get("ray_dirs_prob", 1.0)
        cam_prob = self.geometric_input_config.get("cam_prob", 1.0)
        depth_prob = self.geometric_input_config.get("depth_prob", 1.0)

        if self.training:
            # Training mode: apply probabilistic dropout
            # First, determine if we use geometric inputs at all
            use_geometric_inputs = torch.rand(1).item() < overall_prob

            if use_geometric_inputs:
                # Legacy input_priors support (deterministic)
                if len(self.input_priors) > 0:
                    if "intrinsics" in self.input_priors and intrinsics is not None:
                        model_intrinsics = intrinsics.to(images.device)

                    if "extrinsics" in self.input_priors and poses is not None:
                        model_poses = poses.to(images.device)
                        print(model_poses)

                    if "depths" in self.input_priors and depths_z is not None:
                        model_depths = depths_z.to(images.device)
                else:
                    # Probabilistic conditioning: apply individual probabilities
                    if intrinsics is not None and torch.rand(1).item() < ray_dirs_prob:
                        model_intrinsics = intrinsics.to(images.device)

                    if poses is not None and torch.rand(1).item() < cam_prob:
                        model_poses = poses.to(images.device)

                    if depths_z is not None and torch.rand(1).item() < depth_prob:
                        model_depths = depths_z.to(images.device)
        else:
            # Evaluation mode: use all available inputs (or respect input_priors if specified)
            if len(self.input_priors) > 0:
                if "intrinsics" in self.input_priors and intrinsics is not None:
                    model_intrinsics = intrinsics.to(images.device)

                if "extrinsics" in self.input_priors and poses is not None:
                    model_poses = poses.to(images.device)

                if "depths" in self.input_priors and depths_z is not None:
                    model_depths = depths_z.to(images.device)
            else:
                # Use all available inputs during evaluation
                if intrinsics is not None:
                    model_intrinsics = intrinsics.to(images.device)

                if poses is not None:
                    model_poses = poses.to(images.device)

                if depths_z is not None:
                    model_depths = depths_z.to(images.device)

        # Run the Pow3r-VGGT model with new interface
        with torch.autocast("cuda", dtype=self.dtype):
            model_output = self.model(
                images=images,
                intrinsics=model_intrinsics,
                poses=model_poses,
                depths=model_depths,
                # scale=1.0 / B_scales if B_scales is not None else None,
                scale=None,
                injection_masks=None,
            )

        # Extract predictions from model output
        with torch.autocast("cuda", enabled=False):
            # Get outputs from model
            pose_enc = model_output["pose_enc"]
            depth_map = model_output["depth"]  # (B, V, H, W, 1)
            depth_conf = model_output["depth_conf"]  # (B, V, H, W)
            scale = model_output["scale"]  # Scale predictions

            print(f"DEBUG: scale shape from model = {scale.shape}")

            # Extract scale factor - handle different possible shapes
            # Scale could be (B,), (B, 1), or (B, num_views)
            if scale.ndim == 1:
                # Already (B,)
                metric_scaling_factor = scale
            elif scale.ndim == 2:
                if scale.shape[1] == 1:
                    # (B, 1) -> (B,)
                    metric_scaling_factor = scale.squeeze(-1)
                else:
                    # (B, num_views) -> (B,) by taking mean across views
                    metric_scaling_factor = scale.mean(dim=-1)
            else:
                # Scalar - make it (B,) with batch size 1
                metric_scaling_factor = scale.unsqueeze(0) if scale.ndim == 0 else scale

            # Keep depth_map unscaled - we'll scale outputs explicitly later
            # This matches MapAnything's pattern

            # Extrinsic and intrinsic matrices from pose encoding
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                pose_enc, images.shape[-2:]
            )

            # Convert the output to MapAnything format
            res = []
            for view_idx in range(num_views):
                # Get the extrinsics, intrinsics, depth map for the current view
                curr_view_extrinsic = extrinsic[:, view_idx, ...]
                curr_view_extrinsic = closed_form_inverse_se3(
                    curr_view_extrinsic
                )  # Convert to cam2world
                curr_view_intrinsic = intrinsic[:, view_idx, ...]
                curr_view_depth_z = depth_map[:, view_idx, ...]
                curr_view_depth_z = curr_view_depth_z.squeeze(-1)
                curr_view_confidence = depth_conf[:, view_idx, ...]

                # Get the camera frame pointmaps (z-depth based)
                curr_view_pts3d_cam, _ = depthmap_to_camera_frame(
                    curr_view_depth_z, curr_view_intrinsic
                )

                # Convert the extrinsics to quaternions and translations
                translation = curr_view_extrinsic[..., :3, 3]
                curr_view_cam_translations = translation  # Unscaled - will scale at output
                curr_view_cam_quats = mat_to_quat(curr_view_extrinsic[..., :3, :3])

                # Convert the z depth to depth along ray
                curr_view_depth_along_ray = convert_z_depth_to_depth_along_ray(
                    curr_view_depth_z, curr_view_intrinsic
                )
                curr_view_depth_along_ray = curr_view_depth_along_ray.unsqueeze(-1)

                # Get the ray directions on the unit sphere in the camera frame
                _, curr_view_ray_dirs = get_rays_in_camera_frame(
                    curr_view_intrinsic, height, width, normalize_to_unit_sphere=True
                )

                # Get the pointmaps in world frame
                curr_view_pts3d = (
                    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
                        curr_view_ray_dirs,
                        curr_view_depth_along_ray,
                        curr_view_cam_translations,
                        curr_view_cam_quats,
                    )
                )

                # Append the outputs to the result list in MapAnything format
                # Scale all spatial quantities explicitly (matching MapAnything's pattern)
                res.append(
                    {
                        "pts3d": curr_view_pts3d * metric_scaling_factor.unsqueeze(-1).unsqueeze(-1),
                        "pts3d_cam": curr_view_pts3d_cam * metric_scaling_factor.unsqueeze(-1).unsqueeze(-1),
                        "ray_directions": curr_view_ray_dirs,  # NOT scaled (normalized)
                        "depth_along_ray": curr_view_depth_along_ray * metric_scaling_factor.unsqueeze(-1).unsqueeze(-1),
                        "cam_trans": curr_view_cam_translations * metric_scaling_factor.unsqueeze(-1),
                        "cam_quats": curr_view_cam_quats,  # NOT scaled (normalized)
                        "conf": curr_view_confidence,
                        "metric_scaling_factor": metric_scaling_factor,  # Raw scale for loss
                    }
                )

        return res


def build_intrinsics_from_pose_enc(
    pose_encoding,
    image_size_hw=None,
):
    """Convert a pose encoding back to camera intrinsics from field of view."""
    fov_h = pose_encoding[..., 7]
    fov_w = pose_encoding[..., 8]

    H, W = image_size_hw
    fy = (H / 2.0) / torch.tan(fov_h / 2.0)
    fx = (W / 2.0) / torch.tan(fov_w / 2.0)
    intrinsics = torch.zeros(
        pose_encoding.shape[:2] + (3, 3),
        dtype=pose_encoding.dtype,
        device=pose_encoding.device,
    )
    intrinsics[..., 0, 0] = fx
    intrinsics[..., 1, 1] = fy
    intrinsics[..., 0, 2] = W / 2
    intrinsics[..., 1, 2] = H / 2
    intrinsics[..., 2, 2] = 1.0

    return intrinsics

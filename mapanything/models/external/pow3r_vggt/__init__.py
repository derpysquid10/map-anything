# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Inference wrapper for Pow3r-VGGT
"""

import torch

from mapanything.models.external.pow3r_vggt.models.pow3r_vggt import Pow3rVGGT
from mapanything.models.external.pow3r_vggt.utils.geometry import closed_form_inverse_se3
from mapanything.models.external.pow3r_vggt.utils.pose_enc import pose_encoding_to_extri_intri
from mapanything.models.external.pow3r_vggt.utils.rotation import mat_to_quat
from mapanything.utils.geometry import (
    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap,
    convert_z_depth_to_depth_along_ray,
    depthmap_to_camera_frame,
    get_rays_in_camera_frame,
)

from pathlib import Path
from omegaconf import OmegaConf
import os


class ModelArgs():
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
    ):
        super().__init__()
        self.name = name
        self.torch_hub_force_reload = torch_hub_force_reload
        self.load_custom_ckpt = load_custom_ckpt
        self.custom_ckpt_path = custom_ckpt_path

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
            with open(yaml_file, 'r') as f:
                print(f"opening config file at path: {yaml_file}")
                model_config = OmegaConf.load(f)
            self.model = Pow3rVGGT(ablation=model_config.model.ablation)

            custom_ckpt = torch.load(self.custom_ckpt_path, weights_only=False)
            # Handle different checkpoint formats
            if 'model' in custom_ckpt:
                state_dict = custom_ckpt['model']
                print(f"Loading from checkpoint['model'] (training checkpoint format)")
            elif 'state_dict' in custom_ckpt:
                state_dict = custom_ckpt['state_dict']
                print(f"Loading from checkpoint['state_dict']")
            else:
                state_dict = custom_ckpt
                print(f"Loading checkpoint directly")

            load_result = self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded checkpoint: {load_result}")
            del custom_ckpt, state_dict  # in case it occupies memory

        else:
            args = ModelArgs()
            self.model = Pow3rVGGT(ablation=args)
            self.default_path = "/work/weights/vggt/model.pt"
            custom_ckpt = torch.load(self.default_path, weights_only=False)

            # Handle different checkpoint formats
            if 'model' in custom_ckpt:
                state_dict = custom_ckpt['model']
                print(f"Loading from checkpoint['model'] (training checkpoint format)")
            elif 'state_dict' in custom_ckpt:
                state_dict = custom_ckpt['state_dict']
                print(f"Loading from checkpoint['state_dict']")
            else:
                state_dict = custom_ckpt
                print(f"Loading checkpoint directly")

            load_result = self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded checkpoint: {load_result}")
            del custom_ckpt, state_dict  # in case it occupies memory

        # Add geometric_input_config for RMVD adapter compatibility
        # Pow3r-VGGT always uses conditioning when provided (no dropout)
        # These values will be set by RMVD adapter based on evaluation_conditioning
        self.geometric_input_config = {
            "ray_dirs_prob": 1.0,      # Probability of using intrinsics (ray directions)
            "cam_prob": 1.0,            # Probability of using camera poses
            "overall_prob": 1.0,        # Overall probability of using geometric inputs
            "dropout_prob": 0.0,        # Dropout probability (0 = always use when available)
        }



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
        if "intrinsics" in views[0]:
            # RMVD adapter provides the intrinsic matrix directly
            intrinsics_list = []
            for view in views:
                K = view["intrinsics"]  # (B, 3, 3)
                intrinsics_list.append(K)
            intrinsics = torch.stack(intrinsics_list, dim=1)  # (B, V, 3, 3)

        # Initialize poses as None
        poses = None
        
        # Debug: Print available keys in first view
        print(f"[DEBUG] Available keys in views[0]: {list(views[0].keys())}")
        
        # Check if poses are provided (RMVD provides trans + quats)
        if "camera_pose_trans" in views[0] and "camera_pose_quats" in views[0]:
            print(f"[DEBUG] Found camera_pose_trans and camera_pose_quats format")
            # Convert from translation + quaternion to 3x4 matrix format
            # RMVD provides poses in the format expected by MapAnything models
            from mapanything.models.external.pow3r_vggt.utils.rotation import quat_to_mat

            pose_matrices = []
            for view in views:
                trans = view["camera_pose_trans"]  # (B, 3)
                quats = view["camera_pose_quats"]  # (B, 4) in XYZW format
                rot_mat = quat_to_mat(quats)  # (B, 3, 3)
                # Combine into 3x4 matrix [R | t]
                pose_3x4 = torch.cat([rot_mat, trans.unsqueeze(-1)], dim=-1)  # (B, 3, 4)
                pose_matrices.append(pose_3x4)
            poses = torch.stack(pose_matrices, dim=1)  # (B, V, 3, 4)
            print(f"[DEBUG] Created poses from trans+quats, shape: {poses.shape}")
        elif "extrinsics" in views[0]:
            print(f"[DEBUG] Found extrinsics format")
            # Check if poses are provided as extrinsics matrices
            extrinsics_list = []
            for view in views:
                ext = view["extrinsics"]  # Should be (B, 4, 4) or (B, 3, 4)
                if ext.shape[-2:] == (4, 4):
                    # Convert 4x4 to 3x4 by dropping last row
                    ext = ext[:, :3, :]  # (B, 3, 4)
                extrinsics_list.append(ext)
            poses = torch.stack(extrinsics_list, dim=1)  # (B, V, 3, 4)
            print(f"[DEBUG] Created poses from extrinsics, shape: {poses.shape}")
        else:
            print(f"[DEBUG] No pose data found - poses will be None")

        # Extract depth information if available
        if "depth_z" in views[0]:
            # Extract z-depths from views
            depth_z_list = []
            for view in views:
                depth_z = view["depth_z"]  # (B, H, W, 1)
                depth_z = depth_z.squeeze(-1)  # (B, H, W)
                depth_z_list.append(depth_z)
            depths_z = torch.stack(depth_z_list, dim=1)  # (B, V, H, W)

        if "depth_along_ray" in views[0]:
            # Extract depth-along-ray from views
            depth_along_ray_list = []
            for view in views:
                depth_along_ray = view["depth_along_ray"]  # (B, H, W, 1)
                depth_along_ray = depth_along_ray.squeeze(-1)  # (B, H, W)
                depth_along_ray_list.append(depth_along_ray)
            depths_along_ray = torch.stack(depth_along_ray_list, dim=1)  # (B, V, H, W)

        # Run the Pow3r-VGGT aggregator with conditioning
        with torch.autocast("cuda", dtype=self.dtype):
            aggregated_tokens_list, ps_idx = self.model.aggregator(
                images=images,
                intrinsics=intrinsics,
                poses=poses,
                depths_z=depths_z,
                depths_along_ray=depths_along_ray
            )

        # Run the Camera + Pose Branch and Depth Branch
        with torch.autocast("cuda", enabled=False):
            # Predict Cameras
            pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
            # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
            # Extrinsics Shape: (B, V, 3, 4)
            # Intrinsics Shape: (B, V, 3, 3)
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                pose_enc, images.shape[-2:]
            )

            # Predict Depth Maps
            # Depth Shape: (B, V, H, W, 1)
            # Depth Confidence Shape: (B, V, H, W)
            depth_map, depth_conf = self.model.depth_head(
                aggregated_tokens_list, images, ps_idx
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
                curr_view_cam_translations = curr_view_extrinsic[..., :3, 3]
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
                res.append(
                    {
                        "pts3d": curr_view_pts3d,
                        "pts3d_cam": curr_view_pts3d_cam,
                        "ray_directions": curr_view_ray_dirs,
                        "depth_along_ray": curr_view_depth_along_ray,
                        "cam_trans": curr_view_cam_translations,
                        "cam_quats": curr_view_cam_quats,
                        "conf": curr_view_confidence,
                    }
                )

        return res

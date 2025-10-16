# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any
from pytorch3d.transforms import matrix_to_rotation_6d

from pow3r_vggt.layers import PatchEmbed
from pow3r_vggt.layers.block import Block
from pow3r_vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from pow3r_vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
from pow3r_vggt.layers.prope_attention import _rope_precompute_coeffs, _prepare_apply_fns
from pow3r_vggt.layers.gta_attention import _prepare_apply_fns as _prepare_GTA_apply_fns
from pow3r_vggt.layers.cape_attention import _prepare_apply_fns as _prepare_cape_apply_fns


from pow3r_vggt.layers.prior_encoders import RayEncoder, DepthEncoder, PoseEncoder, PoseEncoder6D, PoseEncoderQuaternion
from pow3r_vggt.utils.raymap import generate_raymap
from pow3r_vggt.utils.pose_enc import mat_to_quat
from pow3r_vggt.utils.rotation import normalize_camera_extrinsics_batch


import pdb

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]



class Pow3rAggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.

    Remember to set model.train() to enable gradient checkpointing to reduce memory usage.

    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
        skip_connections: The layers in attention to add the skip connections to priors before
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        ablation=None
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 and ablation.use_vggt_rope is True else None
        self.position_getter = PositionGetter() if self.rope is not None else None
        
        self.pose_encoder_type = ablation.pose_encoder_type
        self.intrinsics_encoder_type = ablation.intrinsics_encoder_type
        self.skip_connections = ablation.skip_connections
        self.special_attention_skip_connections = ablation.special_attention_skip_connections
        self.encode_pose_in_register = ablation.encode_pose_in_register
        self.head_dim = embed_dim // num_heads
        self.other_ablations = ablation

        self.attention_encoder_types = ["prope", "cape", "gta"]

        pose_embed_dim = 4096 if self.encode_pose_in_register else 1024

        print("="*50)

        if self.pose_encoder_type == None or self.pose_encoder_type == "3x4":
            self.pose_encoder = PoseEncoder(embed_dim=pose_embed_dim)
            print("using regular pose encoder")
            
        elif self.pose_encoder_type == "6D":
            self.pose_encoder = PoseEncoder6D(embed_dim=pose_embed_dim)
            print("using 6D pose encoder")

        elif self.pose_encoder_type == "quaternion":
            self.pose_encoder = PoseEncoderQuaternion(embed_dim=pose_embed_dim)
            print("using quaternion pose encoder")

        elif self.pose_encoder_type == "prope":
            print("using PRoPE encoding...")
            self.pose_encoder = None

        else:
            print(f"{self.pose_encoder_type} not valid, resorting to 3x4 pose encoder")
            
            self.pose_encoder = PoseEncoder(embed_dim=pose_embed_dim)
        print("="*50)

        self.ray_encoder = RayEncoder()
        self.depth_encoder = DepthEncoder()
        

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant = False # hardcoded to False

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def setup_prope(self, poses, intrinsics, H, W, tokens_per_image, B):
        """
        Setup PRoPE (Projected RoPE) encoding arguments.
        
        Args:
            poses (torch.Tensor): Camera poses [B,S , 3, 4]
            intrinsics (torch.Tensor): Camera intrinsics [B, S, 3, 3]
            H, W (int): Image height and width
            tokens_per_image (int): Number of tokens per image
            B (int): Batch size
            
        Returns:
            Dict: PRoPE arguments for attention blocks
        """
        freq_base = 100.0
        freq_scale = 1.0

        patches_x = W // self.patch_size
        patches_y = H // self.patch_size

        # Precompute RoPE coefficients for x and y dimensions
        coeffs_x: Tuple[torch.Tensor, torch.Tensor] = _rope_precompute_coeffs(
            torch.tile(torch.arange(patches_x), (patches_y,)),
            freq_base=freq_base,
            freq_scale=freq_scale,
            feat_dim=self.head_dim // 4,
        )

        coeffs_y: Tuple[torch.Tensor, torch.Tensor] = _rope_precompute_coeffs(
            torch.repeat_interleave(torch.arange(patches_y), patches_x),
            freq_base=freq_base,
            freq_scale=freq_scale,
            feat_dim=self.head_dim // 4,
        )

        B, S = poses.shape[:2]

        # Add bottom row [0, 0, 0, 1] to convert 3x4 to 4x4 homogeneous matrices
        bottom_row = torch.tensor([0, 0, 0, 1], device=poses.device, dtype=poses.dtype)
        bottom_row = bottom_row.expand(B, S, 1, 4)
        prope_input_poses = torch.cat([poses, bottom_row], dim=2)  # (B, S, 4, 4)

        

        # Prepare apply functions for query, key-value, and output projections
        apply_fn_q, apply_fn_kv, apply_fn_o = _prepare_apply_fns(
            head_dim=self.head_dim,
            viewmats=prope_input_poses,
            Ks=intrinsics,
            patches_x=patches_x,
            patches_y=patches_y,
            image_width=W,
            image_height=H,
            coeffs_x=coeffs_x,
            coeffs_y=coeffs_y,
        )

        return {
            "q_encoder": apply_fn_q,
            "kv_encoder": apply_fn_kv,
            "o_encoder": apply_fn_o,
            "tokens_per_image": tokens_per_image,
            "batch_size": B,
        }
    
    def setup_cape(self, poses, tokens_per_image, B):
        """
        Setup GTA encoding arguments.
        
        Args:
            poses (torch.Tensor): Camera poses [B,S , 3, 4]

        Returns:
            Dict: GTA arguments for attention blocks
        """

        B, S = poses.shape[:2]

        # Add bottom row [0, 0, 0, 1] to convert 3x4 to 4x4 homogeneous matrices
        bottom_row = torch.tensor([0, 0, 0, 1], device=poses.device, dtype=poses.dtype)
        bottom_row = bottom_row.expand(B, S, 1, 4)
        gta_input_poses = torch.cat([poses, bottom_row], dim=2)  # (B, S, 4, 4)

        apply_fn_q, apply_fn_k, apply_fn_o = _prepare_cape_apply_fns(
            head_dim=self.head_dim,
            viewmats=gta_input_poses,
        )

        return {
            "q_encoder": apply_fn_q,
            "k_encoder": apply_fn_k,
            "o_encoder": apply_fn_o,
            "tokens_per_image": tokens_per_image,
            "batch_size": B,
        }

    def setup_gta(self, poses, tokens_per_image, B):
        """
        Setup GTA encoding arguments.
        
        Args:
            poses (torch.Tensor): Camera poses [B,S , 3, 4]

        Returns:
            Dict: GTA arguments for attention blocks
        """

        B, S = poses.shape[:2]

        # Add bottom row [0, 0, 0, 1] to convert 3x4 to 4x4 homogeneous matrices
        bottom_row = torch.tensor([0, 0, 0, 1], device=poses.device, dtype=poses.dtype)
        bottom_row = bottom_row.expand(B, S, 1, 4)
        gta_input_poses = torch.cat([poses, bottom_row], dim=2)  # (B, S, 4, 4)

        apply_fn_q, apply_fn_kv, apply_fn_o = _prepare_GTA_apply_fns(
            head_dim=self.head_dim,
            viewmats=gta_input_poses,
        )

        return {
            "q_encoder": apply_fn_q,
            "kv_encoder": apply_fn_kv,
            "o_encoder": apply_fn_o,
            "tokens_per_image": tokens_per_image,
            "batch_size": B,
        }

    def forward(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor = None,
        depths: torch.Tensor = None,
        poses=None,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape


        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = images.to(self._resnet_mean.device)
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # Store encodings for skip connections
        ray_embeddings = None
        pose_encodings = None
        depth_encodings = None
        

        if intrinsics is not None:
            if "raymap" in self.intrinsics_encoder_type:
                print(f"using raymap encoder")
                # First reshape images back to (B, S, C, H, W) for raymap generation
                images_for_raymap = images.view(B, S, C_in, H, W)
                
                # Generate raymaps - returns (B, S, 3, H, W)
                ray_images = generate_raymap(images_for_raymap, intrinsics)
                
                # Reshape to (B*S, 3, H, W) for the encoder
                ray_images = ray_images.view(B * S, 3, H, W)
                ray_embeddings, pos = self.ray_encoder(ray_images)
                patch_tokens += ray_embeddings
            else:
                print("No intrinsics encoder used.")


        if poses is not None:
            poses = poses.to(next(self.parameters()).device)
         
            if self.pose_encoder_type in ["6D", "quaternion", "3x4"]:
                if self.pose_encoder_type == "6D":
                    poses = poses.view(B * S, 3, 4) 
                    rotations = poses[:, :, :3]
                    translations = poses[:, :, 3]
                    rotation_6d = matrix_to_rotation_6d(rotations)
                    poses = torch.cat([rotation_6d, translations], dim=1)

                elif self.pose_encoder_type == "quaternion":
                    poses = poses.view(B * S, 3, 4) 
                    rotations = poses[:, :, :3]
                    translations = poses[:, :, 3]
                    quaternions = mat_to_quat(rotations)
                    poses = torch.cat([quaternions, translations], dim=1)
                
                elif self.pose_encoder_type == "3x4":
                    poses = poses.view(B * S, -1) # Reshape to [B*S, 12] 
                
                pose_encodings = self.pose_encoder(poses)  # (B*S, embed_dim)

                if self.encode_pose_in_register:
                    pose_encodings = pose_encodings.view(B * S, 4, -1)  # (B*S, 4, 1024)
                    register_token += pose_encodings
                else:
                    pose_encodings = pose_encodings.view(B * S, 1, -1)  # (B*S, 1, 1024)  
                    camera_token += pose_encodings
            # else:
            #     print(f"Pose encoder type '{self.pose_encoder_type}' not supported, skipping pose encoding.")

        if depths is not None: # TODO: add ray depth encodings
            # Add channel dimension and reshape from (B, S, H, W) to (B*S, 1, H, W)
            depths = depths.unsqueeze(2)  # (B, S, 1, H, W)
            depths = depths.view(B * S, 1, H, W)
            depth_encodings, _ = self.depth_encoder(depths)
            patch_tokens += depth_encodings

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        special_attention_args = None
        special_attention_type = self.pose_encoder_type if self.pose_encoder_type in self.attention_encoder_types else None
        if self.pose_encoder_type in self.attention_encoder_types:
            if self.pose_encoder_type == "prope" and poses is not None:
                prope_input_intrinsics = intrinsics if "prope" in self.intrinsics_encoder_type else None
                special_attention_args = self.setup_prope(poses, prope_input_intrinsics, H, W, tokens.shape[1], B)

                print("="*50)
                print(f"Setting up Prope Attention with {prope_input_intrinsics is not None} intrinsics")
                print("="*50)

            elif "cape" in self.pose_encoder_type and poses is not None:
                special_attention_args = self.setup_cape(poses, tokens.shape[1], B)
                print("="*50)
                print("Setting up Cape Attention")
                print("="*50)

            elif "gta" in self.pose_encoder_type and poses is not None:
                special_attention_args = self.setup_gta(poses, tokens.shape[1], B)

                print("="*50)
                print("Setting up GTA Attention")
                print("="*50)
            
            else:
                print(f"special encoder type does not exist: {self.pose_encoder_type}, skipping special attention")


        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0 and pos is not None:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []

        for i in range(self.aa_block_num): 
            if self.skip_connections is not None and i in self.skip_connections and i != 0:
                # Apply skip connections: add priors to tokens at specified layers
                
                # Token structure after concatenation: [camera_token, register_token, patch_tokens]
                # - camera_token: [B*S, 1, C] (1 camera token per frame)
                # - register_token: [B*S, num_register_tokens, C] 
                # - patch_tokens: [B*S, num_patch_tokens, C]
                
                num_register_tokens = self.register_token.shape[2]  # Get from the original parameter
                
                # Extract different token types
                tokens = tokens.view(B * S, -1, C)  # Reshape to [B*S, 1+num_register_tokens, num_patch_tokens, C]
                camera_tokens = tokens[:, 0:1, :]  # [B*S, 1, C] - 1 camera token per frame
                register_tokens = tokens[:, 1:self.patch_start_idx, :]  # [B*S, num_register_tokens, C]
                patch_tokens = tokens[:, self.patch_start_idx:, :]  # [B*S, num_patch_tokens, C]
      
                
                if poses is not None and pose_encodings is not None:
                    if self.encode_pose_in_register:
                        register_tokens += pose_encodings
                    else:
                        camera_tokens += pose_encodings
                
                if depths is not None and depth_encodings is not None:
                    patch_tokens = patch_tokens + depth_encodings
                    
                if intrinsics is not None and ray_embeddings is not None:
                    patch_tokens = patch_tokens + ray_embeddings
                
                # Update the tokens with the modified camera and patch tokens
                tokens[:, 0:1, :] = camera_tokens  # Update camera tokens
                tokens[:, 1:self.patch_start_idx, :] = register_tokens
                tokens[:, self.patch_start_idx:, :] = patch_tokens  # Update patch tokens
                tokens = tokens.view(B*S, -1, C)  # Reshape back to [B*S, 1+num_register_tokens+num_patch_tokens, C]
            
            use_special_attention = False
            if special_attention_args is not None:
                
                if self.special_attention_skip_connections == "all": # apply special attention to all layers if not specified
                    use_special_attention = True 

                elif self.special_attention_skip_connections is not None:
                    # Only check 'i in ...' if it's actually a list/iterable, not None
                    if isinstance(self.special_attention_skip_connections, (list, tuple)):
                        use_special_attention = True if i in self.special_attention_skip_connections else False

            for attn_type in self.aa_order:
                
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos, 
                        special_attention_args=special_attention_args, use_special_attention=use_special_attention,
                        special_attention_type=special_attention_type,
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos, 
                        special_attention_args=special_attention_args, use_special_attention=use_special_attention,
                        special_attention_type=special_attention_type,
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        return output_list, self.patch_start_idx

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None, 
                                special_attention_args=None, 
                                use_special_attention=False, 
                                special_attention_type=None
                                ):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(
                    self.frame_blocks[frame_idx],
                    tokens,
                    pos,
                    special_attention_args=special_attention_args,
                    use_reentrant=self.use_reentrant,
                    use_special_attention=use_special_attention,
                    special_attention_type=special_attention_type
                )
            else:
                tokens = self.frame_blocks[frame_idx](
                    tokens,
                    pos=pos,
                    special_attention_args=special_attention_args,
                    use_special_attention=use_special_attention,
                    special_attention_type=special_attention_type
                )
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None, 
                                    special_attention_args=None, 
                                    use_special_attention=False,
                                    special_attention_type=None
                                    ):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(
                    self.global_blocks[global_idx],
                    tokens,
                    pos,
                    special_attention_args=special_attention_args,
                    use_reentrant=self.use_reentrant,
                    use_special_attention=use_special_attention,
                    special_attention_type=special_attention_type
                )
            else:
                tokens = self.global_blocks[global_idx](
                    tokens,
                    pos=pos,
                    special_attention_args=special_attention_args,
                    use_special_attention=use_special_attention,
                    special_attention_type=special_attention_type
                )
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined

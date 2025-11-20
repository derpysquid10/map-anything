# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from mapanything.models.external.pow3r_vggt.layers import Mlp
from mapanything.models.external.pow3r_vggt.layers.block import Block
from mapanything.models.external.pow3r_vggt.heads.head_act import activate_pose
from .modules import MLP
import copy
import time
from mapanything.models.external.pow3r_vggt.heads.FiLM import ResidualFiLM

device = "cuda" if torch.cuda.is_available() else "cpu"




class ScaleHead_MLP_L(nn.Module):
    """
    MLP_LCP
    """
    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
    ):
        super().__init__()

        self.token_norm1024 = nn.LayerNorm(dim_in//2)

        self.mlp = MLP([1024, 512, 256, 64, 1])
        self.proj = nn.Linear(dim_in, 1)  # Will learn attention over tokens

    def forward(self, aggregated_tokens_list: list, cls_token: torch.tensor, patch_tokens: torch.tensor, B: int, S: int) -> float:
        cls_token = cls_token.view(B,S,-1)   # [B,S,1024]
        cls_token_norm = self.token_norm1024(cls_token)
        total_tokens_post = torch.cat((cls_token_norm), dim=-1)
        out = torch.exp(self.mlp(total_tokens_post)).squeeze(-1)

        encoded_feature = torch.mean(out,dim=-1)

        return encoded_feature



class ScaleHead_MLP_LCP(nn.Module):
    """
    MLP_LCP
    """
    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
    ):
        super().__init__()

        self.token_norm1024 = nn.LayerNorm(dim_in//2)
        self.token_norm2048 = nn.LayerNorm(dim_in)

        self.mlp = MLP([5120, 2048, 256, 64, 1])
        self.proj = nn.Linear(dim_in, 1)  # Will learn attention over tokens

    def forward(self, aggregated_tokens_list: list, cls_token: torch.tensor, patch_tokens: torch.tensor, pose_encodings: torch.tensor, ray_embeddings: torch.tensor, B: int, S: int) -> float:
        aggregated_cam_token = aggregated_tokens_list[-1][:,:,0,:]                             # [B,S,2048]
        # aggregated_register_token = aggregated_tokens_list[-1][:,:,1:5,:]                      # [B,S,4,2048]
        aggregated_patch_tokens = aggregated_tokens_list[-1][:,:,5:,:]                         # [B,S,N,2048]
        if ray_embeddings is not None:
            ray_embeddings = ray_embeddings.view(B, S, aggregated_patch_tokens.shape[2], -1)  # [B,S,1,1024]
            ray_embeddings = torch.concat([ray_embeddings,ray_embeddings],dim=-1)  # [B,S,1,2048]
            aggregated_patch_tokens = aggregated_patch_tokens + ray_embeddings

        weights = self.proj(aggregated_patch_tokens)  # Shape: [3, 14, 1258, 1]
        weights = F.softmax(weights, dim=2)

        # Weighted sum
        pooled_patch_tokens = (weights * aggregated_patch_tokens).sum(dim=2)

        if pose_encodings is not None:
            pose_encodings = pose_encodings.view(B, S, -1)  # [B,S,1024]
            pose_encodings = torch.concat([pose_encodings,pose_encodings],dim=-1)
            aggregated_cam_token = aggregated_cam_token + pose_encodings

        cls_token = cls_token.view(B,S,-1)   # [B,S,1024]

        pooled_patch_tokens_norm = self.token_norm2048(pooled_patch_tokens)
        cls_token_norm = self.token_norm1024(cls_token)
        aggregated_cam_token_norm = self.token_norm2048(aggregated_cam_token)

        total_tokens_post = torch.cat((pooled_patch_tokens_norm, aggregated_cam_token_norm, cls_token_norm), dim=-1)  # [B,S,5120]
        out = torch.exp(self.mlp(total_tokens_post)).squeeze(-1)

        encoded_feature = torch.mean(out,dim=-1)

        return encoded_feature



class ScaleHead_MLP_CP(nn.Module):
    """
    MLP_CP
    """
    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
    ):
        super().__init__()

        self.token_norm1024 = nn.LayerNorm(dim_in//2)
        self.token_norm2048 = nn.LayerNorm(dim_in)

        self.mlp = MLP([4096, 2048, 256, 64, 1])
        self.proj = nn.Linear(dim_in, 1)  # Will learn attention over tokens

    def forward(self, aggregated_tokens_list: list, cls_token: torch.tensor, patch_tokens: torch.tensor, pose_encodings: torch.tensor, ray_embeddings: torch.tensor, B: int, S: int) -> float:
        aggregated_cam_token = aggregated_tokens_list[-1][:,:,0,:]                             # [B,S,2048]
        # aggregated_register_token = aggregated_tokens_list[-1][:,:,1:5,:]                      # [B,S,4,2048]
        aggregated_patch_tokens = aggregated_tokens_list[-1][:,:,5:,:]                         # [B,S,N,2048]
        if ray_embeddings is not None:
            ray_embeddings = ray_embeddings.view(B, S, aggregated_patch_tokens.shape[2], -1)  # [B,S,1,1024]
            ray_embeddings = torch.concat([ray_embeddings,ray_embeddings],dim=-1)  # [B,S,1,2048]
            aggregated_patch_tokens = aggregated_patch_tokens + ray_embeddings

        weights = self.proj(aggregated_patch_tokens)  # Shape: [3, 14, 1258, 1]
        weights = F.softmax(weights, dim=2)

        # Weighted sum
        pooled_patch_tokens = (weights * aggregated_patch_tokens).sum(dim=2)

        cls_token = cls_token.view(B,S,-1)   # [B,S,1024]
  #      cls_token_norm = self.token_norm1024(cls_token)

        if pose_encodings is not None:
            pose_encodings = pose_encodings.view(B, S, -1)  # [B,S,1024]
            pose_encodings = torch.concat([pose_encodings,pose_encodings],dim=-1)
            aggregated_cam_token = aggregated_cam_token + pose_encodings

        pooled_patch_tokens_norm = self.token_norm2048(pooled_patch_tokens)
   #     cls_token_norm = self.token_norm1024(cls_token)
        aggregated_cam_token_norm = self.token_norm2048(aggregated_cam_token)

        total_tokens_post = torch.cat((pooled_patch_tokens_norm, aggregated_cam_token_norm), dim=-1)  # [B,S,5120]
        out = torch.exp(self.mlp(total_tokens_post)).squeeze(-1)

        encoded_feature = torch.mean(out,dim=-1)

        return encoded_feature

class ScaleHead_MLP_LCP_noprior(nn.Module):
    """
    MLP_LCP_noprior
    """
    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
    ):
        super().__init__()

        self.token_norm1024 = nn.LayerNorm(dim_in//2)
        self.token_norm2048 = nn.LayerNorm(dim_in)

        self.mlp = MLP([5120, 2048, 256, 64, 1])
        self.proj = nn.Linear(dim_in, 1)  # Will learn attention over tokens

    def forward(self, aggregated_tokens_list: list, cls_token: torch.tensor, patch_tokens: torch.tensor, pose_encodings: torch.tensor, ray_embeddings: torch.tensor, B: int, S: int) -> float:
        aggregated_cam_token = aggregated_tokens_list[-1][:,:,0,:]                             # [B,S,2048]
        # aggregated_register_token = aggregated_tokens_list[-1][:,:,1:5,:]                      # [B,S,4,2048]
        aggregated_patch_tokens = aggregated_tokens_list[-1][:,:,5:,:]                         # [B,S,N,2048]
        weights = self.proj(aggregated_patch_tokens)  # Shape: [3, 14, 1258, 1]
        weights = F.softmax(weights, dim=2)

        # Weighted sum
        pooled_patch_tokens = (weights * aggregated_patch_tokens).sum(dim=2)

        cls_token = cls_token.view(B,S,-1)   # [B,S,1024]

        pooled_patch_tokens_norm = self.token_norm2048(pooled_patch_tokens)
        cls_token_norm = self.token_norm1024(cls_token)
        aggregated_cam_token_norm = self.token_norm2048(aggregated_cam_token)

        total_tokens_post = torch.cat((pooled_patch_tokens_norm, aggregated_cam_token_norm, cls_token_norm), dim=-1)  # [B,S,5120]
        out = torch.exp(self.mlp(total_tokens_post)).squeeze(-1)

        encoded_feature = torch.mean(out,dim=-1)

        return encoded_feature

class ScaleHead_MLP_LC_noray(nn.Module):
    """
    MLP_LCP
    """
    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
    ):
        super().__init__()

        self.token_norm1024 = nn.LayerNorm(dim_in//2)
        self.token_norm2048 = nn.LayerNorm(dim_in)

        self.mlp = MLP([3072, 2048, 256, 64, 1])

    def forward(self, aggregated_tokens_list: list, cls_token: torch.tensor, patch_tokens: torch.tensor, pose_encodings: torch.tensor, ray_embeddings: torch.tensor, B: int, S: int) -> float:
        aggregated_cam_token = aggregated_tokens_list[-1][:,:,0,:]                             # [B,S,2048]
        # aggregated_register_token = aggregated_tokens_list[-1][:,:,1:5,:]                      # [B,S,4,2048]
        aggregated_patch_tokens = aggregated_tokens_list[-1][:,:,5:,:]                         # [B,S,N,2048]

        cls_token = cls_token.view(B,S,-1)   # [B,S,1024]

        if pose_encodings is not None:
            pose_encodings = pose_encodings.view(B, S, -1)  # [B,S,1024]
            pose_encodings = torch.concat([pose_encodings,pose_encodings],dim=-1)
            aggregated_cam_token = aggregated_cam_token + pose_encodings

        cls_token_norm = self.token_norm1024(cls_token)
        aggregated_cam_token_norm = self.token_norm2048(aggregated_cam_token)

        total_tokens_post = torch.cat((aggregated_cam_token_norm, cls_token_norm), dim=-1)  # [B,S,5120]
        out = torch.exp(self.mlp(total_tokens_post)).squeeze(-1)

        encoded_feature = torch.mean(out,dim=-1)

        return encoded_feature

class ScaleHead_MLP_LP(nn.Module):
    """
    MLP_LP
    """
    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
    ):
        super().__init__()

        self.token_norm1024 = nn.LayerNorm(dim_in//2)
        self.token_norm2048 = nn.LayerNorm(dim_in)

        self.mlp = MLP([3072, 2048, 256, 64, 1])
        self.proj = nn.Linear(dim_in, 1)  # Will learn attention over tokens

    def forward(self, aggregated_tokens_list: list, cls_token: torch.tensor, patch_tokens: torch.tensor, pose_encodings: torch.tensor, ray_embeddings: torch.tensor, B: int, S: int) -> float:
        aggregated_cam_token = aggregated_tokens_list[-1][:,:,0,:]                             # [B,S,2048]
        # aggregated_register_token = aggregated_tokens_list[-1][:,:,1:5,:]                      # [B,S,4,2048]
        aggregated_patch_tokens = aggregated_tokens_list[-1][:,:,5:,:]                         # [B,S,N,2048]
        if ray_embeddings is not None:
            ray_embeddings = ray_embeddings.view(B, S, aggregated_patch_tokens.shape[2], -1)  # [B,S,1,1024]
            ray_embeddings = torch.concat([ray_embeddings,ray_embeddings],dim=-1)  # [B,S,1,2048]
            aggregated_patch_tokens = aggregated_patch_tokens + ray_embeddings

        weights = self.proj(aggregated_patch_tokens)  # Shape: [3, 14, 1258, 1]
        weights = F.softmax(weights, dim=2)

        # Weighted sum
        pooled_patch_tokens = (weights * aggregated_patch_tokens).sum(dim=2)
        cls_token = cls_token.view(B,S,-1)   # [B,S,1024]

        if pose_encodings is not None:
            pose_encodings = pose_encodings.view(B, S, -1)  # [B,S,1024]
            cls_token = cls_token + pose_encodings

        pooled_patch_tokens_norm = self.token_norm2048(pooled_patch_tokens)
        cls_token_norm = self.token_norm1024(cls_token)
    #    aggregated_cam_token_norm = self.token_norm2048(aggregated_cam_token)

        total_tokens_post = torch.cat((pooled_patch_tokens_norm, cls_token_norm), dim=-1)  # [B,S,5120]
        out = torch.exp(self.mlp(total_tokens_post)).squeeze(-1)

        encoded_feature = torch.mean(out,dim=-1)

        return encoded_feature


class ScaleHead_MLP_LCP_MoE_SS(nn.Module):
    """
    MLP_LCP
    """
    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
    ):
        super().__init__()

        self.token_norm1024 = nn.LayerNorm(dim_in//2)
        self.token_norm2048 = nn.LayerNorm(dim_in)

        self.mlp_indoor = MLP([5120, 2048, 512, 64, 1])
        self.mlp_outdoor = MLP([5120, 2048, 512, 64, 1])
        self.proj = nn.Linear(dim_in, 1)  # Will learn attention over tokens
        self.gating = GatingNetwork(int(2.5*dim_in))

    def forward(self, aggregated_tokens_list: list, cls_token: torch.tensor, patch_tokens: torch.tensor, pose_encodings: torch.tensor, ray_embeddings: torch.tensor, gt_areas: torch.tensor, data_iter: int, B: int, S: int) -> float:
        aggregated_cam_token = aggregated_tokens_list[-1][:,:,0,:]                             # [B,S,2048]
        # aggregated_register_token = aggregated_tokens_list[-1][:,:,1:5,:]                      # [B,S,4,2048]
        aggregated_patch_tokens = aggregated_tokens_list[-1][:,:,5:,:]                         # [B,S,N,2048]
        if ray_embeddings is not None:
            ray_embeddings = ray_embeddings.view(B, S, aggregated_patch_tokens.shape[2], -1)  # [B,S,1,1024]
            ray_embeddings = torch.concat([ray_embeddings,ray_embeddings],dim=-1)  # [B,S,1,2048]
            aggregated_patch_tokens = aggregated_patch_tokens + ray_embeddings

        weights = self.proj(aggregated_patch_tokens)  # Shape: [3, 14, 1258, 1]
        weights = F.softmax(weights, dim=2)

        # Weighted sum
        pooled_patch_tokens = (weights * aggregated_patch_tokens).sum(dim=2)

        cls_token = cls_token.view(B,S,-1)   # [B,S,1024]
  #      cls_token_norm = self.token_norm1024(cls_token)

        if pose_encodings is not None:
            pose_encodings = pose_encodings.view(B, S, -1)  # [B,S,1024]
            cls_token = cls_token + pose_encodings

        pooled_patch_tokens_norm = self.token_norm2048(pooled_patch_tokens)
        cls_token_norm = self.token_norm1024(cls_token)
        aggregated_cam_token_norm = self.token_norm2048(aggregated_cam_token)

        total_tokens_post = torch.cat((pooled_patch_tokens_norm, aggregated_cam_token_norm, cls_token_norm), dim=-1)  # [B,S,5120]

        ## MoE 
        out_indoor = torch.exp(self.mlp_indoor(total_tokens_post)).squeeze(-1)
        out_outdoor = torch.exp(self.mlp_outdoor(total_tokens_post)).squeeze(-1)

        pred_indoor = torch.mean(out_indoor,dim=-1)
        pred_outdoor = torch.mean(out_outdoor,dim=-1)

        gating_logits = self.gating(total_tokens_post)
        gating_logits_cls = torch.mean(gating_logits,dim=1)
        probs = F.softmax(gating_logits_cls, dim=-1)   # [B, 2]

        if data_iter<10000:
            if random.random() < 1 - data_iter * (1/10000):
                probs = torch.zeros_like(probs)
                probs[:,0] = (gt_areas[:,0]==0).float()
                probs[:,1] = (gt_areas[:,0]==1).float()
            
        pred = (
            probs[:, 0] * pred_indoor +
            probs[:, 1] * pred_outdoor
        )

        return pred, gating_logits_cls
    

# class ScaleHead_MLP_LCP_MoE_SS(nn.Module):
#     """
#     MLP_LCP_MoE_SS
#     """
#     def __init__(
#         self,
#         dim_in: int = 2048,
#         trunk_depth: int = 4,
#         pose_encoding_type: str = "absT_quaR_FoV",
#         num_heads: int = 16,
#         mlp_ratio: int = 4,
#         init_values: float = 0.01,
#         trans_act: str = "linear",
#         quat_act: str = "linear",
#         fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
#     ):
#         super().__init__()

#         self.token_norm1024 = nn.LayerNorm(dim_in//2)
#         self.token_norm2048 = nn.LayerNorm(dim_in)
#      #   self.token_norm6144 = nn.LayerNorm(int(3*dim_in))

#         self.mlp_indoor = MLP([5120, 2048, 512, 64, 1])
#         self.mlp_outdoor = MLP([5120, 2048, 512, 64, 1])
#         self.proj = nn.Linear(dim_in, 1)  # Will learn attention over tokens
#      #   self.proj_patch = nn.Linear(dim_in//2, 1)  # Will learn attention over tokens
#         self.gating = GatingNetwork(int(2.5*dim_in))

#     def forward(self, aggregated_tokens_list: list, cls_token: torch.tensor, patch_tokens: torch.tensor,  pose_encodings: torch.tensor, ray_embeddings: torch.tensor, B: int, S: int) -> float:
#         aggregated_cam_token = aggregated_tokens_list[-1][:,:,0,:]                             # [B,S,2048]
#         # aggregated_register_token = aggregated_tokens_list[-1][:,:,1:5,:]                      # [B,S,4,2048]
#         aggregated_patch_tokens = aggregated_tokens_list[-1][:,:,5:,:]                         # [B,S,N,2048]

#         weights = self.proj(aggregated_patch_tokens)  # Shape: [3, 14, 1258, 1]
#         weights = F.softmax(weights, dim=2)
#         pooled_patch_tokens = (weights * aggregated_patch_tokens).sum(dim=2)

#      #   patch_tokens = patch_tokens.view(B, S, aggregated_patch_tokens.shape[2], -1)  # [B,S,1024]
#      #   weights_patch = self.proj_patch(patch_tokens)
#      #   weights_patch = F.softmax(weights_patch, dim=2)
#      #   pooled_patches = (weights_patch * patch_tokens).sum(dim=2)

#         cls_token = cls_token.view(B,S,-1)   # [B,S,1024]

#         pooled_patch_tokens_norm = self.token_norm2048(pooled_patch_tokens)
#         cls_token_norm = self.token_norm1024(cls_token)
#         aggregated_cam_token_norm = self.token_norm2048(aggregated_cam_token)
#     #    pooled_patches_norm = self.token_norm1024(pooled_patches)

#         total_tokens_post = torch.cat((pooled_patch_tokens_norm, aggregated_cam_token_norm, cls_token_norm), dim=-1)  # [B,S,5120]
#       #  total_tokens_post = self.token_norm6144(total_tokens_post) ## normalize this line
#       #  print("normalize this line")

#         ## MoE 
#         out_indoor = torch.exp(self.mlp_indoor(total_tokens_post)).squeeze(-1)
#         out_outdoor = torch.exp(self.mlp_outdoor(total_tokens_post)).squeeze(-1)

#         pred_indoor = torch.mean(out_indoor,dim=-1)
#         pred_outdoor = torch.mean(out_outdoor,dim=-1)

#         gating_logits = self.gating(total_tokens_post)
#         gating_logits_cls = torch.mean(gating_logits,dim=1)
#         probs = F.softmax(gating_logits_cls, dim=-1)   # [B, 2]

#         pred = (
#             probs[:, 0] * pred_indoor +
#             probs[:, 1] * pred_outdoor
#         )

#         return pred, gating_logits_cls


class ScaleHead_MLP_LCPT_MoE_SS(nn.Module):
    """
    MLP_LCPT_MoE_SS
    """
    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
    ):
        super().__init__()

        self.token_norm1024 = nn.LayerNorm(dim_in//2)
        self.token_norm2048 = nn.LayerNorm(dim_in)
     #   self.token_norm6144 = nn.LayerNorm(int(3*dim_in))

        self.mlp_indoor = MLP([6144, 2048, 512, 64, 1])
        self.mlp_outdoor = MLP([6144, 2048, 512, 64, 1])
        self.proj = nn.Linear(dim_in, 1)  # Will learn attention over tokens
        self.proj_patch = nn.Linear(dim_in//2, 1)  # Will learn attention over tokens
        self.gating = GatingNetwork(int(3*dim_in))

    def forward(self, aggregated_tokens_list: list, cls_token: torch.tensor, patch_tokens: torch.tensor, pose_encodings: torch.tensor, ray_embeddings: torch.tensor,  B: int, S: int) -> float:
        aggregated_cam_token = aggregated_tokens_list[-1][:,:,0,:]                             # [B,S,2048]
        # aggregated_register_token = aggregated_tokens_list[-1][:,:,1:5,:]                      # [B,S,4,2048]
        aggregated_patch_tokens = aggregated_tokens_list[-1][:,:,5:,:]                         # [B,S,N,2048]
        
        weights = self.proj(aggregated_patch_tokens)  # Shape: [3, 14, 1258, 1]
        weights = F.softmax(weights, dim=2)
        pooled_patch_tokens = (weights * aggregated_patch_tokens).sum(dim=2)
        
        patch_tokens = patch_tokens.view(B, S, aggregated_patch_tokens.shape[2], -1)  # [B,S,1024]
        weights_patch = self.proj_patch(patch_tokens)
        weights_patch = F.softmax(weights_patch, dim=2)
        pooled_patches = (weights_patch * patch_tokens).sum(dim=2)

        cls_token = cls_token.view(B,S,-1)   # [B,S,1024]
        
        pooled_patch_tokens_norm = self.token_norm2048(pooled_patch_tokens)
        cls_token_norm = self.token_norm1024(cls_token)
        aggregated_cam_token_norm = self.token_norm2048(aggregated_cam_token)
        pooled_patches_norm = self.token_norm1024(pooled_patches)

        total_tokens_post = torch.cat((pooled_patch_tokens_norm, aggregated_cam_token_norm, cls_token_norm, pooled_patches_norm), dim=-1)  # [B,S,5120]
      #  total_tokens_post = self.token_norm6144(total_tokens_post) ## normalize this line
      #  print("normalize this line")
      
        ## MoE 
        out_indoor = torch.exp(self.mlp_indoor(total_tokens_post)).squeeze(-1)
        out_outdoor = torch.exp(self.mlp_outdoor(total_tokens_post)).squeeze(-1)

        pred_indoor = torch.mean(out_indoor,dim=-1)
        pred_outdoor = torch.mean(out_outdoor,dim=-1)
        
        gating_logits = self.gating(total_tokens_post)
        gating_logits_cls = torch.mean(gating_logits,dim=1)
        probs = F.softmax(gating_logits_cls, dim=-1)   # [B, 2]

        pred = (
            probs[:, 0] * pred_indoor +
            probs[:, 1] * pred_outdoor
        )

        return pred, gating_logits_cls


class ScaleHead_selfAttn_cat_LCPT_MoE_SS(nn.Module):
    """
    selfAttn_cat_LCPT_MoE_SS
    """
    def __init__(
        self,
        max_seq_len: int = 24,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.scale_attention6144 = nn.MultiheadAttention(embed_dim=3*dim_in, num_heads=num_heads, batch_first=True)

        self.token_norm1024 = nn.LayerNorm(dim_in//2)
        self.token_norm2048 = nn.LayerNorm(dim_in)

        self.mlp_indoor = MLP([6144, 2048, 512, 64, 1])
        self.mlp_outdoor = MLP([6144, 2048, 512, 64, 1])
        self.proj = nn.Linear(dim_in, 1)  # Will learn attention over tokens
        self.proj_patch = nn.Linear(dim_in//2, 1)  # Will learn attention over tokens
        self.gating = GatingNetwork(int(3*dim_in))

    def forward(self, aggregated_tokens_list, cls_token, patch_tokens, B, S):
        """
        Forward pass to estimate the scale factor.
        """
        aggregated_cam_token = aggregated_tokens_list[-1][:,:,0,:]                             # [B,S,2048]
        # aggregated_register_token = aggregated_tokens_list[-1][:,:,1:5,:]                      # [B,S,4,2048]
        aggregated_patch_tokens = aggregated_tokens_list[-1][:,:,5:,:]   

        weights = self.proj(aggregated_patch_tokens)  # Shape: [3, 14, 1258, 1]
        weights = F.softmax(weights, dim=2)
        pooled_patch_tokens = (weights * aggregated_patch_tokens).sum(dim=2)

        patch_tokens = patch_tokens.view(B, S, aggregated_patch_tokens.shape[2], -1)  # [B,S,1024]
        weights_patch = self.proj_patch(patch_tokens)
        weights_patch = F.softmax(weights_patch, dim=2)
        pooled_patches = (weights_patch * patch_tokens).sum(dim=2)

        cls_token = cls_token.view(B, S, -1)  # [B,S,1024]

        pooled_patch_tokens_norm = self.token_norm2048(pooled_patch_tokens)
        aggregated_cam_token_norm = self.token_norm2048(aggregated_cam_token)
        cls_token_norm = self.token_norm1024(cls_token)
        pooled_patches_norm = self.token_norm1024(pooled_patches)

        total_tokens_post = torch.cat((pooled_patch_tokens_norm, aggregated_cam_token_norm, cls_token_norm, pooled_patches_norm), dim=-1)  # [B,S,5120]

        encoded_feat, attn_feat = self.scale_attention6144(total_tokens_post, total_tokens_post, total_tokens_post)
        total_tokens_post = total_tokens_post + encoded_feat

        ## MoE
        out_indoor = torch.exp(self.mlp_indoor(total_tokens_post)).squeeze(-1)
        out_outdoor = torch.exp(self.mlp_outdoor(total_tokens_post)).squeeze(-1)

        pred_indoor = torch.mean(out_indoor,dim=-1)
        pred_outdoor = torch.mean(out_outdoor,dim=-1)
        
        gating_logits = self.gating(total_tokens_post)
        gating_logits_cls = torch.mean(gating_logits,dim=1)
        probs = F.softmax(gating_logits_cls, dim=-1)   # [B, 2]

        pred = (
            probs[:, 0] * pred_indoor +
            probs[:, 1] * pred_outdoor
        )

        return pred, gating_logits_cls  
    

class ScaleHead_Tr_crossAttn_LCPT_MLP_camera_aware_naive_MoE_SS(nn.Module):
    """
    crossAttn_LCPT_MLP_camera_aware_naive_MoE_SS
    """
    def __init__(
        self,
        max_seq_len: int = 24,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.scale_attention2048_cam = nn.MultiheadAttention(embed_dim=dim_in, num_heads=num_heads, batch_first=True)
        self.scale_attention1024_cls = nn.MultiheadAttention(embed_dim=dim_in//2, num_heads=num_heads, batch_first=True)

        self.token_norm2048 = nn.LayerNorm(dim_in)
        self.token_norm1024 = nn.LayerNorm(dim_in//2)

        self.to_gamma_beta = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(),
            nn.Linear(dim_in, dim_in)   # outputs [gamma, beta]
        )

        self.mlp_indoor = MLP([6144, 2048, 512, 64, 1])
        self.mlp_outdoor = MLP([6144, 2048, 512, 64, 1])
        self.proj = nn.Linear(dim_in, 1)  # Will learn attention over tokens
        self.proj_patch = nn.Linear(dim_in//2, 1)  # Will learn attention over tokens
        self.gating = GatingNetwork(int(3*dim_in))

    def forward(self, aggregated_tokens_list, cls_token, patch_tokens, B, S):
        """
        Forward pass to estimate the scale factor.
        """
        aggregated_cam_token = aggregated_tokens_list[-1][:,:,0,:]                             # [B,S,2048]
        # aggregated_register_token = aggregated_tokens_list[-1][:,:,1:5,:]                      # [B,S,4,2048]
        aggregated_patch_tokens = aggregated_tokens_list[-1][:,:,5:,:]   
        weights = self.proj(aggregated_patch_tokens)  # Shape: [3, 14, 1258, 1]
        weights = F.softmax(weights, dim=2)

        # Weighted sum
        pooled_patch_tokens = (weights * aggregated_patch_tokens).sum(dim=2)
        ## 
        patch_tokens = patch_tokens.view(B, S, aggregated_patch_tokens.shape[2], -1)  # [B,S,1024]
        weights_patch = self.proj_patch(patch_tokens)
        weights_patch = F.softmax(weights_patch, dim=2)
        pooled_patches = (weights_patch * patch_tokens).sum(dim=2)

        cls_token = cls_token.view(B, S, -1)  # [B,S,1024]

        cls_token_norm = self.token_norm1024(cls_token)
        pooled_patch_tokens_norm = self.token_norm2048(pooled_patch_tokens)
        aggregated_cam_token_norm = self.token_norm2048(aggregated_cam_token)
        pooled_patches_norm = self.token_norm1024(pooled_patches)


        ## FiLM conditioning on camera token
        gamma_beta = self.to_gamma_beta(aggregated_cam_token_norm)
        gamma_beta = torch.mean(gamma_beta, dim=1)   # [B, 2*D]
        gamma, beta = gamma_beta.chunk(2, dim=-1)    # [B, D], [B, D]

        # reshape to broadcast over tokens
        gamma = gamma.unsqueeze(1)   # [B, 1, D]
        beta  = beta.unsqueeze(1)    # [B, 1, D]

        # FiLM modulation: scale + shift
        ##
        cls_token_norm = gamma * cls_token_norm + beta
        total_tokens_post = torch.cat((aggregated_cam_token_norm, pooled_patch_tokens_norm, cls_token_norm, pooled_patches_norm), dim=-1)  # [B,S,6144]

        ## MoE
        out_indoor = torch.exp(self.mlp_indoor(total_tokens_post)).squeeze(-1)
        out_outdoor = torch.exp(self.mlp_outdoor(total_tokens_post)).squeeze(-1)

        pred_indoor = torch.mean(out_indoor,dim=-1)
        pred_outdoor = torch.mean(out_outdoor,dim=-1)
        
        gating_logits = self.gating(total_tokens_post)
        gating_logits_cls = torch.mean(gating_logits,dim=1)
        probs = F.softmax(gating_logits_cls, dim=-1)   # [B, 2]

        pred = (
            probs[:, 0] * pred_indoor +
            probs[:, 1] * pred_outdoor
        )

        return pred, gating_logits_cls  
    
class ScaleHead_MLP_LCPT_MoE_SS_crossFiLM(nn.Module):
    """
    MLP_LCPT_MoE_SS_crossFiLM
    """
    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
    ):
        super().__init__()

        self.token_norm1024 = nn.LayerNorm(dim_in//2)
        self.token_norm2048 = nn.LayerNorm(dim_in)

        self.scale_attention2048_cam = nn.MultiheadAttention(embed_dim=dim_in, num_heads=num_heads, batch_first=True)

        self.mlp_indoor = MLP([7168, 2048, 512, 64, 1])
        self.mlp_outdoor = MLP([7168, 2048, 512, 64, 1])
        self.proj = nn.Linear(dim_in, 1)  # Will learn attention over tokens
        self.proj_patch = nn.Linear(dim_in//2, 1)  # Will learn attention over tokens
        self.gating = GatingNetwork(int(3.5*dim_in))

    def forward(self, aggregated_tokens_list: list, cls_token: torch.tensor, patch_tokens: torch.tensor, B: int, S: int) -> float:
        aggregated_cam_token = aggregated_tokens_list[-1][:,:,0,:]                             # [B,S,2048]
        # aggregated_register_token = aggregated_tokens_list[-1][:,:,1:5,:]                      # [B,S,4,2048]
        aggregated_patch_tokens = aggregated_tokens_list[-1][:,:,5:,:]                         # [B,S,N,2048]
        
        weights = self.proj(aggregated_patch_tokens)  # Shape: [3, 14, 1258, 1]
        weights = F.softmax(weights, dim=2)
        pooled_patch_tokens = (weights * aggregated_patch_tokens).sum(dim=2)
        
        patch_tokens = patch_tokens.view(B, S, aggregated_patch_tokens.shape[2], -1)  # [B,S,1024]
        weights_patch = self.proj_patch(patch_tokens)
        weights_patch = F.softmax(weights_patch, dim=2)
        pooled_patches = (weights_patch * patch_tokens).sum(dim=2)

        cls_token = cls_token.view(B,S,-1)   # [B,S,1024]
        cls_token = torch.cat((cls_token,cls_token),dim=-1)  # [B,S,2048]

        pooled_patch_tokens_norm = self.token_norm2048(pooled_patch_tokens)
        cls_token_norm = self.token_norm2048(cls_token)
        aggregated_cam_token_norm = self.token_norm2048(aggregated_cam_token)
        pooled_patches_norm = self.token_norm1024(pooled_patches)

        encoded_cls_token, attn_map = self.scale_attention2048_cam(cls_token_norm, aggregated_cam_token_norm, aggregated_cam_token_norm)
        cls_token_norm = cls_token_norm + encoded_cls_token

        total_tokens_post = torch.cat((pooled_patch_tokens_norm, aggregated_cam_token_norm, cls_token_norm, pooled_patches_norm), dim=-1)  # [B,S,5120]
      
        ## MoE 
        out_indoor = torch.exp(self.mlp_indoor(total_tokens_post)).squeeze(-1)
        out_outdoor = torch.exp(self.mlp_outdoor(total_tokens_post)).squeeze(-1)

        pred_indoor = torch.mean(out_indoor,dim=-1)
        pred_outdoor = torch.mean(out_outdoor,dim=-1)
        
        gating_logits = self.gating(total_tokens_post)
        gating_logits_cls = torch.mean(gating_logits,dim=1)
        probs = F.softmax(gating_logits_cls, dim=-1)   # [B, 2]

        pred = (
            probs[:, 0] * pred_indoor +
            probs[:, 1] * pred_outdoor
        )

        return pred, gating_logits_cls


class ScaleHead_selfAttn_cat_LCPT(nn.Module):
    """
    selfAttn_cat_LCPT
    """
    def __init__(
        self,
        max_seq_len: int = 24,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.scale_attention6144 = nn.MultiheadAttention(embed_dim=3*dim_in, num_heads=num_heads, batch_first=True)

        self.token_norm1024 = nn.LayerNorm(dim_in//2)
        self.token_norm2048 = nn.LayerNorm(dim_in)

        self.mlp = MLP([6144, 2048, 256, 64, 1])
        self.proj = nn.Linear(dim_in, 1)  # Will learn attention over tokens
        self.proj_patch = nn.Linear(dim_in//2, 1)  # Will learn attention over tokens

    def forward(self, aggregated_tokens_list, cls_token, patch_tokens, B, S):
        """
        Forward pass to estimate the scale factor.
        """
        aggregated_cam_token = aggregated_tokens_list[-1][:,:,0,:]                             # [B,S,2048]
        # aggregated_register_token = aggregated_tokens_list[-1][:,:,1:5,:]                      # [B,S,4,2048]
        aggregated_patch_tokens = aggregated_tokens_list[-1][:,:,5:,:]   

        weights = self.proj(aggregated_patch_tokens)  # Shape: [3, 14, 1258, 1]
        weights = F.softmax(weights, dim=2)
        pooled_patch_tokens = (weights * aggregated_patch_tokens).sum(dim=2)

        patch_tokens = patch_tokens.view(B, S, aggregated_patch_tokens.shape[2], -1)  # [B,S,1024]
        weights_patch = self.proj_patch(patch_tokens)
        weights_patch = F.softmax(weights_patch, dim=2)
        pooled_patches = (weights_patch * patch_tokens).sum(dim=2)

        cls_token = cls_token.view(B, S, -1)  # [B,S,1024]

        pooled_patch_tokens_norm = self.token_norm2048(pooled_patch_tokens)
        aggregated_cam_token_norm = self.token_norm2048(aggregated_cam_token)
        cls_token_norm = self.token_norm1024(cls_token)
        pooled_patches_norm = self.token_norm1024(pooled_patches)

        total_tokens_post = torch.cat((pooled_patch_tokens_norm, aggregated_cam_token_norm, cls_token_norm, pooled_patches_norm), dim=-1)  # [B,S,5120]

        encoded_feat, attn_feat = self.scale_attention6144(total_tokens_post, total_tokens_post, total_tokens_post)
        total_tokens_post = total_tokens_post + encoded_feat

        total_tokens_post = torch.cat((pooled_patch_tokens_norm, aggregated_cam_token_norm, cls_token_norm, pooled_patches_norm), dim=-1)  # [B,S,6144]
        out = torch.exp(self.mlp(total_tokens_post)).squeeze(-1)
        encoded_feature = torch.mean(out,dim=-1)

        return encoded_feature
    
class ScaleHead_MLP_LCPT(nn.Module):
    """
    MLP_LCPT
    """
    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
    ):
        super().__init__()

        self.token_norm1024 = nn.LayerNorm(dim_in//2)
        self.token_norm2048 = nn.LayerNorm(dim_in)

        self.mlp = MLP([6144, 2048, 256, 64, 1])
        self.proj = nn.Linear(dim_in, 1)
        self.proj_patch = nn.Linear(dim_in//2, 1)

    def forward(self, aggregated_tokens_list: list, cls_token: torch.tensor, patch_tokens: torch.tensor, pose_encodings: torch.tensor, ray_embeddings: torch.tensor,  B: int, S: int) -> float:
        aggregated_cam_token = aggregated_tokens_list[-1][:,:,0,:]                             # [B,S,2048]
        # aggregated_register_token = aggregated_tokens_list[-1][:,:,1:5,:]                      # [B,S,4,2048]
        aggregated_patch_tokens = aggregated_tokens_list[-1][:,:,5:,:]                         # [B,S,N,2048]
        weights = self.proj(aggregated_patch_tokens)  # Shape: [3, 14, 1258, 1]
        weights = F.softmax(weights, dim=2)

        # Weighted sum
        pooled_patch_tokens = (weights * aggregated_patch_tokens).sum(dim=2)

        patch_tokens = patch_tokens.view(B, S, aggregated_patch_tokens.shape[2], -1)  # [B,S,1024]
        weights_patch = self.proj_patch(patch_tokens)
        weights_patch = F.softmax(weights_patch, dim=2)
        pooled_patches = (weights_patch * patch_tokens).sum(dim=2)

        cls_token = cls_token.view(B,S,-1)                                                     # [B,S,1024]

        pooled_patch_tokens_norm = self.token_norm2048(pooled_patch_tokens)
        cls_token_norm = self.token_norm1024(cls_token)
        aggregated_cam_token_norm = self.token_norm2048(aggregated_cam_token)
        pooled_patches_norm = self.token_norm1024(pooled_patches)

        total_tokens_post = torch.cat((pooled_patch_tokens_norm, aggregated_cam_token_norm, cls_token_norm, pooled_patches_norm), dim=-1)  # [B,S,6144]
        out = torch.exp(self.mlp(total_tokens_post)).squeeze(-1)
        encoded_feature = torch.mean(out,dim=-1)

        return encoded_feature
    


class ScaleHead_selfAttn_sep_LCP_MoE_SS(nn.Module):
    """
    selfAttn_sep_LCP_MoE_SS
    """
    def __init__(
        self,
        max_seq_len: int = 24,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.scale_attention2048 = nn.MultiheadAttention(embed_dim=dim_in, num_heads=num_heads, batch_first=True)
        self.scale_attention1024 = nn.MultiheadAttention(embed_dim=int(dim_in//2), num_heads=num_heads, batch_first=True)

        self.token_norm1024 = nn.LayerNorm(dim_in//2)
        self.token_norm2048 = nn.LayerNorm(dim_in)
        self.token_norm5120 = nn.LayerNorm(int(2.5*dim_in))

        self.mlp_indoor = MLP([5120, 1024, 256, 64, 1])
        self.mlp_outdoor = MLP([5120, 1024, 256, 64, 1])

        self.proj = nn.Linear(dim_in, 1)  # Will learn attention over tokens
        self.gating = GatingNetwork(int(2.5*dim_in))

    def forward(self, aggregated_tokens_list, cls_token, patch_tokens, B, S, phase):
        """
        Forward pass to estimate the scale factor.
        """
        aggregated_cam_token = aggregated_tokens_list[-1][:,:,0,:]                             # [B,S,2048]
        # aggregated_register_token = aggregated_tokens_list[-1][:,:,1:5,:]                      # [B,S,4,2048]
        aggregated_patch_tokens = aggregated_tokens_list[-1][:,:,5:,:]   
        weights = self.proj(aggregated_patch_tokens)  # Shape: [3, 14, 1258, 1]
        weights = F.softmax(weights, dim=2)

        # Weighted sum
        pooled_patch_tokens = (weights * aggregated_patch_tokens).sum(dim=2)
        cls_token = cls_token.view(B, S, -1)  # [B,S,1024]

        pooled_patch_tokens_norm = self.token_norm2048(pooled_patch_tokens)
        encoded_feat_patch, attn_feat_path = self.scale_attention2048(pooled_patch_tokens_norm,pooled_patch_tokens_norm,pooled_patch_tokens_norm)
        pooled_patch_tokens_norm = pooled_patch_tokens_norm + encoded_feat_patch

        aggregated_cam_token_norm = self.token_norm2048(aggregated_cam_token)
        encoded_feat_cam, attn_feat_cam = self.scale_attention2048(aggregated_cam_token_norm,aggregated_cam_token_norm,aggregated_cam_token_norm)
        aggregated_cam_token_norm = aggregated_cam_token_norm + encoded_feat_cam

        cls_token_norm = self.token_norm1024(cls_token)
        encoded_feat_cls, attn_feat_cls = self.scale_attention1024(cls_token_norm,cls_token_norm,cls_token_norm)
        cls_token_norm = cls_token_norm + encoded_feat_cls

        total_tokens_post = torch.cat((pooled_patch_tokens_norm, aggregated_cam_token_norm, cls_token_norm), dim=-1)  # [B,S,5120]

        ## MoE
        out_indoor = torch.exp(self.mlp_indoor(total_tokens_post)).squeeze(-1)
        out_outdoor = torch.exp(self.mlp_outdoor(total_tokens_post)).squeeze(-1)

        pred_indoor = torch.mean(out_indoor,dim=-1)
        pred_outdoor = torch.mean(out_outdoor,dim=-1)
        
        gating_logits = self.gating(total_tokens_post)
        gating_logits_cls = torch.mean(gating_logits,dim=1)
        probs = F.softmax(gating_logits_cls, dim=-1)   # [B, 2]

        pred = (
            probs[:, 0] * pred_indoor +
            probs[:, 1] * pred_outdoor
        )

        return pred, gating_logits_cls  



class ScaleHead_selfattn_crossAttn_MLP_LCP_MoE_SS(nn.Module):
    """
    selfattn_crossAttn_MLP_LCP_MoE_SS
    """
    def __init__(
        self,
        max_seq_len: int = 24,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.scale_self_attention2048_cls = nn.MultiheadAttention(embed_dim=dim_in, num_heads=num_heads, batch_first=True)
        self.scale_self_attention2048_ppt = nn.MultiheadAttention(embed_dim=dim_in, num_heads=num_heads, batch_first=True)
        self.scale_self_attention2048_cam = nn.MultiheadAttention(embed_dim=dim_in, num_heads=num_heads, batch_first=True)

        self.scale_cross_attention2048_cls_cam = nn.MultiheadAttention(embed_dim=dim_in, num_heads=num_heads, batch_first=True)
        self.scale_cross_attention2048_cls_ppt = nn.MultiheadAttention(embed_dim=dim_in, num_heads=num_heads, batch_first=True)

        self.token_norm2048 = nn.LayerNorm(dim_in)
        self.token_norm6144 = nn.LayerNorm(int(3*dim_in))

        self.mlp_indoor = MLP([int(3*2048), 2048, 512, 64, 1])
        self.mlp_outdoor = MLP([int(3*2048), 2048, 512, 64, 1])

        self.proj = nn.Linear(dim_in, 1)  # Will learn attention over tokens
        self.proj_patch = nn.Linear(dim_in//2, 1)  # Will learn attention over tokens

        self.gating = GatingNetwork(int(3*dim_in))

    def forward(self, aggregated_tokens_list, cls_token, patch_tokens, B, S, phase):
        """
        Forward pass to estimate the scale factor.
        """
        aggregated_cam_token = aggregated_tokens_list[-1][:,:,0,:]                             # [B,S,2048]
        # aggregated_register_token = aggregated_tokens_list[-1][:,:,1:5,:]                      # [B,S,4,2048]
        aggregated_patch_tokens = aggregated_tokens_list[-1][:,:,5:,:]   
        weights = self.proj(aggregated_patch_tokens)  # Shape: [3, 14, 1258, 1]
        weights = F.softmax(weights, dim=2)

        # Weighted sum
        pooled_patch_tokens = (weights * aggregated_patch_tokens).sum(dim=2)

        patch_tokens = patch_tokens.view(B, S, aggregated_patch_tokens.shape[2], -1)  # [B,S,1024]
        weights_patch = self.proj_patch(patch_tokens)
        weights_patch = F.softmax(weights_patch, dim=2)
        pooled_patches = (weights_patch * patch_tokens).sum(dim=2)
        ##
        cls_token = cls_token.view(B, S, -1)  # [B,S,1024]
        cls_token = torch.cat((cls_token,cls_token),dim=-1)  # [B,S,2048]
        ## normalization
        cls_token_norm = self.token_norm2048(cls_token)
        pooled_patch_tokens_norm = self.token_norm2048(pooled_patch_tokens)
        aggregated_cam_token_norm = self.token_norm2048(aggregated_cam_token)

        ## self attention
        encoded_feat_self_cls, attn_feat_self_cls = self.scale_self_attention2048_cls(cls_token_norm, cls_token_norm, cls_token_norm)
        cls_token_norm = cls_token_norm + encoded_feat_self_cls

        encoded_feat_self_ppt, attn_feat_self_ppt = self.scale_self_attention2048_ppt(pooled_patch_tokens_norm, pooled_patch_tokens_norm, pooled_patch_tokens_norm)
        pooled_patch_tokens_norm = pooled_patch_tokens_norm + encoded_feat_self_ppt

        encoded_feat_self_cam, attn_feat_self_cam = self.scale_self_attention2048_cam(aggregated_cam_token_norm, aggregated_cam_token_norm, aggregated_cam_token_norm)
        aggregated_cam_token_norm = aggregated_cam_token_norm + encoded_feat_self_cam
        ## normalization
        cls_token_norm = self.token_norm2048(cls_token_norm)
        pooled_patch_tokens_norm = self.token_norm2048(pooled_patch_tokens_norm)
        aggregated_cam_token_norm = self.token_norm2048(aggregated_cam_token_norm)
        ## cross attention
        encoded_feat_cls_cam, attn_feat_cls_cam = self.scale_cross_attention2048_cls_cam(cls_token_norm, aggregated_cam_token_norm, aggregated_cam_token_norm)
        cls_token_norm = cls_token_norm + encoded_feat_cls_cam

        encoded_feat_cls_ppt, attn_feat_cls_ppt = self.scale_cross_attention2048_cls_ppt(cls_token_norm, pooled_patch_tokens_norm, pooled_patch_tokens_norm)
        cls_token_norm = cls_token_norm + encoded_feat_cls_ppt

        ## normalization
        total_tokens_post = torch.cat((cls_token_norm, aggregated_cam_token_norm, pooled_patch_tokens_norm), dim=-1)  # [B,S,6144]
        total_tokens_post = self.token_norm6144(total_tokens_post)

        ## MoE
        out_indoor = torch.exp(self.mlp_indoor(total_tokens_post)).squeeze(-1)
        out_outdoor = torch.exp(self.mlp_outdoor(total_tokens_post)).squeeze(-1)

        pred_indoor = torch.mean(out_indoor,dim=-1)
        pred_outdoor = torch.mean(out_outdoor,dim=-1)
        
        gating_logits = self.gating(total_tokens_post)
        gating_logits_cls = torch.mean(gating_logits,dim=1)
        probs = F.softmax(gating_logits_cls, dim=-1)   # [B, 2]

        pred = (
            probs[:, 0] * pred_indoor +
            probs[:, 1] * pred_outdoor
        )

        return pred, gating_logits_cls
        

class ScaleHead_Tr_selfattn_crossAttn_MLP_LCP(nn.Module):
    """
    selfattn_crossAttn_MLP_LCP
    """
    def __init__(
        self,
        max_seq_len: int = 24,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.scale_self_attention2048_cls = nn.MultiheadAttention(embed_dim=dim_in, num_heads=num_heads, batch_first=True)
        self.scale_self_attention2048_ppt = nn.MultiheadAttention(embed_dim=dim_in, num_heads=num_heads, batch_first=True)
        self.scale_self_attention2048_cam = nn.MultiheadAttention(embed_dim=dim_in, num_heads=num_heads, batch_first=True)

        self.scale_cross_attention2048_cls_cam = nn.MultiheadAttention(embed_dim=dim_in, num_heads=num_heads, batch_first=True)
        self.scale_cross_attention2048_cls_ppt = nn.MultiheadAttention(embed_dim=dim_in, num_heads=num_heads, batch_first=True)

        self.token_norm1024 = nn.LayerNorm(dim_in//2)
        self.token_norm2048 = nn.LayerNorm(dim_in)
        self.token_norm6144 = nn.LayerNorm(int(3*dim_in))

        self.mlp = MLP([int(3*2048), 2048, 512, 64, 1])
        self.proj = nn.Linear(dim_in, 1)  # Will learn attention over tokens
        self.proj_patch = nn.Linear(dim_in//2, 1)  # Will learn attention over tokens

    def forward(self, aggregated_tokens_list, cls_token, patch_tokens, B, S):
        """
        Forward pass to estimate the scale factor.
        """
        aggregated_cam_token = aggregated_tokens_list[-1][:,:,0,:]                             # [B,S,2048]
        # aggregated_register_token = aggregated_tokens_list[-1][:,:,1:5,:]                      # [B,S,4,2048]
        aggregated_patch_tokens = aggregated_tokens_list[-1][:,:,5:,:]   
        weights = self.proj(aggregated_patch_tokens)  # Shape: [3, 14, 1258, 1]
        weights = F.softmax(weights, dim=2)

        # Weighted sum
        pooled_patch_tokens = (weights * aggregated_patch_tokens).sum(dim=2)

        patch_tokens = patch_tokens.view(B, S, aggregated_patch_tokens.shape[2], -1)  # [B,S,1024]
        weights_patch = self.proj_patch(patch_tokens)
        weights_patch = F.softmax(weights_patch, dim=2)
        pooled_patches = (weights_patch * patch_tokens).sum(dim=2)
        ##
        cls_token = cls_token.view(B, S, -1)  # [B,S,1024]
        cls_token = torch.cat((cls_token,cls_token),dim=-1)  # [B,S,2048]
        ## normalization
        cls_token_norm = self.token_norm2048(cls_token)
        pooled_patch_tokens_norm = self.token_norm2048(pooled_patch_tokens)
        aggregated_cam_token_norm = self.token_norm2048(aggregated_cam_token)

        ## self attention
        encoded_feat_self_cls, attn_feat_self_cls = self.scale_self_attention2048_cls(cls_token_norm, cls_token_norm, cls_token_norm)
        cls_token_norm = cls_token_norm + encoded_feat_self_cls

        encoded_feat_self_ppt, attn_feat_self_ppt = self.scale_self_attention2048_ppt(pooled_patch_tokens_norm, pooled_patch_tokens_norm, pooled_patch_tokens_norm)
        pooled_patch_tokens_norm = pooled_patch_tokens_norm + encoded_feat_self_ppt

        encoded_feat_self_cam, attn_feat_self_cam = self.scale_self_attention2048_cam(aggregated_cam_token_norm, aggregated_cam_token_norm, aggregated_cam_token_norm)
        aggregated_cam_token_norm = aggregated_cam_token_norm + encoded_feat_self_cam
        ## normalization
        cls_token_norm = self.token_norm2048(cls_token_norm)
        pooled_patch_tokens_norm = self.token_norm2048(pooled_patch_tokens_norm)
        aggregated_cam_token_norm = self.token_norm2048(aggregated_cam_token_norm)
        ## cross attention
        encoded_feat_cls_cam, attn_feat_cls_cam = self.scale_cross_attention2048_cls_cam(cls_token_norm, aggregated_cam_token_norm, aggregated_cam_token_norm)
        cls_token_norm = cls_token_norm + encoded_feat_cls_cam

        encoded_feat_cls_ppt, attn_feat_cls_ppt = self.scale_cross_attention2048_cls_ppt(cls_token_norm, pooled_patch_tokens_norm, pooled_patch_tokens_norm)
        cls_token_norm = cls_token_norm + encoded_feat_cls_ppt

        ## Final Step
        total_tokens_post = torch.cat((cls_token_norm, aggregated_cam_token_norm, pooled_patch_tokens_norm), dim=-1)  # [B,S,6144]
        total_tokens_post = self.token_norm6144(total_tokens_post)
        out = torch.exp(self.mlp(total_tokens_post)).squeeze(-1)
        encoded_feature = torch.mean(out,dim=-1)

        return encoded_feature 
    
class ScaleHead_Tr_crossAttn_MLP_LCP(nn.Module):
    """
    crossAttn_MLP_LCP_nonshared
    """
    def __init__(
        self,
        max_seq_len: int = 24,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.scale_attention2048_cam = nn.MultiheadAttention(embed_dim=dim_in, num_heads=num_heads, batch_first=True)
        self.scale_attention2048_cls = nn.MultiheadAttention(embed_dim=dim_in, num_heads=num_heads, batch_first=True)

        self.token_norm2048 = nn.LayerNorm(dim_in)
        self.token_norm6144 = nn.LayerNorm(3*dim_in)
        self.token_norm6912 = nn.LayerNorm(6912)

        self.mlp = MLP([3*2048, 2048, 512, 64, 1])
        self.proj = nn.Linear(dim_in, 1)  # Will learn attention over tokens

    def forward(self, aggregated_tokens_list, cls_token, patch_tokens, B, S):
        """
        Forward pass to estimate the scale factor.
        """
        aggregated_cam_token = aggregated_tokens_list[-1][:,:,0,:]                             # [B,S,2048]
        # aggregated_register_token = aggregated_tokens_list[-1][:,:,1:5,:]                      # [B,S,4,2048]
        aggregated_patch_tokens = aggregated_tokens_list[-1][:,:,5:,:]   
        weights = self.proj(aggregated_patch_tokens)  # Shape: [3, 14, 1258, 1]
        weights = F.softmax(weights, dim=2)

        # Weighted sum
        pooled_patch_tokens = (weights * aggregated_patch_tokens).sum(dim=2)

        cls_token = cls_token.view(B, S, -1)  # [B,S,1024]
        cls_token = torch.cat((cls_token,cls_token),dim=-1)  # [B,S,2048]

        cls_token_norm = self.token_norm2048(cls_token)
        pooled_patch_tokens_norm = self.token_norm2048(pooled_patch_tokens)
        aggregated_cam_token_norm = self.token_norm2048(aggregated_cam_token)

        encoded_feat_cam, attn_feat_cam = self.scale_attention2048_cam(aggregated_cam_token_norm, pooled_patch_tokens_norm, pooled_patch_tokens_norm)
        aggregated_cam_token_norm = aggregated_cam_token_norm + encoded_feat_cam

        encoded_feat_cls, attn_feat_cls = self.scale_attention2048_cls(cls_token_norm, pooled_patch_tokens_norm, pooled_patch_tokens_norm)
        cls_token_norm = cls_token_norm + encoded_feat_cls

        total_tokens_post = torch.cat((aggregated_cam_token_norm, cls_token_norm, pooled_patch_tokens_norm), dim=-1)  # [B,S,6144]
        total_tokens_post = self.token_norm6144(total_tokens_post)
        out = torch.exp(self.mlp(total_tokens_post)).squeeze(-1)
        encoded_feature = torch.mean(out,dim=-1)

        return encoded_feature 
    

class ScaleHead_Tr_crossAttn_LCP_MLP_ResidualFiLM(nn.Module):
    """
    crossAttn_LCP_MLP_ResidualFiLM
    """
    def __init__(
        self,
        max_seq_len: int = 24,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.scale_attention2048_cam = nn.MultiheadAttention(embed_dim=dim_in, num_heads=num_heads, batch_first=True)
        self.scale_attention2048_cls = nn.MultiheadAttention(embed_dim=dim_in, num_heads=num_heads, batch_first=True)

        self.token_norm2048 = nn.LayerNorm(dim_in)
        self.token_norm6144 = nn.LayerNorm(3*dim_in)

        self.mlp = MLP([3*2048, 2048, 512, 64, 1])
        self.proj = nn.Linear(dim_in, 1)  # Will learn attention over tokens

        self.FiLM = ResidualFiLM(dim_in, dim_in, dim_in, use_gate=True)

    def forward(self, aggregated_tokens_list, cls_token, patch_tokens, gt_scale, areas, B, S):
        """
        Forward pass to estimate the scale factor.
        """
        aggregated_cam_token = aggregated_tokens_list[-1][:,:,0,:]                             # [B,S,2048]
        # aggregated_register_token = aggregated_tokens_list[-1][:,:,1:5,:]                      # [B,S,4,2048]
        aggregated_patch_tokens = aggregated_tokens_list[-1][:,:,5:,:]   
        weights = self.proj(aggregated_patch_tokens)  # Shape: [3, 14, 1258, 1]
        weights = F.softmax(weights, dim=2)

        # Weighted sum
        pooled_patch_tokens = (weights * aggregated_patch_tokens).sum(dim=2)

        cls_token = self.FiLM(cls_token.view(B*S,-1), aggregated_cam_token.view(B*S,-1))
        ##
        cls_token = cls_token.view(B, S, -1)  # [B,S,1024]

        cls_token = torch.cat((cls_token,cls_token),dim=-1)  # [B,S,2048]

        cls_token_norm = self.token_norm2048(cls_token)
        pooled_patch_tokens_norm = self.token_norm2048(pooled_patch_tokens)
        aggregated_cam_token_norm = self.token_norm2048(aggregated_cam_token)

        encoded_feat_cam, attn_feat_cam = self.scale_attention2048_cam(aggregated_cam_token_norm, pooled_patch_tokens_norm, pooled_patch_tokens_norm)
        aggregated_cam_token_norm = aggregated_cam_token_norm + encoded_feat_cam

        encoded_feat_cls, attn_feat_cls = self.scale_attention2048_cls(cls_token_norm, pooled_patch_tokens_norm, pooled_patch_tokens_norm)
        cls_token_norm = cls_token_norm + encoded_feat_cls

        total_tokens_post = torch.cat((aggregated_cam_token_norm, cls_token_norm, pooled_patch_tokens_norm), dim=-1)  # [B,S,6144]
        total_tokens_post = self.token_norm6144(total_tokens_post)
        out = torch.exp(self.mlp(total_tokens_post)).squeeze(-1)
        encoded_feature = torch.mean(out,dim=-1)

        return encoded_feature 
    

class ScaleHead_Tr_selfattn_crossAttn_MLP_LCPT(nn.Module):
    """
    selfattn_crossAttn_MLP_LCPT
    """
    def __init__(
        self,
        max_seq_len: int = 24,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.scale_self_attention2048_cls = nn.MultiheadAttention(embed_dim=dim_in, num_heads=num_heads, batch_first=True)
        self.scale_self_attention2048_ppt = nn.MultiheadAttention(embed_dim=dim_in, num_heads=num_heads, batch_first=True)
        self.scale_self_attention2048_cam = nn.MultiheadAttention(embed_dim=dim_in, num_heads=num_heads, batch_first=True)
        self.scale_self_attention2048_pat = nn.MultiheadAttention(embed_dim=dim_in//2, num_heads=num_heads, batch_first=True)

        self.scale_cross_attention2048_cls = nn.MultiheadAttention(embed_dim=dim_in, num_heads=num_heads, batch_first=True)
        self.scale_cross_attention2048_cam = nn.MultiheadAttention(embed_dim=dim_in, num_heads=num_heads, batch_first=True)

        self.token_norm1024 = nn.LayerNorm(dim_in//2)
        self.token_norm2048 = nn.LayerNorm(dim_in)
        self.token_norm7168 = nn.LayerNorm(int(3.5*dim_in))

        self.mlp = MLP([int(3.5*2048), 2048, 512, 64, 1])
        self.proj = nn.Linear(dim_in, 1)  # Will learn attention over tokens
        self.proj_patch = nn.Linear(dim_in//2, 1)  # Will learn attention over tokens

    def forward(self, aggregated_tokens_list, cls_token, patch_tokens, B, S):
        """
        Forward pass to estimate the scale factor.
        """
        aggregated_cam_token = aggregated_tokens_list[-1][:,:,0,:]                             # [B,S,2048]
        # aggregated_register_token = aggregated_tokens_list[-1][:,:,1:5,:]                      # [B,S,4,2048]
        aggregated_patch_tokens = aggregated_tokens_list[-1][:,:,5:,:]   
        weights = self.proj(aggregated_patch_tokens)  # Shape: [3, 14, 1258, 1]
        weights = F.softmax(weights, dim=2)

        # Weighted sum
        pooled_patch_tokens = (weights * aggregated_patch_tokens).sum(dim=2)

        patch_tokens = patch_tokens.view(B, S, aggregated_patch_tokens.shape[2], -1)  # [B,S,1024]
        weights_patch = self.proj_patch(patch_tokens)
        weights_patch = F.softmax(weights_patch, dim=2)
        pooled_patches = (weights_patch * patch_tokens).sum(dim=2)
        ##
        cls_token = cls_token.view(B, S, -1)  # [B,S,1024]
        cls_token = torch.cat((cls_token,cls_token),dim=-1)  # [B,S,2048]

        cls_token_norm = self.token_norm2048(cls_token)
        pooled_patch_tokens_norm = self.token_norm2048(pooled_patch_tokens)
        aggregated_cam_token_norm = self.token_norm2048(aggregated_cam_token)
        pooled_patches_norm = self.token_norm1024(pooled_patches)

        ###
        encoded_feat_self_cls, attn_feat_self_cls = self.scale_self_attention2048_cls(cls_token_norm, cls_token_norm, cls_token_norm)
        cls_token_norm = cls_token_norm + encoded_feat_self_cls

        encoded_feat_self_ppt, attn_feat_self_ppt = self.scale_self_attention2048_ppt(pooled_patch_tokens_norm, pooled_patch_tokens_norm, pooled_patch_tokens_norm)
        pooled_patch_tokens_norm = pooled_patch_tokens_norm + encoded_feat_self_ppt

        encoded_feat_self_cam, attn_feat_self_cam = self.scale_self_attention2048_cam(aggregated_cam_token_norm, aggregated_cam_token_norm, aggregated_cam_token_norm)
        aggregated_cam_token_norm = aggregated_cam_token_norm + encoded_feat_self_cam

        encoded_feat_self_pat, attn_feat_self_pat = self.scale_self_attention2048_pat(pooled_patches_norm, pooled_patches_norm, pooled_patches_norm)
        pooled_patches_norm = pooled_patches_norm + encoded_feat_self_pat
        ##

        cls_token_norm = self.token_norm2048(cls_token_norm)
        pooled_patch_tokens_norm = self.token_norm2048(pooled_patch_tokens_norm)
        aggregated_cam_token_norm = self.token_norm2048(aggregated_cam_token_norm)
        pooled_patches_norm = self.token_norm1024(pooled_patches_norm)

        ###
        encoded_feat_cam, attn_feat_cam = self.scale_cross_attention2048_cam(aggregated_cam_token_norm, pooled_patch_tokens_norm, pooled_patch_tokens_norm)
        aggregated_cam_token_norm = aggregated_cam_token_norm + encoded_feat_cam

        encoded_feat_cls, attn_feat_cls = self.scale_cross_attention2048_cls(cls_token_norm, pooled_patch_tokens_norm, pooled_patch_tokens_norm)
        cls_token_norm = cls_token_norm + encoded_feat_cls

        total_tokens_post = torch.cat((aggregated_cam_token_norm, cls_token_norm, pooled_patch_tokens_norm,pooled_patches_norm), dim=-1)  # [B,S,6144]
        total_tokens_post = self.token_norm7168(total_tokens_post)
        out = torch.exp(self.mlp(total_tokens_post)).squeeze(-1)
        encoded_feature = torch.mean(out,dim=-1)

        return encoded_feature 

class ScaleHead_Tr_selfAttn_sep_LCPM(nn.Module):
    """
    selfAttn_sep_LCPM
    """
    def __init__(
        self,
        max_seq_len: int = 24,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.scale_attention2048 = nn.MultiheadAttention(embed_dim=dim_in, num_heads=num_heads, batch_first=True)
        self.scale_attention1024 = nn.MultiheadAttention(embed_dim=int(dim_in//2), num_heads=num_heads, batch_first=True)
        self.scale_attention768 = nn.MultiheadAttention(embed_dim=256*3, num_heads=num_heads, batch_first=True)

        self.token_norm1024 = nn.LayerNorm(dim_in//2)
        self.token_norm2048 = nn.LayerNorm(dim_in)
        self.token_norm5888 = nn.LayerNorm(int(2.5*dim_in+3*256))
        self.token_norm768 = nn.LayerNorm(3*256)

        self.mlp = MLP([int(2.5*dim_in+3*256), 1024, 256, 64, 1])
        self.proj = nn.Linear(dim_in, 1)  # Will learn attention over tokens

    def forward(self, aggregated_tokens_list, cls_token, patch_tokens, t_feat, r_feat, s_tok, B, S):
        """
        Forward pass to estimate the scale factor.
        """
        aggregated_cam_token = aggregated_tokens_list[-1][:,:,0,:]                             # [B,S,2048]
        # aggregated_register_token = aggregated_tokens_list[-1][:,:,1:5,:]                      # [B,S,4,2048]
        aggregated_patch_tokens = aggregated_tokens_list[-1][:,:,5:,:]   
        weights = self.proj(aggregated_patch_tokens)  # Shape: [3, 14, 1258, 1]
        weights = F.softmax(weights, dim=2)

        # Weighted sum
        pooled_patch_tokens = (weights * aggregated_patch_tokens).sum(dim=2)

        cls_token = cls_token.view(B, S, -1)  # [B,S,1024]

        pooled_patch_tokens_norm = self.token_norm2048(pooled_patch_tokens)
        encoded_feat_patch, attn_feat_path = self.scale_attention2048(pooled_patch_tokens_norm,pooled_patch_tokens_norm,pooled_patch_tokens_norm)
        pooled_patch_tokens_norm = pooled_patch_tokens_norm + encoded_feat_patch

        aggregated_cam_token_norm = self.token_norm2048(aggregated_cam_token)
        encoded_feat_cam, attn_feat_cam = self.scale_attention2048(aggregated_cam_token_norm,aggregated_cam_token_norm,aggregated_cam_token_norm)
        aggregated_cam_token_norm = aggregated_cam_token_norm + encoded_feat_cam

        cls_token_norm = self.token_norm1024(cls_token)
        encoded_feat_cls, attn_feat_cls = self.scale_attention1024(cls_token_norm,cls_token_norm,cls_token_norm)
        cls_token_norm = cls_token_norm + encoded_feat_cls

        ##
        s_expanded_tok = s_tok.expand(r_feat.shape)
        
        total_pose_feat = torch.cat((t_feat, r_feat, s_expanded_tok), dim=-1)  # [B,S,3*256]
        total_pose_feat_norm = self.token_norm768(total_pose_feat)

        encoded_feat_pose, attn_feat_pose = self.scale_attention768(total_pose_feat_norm,total_pose_feat_norm,total_pose_feat_norm)
        total_pose_feat_norm = total_pose_feat_norm + encoded_feat_pose

        ##
        total_tokens_post = torch.cat((pooled_patch_tokens_norm, aggregated_cam_token_norm, cls_token_norm, total_pose_feat_norm), dim=-1)  # [B,S,5120]
        total_tokens_post = self.token_norm5888(total_tokens_post)
        out = torch.exp(self.mlp(total_tokens_post)).squeeze(-1)
        encoded_feature = torch.mean(out,dim=-1)

        return encoded_feature 
    

class ScaleHead_MLP_LCPM(nn.Module):
    """
    MLP_LCPM
    """
    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
    ):
        super().__init__()

        self.token_norm1024 = nn.LayerNorm(dim_in//2)
        self.token_norm2048 = nn.LayerNorm(dim_in)
        self.token_norm256 = nn.LayerNorm(256)
        self.token_norm5888 = nn.LayerNorm(int(2.5*dim_in+3*256))

        self.mlp = MLP([int(2.5*dim_in+3*256), 1024, 256, 64, 1])
        self.proj = nn.Linear(dim_in, 1)  # Will learn attention over tokens

    def forward(self, aggregated_tokens_list: list, cls_token: torch.tensor, patch_tokens: torch.tensor, t_feat: torch.tensor, r_feat: torch.tensor, s_tok: torch.tensor, B: int, S: int) -> float:
        aggregated_cam_token = aggregated_tokens_list[-1][:,:,0,:]                             # [B,S,2048]
        # aggregated_register_token = aggregated_tokens_list[-1][:,:,1:5,:]                      # [B,S,4,2048]
        aggregated_patch_tokens = aggregated_tokens_list[-1][:,:,5:,:]                         # [B,S,N,2048]
        weights = self.proj(aggregated_patch_tokens)  # Shape: [3, 14, 1258, 1]
        weights = F.softmax(weights, dim=2)

        # Weighted sum
        pooled_patch_tokens = (weights * aggregated_patch_tokens).sum(dim=2)

        cls_token = cls_token.view(B,S,-1)   # [B,S,1024]
        
        pooled_patch_tokens_norm = self.token_norm2048(pooled_patch_tokens)
        cls_token_norm = self.token_norm1024(cls_token)
        aggregated_cam_token_norm = self.token_norm2048(aggregated_cam_token)

        t_feat_norm = self.token_norm256(t_feat)
        r_feat_norm = self.token_norm256(r_feat)
        s_tok_norm = self.token_norm256(s_tok)
        s_expanded_tok_norm = s_tok_norm.expand(r_feat_norm.shape)

        total_tokens_post = torch.cat((pooled_patch_tokens_norm, aggregated_cam_token_norm, cls_token_norm, t_feat_norm, r_feat_norm, s_expanded_tok_norm), dim=-1)  # [B,S,5120]
        total_tokens_post = self.token_norm5888(total_tokens_post)
        out = torch.exp(self.mlp(total_tokens_post)).squeeze(-1)
        encoded_feature = torch.mean(out,dim=-1)

        return encoded_feature


    
class ScaleHead_Tr_crossAttn_LCPT_MLP_v2(nn.Module):
    """
    crossAttn_LCPT_MLP_v2
    """
    def __init__(
        self,
        max_seq_len: int = 24,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.scale_attention2048_cam = nn.MultiheadAttention(embed_dim=dim_in, num_heads=num_heads, batch_first=True)
        self.scale_attention2048_cls = nn.MultiheadAttention(embed_dim=dim_in, num_heads=num_heads, batch_first=True)
  #      self.token_norm1024 = nn.LayerNorm(dim_in//2)
        self.token_norm2048 = nn.LayerNorm(dim_in)
        self.token_norm6144 = nn.LayerNorm(int(3*dim_in))

        self.mlp = MLP([int(3*2048), 2048, 512, 64, 1])
        self.proj = nn.Linear(dim_in, 1)  # Will learn attention over tokens
        self.proj_patch = nn.Linear(dim_in//2, 1)  # Will learn attention over tokens

    def forward(self, aggregated_tokens_list, cls_token, patch_tokens, gt_scale, areas, B, S):
        """
        Forward pass to estimate the scale factor.
        """
        aggregated_cam_token = aggregated_tokens_list[-1][:,:,0,:]                             # [B,S,2048]
        # aggregated_register_token = aggregated_tokens_list[-1][:,:,1:5,:]                      # [B,S,4,2048]
        aggregated_patch_tokens = aggregated_tokens_list[-1][:,:,5:,:]   
        weights = self.proj(aggregated_patch_tokens)  # Shape: [3, 14, 1258, 1]
        weights = F.softmax(weights, dim=2)

        # Weighted sum
        pooled_patch_tokens = (weights * aggregated_patch_tokens).sum(dim=2)

        patch_tokens = patch_tokens.view(B, S, aggregated_patch_tokens.shape[2], -1)  # [B,S,1024]
        weights_patch = self.proj_patch(patch_tokens)
        weights_patch = F.softmax(weights_patch, dim=2)
        pooled_patches = (weights_patch * patch_tokens).sum(dim=2)

        ##
        cls_token = cls_token.view(B, S, -1)  # [B,S,1024]
        cls_patch_token = torch.cat((cls_token,pooled_patches),dim=-1)  # [B,S,2048]

        cls_patch_token_norm = self.token_norm2048(cls_patch_token)
        pooled_patch_tokens_norm = self.token_norm2048(pooled_patch_tokens)
        aggregated_cam_token_norm = self.token_norm2048(aggregated_cam_token)
   ##     pooled_patches_norm = self.token_norm1024(pooled_patches)

        encoded_feat_cam, attn_feat_cam = self.scale_attention2048_cam(aggregated_cam_token_norm, pooled_patch_tokens_norm, pooled_patch_tokens_norm)
        aggregated_cam_token_norm = aggregated_cam_token_norm + encoded_feat_cam

        encoded_feat_cls_patch, attn_feat_cls = self.scale_attention2048_cls(cls_patch_token_norm, pooled_patch_tokens_norm, pooled_patch_tokens_norm)
        cls_patch_token_norm = cls_patch_token_norm + encoded_feat_cls_patch

        total_tokens_post = torch.cat((aggregated_cam_token_norm, cls_patch_token_norm, pooled_patch_tokens_norm), dim=-1)  # [B,S,6144]
        total_tokens_post = self.token_norm6144(total_tokens_post)
        out = torch.exp(self.mlp(total_tokens_post)).squeeze(-1)
        encoded_feature = torch.mean(out,dim=-1)

        return encoded_feature 

  

class ScaleHead_Tr_crossAttn_LCPT_MLP(nn.Module):
    """
    crossAttn_LCPT_MLP
    """
    def __init__(
        self,
        max_seq_len: int = 24,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.scale_attention2048_cls = nn.MultiheadAttention(embed_dim=dim_in, num_heads=num_heads, batch_first=True)
        self.scale_attention2048_cam = nn.MultiheadAttention(embed_dim=dim_in, num_heads=num_heads, batch_first=True)

        self.token_norm1024 = nn.LayerNorm(dim_in//2)
        self.token_norm2048 = nn.LayerNorm(dim_in)
        self.token_norm7168 = nn.LayerNorm(int(3.5*dim_in))

        self.mlp = MLP([int(3.5*2048), 2048, 512, 64, 1])
        self.proj = nn.Linear(dim_in, 1)  # Will learn attention over tokens
        self.proj_patch = nn.Linear(dim_in//2, 1)  # Will learn attention over tokens

    def forward(self, aggregated_tokens_list, cls_token, patch_tokens,gt_scale, areas, B, S):
        """
        Forward pass to estimate the scale factor.
        """
        aggregated_cam_token = aggregated_tokens_list[-1][:,:,0,:]                             # [B,S,2048]
        # aggregated_register_token = aggregated_tokens_list[-1][:,:,1:5,:]                      # [B,S,4,2048]
        aggregated_patch_tokens = aggregated_tokens_list[-1][:,:,5:,:]   
        weights = self.proj(aggregated_patch_tokens)  # Shape: [3, 14, 1258, 1]
        weights = F.softmax(weights, dim=2)

        # Weighted sum
        pooled_patch_tokens = (weights * aggregated_patch_tokens).sum(dim=2)

        patch_tokens = patch_tokens.view(B, S, aggregated_patch_tokens.shape[2], -1)  # [B,S,1024]
        weights_patch = self.proj_patch(patch_tokens)
        weights_patch = F.softmax(weights_patch, dim=2)
        pooled_patches = (weights_patch * patch_tokens).sum(dim=2)
        ##
        cls_token = cls_token.view(B, S, -1)  # [B,S,1024]
        cls_token = torch.cat((cls_token,cls_token),dim=-1)  # [B,S,2048]

        cls_token_norm = self.token_norm2048(cls_token)
        pooled_patch_tokens_norm = self.token_norm2048(pooled_patch_tokens)
        aggregated_cam_token_norm = self.token_norm2048(aggregated_cam_token)
        pooled_patches_norm = self.token_norm1024(pooled_patches)

        encoded_feat_cam, attn_feat_cam = self.scale_attention2048_cls(aggregated_cam_token_norm, pooled_patch_tokens_norm, pooled_patch_tokens_norm)
        aggregated_cam_token_norm = aggregated_cam_token_norm + encoded_feat_cam

        encoded_feat_cls, attn_feat_cls = self.scale_attention2048_cam(cls_token_norm, pooled_patch_tokens_norm, pooled_patch_tokens_norm)
        cls_token_norm = cls_token_norm + encoded_feat_cls

        total_tokens_post = torch.cat((aggregated_cam_token_norm, cls_token_norm, pooled_patch_tokens_norm,pooled_patches_norm), dim=-1)  # [B,S,6144]
        total_tokens_post = self.token_norm7168(total_tokens_post)
        out = torch.exp(self.mlp(total_tokens_post)).squeeze(-1)
        encoded_feature = torch.mean(out,dim=-1)

        return encoded_feature 


class ScaleHead_Tr_selfAttn_cat_LCPT(nn.Module):
    """
    selfAttn_cat_LCPT
    """
    def __init__(
        self,
        max_seq_len: int = 24,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
    ):
        super().__init__()

        self.scale_attention = nn.MultiheadAttention(embed_dim=3*dim_in, num_heads=num_heads, batch_first=True)
        self.token_norm1024 = nn.LayerNorm(dim_in//2)
        self.token_norm2048 = nn.LayerNorm(dim_in)
        self.token_norm6144 = nn.LayerNorm(int(3*dim_in))
        self.mlp = MLP([6144, 2048, 512, 64, 1])
        self.proj = nn.Linear(dim_in, 1)
        self.proj_patch = nn.Linear(dim_in//2, 1)

    def forward(self, aggregated_tokens_list, cls_token, patch_tokens, gt_scale, areas, B, S):
        """
        Forward pass to estimate the scale factor.
        """
        aggregated_cam_token = aggregated_tokens_list[-1][:,:,0,:]                            
        # aggregated_register_token = aggregated_tokens_list[-1][:,:,1:5,:]                      
        aggregated_patch_tokens = aggregated_tokens_list[-1][:,:,5:,:]   
        weights = self.proj(aggregated_patch_tokens) 
        weights = F.softmax(weights, dim=2)

        # Weighted sum
        pooled_patch_tokens = (weights * aggregated_patch_tokens).sum(dim=2)

        patch_tokens = patch_tokens.view(B, S, aggregated_patch_tokens.shape[2], -1) 
        weights_patch = self.proj_patch(patch_tokens)
        weights_patch = F.softmax(weights_patch, dim=2)
        pooled_patches = (weights_patch * patch_tokens).sum(dim=2)

        cls_token = cls_token.view(B, S, -1)

        total_tokens = torch.cat((self.token_norm2048(pooled_patch_tokens),self.token_norm2048(aggregated_cam_token),self.token_norm1024(cls_token),self.token_norm1024(pooled_patches)), dim=-1)  # [B,S,6144]

        total_tokens_norm = self.token_norm6144(total_tokens)
        encoded_feat_total, attn_feat_aggr = self.scale_attention(total_tokens_norm,total_tokens_norm,total_tokens_norm)
        total_tokens_post = total_tokens_norm + encoded_feat_total

        total_tokens_post = self.token_norm6144(total_tokens_post)
        out = torch.exp(self.mlp(total_tokens_post)).squeeze(-1)
        encoded_feature = torch.mean(out,dim=-1)

        return encoded_feature
    
class ScaleHead_Tr(nn.Module):
    """
    selfAttn_sep_LCPT
    """
    def __init__(
        self,
        max_seq_len: int = 24,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.scale_attention2048_ppt = nn.MultiheadAttention(embed_dim=dim_in, num_heads=num_heads, batch_first=True)
        self.scale_attention2048_cam = nn.MultiheadAttention(embed_dim=dim_in, num_heads=num_heads, batch_first=True)

        self.scale_attention1024_cls = nn.MultiheadAttention(embed_dim=int(dim_in//2), num_heads=num_heads, batch_first=True)
        self.scale_attention1024_pat = nn.MultiheadAttention(embed_dim=int(dim_in//2), num_heads=num_heads, batch_first=True)

        self.token_norm1024 = nn.LayerNorm(dim_in//2)
        self.token_norm2048 = nn.LayerNorm(dim_in)
        self.token_norm6144 = nn.LayerNorm(3*dim_in)

        self.mlp = MLP([6144, 2048, 256, 64, 1])
        self.proj = nn.Linear(dim_in, 1)  # Will learn attention over tokens
        self.proj_patch = nn.Linear(dim_in//2, 1)  # Will learn attention over tokens

    def forward(self, aggregated_tokens_list, cls_token, patch_tokens, B, S):
        """
        Forward pass to estimate the scale factor.
        """
        aggregated_cam_token = aggregated_tokens_list[-1][:,:,0,:]                             # [B,S,2048]
        # aggregated_register_token = aggregated_tokens_list[-1][:,:,1:5,:]                      # [B,S,4,2048]
        aggregated_patch_tokens = aggregated_tokens_list[-1][:,:,5:,:]   
        weights = self.proj(aggregated_patch_tokens)  # Shape: [3, 14, 1258, 1]
        weights = F.softmax(weights, dim=2)

        # Weighted sum
        pooled_patch_tokens = (weights * aggregated_patch_tokens).sum(dim=2)

        patch_tokens = patch_tokens.view(B, S, aggregated_patch_tokens.shape[2], -1)  # [B,S,1024]
        weights_patch = self.proj_patch(patch_tokens)
        weights_patch = F.softmax(weights_patch, dim=2)
        pooled_patches = (weights_patch * patch_tokens).sum(dim=2)

        cls_token = cls_token.view(B, S, -1)  # [B,S,1024]

        pooled_patch_tokens_norm = self.token_norm2048(pooled_patch_tokens)
        encoded_feat_aggrpatch, attn_feat_aggrpatch = self.scale_attention2048_ppt(pooled_patch_tokens_norm,pooled_patch_tokens_norm,pooled_patch_tokens_norm)
        pooled_patch_aggrtokens_norm = pooled_patch_tokens_norm + encoded_feat_aggrpatch

        aggregated_cam_token_norm = self.token_norm2048(aggregated_cam_token)
        encoded_feat_cam, attn_feat_cam = self.scale_attention2048_cam(aggregated_cam_token_norm,aggregated_cam_token_norm,aggregated_cam_token_norm)
        aggregated_cam_token_norm = aggregated_cam_token_norm + encoded_feat_cam

        cls_token_norm = self.token_norm1024(cls_token)
        encoded_feat_cls, attn_feat_cls = self.scale_attention1024_cls(cls_token_norm,cls_token_norm,cls_token_norm)
        cls_token_norm = cls_token_norm + encoded_feat_cls

        pooled_patches_norm = self.token_norm1024(pooled_patches)
        encoded_feat_patch, attn_feat_patch = self.scale_attention1024_pat(pooled_patches_norm,pooled_patches_norm,pooled_patches_norm)
        pooled_patches_norm = pooled_patches_norm + encoded_feat_patch

        total_tokens_post = torch.cat((pooled_patch_aggrtokens_norm, aggregated_cam_token_norm, cls_token_norm, pooled_patches_norm), dim=-1)  # [B,S,6144]
        total_tokens_post = self.token_norm6144(total_tokens_post)
        out = torch.exp(self.mlp(total_tokens_post)).squeeze(-1)
        encoded_feature = torch.mean(out,dim=-1)

        return encoded_feature 


class ScaleHead_Tr_selfAttn_sep_LCP(nn.Module):
    """
    selfAttn_sep_LCP
    """
    def __init__(
        self,
        max_seq_len: int = 24,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.scale_attention2048 = nn.MultiheadAttention(embed_dim=dim_in, num_heads=num_heads, batch_first=True)
        self.scale_attention1024 = nn.MultiheadAttention(embed_dim=int(dim_in//2), num_heads=num_heads, batch_first=True)

        self.token_norm1024 = nn.LayerNorm(dim_in//2)
        self.token_norm2048 = nn.LayerNorm(dim_in)
        self.token_norm5120 = nn.LayerNorm(int(2.5*dim_in))

        self.mlp = MLP([5120, 1024, 256, 64, 1])
        self.proj = nn.Linear(dim_in, 1)  # Will learn attention over tokens

    def forward(self, aggregated_tokens_list, cls_token, patch_tokens, B, S):
        """
        Forward pass to estimate the scale factor.
        """
        aggregated_cam_token = aggregated_tokens_list[-1][:,:,0,:]                             # [B,S,2048]
        # aggregated_register_token = aggregated_tokens_list[-1][:,:,1:5,:]                      # [B,S,4,2048]
        aggregated_patch_tokens = aggregated_tokens_list[-1][:,:,5:,:]   
        weights = self.proj(aggregated_patch_tokens)  # Shape: [3, 14, 1258, 1]
        weights = F.softmax(weights, dim=2)

        # Weighted sum
        pooled_patch_tokens = (weights * aggregated_patch_tokens).sum(dim=2)

        cls_token = cls_token.view(B, S, -1)  # [B,S,1024]

        pooled_patch_tokens_norm = self.token_norm2048(pooled_patch_tokens)
        encoded_feat_patch, attn_feat_path = self.scale_attention2048(pooled_patch_tokens_norm,pooled_patch_tokens_norm,pooled_patch_tokens_norm)
        pooled_patch_tokens_norm = pooled_patch_tokens_norm + encoded_feat_patch

        aggregated_cam_token_norm = self.token_norm2048(aggregated_cam_token)
        encoded_feat_cam, attn_feat_cam = self.scale_attention2048(aggregated_cam_token_norm,aggregated_cam_token_norm,aggregated_cam_token_norm)
        aggregated_cam_token_norm = aggregated_cam_token_norm + encoded_feat_cam

        cls_token_norm = self.token_norm1024(cls_token)
        encoded_feat_cls, attn_feat_cls = self.scale_attention1024(cls_token_norm,cls_token_norm,cls_token_norm)
        cls_token_norm = cls_token_norm + encoded_feat_cls

        total_tokens_post = torch.cat((pooled_patch_tokens_norm, aggregated_cam_token_norm, cls_token_norm), dim=-1)  # [B,S,5120]
        total_tokens_post = self.token_norm5120(total_tokens_post)
        out = torch.exp(self.mlp(total_tokens_post)).squeeze(-1)
        encoded_feature = torch.mean(out,dim=-1)

        return encoded_feature 
  
class ScaleHead_Tr_selfAttn_cat_LCP(nn.Module):
    def __init__(
        self,
        max_seq_len: int = 24,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.scale_attention = nn.MultiheadAttention(embed_dim=int(2.5*dim_in), num_heads=num_heads, batch_first=True)
        self.token_norm1024 = nn.LayerNorm(dim_in//2)
        self.token_norm2048 = nn.LayerNorm(dim_in)
        self.token_norm5120 = nn.LayerNorm(int(2.5*dim_in))
        self.mlp = MLP([5120, 1024, 256, 64, 1])
        self.proj = nn.Linear(dim_in, 1)  # Will learn attention over tokens

    def forward(self, aggregated_tokens_list, cls_token, patch_tokens, B, S):
        """
        Forward pass to estimate the scale factor.
        """
        aggregated_cam_token = aggregated_tokens_list[-1][:,:,0,:]                             # [B,S,2048]
        # aggregated_register_token = aggregated_tokens_list[-1][:,:,1:5,:]                      # [B,S,4,2048]
        aggregated_patch_tokens = aggregated_tokens_list[-1][:,:,5:,:]   
        weights = self.proj(aggregated_patch_tokens)  # Shape: [3, 14, 1258, 1]
        weights = F.softmax(weights, dim=2)

        # Weighted sum
        pooled_patch_tokens = (weights * aggregated_patch_tokens).sum(dim=2)

        cls_token = cls_token.view(B, S, -1)  # [B,S,1024]

        total_tokens = torch.cat((self.token_norm2048(pooled_patch_tokens),self.token_norm2048(aggregated_cam_token), self.token_norm1024(cls_token)), dim=-1)  # [B,S,5070]

        total_tokens_norm = self.token_norm5120(total_tokens)
        encoded_feat_total, attn_feat_aggr = self.scale_attention(total_tokens_norm,total_tokens_norm,total_tokens_norm)
        total_tokens_post = total_tokens_norm + encoded_feat_total

        total_tokens_post = self.token_norm5120(total_tokens_post)
        out = torch.exp(self.mlp(total_tokens_post)).squeeze(-1)
        encoded_feature = torch.mean(out,dim=-1)

        return encoded_feature

class GatingNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),       # Output: [p_indoor, p_outdoor]
     #       nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.classifier(x)

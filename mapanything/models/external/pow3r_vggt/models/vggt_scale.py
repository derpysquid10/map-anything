# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from pow3r_vggt.models.aggregator import Aggregator
from pow3r_vggt.heads.camera_head import CameraHead
from pow3r_vggt.heads.dpt_head import DPTHead
from pow3r_vggt.heads.track_head import TrackHead
from pow3r_vggt.heads.scale_head import ScaleHead, ScaleHead_Tr, ScaleHead_MoE
from transformers import CLIPTokenizer, CLIPTextModel, CLIPModel
import numpy as np
import torch.nn.functional as F
from pow3r_vggt.heads.caminfo_head import PoseEncoder, IntrinsicsEncoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VGGTS(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True, enable_scale=True):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None
      #  self.scale_head = ScaleHead(dim_in=2 * embed_dim) if enable_scale else None
      #  self.scale_head = ScaleHead_Tr(dim_in=2 * embed_dim) if enable_scale else None
        self.scale_head = ScaleHead_MoE(dim_in=2 * embed_dim) if enable_scale else None
      #  self.campose_head = PoseEncoder(num_frequencies=8, out_dim=256)
   ##     self.camintr_head = IntrinsicsEncoder(num_frequencies=8, out_dim=256)
     ##   self._load_clip()

    def _load_clip(self):
        self.scale_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)

        for param in clip_model.vision_model.parameters():
            param.requires_grad = False

        # Extract only the text parts
        self.scale_text_encoder = clip_model.text_model
        self.scale_text_projection = clip_model.text_projection

        self.text = [" scene, scale is "]
        self.area_map = {0:"Indoor", 1:"Outdoor"}

    def gen_text_features(self, areas, gt_scale):
        # Creat text prompt
        gt_scale = gt_scale.cpu().numpy()
        envs = areas[:,0].cpu().numpy().astype(int)

        envs_str = np.vectorize(self.area_map.get)(envs.astype(int))

        # Round scales and format as strings
        scales_str = np.floor(gt_scale * 10) / 10   # round *down* to 1 decimal
        scales_str = np.char.mod('%.1f', scales_str)

        # Build final sentences efficiently
        sentence = np.core.defchararray.add(
                   np.core.defchararray.add(envs_str, self.text[0]),scales_str).tolist()

        # Tokenize input
        tokenized_data = self.scale_tokenizer(sentence, return_tensors="pt", padding=True)
        input_ids = tokenized_data['input_ids']
        attention_mask = tokenized_data['attention_mask']

        outputs = self.scale_text_encoder(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
        pooled_output = outputs.pooler_output  # [CLS] representation
        text_features = self.scale_text_projection(pooled_output)

        return text_features

    def forward(self, images: torch.Tensor, cam_extr: torch.tensor, cam_intr: torch.tensor, query_points: torch.Tensor = None, gt_scale: torch.Tensor = None, areas:torch.Tensor = None, phase: str = ""): #, text_features:torch.Tensor = None): # pose: torch.Tensor = None
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """        
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
            
        B, S, C, H, W = images.shape

        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx, cls_token, patch_tokens = self.aggregator(images)

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list
                
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

            # compute the camera extrinsics token
         #   t_feat, r_feat, s_tok, _ = self.campose_head(cam_extr[:,:,:,0:3], cam_extr[:,:,:,3], anchor_idx=0)
            # compute the camera intrinsics token
            #intrin_feat, res_tok, dist_tok, phys_tok = self.camintr_head(cam_intr[:,:,0,0], cam_intr[:,:,1,1], cam_intr[:,:,0,2], cam_intr[:,:,1,2], H, W)
         #   scale = self.scale_head(aggregated_tokens_list, cls_token, patch_tokens, t_feat, r_feat, s_tok, B, S) # gt_scale, areas, B, S)  # for scale head ##poses

        ## Normal Head
         #   scale = self.scale_head(aggregated_tokens_list, cls_token, patch_tokens, B, S) # gt_scale, areas, B, S)  # for scale head ##poses

            ## MoE
            scale, cls_pred = self.scale_head(aggregated_tokens_list, cls_token, patch_tokens, B, S) #, phase) # gt_scale, areas, B, S)  # for scale head ##poses
            predictions["cls_pred"] = cls_pred  # gating network output for MoE

            ## Text Supervision
           #  scale, vis_feat = self.scale_head(aggregated_tokens_list, cls_token, patch_tokens, B, S) # gt_scale, areas, B, S)  # for scale head ##poses
           #  text_features = self.gen_text_features(areas, gt_scale)
           #  text_features = F.normalize(text_features,dim=-1) #.clone().detach()
           #  predictions["text_feat"] = text_features 
           #  predictions["vis_feat"] = vis_feat

            predictions["scale"] = scale

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions


# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

from torch import Tensor
from torch import nn
import torch.nn.functional as F
import torch

XFORMERS_AVAILABLE = False

import pdb

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope


    def forward(
        self, 
        x: Tensor, 
        pos=None, 
        special_attention_args=None, 
        use_special_attention=False,
        special_attention_type=None
    ) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)


        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
                

        if special_attention_args and use_special_attention:
            x = self.special_attention(q, k, v, special_attention_args, special_attention_type)
        else:
            x = self.original_attention(q, k, v)
    

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def original_attention(self, q, k, v):
        """original attention implementation of VGGT"""
        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        return x
    
    def special_attention(self, q, k, v, special_attention_args, special_attention_type):
        """Dispatch to the appropriate special attention type"""
        if special_attention_type == "prope":
            return self.prope_attention(q, k, v, special_attention_args)
        elif special_attention_type == "cape":
            return self.cape_attention(q, k, v, special_attention_args)
        elif special_attention_type == "gta":
            return self.gta_attention(q, k, v, special_attention_args)
        else:

            raise ValueError(f"Unknown special attention type: {special_attention_type}")
    
    def cape_attention(self, q, k, v, cape_args):
        q_fn = cape_args["q_encoder"]
        k_fn = cape_args["k_encoder"]
        o_fn = cape_args["o_encoder"]

        tokens_per_image = cape_args["tokens_per_image"]
        batch_size = cape_args["batch_size"]

        assert q.shape[2] % tokens_per_image == 0

        change_to_global_flag = False
        if q.shape[2] == tokens_per_image:
            q = apply_special_encoding(q, q_fn, batch_size)
            k = apply_special_encoding(k, k_fn, batch_size)

        else:
            change_to_global_flag = True
            a, b, c, d = q.shape

            q_reshaped = q.reshape(-1, q.shape[1], tokens_per_image, q.shape[3])
            k_reshaped = k.reshape(-1, k.shape[1], tokens_per_image, k.shape[3])

            q_with_prope = apply_special_encoding(q_reshaped, q_fn, batch_size)
            k_with_prope = apply_special_encoding(k_reshaped, k_fn, batch_size)

            q = q_with_prope.reshape(a, b, c, d)
            k = k_with_prope.reshape(a, b, c, d)
        
        output = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)

        if not change_to_global_flag:
            output = apply_special_encoding(output, o_fn, batch_size)
        
        else:
            output_reshaped = output.reshape(-1, output.shape[1], tokens_per_image, output.shape[3])
            output_with_prope = apply_special_encoding(output_reshaped, o_fn, batch_size)
            output = output_with_prope.reshape(a, b, c, d)

        return output
    
    def gta_attention(self, q, k, v, gta_args):
        q_fn = gta_args["q_encoder"]
        kv_fn = gta_args["kv_encoder"]
        o_fn = gta_args["o_encoder"]

        tokens_per_image = gta_args["tokens_per_image"]
        batch_size = gta_args["batch_size"]

        assert q.shape[2] % tokens_per_image == 0

        change_to_global_flag = False
        if q.shape[2] == tokens_per_image:
            q = apply_special_encoding(q, q_fn, batch_size)
            k = apply_special_encoding(k, kv_fn, batch_size)
            v = apply_special_encoding(v, kv_fn, batch_size)

        else:
            change_to_global_flag = True
            a, b, c, d = q.shape

            q_reshaped = q.reshape(-1, q.shape[1], tokens_per_image, q.shape[3])
            k_reshaped = k.reshape(-1, k.shape[1], tokens_per_image, k.shape[3])
            v_reshaped = v.reshape(-1, v.shape[1], tokens_per_image, v.shape[3])

            q_with_prope = apply_special_encoding(q_reshaped, q_fn, batch_size)
            k_with_prope = apply_special_encoding(k_reshaped, kv_fn, batch_size)
            v_with_prope = apply_special_encoding(v_reshaped, kv_fn, batch_size)

            q = q_with_prope.reshape(a, b, c, d)
            k = k_with_prope.reshape(a, b, c, d)
            v = v_with_prope.reshape(a, b, c, d)      
        
        output = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)

        if not change_to_global_flag:
            output = apply_special_encoding(output, o_fn, batch_size)
        
        else:
            output_reshaped = output.reshape(-1, output.shape[1], tokens_per_image, output.shape[3])
            output_with_prope = apply_special_encoding(output_reshaped, o_fn, batch_size)
            output = output_with_prope.reshape(a, b, c, d)

        return output


    def prope_attention(self, q, k, v, prope_args):
        
        q_fn = prope_args["q_encoder"]
        kv_fn = prope_args["kv_encoder"]
        o_fn = prope_args["o_encoder"]

        tokens_per_image = prope_args["tokens_per_image"]
        batch_size = prope_args["batch_size"]

        

        assert q.shape[2] % tokens_per_image == 0

        change_to_global_flag = False
        if q.shape[2] == tokens_per_image:
            q = apply_special_encoding(q, q_fn, batch_size)
            k = apply_special_encoding(k, kv_fn, batch_size)
            v = apply_special_encoding(v, kv_fn, batch_size)

        else:
            change_to_global_flag = True
            a, b, c, d = q.shape

            q_reshaped = q.reshape(-1, q.shape[1], tokens_per_image, q.shape[3])
            k_reshaped = k.reshape(-1, k.shape[1], tokens_per_image, k.shape[3])
            v_reshaped = v.reshape(-1, v.shape[1], tokens_per_image, v.shape[3])

            q_with_prope = apply_special_encoding(q_reshaped, q_fn, batch_size)
            k_with_prope = apply_special_encoding(k_reshaped, kv_fn, batch_size)
            v_with_prope = apply_special_encoding(v_reshaped, kv_fn, batch_size)

            q = q_with_prope.reshape(a, b, c, d)
            k = k_with_prope.reshape(a, b, c, d)
            v = v_with_prope.reshape(a, b, c, d)      
        
        output = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)

        if not change_to_global_flag:
            output = apply_special_encoding(output, o_fn, batch_size)
        
        else:
            output_reshaped = output.reshape(-1, output.shape[1], tokens_per_image, output.shape[3])
            output_with_prope = apply_special_encoding(output_reshaped, o_fn, batch_size)
            output = output_with_prope.reshape(a, b, c, d)

        return output
        

def apply_special_encoding(tokens, function, batch_size):
    """ Assume tokens are in shape(B*S, num_heads, P, head_dim), outputs in same dim"""
    patch_tokens = tokens[:, :, 5:, :] 
    BS, num_heads, num_patch_tokens, head_dim = patch_tokens.shape
    patch_tokens = patch_tokens.reshape(batch_size, patch_tokens.shape[1], -1, patch_tokens.shape[3])
    prope_tokens = function(patch_tokens)
    prope_tokens = prope_tokens.reshape(BS, num_heads, num_patch_tokens, head_dim)

    # Use torch.cat to avoid in-place modification of view
    tokens = torch.cat([tokens[:, :, :5, :], prope_tokens], dim=2)
    return tokens



class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None, 
        special_attention_args=None, 
        use_special_attention=False,
        special_attention_type=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

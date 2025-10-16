import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualFiLM(nn.Module):
    def __init__(self, token_dim, cam_dim, hidden_dim=128, use_gate=True):
        super().__init__()
        self.use_gate = use_gate
        out_dim = token_dim + (1 if use_gate else 0)  # gamma, beta, [alpha]

        self.mlp = nn.Sequential(
            nn.Linear(cam_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

        # small learnable scaling for stability
        self.gamma_scale = nn.Parameter(torch.tensor(0.1))
        self.beta_scale  = nn.Parameter(torch.tensor(0.1))

    def forward(self, cls_token, cam_token):
        """
        cls_token: [B, D]
        cam_token: [B, D]
        """
        B, D = cls_token.shape
        pred = self.mlp(cam_token)  # [B, 2D(+1)]

        if self.use_gate:
            gamma, beta, alpha = torch.split(pred, [D, D, 1], dim=-1)
            alpha = torch.sigmoid(alpha)
        else:
            gamma, beta = torch.split(pred, [D, D], dim=-1)
            alpha = 1.0

        gamma = torch.tanh(gamma) * self.gamma_scale
        beta  = torch.tanh(beta)  * self.beta_scale

        mod = (1 + gamma) * cls_token + beta
        out = (1 - alpha) * cls_token + alpha * mod
        return out
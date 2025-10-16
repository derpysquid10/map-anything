import torch
import torch.nn as nn
import math

def fourier_features(x, num_frequencies=8):
    """
    x: (..., 3) in roughly O(1) range
    return: (..., 3 * 2 * num_frequencies)
    """
    # [B,N,3, F] -> sin/cos concat
    freqs = 2.0 ** torch.arange(num_frequencies, device=x.device, dtype=x.dtype) * math.pi
    # (..., 3, F)
    xb = x.unsqueeze(-1) * freqs  # broadcast
    return torch.cat([torch.sin(xb), torch.cos(xb)], dim=-1).reshape(*x.shape[:-1], -1)

def rotation_matrix_to_6d(R):
    """
    Zhou et al. 'On the Continuity of Rotation Representations...'
    R: (..., 3, 3)
    returns (..., 6)
    """
    return torch.cat([R[..., :3, 0], R[..., :3, 1]], dim=-1)

class PoseEncoder(nn.Module):
    def __init__(self, num_frequencies=8, out_dim=256):
        super().__init__()
        self.num_frequencies = num_frequencies
        # translation head
        in_t = 3 * 2 * num_frequencies + 3 + 1  # Fourier(tilde t) + direction u + log-mag m
        self.t_mlp = nn.Sequential(
            nn.Linear(in_t, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim),
        )
        # rotation head (6D -> out_dim)
        self.r_mlp = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
        )
        # scene scale token (scalar -> out_dim)
        self.s_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, Rex, tex, anchor_idx=0, scene_scale=None, eps=1e-8):
        """
        Rex: (B, N, 3, 3), tex: (B, N, 3) in meters (camera extrinsics, world to camera).
        anchor_idx: choose anchor camera per scene
        scene_scale: (B,) in meters, if None compute median baseline
        returns:
          t_feat: (B, N, D), r_feat: (B, N, D), s_tok: (B, 1, D)
        """
        B, N = tex.shape[:2]

        T = torch.eye(4, dtype=Rex.dtype, device=Rex.device).view(1,1,4,4).expand(B, N, 4, 4).clone()
        T[..., :3, :3] = Rex
        T[..., :3,  3] = tex
        T_cam2world = torch.linalg.inv(T)
       
        R = T_cam2world[:,:,0:3,0:3]
        t = T_cam2world[:,:,0:3,3]

        R0 = R[:, anchor_idx]                      # (B,3,3)
        t0 = t[:, anchor_idx]                      # (B,3)

        # align to anchor frame
        Rt = R0.transpose(-1, -2)                  # (B,3,3)
        t_rel = torch.einsum('bij, bnj -> bni', Rt, (t - t0[:,None,:]))   # (B,N,3)
        R_rel = torch.einsum('bij, bnjk -> bnik', Rt, R)        # (B,N,3,3)

        # compute scene scale if not provided: median pairwise baseline in anchor frame
        if scene_scale is None:
            # baselines to anchor as a simple, stable proxy
            d = torch.norm(t_rel, dim=-1)  # (B,N)
            s_scene = torch.median(d, dim=1).values.clamp_min(5e-2)  # avoid too small
        else:
            s_scene = scene_scale

        # normalize translation
        t_norm = t_rel / s_scene[:, None, None]    # (B,N,3), ~O(1)

        # direction + log-magnitude (in meters, robust)
        mag = torch.norm(t_rel, dim=-1, keepdim=True)               # (B,N,1)
        u = t_rel / (mag + eps)                                     # (B,N,3)
        m = torch.log(mag + 1e-6)                                   # (B,N,1)  # log meters

        # Fourier features on normalized translation
        ft = fourier_features(t_norm, num_frequencies=self.num_frequencies)  # (B,N, 3*2*K)

        # pack translation embedding
        t_in = torch.cat([ft, u, m], dim=-1)
        t_feat = self.t_mlp(t_in)                                   # (B,N,D)

        # rotation 6D
        r6 = rotation_matrix_to_6d(R_rel)                            # (B,N,6)
        r_feat = self.r_mlp(r6)                                      # (B,N,D)

        # scene scale token (log meters)
        s_tok_in = torch.log(s_scene[:, None, None] + 1e-6)          # (B,1,1)
        s_tok = self.s_mlp(s_tok_in)                                 # (B,1,D)

        return t_feat, r_feat, s_tok, s_scene  # return s_scene in order to recover the output unit to meter


class IntrinsicsEncoder(nn.Module):
    """
    Input: fx, fy, cx, cy, H, W, (optional) dist=(k1,k2,k3,p1,p2)
    Output: intrinsics_feat: (B,N,D), res_tok: (B,N,D), (optional) phys_tok: (B,N,D)
    """
    def __init__(self, num_frequencies=8, out_dim=256, use_fov=True, use_distortion=True):
        super().__init__()
        self.use_fov = use_fov
        self.use_distortion = use_distortion
        self.num_frequencies = num_frequencies

        base_dim = 5  # [fx/W, fy/H, (cx-W/2)/W, (cy-H/2)/H, H/W]
        if use_fov:
            base_dim += 2  # [fov_x, fov_y]
        in_dim = base_dim

        self.intrin_mlp = nn.Sequential(
            nn.Linear(in_dim * 2 * num_frequencies, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.res_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_dim),
            nn.LayerNorm(out_dim),
        )
        if use_distortion:
            self.dist_mlp = nn.Sequential(
                nn.Linear(5, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, out_dim),
                nn.LayerNorm(out_dim),
            )

    def forward(self, fx, fy, cx, cy, H, W, dist=None, pixel_pitch_mm=None, sensor_diag_mm=None):
        """
        fx,...H,W are (B,N) or can be borarcasted to (B,N)
        dist: (B,N,5) optional [k1,k2,k3,p1,p2]
        pixel_pitch_mm/sensor_diag_mm: (B,N) optional
        """
        dtype = fx.dtype
        device = fx.device
        W = torch.full(fx.shape, W, dtype=torch.float32, device = device)
        H = torch.full(fx.shape, H, dtype=torch.float32, device = device)

        fx_n = fx / W
        fy_n = fy / H
        cx_n = (cx - 0.5 * W) / W
        cy_n = (cy - 0.5 * H) / H
        aspect = H / (W + 1e-8)

        feats = [fx_n, fy_n, cx_n, cy_n, aspect]
        if self.use_fov:
            fov_x = 2.0 * torch.atan(1.0 / (2.0 * fx_n.clamp_min(1e-8)))
            fov_y = 2.0 * torch.atan(1.0 / (2.0 * fy_n.clamp_min(1e-8)))
            feats += [fov_x, fov_y]

        x = torch.stack(feats, dim=-1)  # (B,N,D0)
        x = fourier_features(x, self.num_frequencies)  # (B,N, 2*F*D0)
        intrin_feat = self.intrin_mlp(x)           # (B,N,D)

        # resolution token：log(diagonal_px)
        diag_px = torch.sqrt(W**2 + H**2).clamp_min(1.0)
        res_tok = self.res_mlp(torch.log(diag_px).unsqueeze(-1))  # (B,N,1)->(B,N,D)

        # optional: dist token
        dist_tok = None
        if self.use_distortion and dist is not None:
            alpha = 1.0
            dist_bounded = torch.tanh(alpha * dist)   # (B,N,5)
            dist_tok = self.dist_mlp(dist_bounded)    # (B,N,D)

        # optional: physical pixel or CCD size token
        phys_tok = None
        if (pixel_pitch_mm is not None) or (sensor_diag_mm is not None):
            if sensor_diag_mm is None:
                sensor_diag_mm = diag_px * (pixel_pitch_mm + 1e-8)
            phys_tok = self.res_mlp(torch.log(sensor_diag_mm.clamp_min(1e-6)).unsqueeze(-1))

        return intrin_feat, res_tok, dist_tok, phys_tok
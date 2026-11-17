"""
Detection and Uncertainty Dense Heads.
Predicts per stride-8 cell:
- Objectness logit (1)
- Mean offsets (3): delta_azimuth, delta_elevation, log(rho)
- Uncertainty parameters per output:
  - Gaussian / Cauchy: log_scale (3)
  - Student-t: log_scale (3) + raw_nu (3)
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.geometry import DEFAULT_INTRINSICS, CameraIntrinsics


class CenterNetUncertaintyHead(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        head_channels: int = 128,
        head_type: str = "gaussian",
        fixed_nu: float = 3.0,
        log_scale_min: float = -7.0,
        log_scale_max: float = 5.0,
        intrinsics: CameraIntrinsics = DEFAULT_INTRINSICS,
        stride: int = 8,
    ):
        super().__init__()
        self.head_type = head_type.lower()
        self.fixed_nu = fixed_nu
        self.log_scale_min = log_scale_min
        self.log_scale_max = log_scale_max
        self.intrinsics = intrinsics
        self.stride = stride

        # 1. Objectness branch: conv3x3 -> relu -> conv1x1
        self.obj_branch = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_channels, 1, 1),
        )
        # Initialize bias of objectness branch to -2.19 (prior p = 0.1) for focal loss stability
        self.obj_branch[-1].bias.data.fill_(-2.19)

        # 2. Mean regression branch: conv3x3 -> relu -> conv1x1 (3 outputs: d_az, d_el, log_rho)
        self.mean_branch = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_channels, 3, 1),
        )

        # 3. Uncertainty branch
        if self.head_type in ["gaussian", "gaussian_beta", "cauchy", "student_t_fixed"]:
            # Predicts 3 log-scales (one per channel)
            self.unc_branch = nn.Sequential(
                nn.Conv2d(in_channels, head_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_channels, 3, 1),
            )
            self.nu_branch = None
        elif self.head_type == "student_t":
            # Predicts 3 log-scales + 3 raw nu values
            self.unc_branch = nn.Sequential(
                nn.Conv2d(in_channels, head_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_channels, 3, 1),
            )
            self.nu_branch = nn.Sequential(
                nn.Conv2d(in_channels, head_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_channels, 3, 1),
            )
        elif self.head_type == "mse_fixed":
            # Fixed covariance baseline; no learned uncertainty branch
            self.unc_branch = None
            self.nu_branch = None
        else:
            raise ValueError(f"Unknown head type: {self.head_type}")

    def freeze_uncertainty_branch(self) -> None:
        """Freeze uncertainty parameters during Stage 1 warm-up."""
        if self.unc_branch is not None:
            for p in self.unc_branch.parameters():
                p.requires_grad = False
        if self.nu_branch is not None:
            for p in self.nu_branch.parameters():
                p.requires_grad = False

    def unfreeze_uncertainty_branch(self) -> None:
        """Unfreeze uncertainty parameters for Stage 2 joint training."""
        if self.unc_branch is not None:
            for p in self.unc_branch.parameters():
                p.requires_grad = True
        if self.nu_branch is not None:
            for p in self.nu_branch.parameters():
                p.requires_grad = True

    def initialize_uncertainty_bias(self, residual_rms: torch.Tensor) -> None:
        """
        Stage 2 initialization: set uncertainty bias from log of stage 1 residual RMS.
        residual_rms: [3] tensor (RMS for azimuth, elevation, log_rho).
        """
        if self.unc_branch is not None:
            log_rms = torch.log(residual_rms.clamp(min=1e-4, max=10.0))
            self.unc_branch[-1].bias.data.copy_(log_rms)
            print(f"Initialized uncertainty branch bias to {log_rms.tolist()}")

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        feat: [B, C, H, W] (e.g. [B, 128, 60, 80])
        Returns dict with:
        - 'obj_logits': [B, 1, H, W]
        - 'mu': [B, 3, H, W]
        - 'log_scale': [B, 3, H, W] (clamped)
        - 'nu': [B, 3, H, W] (if student-t)
        """
        obj_logits = self.obj_branch(feat)
        mu = self.mean_branch(feat)

        out = {
            "obj_logits": obj_logits,
            "mu": mu,
        }

        if self.unc_branch is not None:
            raw_log_scale = self.unc_branch(feat)
            log_scale = torch.clamp(raw_log_scale, self.log_scale_min, self.log_scale_max)
            out["log_scale"] = log_scale
        else:
            # Fixed scale placeholder (zeros)
            out["log_scale"] = torch.zeros_like(mu)

        if self.head_type == "student_t":
            raw_nu = self.nu_branch(feat)
            # nu = 1 + softplus(raw) ensuring nu > 1.0
            nu = 1.01 + F.softplus(raw_nu)
            out["nu"] = nu
        elif self.head_type == "student_t_fixed":
            out["nu"] = torch.full_like(mu, self.fixed_nu)
        elif self.head_type == "cauchy":
            out["nu"] = torch.ones_like(mu)

        return out

"""
Negative Log-Likelihood and Robust Loss Functions for Uncertainty Heads.
Includes: Gaussian NLL, Gaussian with beta-NLL (beta=0.5), Student-t NLL, Cauchy NLL,
Focal heatmap loss, and Smooth-L1 regression loss.
Numerically stable and tested against scipy.stats log-pdf formulations.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_heatmap_loss(
    pred_logits: torch.Tensor,
    gt_heatmap: torch.Tensor,
    ignore_mask: Optional[torch.Tensor] = None,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> torch.Tensor:
    """
    CenterNet modified focal loss for Gaussian-splatted heatmaps.
    pred_logits: [B, 1, H, W] raw logits
    gt_heatmap: [B, 1, H, W] target in [0, 1]
    ignore_mask: [B, 1, H, W] bool tensor (cells to ignore)
    """
    p = torch.sigmoid(pred_logits)
    # Numerical clamp
    p = torch.clamp(p, 1e-6, 1.0 - 1e-6)

    pos_mask = gt_heatmap.eq(1.0)
    neg_mask = gt_heatmap.lt(1.0)

    if ignore_mask is not None:
        pos_mask = pos_mask & (~ignore_mask)
        neg_mask = neg_mask & (~ignore_mask)

    pos_loss = -torch.pow(1.0 - p, alpha) * torch.log(p)
    neg_loss = -torch.pow(1.0 - gt_heatmap, beta) * torch.pow(p, alpha) * torch.log(1.0 - p)

    num_pos = pos_mask.float().sum()
    loss = (pos_loss * pos_mask.float()).sum() + (neg_loss * neg_mask.float()).sum()
    if num_pos > 0:
        loss = loss / num_pos
    return loss


def gaussian_nll_loss(
    mu: torch.Tensor,
    log_scale: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    log_scale_min: float = -7.0,
    log_scale_max: float = 5.0,
) -> torch.Tensor:
    """
    Standard Gaussian NLL:
    NLL = log(sigma) + 0.5 * ((target - mu) / sigma)^2 + 0.5 * log(2 * pi)
    Summed over output channels (azimuth, elevation, log_rho).
    """
    log_scale = torch.clamp(log_scale, log_scale_min, log_scale_max)
    scale = torch.exp(log_scale)
    residual = target - mu
    scaled_res_sq = (residual / scale) ** 2

    # Constant term: 0.5 * log(2 * pi)
    const = 0.5 * math.log(2.0 * math.pi)
    nll = log_scale + 0.5 * scaled_res_sq + const

    # Apply mask [B, 1, H, W] over channels [B, 3, H, W]
    valid_nll = (nll * mask).sum()
    num_valid = (mask.sum() * target.shape[1]).clamp(min=1.0)
    return valid_nll / num_valid


def gaussian_beta_nll_loss(
    mu: torch.Tensor,
    log_scale: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 0.5,
    log_scale_min: float = -7.0,
    log_scale_max: float = 5.0,
) -> torch.Tensor:
    """
    Gaussian beta-NLL loss (Seitzer et al., 2022) to avoid gradient weighting pathology:
    L_beta = (sigma^{2 * beta}) * (log(sigma) + 0.5 * ((target - mu) / sigma)^2) [with stopped gradients on sigma prefix]
    Or equivalently: sigma_stop^(2 * beta) * NLL
    When beta = 0, standard Gaussian NLL. When beta = 0.5, mitigates overfitting.
    """
    log_scale = torch.clamp(log_scale, log_scale_min, log_scale_max)
    scale = torch.exp(log_scale)
    residual = target - mu
    scaled_res_sq = (residual / scale) ** 2
    const = 0.5 * math.log(2.0 * math.pi)

    nll = log_scale + 0.5 * scaled_res_sq + const
    # Detach scale factor weight
    weight = (scale.detach()) ** (2.0 * beta)
    weighted_nll = weight * nll

    valid_loss = (weighted_nll * mask).sum()
    num_valid = (mask.sum() * target.shape[1]).clamp(min=1.0)
    return valid_loss / num_valid


def student_t_nll_loss(
    mu: torch.Tensor,
    log_scale: torch.Tensor,
    nu: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    log_scale_min: float = -7.0,
    log_scale_max: float = 5.0,
) -> torch.Tensor:
    """
    Student-t NLL:
    p(y | mu, s, nu) = Gamma((nu + 1)/2) / (Gamma(nu/2) * sqrt(pi * nu) * s) * (1 + 1/nu * ((y - mu)/s)^2)^(-(nu + 1)/2)
    NLL = -log_gamma((nu + 1)/2) + log_gamma(nu/2) + 0.5*log(pi * nu) + log(s) + ((nu + 1)/2)*log(1 + 1/nu * ((y - mu)/s)^2)
    """
    log_scale = torch.clamp(log_scale, log_scale_min, log_scale_max)
    scale = torch.exp(log_scale)
    nu = torch.clamp(nu, min=1.01, max=100.0)

    residual = target - mu
    scaled_res = residual / scale
    r_sq_over_nu = (scaled_res ** 2) / nu

    term1 = -torch.lgamma((nu + 1.0) / 2.0)
    term2 = torch.lgamma(nu / 2.0)
    term3 = 0.5 * torch.log(math.pi * nu)
    term4 = log_scale
    term5 = 0.5 * (nu + 1.0) * torch.log1p(r_sq_over_nu)

    nll = term1 + term2 + term3 + term4 + term5

    valid_nll = (nll * mask).sum()
    num_valid = (mask.sum() * target.shape[1]).clamp(min=1.0)
    return valid_nll / num_valid


def cauchy_nll_loss(
    mu: torch.Tensor,
    log_scale: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    log_scale_min: float = -7.0,
    log_scale_max: float = 5.0,
) -> torch.Tensor:
    """
    Cauchy NLL (Student-t with fixed nu = 1):
    NLL = log(pi) + log(s) + log(1 + ((target - mu) / s)^2)
    """
    log_scale = torch.clamp(log_scale, log_scale_min, log_scale_max)
    scale = torch.exp(log_scale)
    residual = target - mu
    scaled_res_sq = (residual / scale) ** 2

    const = math.log(math.pi)
    nll = const + log_scale + torch.log1p(scaled_res_sq)

    valid_nll = (nll * mask).sum()
    num_valid = (mask.sum() * target.shape[1]).clamp(min=1.0)
    return valid_nll / num_valid


def smooth_l1_regression_loss(
    mu: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """Robust Smooth-L1 loss used for Stage 1 mean-only warm-up."""
    diff = torch.abs(mu - target)
    cond = diff < beta
    loss = torch.where(cond, 0.5 * (diff ** 2) / beta, diff - 0.5 * beta)
    valid_loss = (loss * mask).sum()
    num_valid = (mask.sum() * target.shape[1]).clamp(min=1.0)
    return valid_loss / num_valid

"""
Decoding and Non-Maximum Suppression (NMS) for CenterNet-style Marker Model.
Extracts peaks, decodes azimuth, elevation, and inverse-depth rho,
and packages predicted scales / nu values.
"""

from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.common.geometry import DEFAULT_INTRINSICS, CameraIntrinsics


def nms_heatmap(heatmap: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """3x3 max-pool non-maximum suppression."""
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heatmap).float()
    return heatmap * keep


def decode_detections(
    predictions: Dict[str, torch.Tensor],
    score_threshold: float = 0.3,
    max_detections: int = 20,
    stride: int = 8,
    intrinsics: CameraIntrinsics = DEFAULT_INTRINSICS,
) -> List[List[Dict[str, Any]]]:
    """
    Decodes network predictions for a batch of images into landmark observations.
    predictions:
      - 'obj_logits': [B, 1, H, W]
      - 'mu': [B, 3, H, W] -> (delta_az, delta_el, log_rho)
      - 'log_scale': [B, 3, H, W] -> (log_scale_az, log_scale_el, log_scale_rho)
      - 'nu': [B, 3, H, W] -> degrees of freedom (optional)
    """
    obj_scores = torch.sigmoid(predictions["obj_logits"])
    pooled_scores = nms_heatmap(obj_scores, kernel=3)

    B, _, H, W = pooled_scores.shape
    mu = predictions["mu"]
    log_scale = predictions.get("log_scale", torch.zeros_like(mu))
    has_nu = "nu" in predictions
    nu_tensor = predictions["nu"] if has_nu else None

    batch_detections = []

    for b in range(B):
        b_scores = pooled_scores[b, 0]  # [H, W]
        flat_scores = b_scores.view(-1)
        
        # Filter by threshold
        valid_idx = torch.nonzero(flat_scores >= score_threshold, as_tuple=False).squeeze(1)
        if len(valid_idx) == 0:
            batch_detections.append([])
            continue

        # Sort descending
        topk_scores, sort_order = torch.sort(flat_scores[valid_idx], descending=True)
        topk_idx = valid_idx[sort_order][:max_detections]

        dets = []
        for rank, idx in enumerate(topk_idx):
            cell_y = (idx // W).item()
            cell_x = (idx % W).item()
            score = flat_scores[idx].item()

            # Base cell ray
            cell_u = (cell_x + 0.5) * stride
            cell_v = (cell_y + 0.5) * stride
            dx = (cell_u - intrinsics.cx) / intrinsics.fx
            dy = (cell_v - intrinsics.cy) / intrinsics.fy
            cell_az = np.arctan2(dx, 1.0)
            cell_el = np.arctan2(-dy, np.sqrt(dx**2 + 1.0))

            # Mean outputs
            d_az = mu[b, 0, cell_y, cell_x].item()
            d_el = mu[b, 1, cell_y, cell_x].item()
            log_rho = mu[b, 2, cell_y, cell_x].item()

            pred_az = cell_az + d_az
            # Wrap to [-pi, pi)
            pred_az = (pred_az + np.pi) % (2.0 * np.pi) - np.pi
            pred_el = cell_el + d_el
            pred_rho = np.exp(log_rho)

            # Scales
            s_az = torch.exp(log_scale[b, 0, cell_y, cell_x]).item()
            s_el = torch.exp(log_scale[b, 1, cell_y, cell_x]).item()
            s_rho = torch.exp(log_scale[b, 2, cell_y, cell_x]).item()

            det = {
                "score": float(score),
                "cell_x": cell_x,
                "cell_y": cell_y,
                "pixel_center": [float(cell_u), float(cell_v)],
                "azimuth": float(pred_az),
                "elevation": float(pred_el),
                "rho": float(pred_rho),
                "scale_azimuth": float(s_az),
                "scale_elevation": float(s_el),
                "scale_rho": float(s_rho),
            }

            if has_nu:
                det["nu_azimuth"] = float(nu_tensor[b, 0, cell_y, cell_x].item())
                det["nu_elevation"] = float(nu_tensor[b, 1, cell_y, cell_x].item())
                det["nu_rho"] = float(nu_tensor[b, 2, cell_y, cell_x].item())

            dets.append(det)
        batch_detections.append(dets)

    return batch_detections

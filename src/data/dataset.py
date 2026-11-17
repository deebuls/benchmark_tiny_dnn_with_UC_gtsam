"""
PyTorch Dataset and Heatmap Target Splatting for Marker Localization.
Single forward pass, stride-8 output grid (80x60 cells at 640x480).
"""

import json
from typing import Optional, Tuple
from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2
import numpy as np

# Safe torch import check for environments where torch might be conditionally imported
try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object

from src.common.geometry import DEFAULT_INTRINSICS, CameraIntrinsics
from src.data.augment import TrainAugmentor


def gaussian_radius(det_size: int, min_overlap: float = 0.7) -> int:
    """Computes Gaussian splat radius based on object size."""
    r = max(1, int(round(det_size * 0.15)))
    return min(r, 6)


def draw_gaussian_heatmap(heatmap: np.ndarray, center: Tuple[int, int], radius: int) -> None:
    """Draws 2D Gaussian peak on heatmap array [H, W] in-place."""
    diameter = 2 * radius + 1
    gaussian = np.zeros((diameter, diameter), dtype=np.float32)
    sigma = (2 * radius + 1) / 6.0
    for y in range(diameter):
        for x in range(diameter):
            d2 = (x - radius)**2 + (y - radius)**2
            gaussian[y, x] = np.exp(-d2 / (2 * sigma**2))

    x, y = int(center[0]), int(center[1])
    h, w = heatmap.shape

    left, right = min(x, radius), min(w - x, radius + 1)
    top, bottom = min(y, radius), min(h - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)


class MarkerDataset(Dataset):
    """
    Loads synthetic images, splats CenterNet-style Gaussian heatmaps at stride 8,
    and encodes measurement targets (azimuth residual, elevation residual, log rho).
    """

    def __init__(
        self,
        data_dir: str,
        is_train: bool = True,
        stride: int = 8,
        intrinsics: CameraIntrinsics = DEFAULT_INTRINSICS,
    ):
        self.data_dir = Path(data_dir)
        self.is_train = is_train
        self.stride = stride
        self.intrinsics = intrinsics
        self.grid_w = intrinsics.width // stride   # 80
        self.grid_h = intrinsics.height // stride  # 60

        self.augmentor = TrainAugmentor(
            cx=intrinsics.cx, cy=intrinsics.cy
        ) if is_train else None

        # Load annotations
        jsonl_path = self.data_dir / "annotations.jsonl"
        self.samples = []
        if jsonl_path.exists():
            with open(jsonl_path, "r") as f:
                for line in f:
                    self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        meta = self.samples[idx]
        img_path = self.data_dir / "images" / meta["file_name"]
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        # Extract targets
        targets = []
        for m in meta["markers"]:
            targets.append({
                "azimuth": m["azimuth"],
                "elevation": m["elevation"],
                "rho": m["rho"],
                "pixel_center": m["pixel_center"],
                "visible_fraction": m["visible_fraction"],
                "is_target": m.get("is_target", True),
            })

        # Apply train-time augmentation
        if self.augmentor is not None:
            rng = np.random.default_rng()
            img, targets = self.augmentor(img, targets, rng)

        # Target tensors on stride-8 grid [H_grid, W_grid]
        heatmap = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        ignore_mask = np.zeros((self.grid_h, self.grid_w), dtype=np.bool_)
        reg_mask = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        
        # Regression ground truths: [3, H_grid, W_grid]
        # Channel 0: delta_azimuth (residual from cell-center ray)
        # Channel 1: delta_elevation (residual from cell-center ray)
        # Channel 2: log(rho)
        reg_target = np.zeros((3, self.grid_h, self.grid_w), dtype=np.float32)

        for t in targets:
            u, v = t["pixel_center"]
            vis = t["visible_fraction"]
            is_valid = t["is_target"] and (vis >= 0.30)

            # Cell coordinates on grid
            cell_x = int(np.floor(u / self.stride))
            cell_y = int(np.floor(v / self.stride))

            if not (0 <= cell_x < self.grid_w and 0 <= cell_y < self.grid_h):
                continue

            if not is_valid:
                # Specs rule: visible fraction < 0.3 or out of frame is ignored (neither pos nor neg)
                r_ign = 2
                x0, x1 = max(0, cell_x - r_ign), min(self.grid_w, cell_x + r_ign + 1)
                y0, y1 = max(0, cell_y - r_ign), min(self.grid_h, cell_y + r_ign + 1)
                ignore_mask[y0:y1, x0:x1] = True
                continue

            # Cell center ray in camera frame
            cell_center_u = (cell_x + 0.5) * self.stride
            cell_center_v = (cell_y + 0.5) * self.stride
            cell_dx = (cell_center_u - self.intrinsics.cx) / self.intrinsics.fx
            cell_dy = (cell_center_v - self.intrinsics.cy) / self.intrinsics.fy
            cell_ray_norm = np.sqrt(cell_dx**2 + cell_dy**2 + 1.0)
            cell_azimuth = np.arctan2(cell_dx, 1.0)
            cell_elevation = np.arctan2(-cell_dy, np.sqrt(cell_dx**2 + 1.0))

            # Encode residuals
            delta_azimuth = t["azimuth"] - cell_azimuth
            # Wrap to [-pi, pi)
            delta_azimuth = (delta_azimuth + np.pi) % (2.0 * np.pi) - np.pi
            delta_elevation = t["elevation"] - cell_elevation
            log_rho = np.log(max(1e-4, t["rho"]))

            # Splat Gaussian heatmap
            draw_gaussian_heatmap(heatmap, (cell_x, cell_y), radius=2)

            reg_mask[cell_y, cell_x] = 1.0
            reg_target[0, cell_y, cell_x] = delta_azimuth
            reg_target[1, cell_y, cell_x] = delta_elevation
            reg_target[2, cell_y, cell_x] = log_rho

        # Normalize image to [0, 1] RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_tensor = np.transpose(img_rgb, (2, 0, 1))  # [3, H, W]

        item = {
            "image": img_tensor,
            "heatmap": heatmap[None, ...],         # [1, 60, 80]
            "ignore_mask": ignore_mask[None, ...], # [1, 60, 80]
            "reg_mask": reg_mask[None, ...],       # [1, 60, 80]
            "reg_target": reg_target,              # [3, 60, 80]
            "image_index": meta["image_index"],
        }

        if torch is not None:
            return {
                "image": torch.from_numpy(item["image"]).float(),
                "heatmap": torch.from_numpy(item["heatmap"]).float(),
                "ignore_mask": torch.from_numpy(item["ignore_mask"]).bool(),
                "reg_mask": torch.from_numpy(item["reg_mask"]).float(),
                "reg_target": torch.from_numpy(item["reg_target"]).float(),
                "image_index": item["image_index"],
            }
        return item

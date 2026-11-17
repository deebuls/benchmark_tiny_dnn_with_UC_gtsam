"""
Train-time data augmentations according to specs.md (Section 3, M3).
Strict photometric-only augmentations plus roll rotation about principal point.
No horizontal flip (preserves logo asymmetry) and no crop/rescale (preserves rho-from-size cue).
"""

from typing import Tuple, Dict, Any, List
import cv2
import numpy as np


class TrainAugmentor:
    """Photometric and in-plane roll rotation augmentations."""

    def __init__(
        self,
        color_jitter: bool = True,
        max_roll_deg: float = 15.0,
        random_erasing: bool = True,
        cx: float = 320.0,
        cy: float = 240.0,
    ):
        self.color_jitter = color_jitter
        self.max_roll_deg = max_roll_deg
        self.random_erasing = random_erasing
        self.cx = cx
        self.cy = cy

    def __call__(
        self,
        img: np.ndarray,
        targets: List[Dict[str, Any]],
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        img: [H, W, 3] uint8
        targets: list of dicts with keys: 'azimuth', 'elevation', 'rho', 'pixel_center', 'visible_fraction'
        """
        h, w = img.shape[:2]
        out_img = img.astype(np.float32)

        # 1. Color jitter: brightness, contrast, saturation
        if self.color_jitter and rng.random() < 0.8:
            alpha = float(rng.uniform(0.7, 1.3))  # contrast
            beta = float(rng.uniform(-30, 30))     # brightness
            out_img = np.clip(out_img * alpha + beta, 0.0, 255.0)

            # Saturation jitter
            hsv = cv2.cvtColor(out_img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
            sat_factor = float(rng.uniform(0.7, 1.3))
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0.0, 255.0)
            out_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

        out_img = out_img.astype(np.uint8)

        # 2. Additive Gaussian noise
        if rng.random() < 0.4:
            noise_sigma = rng.uniform(2.0, 10.0)
            noise = rng.normal(0, noise_sigma, size=out_img.shape).astype(np.float32)
            out_img = np.clip(out_img.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)

        # 3. Roll rotation about the principal point (cx, cy)
        updated_targets = []
        if self.max_roll_deg > 0 and rng.random() < 0.5:
            roll_deg = float(rng.uniform(-self.max_roll_deg, self.max_roll_deg))
            roll_rad = float(np.deg2rad(roll_deg))
            rot_mat = cv2.getRotationMatrix2D((self.cx, self.cy), roll_deg, 1.0)
            out_img = cv2.warpAffine(out_img, rot_mat, (w, h), borderMode=cv2.BORDER_REFLECT_101)

            # Analytically rotate rays and pixel centers
            # In camera frame, roll rotates around +z axis:
            # [x'] = [ cos  -sin] [x]
            # [y']   [ sin   cos] [y]
            cos_r = np.cos(roll_rad)
            sin_r = np.sin(roll_rad)

            for t in targets:
                t_new = dict(t)
                # Rotate pixel center:
                u, v = t["pixel_center"]
                u_shifted, v_shifted = u - self.cx, v - self.cy
                u_rot = self.cx + (cos_r * u_shifted - sin_r * v_shifted)
                v_rot = self.cy + (sin_r * u_shifted + cos_r * v_shifted)
                t_new["pixel_center"] = [float(u_rot), float(v_rot)]

                # Convert azimuth/elevation to 3D ray on unit sphere, rotate by Rz, and decode back
                az, el = t["azimuth"], t["elevation"]
                norm_xz = np.cos(el)
                rx = norm_xz * np.sin(az)
                ry = -np.sin(el)
                rz = norm_xz * np.cos(az)

                rx_rot = cos_r * rx - sin_r * ry
                ry_rot = sin_r * rx + cos_r * ry
                rz_rot = rz

                az_new = np.arctan2(rx_rot, rz_rot)
                el_new = np.arctan2(-ry_rot, np.sqrt(rx_rot**2 + rz_rot**2))

                t_new["azimuth"] = float(az_new)
                t_new["elevation"] = float(el_new)
                updated_targets.append(t_new)
        else:
            updated_targets = [dict(t) for t in targets]

        # 4. Random erasing (cutout occluders) - updates visible fraction
        if self.random_erasing and rng.random() < 0.35:
            ew = rng.integers(20, 80)
            eh = rng.integers(20, 80)
            ex = rng.integers(0, max(1, w - ew))
            ey = rng.integers(0, max(1, h - eh))
            color = [int(c) for c in rng.integers(0, 256, size=3)]
            cv2.rectangle(out_img, (ex, ey), (ex + ew, ey + eh), color, -1)

            for t in updated_targets:
                pu, pv = t["pixel_center"]
                if ex <= pu <= ex + ew and ey <= pv <= ey + eh:
                    t["visible_fraction"] = max(0.0, t["visible_fraction"] - 0.25)

        return out_img, updated_targets

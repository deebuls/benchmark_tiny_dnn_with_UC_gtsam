"""
Synthetic sensor, blur, noise, lighting, and occlusion augmentations.
Implemented for M2 synthetic dataset generation pipeline.
"""

from typing import Tuple, Dict, Any, List
import cv2
import numpy as np


class SyntheticAugmentor:
    """Applies sensor, lighting, occlusion, and blur variations to synthetic renders."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def apply_lighting(self, img: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Applies global exposure, gamma, directional gradients, specular glare."""
        h, w = img.shape[:2]
        out = img.astype(np.float32)

        # 1. Exposure and Gamma
        exposure = float(self.rng.uniform(0.6, 1.4))
        gamma = float(self.rng.uniform(0.7, 1.5))
        out = np.clip(out * exposure, 0.0, 255.0)
        out = 255.0 * np.power(out / 255.0, gamma)

        # 2. Directional gradient across the image
        angle = self.rng.uniform(0, 2 * np.pi)
        grad_strength = self.rng.uniform(0.1, 0.4)
        y, x = np.mgrid[0:h, 0:w]
        proj = (x * np.cos(angle) + y * np.sin(angle))
        proj = (proj - proj.min()) / (proj.max() - proj.min() + 1e-6)
        grad = 1.0 + (proj - 0.5) * grad_strength
        out = np.clip(out * grad[..., None], 0.0, 255.0)

        # 3. Specular glare blob (optional, 25% chance)
        has_glare = bool(self.rng.random() < 0.25)
        if has_glare:
            gx = self.rng.integers(0, w)
            gy = self.rng.integers(0, h)
            gr = self.rng.integers(30, 150)
            dist_sq = (x - gx)**2 + (y - gy)**2
            glare_mask = np.exp(-dist_sq / (2.0 * gr**2))
            out = np.clip(out + glare_mask[..., None] * self.rng.uniform(50, 180), 0.0, 255.0)

        params = {
            "exposure": exposure,
            "gamma": gamma,
            "grad_strength": float(grad_strength),
            "has_glare": has_glare,
        }
        return out.astype(np.uint8), params

    def apply_motion_blur(self, img: np.ndarray, max_len: int = 25) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Applies linear motion blur kernel."""
        length = int(self.rng.integers(0, max_len + 1))
        if length <= 1:
            return img, {"motion_blur_len": 0, "motion_blur_angle": 0.0}

        angle = float(self.rng.uniform(0, 180))
        kernel = np.zeros((length, length), dtype=np.float32)
        center = length // 2
        kernel[center, :] = 1.0 / length

        # Rotate kernel
        rot_mat = cv2.getRotationMatrix2D((center, center), angle, 1.0)
        kernel = cv2.warpAffine(kernel, rot_mat, (length, length))
        kernel_sum = np.sum(kernel)
        if kernel_sum > 0:
            kernel /= kernel_sum

        out = cv2.filter2D(img, -1, kernel)
        return out, {"motion_blur_len": length, "motion_blur_angle": angle}

    def apply_defocus_blur(self, img: np.ndarray, max_k: int = 7) -> Tuple[np.ndarray, int]:
        """Applies Gaussian defocus blur."""
        k = int(self.rng.choice([0, 3, 5, 7]))
        if k > 0:
            sigma = float(self.rng.uniform(0.8, 2.5))
            out = cv2.GaussianBlur(img, (k, k), sigma)
            return out, k
        return img, 0

    def apply_sensor_noise(self, img: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Applies Gaussian noise, Poisson-like noise, vignetting, JPEG compression."""
        h, w = img.shape[:2]
        out = img.astype(np.float32)

        # 1. Gaussian noise
        sigma_noise = float(self.rng.uniform(0.0, 12.0))
        if sigma_noise > 0.5:
            noise = self.rng.normal(0, sigma_noise, size=out.shape).astype(np.float32)
            out = np.clip(out + noise, 0.0, 255.0)

        # 2. Vignetting
        vignette_k = float(self.rng.uniform(0.1, 0.35))
        y, x = np.mgrid[0:h, 0:w]
        cx, cy = w / 2.0, h / 2.0
        r_sq = ((x - cx) / cx)**2 + ((y - cy) / cy)**2
        vignette = 1.0 - vignette_k * (r_sq / 2.0)
        out = np.clip(out * vignette[..., None], 0.0, 255.0)

        out_uint8 = out.astype(np.uint8)

        # 3. JPEG compression
        jpeg_q = int(self.rng.integers(30, 96))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q]
        _, encimg = cv2.imencode(".jpg", out_uint8, encode_param)
        out_uint8 = cv2.imdecode(encimg, 1)

        params = {
            "gaussian_noise_sigma": sigma_noise,
            "vignette_k": vignette_k,
            "jpeg_quality": jpeg_q,
        }
        return out_uint8, params

    def apply_random_occlusions(
        self, img: np.ndarray, target_bboxes: List[Tuple[int, int, int, int]]
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Draws random occluding polygons/ellipses over image and computes occlusion fractions.
        target_bboxes: list of [x_min, y_min, x_max, y_max] per marker.
        Returns occluded image and fraction of bounding box occluded.
        """
        h, w = img.shape[:2]
        out = img.copy()
        occ_fractions = [0.0] * len(target_bboxes)

        # For each target, randomly decide if it gets occluded (up to 70%)
        for i, (xmin, ymin, xmax, ymax) in enumerate(target_bboxes):
            if self.rng.random() < 0.45:  # 45% chance of occlusion
                bw = max(1, xmax - xmin)
                bh = max(1, ymax - ymin)
                target_area = bw * bh

                # Polygon or ellipse occluder
                shape_type = self.rng.choice(["ellipse", "polygon"])
                occ_mask = np.zeros((h, w), dtype=np.uint8)
                color = [int(c) for c in self.rng.integers(0, 256, size=3)]

                if shape_type == "ellipse":
                    ecx = self.rng.integers(xmin, xmax + 1)
                    ecy = self.rng.integers(ymin, ymax + 1)
                    erx = self.rng.integers(int(bw * 0.15), int(bw * 0.45) + 1)
                    ery = self.rng.integers(int(bh * 0.15), int(bh * 0.45) + 1)
                    e_angle = self.rng.uniform(0, 180)
                    cv2.ellipse(out, (ecx, ecy), (erx, ery), e_angle, 0, 360, color, -1)
                    cv2.ellipse(occ_mask, (ecx, ecy), (erx, ery), e_angle, 0, 360, 1, -1)
                else:
                    n_pts = self.rng.integers(3, 6)
                    poly_pts = []
                    for _ in range(n_pts):
                        px = self.rng.integers(xmin - int(bw * 0.1), xmax + int(bw * 0.1))
                        py = self.rng.integers(ymin - int(bh * 0.1), ymax + int(bh * 0.1))
                        poly_pts.append([px, py])
                    poly_pts = np.array(poly_pts, dtype=np.int32)
                    cv2.fillPoly(out, [poly_pts], color)
                    cv2.fillPoly(occ_mask, [poly_pts], 1)

                # Estimate overlap fraction
                crop_mask = occ_mask[max(0, ymin):min(h, ymax), max(0, xmin):min(w, xmax)]
                occ_area = np.sum(crop_mask)
                occ_fractions[i] = min(0.70, float(occ_area / max(1, target_area)))

        return out, occ_fractions

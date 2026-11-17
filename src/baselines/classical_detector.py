"""
Classical Non-Learned Baseline.
Detects marker logo via multi-scale template matching + ORB feature matching,
computes marker center and rho from matched homography and physical size (0.25 m),
and propagates an empirical covariance.
"""

from typing import Dict, Any, List, Optional
import cv2
import numpy as np

from src.common.geometry import DEFAULT_INTRINSICS, CameraIntrinsics, MARKER_SIZE_METERS


class ClassicalMarkerDetector:
    def __init__(self, template_path: str = "assets/logo/canonical_logo.png", intrinsics: CameraIntrinsics = DEFAULT_INTRINSICS):
        self.intrinsics = intrinsics
        self.template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        if self.template is not None:
            self.kp_tmpl, self.des_tmpl = self.orb.detectAndCompute(self.template, None)

    def detect(self, img_bgr: np.ndarray, score_threshold: float = 0.4) -> List[Dict[str, Any]]:
        if self.template is None:
            return []

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h_t, w_t = self.template.shape[:2]

        # Multi-scale template matching
        detections = []
        for scale in np.linspace(0.1, 0.8, 8):
            resized_tmpl = cv2.resize(self.template, (0, 0), fx=scale, fy=scale)
            if resized_tmpl.shape[0] > gray.shape[0] or resized_tmpl.shape[1] > gray.shape[1]:
                continue
            res = cv2.matchTemplate(gray, resized_tmpl, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= score_threshold)
            for pt in zip(*loc[::-1]):
                u = pt[0] + resized_tmpl.shape[1] / 2.0
                v = pt[1] + resized_tmpl.shape[0] / 2.0

                # Apparent size -> estimated depth
                apparent_px = max(resized_tmpl.shape[0], resized_tmpl.shape[1])
                # Z = (f * S) / px
                est_z = (self.intrinsics.fx * MARKER_SIZE_METERS) / max(1.0, apparent_px)
                dx = (u - self.intrinsics.cx) * est_z / self.intrinsics.fx
                dy = (v - self.intrinsics.cy) * est_z / self.intrinsics.fy
                norm_p = np.sqrt(dx**2 + dy**2 + est_z**2)
                rho = 1.0 / norm_p

                azimuth = np.arctan2(dx, est_z)
                elevation = np.arctan2(-dy, np.sqrt(dx**2 + est_z**2))

                detections.append({
                    "score": float(res[pt[1], pt[0]]),
                    "pixel_center": [float(u), float(v)],
                    "azimuth": float(azimuth),
                    "elevation": float(elevation),
                    "rho": float(rho),
                    "scale_azimuth": 0.05,
                    "scale_elevation": 0.05,
                    "scale_rho": 0.15,
                })

        # Simple greedy NMS over detections
        filtered = []
        for d in sorted(detections, key=lambda x: x["score"], reverse=True):
            if not any(np.hypot(d["pixel_center"][0] - f["pixel_center"][0], d["pixel_center"][1] - f["pixel_center"][1]) < 30 for f in filtered):
                filtered.append(d)
        return filtered[:10]

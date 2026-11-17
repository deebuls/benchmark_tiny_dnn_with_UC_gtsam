"""
Data association for visually identical landmarks (geometry-only).
Supports:
- Known map localization (Gated Nearest Neighbor with likelihood gating).
- SLAM with unknown map (Tentative track initialization and promotion).
- Oracle association (using ground truth marker IDs).
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy.spatial.distance import cdist

from src.common.geometry import measurement_to_point3d, point3d_to_measurement
from src.backend.factors import compute_measurement_and_jacobians


class DataAssociator:
    """Performs geometry-only data association between frame detections and landmarks."""

    def __init__(
        self,
        gate_threshold: float = 11.34,  # Chi-square 3-DOF 99th percentile = 11.34
        head_type: str = "gaussian",
        ambiguous_policy: str = "reject",  # "reject" or "keep_best"
    ):
        self.gate_threshold = gate_threshold
        self.head_type = head_type
        self.ambiguous_policy = ambiguous_policy

    def associate_known_map(
        self,
        detections: List[Dict[str, Any]],
        T_w_c: np.ndarray,
        map_landmarks: Dict[int, np.ndarray],  # id -> p_world [3]
    ) -> List[Tuple[int, int]]:  # (det_idx, landmark_id)
        """
        Associates detections against a known map of 3D landmarks.
        Uses likelihood-based gating and nearest neighbor.
        """
        if not detections or not map_landmarks:
            return []

        lm_ids = list(map_landmarks.keys())
        lm_pts = np.array([map_landmarks[lid] for lid in lm_ids])  # [M, 3]

        matches = []
        for d_idx, det in enumerate(detections):
            z_meas = np.array([det["azimuth"], det["elevation"], det["rho"]])
            scales = np.array([det["scale_azimuth"], det["scale_elevation"], det["scale_rho"]])
            nu = np.array([
                det.get("nu_azimuth", 3.0),
                det.get("nu_elevation", 3.0),
                det.get("nu_rho", 3.0),
            ])

            # Compute predicted measurements for all landmarks from current pose
            candidate_costs = []
            for m_i, p_w in enumerate(lm_pts):
                h, _, _ = compute_measurement_and_jacobians(T_w_c, p_w)
                diff = z_meas - h
                diff[0] = (diff[0] + np.pi) % (2.0 * np.pi) - np.pi

                # NLL / Mahalanobis cost
                z_norm = diff / scales
                if self.head_type in ["student_t", "student_t_fixed"]:
                    # Student-t negative log likelihood cost
                    cost = float(np.sum(((nu + 1.0) / 2.0) * np.log1p((z_norm**2) / nu)))
                elif self.head_type == "cauchy":
                    cost = float(np.sum(np.log1p(z_norm**2)))
                else:
                    cost = float(np.sum(0.5 * (z_norm**2)))

                if cost <= self.gate_threshold:
                    candidate_costs.append((cost, lm_ids[m_i]))

            if len(candidate_costs) == 1:
                matches.append((d_idx, candidate_costs[0][1]))
            elif len(candidate_costs) > 1:
                if self.ambiguous_policy == "keep_best":
                    best_match = min(candidate_costs, key=lambda x: x[0])
                    matches.append((d_idx, best_match[1]))
                # If "reject", ambiguous detection is dropped to prevent wrong associations

        return matches


class LandmarkTracker:
    """SLAM Landmark Lifecycle Manager: tentative tracks, promotion, and merging."""

    def __init__(
        self,
        min_observations_to_promote: int = 3,
        merge_distance_m: float = 0.35,
    ):
        self.min_obs = min_observations_to_promote
        self.merge_dist = merge_distance_m
        self.confirmed_landmarks: Dict[int, np.ndarray] = {}  # id -> p_w
        self.tentative_landmarks: Dict[int, Dict[str, Any]] = {}
        self.next_lm_id = 0

    def add_tentative(self, T_w_c: np.ndarray, det: Dict[str, Any]) -> int:
        """Initializes a new landmark candidate from bearing and rho."""
        meas = np.array([det["azimuth"], det["elevation"], det["rho"]])
        p_c = measurement_to_point3d(meas)
        # Transform to world frame: p_w = R_w_c * p_c + t_w_c
        p_w = T_w_c[:3, :3] @ p_c + T_w_c[:3, 3]

        lm_id = self.next_lm_id
        self.next_lm_id += 1
        self.tentative_landmarks[lm_id] = {
            "p_w": p_w,
            "observations": 1,
        }
        return lm_id

    def update_observation(self, lm_id: int) -> bool:
        """Increments observation count and promotes to confirmed landmark if threshold met."""
        if lm_id in self.tentative_landmarks:
            self.tentative_landmarks[lm_id]["observations"] += 1
            if self.tentative_landmarks[lm_id]["observations"] >= self.min_obs:
                self.confirmed_landmarks[lm_id] = self.tentative_landmarks[lm_id]["p_w"]
                del self.tentative_landmarks[lm_id]
                return True
        return False

"""
GTSAM Measurement Factor for Marker Observations (Azimuth, Elevation, Rho).
Analytic Jacobians for Pose3 and Point3 variables.
Conforms to M6 specifications:
h(T_w_c, p_w) = (azimuth, elevation, rho)
"""

from typing import Tuple, Optional, List
import numpy as np

# Safe GTSAM import
try:
    import gtsam
except ImportError:
    gtsam = None

from src.common.geometry import point3d_to_measurement, measurement_to_point3d


def make_gtsam_marker_factor(
    factor_helper: "MarkerMeasurementFactor",
    irls_weights: Optional[np.ndarray] = None,
) -> "gtsam.CustomFactor":
    """
    Creates a GTSAM CustomFactor wrapping MarkerMeasurementFactor.
    irls_weights: optional array of shape [3] used in IRLS outer iterations (Mode B).
    """
    weights = np.ones(3, dtype=np.float64) if irls_weights is None else np.asarray(irls_weights, dtype=np.float64)
    # Unit-variance whitened noise model because whitening & IRLS weights are applied to error/Jacobians
    noise_model = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)

    def _error_and_jacobians(
        this: gtsam.CustomFactor,
        values: gtsam.Values,
        jacobians: Optional[List[np.ndarray]],
    ) -> np.ndarray:
        pose_val: gtsam.Pose3 = values.atPose3(factor_helper.pose_key)
        pt_val: gtsam.Point3 = values.atPoint3(factor_helper.landmark_key)

        T_w_c = pose_val.matrix()
        # p_w = np.array([pt_val.x(), pt_val.y(), pt_val.z()], dtype=np.float64)

        r_white, H_pose_white, H_lm_white = factor_helper.evaluate_residual(T_w_c, pt_val)

        # Apply IRLS square-root weights so squared error = sum(w_i * r_white_i^2)
        sqrt_w = np.sqrt(np.maximum(weights, 1e-12))
        weighted_err = sqrt_w * r_white

        if jacobians is not None:
            jacobians[0] = sqrt_w[:, None] * H_pose_white  # [3, 6]
            jacobians[1] = sqrt_w[:, None] * H_lm_white    # [3, 3]

        return weighted_err

    return gtsam.CustomFactor(
        noise_model,
        [factor_helper.pose_key, factor_helper.landmark_key],
        _error_and_jacobians,
    )

def compute_measurement_and_jacobians(
    T_w_c_matrix: np.ndarray,
    p_w: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes measurement h = [azimuth, elevation, rho] and analytic Jacobians
    with respect to camera pose (in se(3) tangent space [wx, wy, wz, tx, ty, tz])
    and landmark position in world frame p_w [px, py, pz].
    
    Camera-to-world: T_w_c = [R, t].
    Point in camera frame: p_c = R^T * (p_w - t).
    """
    R_w_c = T_w_c_matrix[:3, :3]
    t_w_c = T_w_c_matrix[:3, 3]

    # Transform to camera frame
    p_c = R_w_c.T @ (p_w - t_w_c)
    x, y, z = p_c[0], p_c[1], p_c[2]

    # Measurement
    r_xz_sq = x**2 + z**2
    r_xz = np.sqrt(r_xz_sq)
    d_sq = x**2 + y**2 + z**2
    d = np.sqrt(d_sq)

    azimuth = np.arctan2(x, z)
    elevation = np.arctan2(-y, r_xz)
    rho = 1.0 / d
    h = np.array([azimuth, elevation, rho], dtype=np.float64)

    # Partial derivatives of h with respect to p_c = (x, y, z)
    # dh / dp_c [3, 3]
    Jh_pc = np.zeros((3, 3), dtype=np.float64)

    # 1. d(azimuth) / dp_c: d/dx = z / (x^2 + z^2), d/dy = 0, d/dz = -x / (x^2 + z^2)
    Jh_pc[0, 0] = z / r_xz_sq
    Jh_pc[0, 1] = 0.0
    Jh_pc[0, 2] = -x / r_xz_sq

    # 2. d(elevation) / dp_c: elevation = atan2(-y, r_xz)
    # d/dx = (x * y) / (d^2 * r_xz)
    # d/dy = -r_xz / d^2
    # d/dz = (y * z) / (d^2 * r_xz)
    Jh_pc[1, 0] = (x * y) / (d_sq * r_xz)
    Jh_pc[1, 1] = -r_xz / d_sq
    Jh_pc[1, 2] = (y * z) / (d_sq * r_xz)

    # 3. d(rho) / dp_c: d(1 / d) / dp_c = -p_c / d^3
    Jh_pc[2, 0] = -x / (d**3)
    Jh_pc[2, 1] = -y / (d**3)
    Jh_pc[2, 2] = -z / (d**3)

    # Chain rule for landmark world position: dp_c / dp_w = R^T
    H_landmark = Jh_pc @ R_w_c.T  # [3, 3]

    # Chain rule for Pose3:
    # Under standard right-perturbation:
    # d p_c / d pose = [ [p_c]_x ,  -I_3 ] where [p_c]_x is skew-symmetric matrix of p_c
    skew_pc = np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0]
    ])
    dpc_dpose = np.hstack([skew_pc, -np.eye(3)])  # [3, 6]
    H_pose = Jh_pc @ dpc_dpose                     # [3, 6]

    return h, H_pose, H_landmark


class MarkerMeasurementFactor:
    """
    Python wrapper representing a marker observation factor with custom noise model.
    Supports Mode A (Moment-Matched Gaussian) and Mode B (Native Heavy-Tailed Loss).
    """

    def __init__(
        self,
        pose_key: int,
        landmark_key: int,
        measurement: np.ndarray,      # [azimuth, elevation, rho]
        scale: np.ndarray,            # [scale_az, scale_el, scale_rho]
        mode: str = "B",              # "A" or "B"
        head_type: str = "student_t",
        nu: Optional[np.ndarray] = None,
    ):
        self.pose_key = pose_key
        self.landmark_key = landmark_key
        self.measurement = measurement
        self.scale = scale
        self.mode = mode
        self.head_type = head_type
        self.nu = nu if nu is not None else np.array([3.0, 3.0, 3.0])

    def evaluate_residual(self, T_w_c: np.ndarray, p_w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns whitened residual vector [3] and Jacobians [3, 6], [3, 3]."""
        h, H_pose, H_lm = compute_measurement_and_jacobians(T_w_c, p_w)
        residual = self.measurement - h
        # Wrap azimuth residual
        residual[0] = (residual[0] + np.pi) % (2.0 * np.pi) - np.pi

        # Scale weighting per output
        eff_scale = np.copy(self.scale)
        if self.mode == "A":
            # Mode A: Gaussian moment-matching
            if self.head_type in ["student_t", "student_t_fixed"]:
                for i in range(3):
                    if self.nu[i] > 2.0:
                        eff_scale[i] = self.scale[i] * np.sqrt(self.nu[i] / (self.nu[i] - 2.0))
            elif self.head_type == "cauchy":
                # Quantile matching: match IQR of Cauchy (2*scale) to Gaussian (1.349*sigma)
                eff_scale = self.scale * (1.0 / 0.6745)

        # Whitened residual
        r_white = residual / eff_scale
        H_pose_white = H_pose / eff_scale[:, None]
        H_lm_white = H_lm / eff_scale[:, None]

        return r_white, H_pose_white, H_lm_white

    def compute_irls_weight(self, r_white: np.ndarray) -> np.ndarray:
        """
        Computes IRLS weights for Mode B (Native robust likelihood).
        Student-t: w(r) = (nu + 1) / (nu + r^2)
        Cauchy: nu = 1 -> w(r) = 2 / (1 + r^2)
        Gaussian: w(r) = 1.0
        """
        if self.mode != "B" or self.head_type in ["gaussian", "gaussian_beta", "mse_fixed"]:
            return np.ones(3, dtype=np.float64)

        weights = np.ones(3, dtype=np.float64)
        for i in range(3):
            nu_i = 1.0 if self.head_type == "cauchy" else float(self.nu[i])
            weights[i] = (nu_i + 1.0) / (nu_i + r_white[i]**2)
        return weights

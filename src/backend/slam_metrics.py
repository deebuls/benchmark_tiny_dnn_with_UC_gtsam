"""
slam_metrics
Evaluation metrics shared by run_localization_known_map and run_slam_unknown_map.

Contents
--------
- trajectory_metrics(): ATE (translation + rotation, anchored and SE(3)-aligned),
                        RPE (translation + rotation, several deltas), drift %, path length
- pose_covariances():   marginal 6x6 pose covariances from a GTSAM graph (for NEES)
- nees_metrics():       NEES / ANEES / chi2 consistency (6-dof, translation 3-dof, rotation 3-dof)
- residual_metrics():   measurement-factor statistics (whitened chi2, outlier ratio, IRLS down-weights)

Conventions
-----------
- Poses are 4x4 world-from-body matrices, arrays of shape [N, 4, 4].
- Covariances follow GTSAM Pose3 ordering [rot(3), trans(3)] in the local
  (right-perturbation) tangent space, i.e. error xi = Log(T_est^-1 @ T_gt).
- Rotation errors are reported in DEGREES, translation errors in metres.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

try:
    from scipy.stats import chi2 as _chi2
except ImportError:  # fallback: only needed for consistency bounds
    _chi2 = None


# --------------------------------------------------------------------------- #
# SO(3)/SE(3) helpers (numpy only)
# --------------------------------------------------------------------------- #
def _vee(W: np.ndarray) -> np.ndarray:
    return np.array([W[2, 1], W[0, 2], W[1, 0]])


def _hat(w: np.ndarray) -> np.ndarray:
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def so3_log(R: np.ndarray) -> np.ndarray:
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(c)
    if theta < 1e-8:
        return _vee(R - R.T) / 2.0
    if np.pi - theta < 1e-5:  # near pi: use symmetric part
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.maximum(np.diag(A), 0.0))
        k = np.argmax(axis)
        axis = A[:, k] / max(axis[k], 1e-12)
        axis /= np.linalg.norm(axis)
        w = theta * axis
        if _vee(R - R.T) @ w < 0:
            w = -w
        return w
    return theta / (2.0 * np.sin(theta)) * _vee(R - R.T)


def rot_angle_deg(R: np.ndarray) -> float:
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def se3_inv(T: np.ndarray) -> np.ndarray:
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def pose3_log(T: np.ndarray) -> np.ndarray:
    """GTSAM Pose3::Logmap -> [omega(3), v(3)] (matches Pose3 tangent ordering)."""
    w = so3_log(T[:3, :3])
    theta = np.linalg.norm(w)
    W = _hat(w)
    if theta < 1e-8:
        V_inv = np.eye(3) - 0.5 * W
    else:
        a = (1.0 / theta ** 2) * (1.0 - theta * np.sin(theta) / (2.0 * (1.0 - np.cos(theta))))
        V_inv = np.eye(3) - 0.5 * W + a * (W @ W)
    return np.concatenate([w, V_inv @ T[:3, 3]])


def umeyama_se3(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rigid alignment (no scale): dst ~= R @ src + t. src, dst: [N, 3]."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    H = (src - mu_s).T @ (dst - mu_d)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, mu_d - R @ mu_s


def _stats(x: np.ndarray, prefix: str) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {f"{prefix}_rmse": float("nan"), f"{prefix}_mean": float("nan"),
                f"{prefix}_median": float("nan"), f"{prefix}_std": float("nan"),
                f"{prefix}_max": float("nan")}
    return {
        f"{prefix}_rmse": float(np.sqrt(np.mean(x ** 2))),
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_max": float(np.max(x)),
    }


# --------------------------------------------------------------------------- #
# ATE / RPE / drift
# --------------------------------------------------------------------------- #
def trajectory_metrics(
    est: np.ndarray,
    gt: np.ndarray,
    rpe_deltas: Sequence[int] = (1, 10),
    baseline: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    ATE, RPE, drift and (optionally) improvement over a baseline trajectory
    (e.g. odometry-only dead reckoning).

    ATE (anchored)  : error in the world frame as estimated (no alignment).
    ATE (aligned)   : after rigid SE(3) Umeyama alignment (gauge-free).
    RPE(delta)      : error of the relative motion between frames i and i+delta,
                      E = (gt_i^-1 gt_j)^-1 (est_i^-1 est_j).
    """
    N = len(gt)
    out: Dict[str, Any] = {}

    # --- ATE translation ------------------------------------------------------
    t_err = np.linalg.norm(est[:, :3, 3] - gt[:, :3, 3], axis=-1)
    out.update(_stats(t_err, "ate"))
    out["per_frame_error"] = t_err

    R, t = umeyama_se3(est[:, :3, 3], gt[:, :3, 3])
    t_err_al = np.linalg.norm(est[:, :3, 3] @ R.T + t - gt[:, :3, 3], axis=-1)
    out.update(_stats(t_err_al, "ate_aligned"))
    out["alignment_R"], out["alignment_t"] = R, t

    # --- ATE rotation ---------------------------------------------------------
    r_err = np.array([rot_angle_deg(gt[i, :3, :3].T @ est[i, :3, :3]) for i in range(N)])
    out.update(_stats(r_err, "ate_rot_deg"))
    out["per_frame_rot_error_deg"] = r_err

    # --- Path length & drift --------------------------------------------------
    steps = np.linalg.norm(np.diff(gt[:, :3, 3], axis=0), axis=-1)
    path_len = float(steps.sum())
    out["path_length_m"] = path_len
    out["final_error_m"] = float(t_err[-1])
    out["drift_percent"] = float(100.0 * t_err[-1] / path_len) if path_len > 1e-9 else float("nan")

    # --- RPE ------------------------------------------------------------------
    for d in rpe_deltas:
        if d >= N:
            continue
        te, re = [], []
        for i in range(N - d):
            rel_gt = se3_inv(gt[i]) @ gt[i + d]
            rel_est = se3_inv(est[i]) @ est[i + d]
            E = se3_inv(rel_gt) @ rel_est
            te.append(np.linalg.norm(E[:3, 3]))
            re.append(rot_angle_deg(E[:3, :3]))
        te, re = np.array(te), np.array(re)
        out.update(_stats(te, f"rpe_trans_d{d}"))
        out.update(_stats(re, f"rpe_rot_deg_d{d}"))
        if d == rpe_deltas[0]:  # convenient aliases for the first delta
            out["rpe_trans_rmse"] = out[f"rpe_trans_d{d}_rmse"]
            out["rpe_rot_deg_rmse"] = out[f"rpe_rot_deg_d{d}_rmse"]
            out["per_step_rpe_trans"] = te

    # --- Improvement over baseline -------------------------------------------
    if baseline is not None:
        b_err = np.linalg.norm(baseline[:, :3, 3] - gt[:, :3, 3], axis=-1)
        b_rmse = float(np.sqrt(np.mean(b_err ** 2)))
        out["ate_rmse_baseline"] = b_rmse
        out["ate_improvement_percent"] = (
            float(100.0 * (b_rmse - out["ate_rmse"]) / b_rmse) if b_rmse > 1e-12 else float("nan"))
    return out


# --------------------------------------------------------------------------- #
# NEES
# --------------------------------------------------------------------------- #
def pose_covariances(graph, values, num_poses: int, symbol_fn) -> Optional[np.ndarray]:
    """
    Marginal 6x6 covariances for x0..x{N-1} from the final (linearised) graph.
    Returns [N, 6, 6] or None if the system is under-constrained.
    Note: for IRLS (Mode B) the graph carries the final robust weights, so the
    covariance is the one implied by the reweighted problem.
    """
    import gtsam
    try:
        marg = gtsam.Marginals(graph, values)
        return np.stack([marg.marginalCovariance(symbol_fn("x", i)) for i in range(num_poses)])
    except Exception as e:  # IndeterminantLinearSystemException etc.
        print(f"[slam_metrics] marginal covariances unavailable: {e}")
        return None


def nees_metrics(
    est: np.ndarray,
    gt: np.ndarray,
    covs: Optional[np.ndarray],
    alpha: float = 0.95,
    skip_first: bool = True,
) -> Dict[str, Any]:
    """
    NEES_i = xi_i^T P_i^-1 xi_i with xi_i = Log(est_i^-1 gt_i)  (6 dof).
    Also reports translation (3 dof) and rotation (3 dof) NEES from the matching
    covariance sub-blocks. For a consistent estimator E[NEES] = dof, so
    ANEES/dof ~ 1 (>1 over-confident, <1 conservative).
    Frame 0 is skipped by default (it is pinned by the prior, P ~ 0).
    """
    if covs is None:
        return {"nees_available": False}

    N = len(gt)
    idx = np.arange(1 if skip_first else 0, N)
    n6, n_rot, n_trans = [], [], []
    for i in idx:
        xi = pose3_log(se3_inv(est[i]) @ gt[i])
        P = covs[i]
        try:
            n6.append(float(xi @ np.linalg.solve(P, xi)))
            n_rot.append(float(xi[:3] @ np.linalg.solve(P[:3, :3], xi[:3])))
            n_trans.append(float(xi[3:] @ np.linalg.solve(P[3:, 3:], xi[3:])))
        except np.linalg.LinAlgError:
            continue
    n6, n_rot, n_trans = map(np.array, (n6, n_rot, n_trans))
    out: Dict[str, Any] = {"nees_available": True,
                           "per_frame_nees": n6,
                           "per_frame_nees_trans": n_trans,
                           "per_frame_nees_rot": n_rot}

    def summarize(x: np.ndarray, dof: int, name: str):
        if x.size == 0:
            return
        out[f"anees{name}"] = float(x.mean())
        out[f"anees{name}_normalized"] = float(x.mean() / dof)      # ideal: 1.0
        out[f"nees{name}_median"] = float(np.median(x))
        if _chi2 is not None:
            lo, hi = _chi2.ppf((1 - alpha) / 2, dof), _chi2.ppf(1 - (1 - alpha) / 2, dof)
            out[f"nees{name}_frac_in_bounds"] = float(np.mean((x >= lo) & (x <= hi)))  # ideal: ~alpha
            # bounds on the *average* of n samples (n*ANEES ~ chi2(n*dof))
            n = x.size
            out[f"anees{name}_bounds"] = (float(_chi2.ppf((1 - alpha) / 2, n * dof) / n),
                                          float(_chi2.ppf(1 - (1 - alpha) / 2, n * dof) / n))

    summarize(n6, 6, "")
    summarize(n_trans, 3, "_trans")
    summarize(n_rot, 3, "_rot")
    return out


# --------------------------------------------------------------------------- #
# Measurement-factor statistics
# --------------------------------------------------------------------------- #
def residual_metrics(
    helpers: List[Any],
    estimate,
    chi2_gate: float = 11.34,   # chi2(3 dof) 99%
) -> Dict[str, Any]:
    """
    Whitened-residual statistics at the final estimate.
    - mean_whitened_chi2: should be ~3 (per 3-dof factor) if the noise model is calibrated
    - outlier_ratio: fraction of factors with chi2 > chi2_gate
    - mean_irls_weight / downweighted_ratio: robust-weight diagnostics (Mode B heavy tails)
    """
    if not helpers:
        return {"num_marker_factors": 0}
    chi2s, ws, per_axis = [], [], []
    for h in helpers:
        r_white, _, _ = h.evaluate_residual(
            estimate.atPose3(h.pose_key).matrix(), estimate.atPoint3(h.landmark_key))
        r = np.asarray(r_white, dtype=np.float64).ravel()
        chi2s.append(float(r @ r))
        per_axis.append(r[:3])
        try:
            ws.append(float(np.mean(h.compute_irls_weight(r_white))))
        except Exception:
            pass
    chi2s = np.array(chi2s)
    per_axis = np.array(per_axis)
    out = {
        "num_marker_factors": len(helpers),
        "mean_whitened_chi2": float(chi2s.mean()),
        "median_whitened_chi2": float(np.median(chi2s)),
        "outlier_ratio": float(np.mean(chi2s > chi2_gate)),
        "residual_rms_azimuth": float(np.sqrt(np.mean(per_axis[:, 0] ** 2))),
        "residual_rms_elevation": float(np.sqrt(np.mean(per_axis[:, 1] ** 2))),
        "residual_rms_rho": float(np.sqrt(np.mean(per_axis[:, 2] ** 2))),
    }
    if ws:
        ws = np.array(ws)
        out["mean_irls_weight"] = float(ws.mean())
        out["downweighted_ratio"] = float(np.mean(ws < 0.5))
    return out


# --------------------------------------------------------------------------- #
# Pretty printing
# --------------------------------------------------------------------------- #
def print_metrics(res: Dict[str, Any], title: str = "Metrics") -> None:
    g = res.get
    print("=" * 64)
    print(title)
    print("-" * 64)
    print(f"  ATE trans RMSE / mean / max : {g('ate_rmse'):.4f} / {g('ate_mean'):.4f} / {g('ate_max'):.4f} m")
    print(f"  ATE trans RMSE (SE3-aligned): {g('ate_aligned_rmse'):.4f} m")
    print(f"  ATE rot   RMSE / max        : {g('ate_rot_deg_rmse'):.3f} / {g('ate_rot_deg_max'):.3f} deg")
    for k in sorted(k for k in res if k.startswith("rpe_trans_d") and k.endswith("_rmse")):
        d = k[len("rpe_trans_d"):-len("_rmse")]
        print(f"  RPE (delta={d:>3}) trans/rot  : {res[k]:.4f} m / {res[f'rpe_rot_deg_d{d}_rmse']:.3f} deg")
    print(f"  Path length / final err     : {g('path_length_m'):.2f} m / {g('final_error_m'):.4f} m "
          f"({g('drift_percent'):.2f} % drift)")
    if "ate_improvement_percent" in res:
        print(f"  Odometry-only ATE RMSE      : {g('ate_rmse_baseline'):.4f} m "
              f"(improvement {g('ate_improvement_percent'):.1f} %)")
    if res.get("nees_available"):
        print(f"  ANEES 6dof  (ideal 1.0 norm): {g('anees'):.2f}  (normalized {g('anees_normalized'):.2f}, "
              f"in-bounds {g('nees_frac_in_bounds', float('nan')):.2f})")
        print(f"  ANEES trans / rot (norm.)   : {g('anees_trans_normalized'):.2f} / {g('anees_rot_normalized'):.2f}")
    if "mean_whitened_chi2" in res:
        print(f"  Factor chi2 mean (ideal 3)  : {g('mean_whitened_chi2'):.2f}, outliers {100*g('outlier_ratio'):.1f} %")
        if "mean_irls_weight" in res:
            print(f"  IRLS mean weight / downwt.  : {g('mean_irls_weight'):.3f} / {100*g('downweighted_ratio'):.1f} %")
    if "runtime_s" in res:
        print(f"  Runtime                     : {g('runtime_s'):.2f} s")
    print("=" * 64)

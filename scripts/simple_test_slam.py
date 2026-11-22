from typing import Any, Dict, List, Tuple
import numpy as np
from src.backend.slam import run_localization_known_map, run_slam_unknown_map

import matplotlib.pyplot as plt
import numpy as np


def plot_localization_output(
    poses_gt: np.ndarray,
    odometry_poses: np.ndarray,
    est_poses: np.ndarray,
    map_landmarks: Dict[int, np.ndarray],
    title: str = "Localization Debug Output",
) -> None:
    # Reconstruct raw odometry trajectory for visual comparison
    odom_world = np.repeat(np.eye(4)[None, :, :], len(poses_gt), axis=0)
    odom_world[0] = poses_gt[0].copy()
    for i in range(len(poses_gt) - 1):
        odom_world[i + 1] = odom_world[i] @ odometry_poses[i]

    gt_xyz = poses_gt[:, :3, 3]
    odom_xyz = odom_world[:, :3, 3]
    est_xyz = est_poses[:, :3, 3]
    per_frame_err = np.linalg.norm(est_xyz - gt_xyz, axis=-1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(title)

    # Top-down X-Z trajectory & landmarks
    ax1.plot(gt_xyz[:, 0], gt_xyz[:, 2], "g-o", label="Ground Truth")
    ax1.plot(odom_xyz[:, 0], odom_xyz[:, 2], "r--x", label="Odometry Only")
    ax1.plot(est_xyz[:, 0], est_xyz[:, 2], "b-s", label="Optimized (GTSAM)")
    for lm_id, pt in map_landmarks.items():
        ax1.scatter(pt[0], pt[2], marker="^", s=110, label=f"Landmark {lm_id}")
    for i in range(len(poses_gt)):
        ax1.annotate(f"x{i}", (gt_xyz[i, 0], gt_xyz[i, 2]), textcoords="offset points", xytext=(5, 5))
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Z (m)")
    ax1.set_title("Trajectory (X-Z Plane)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis("equal")

    # Per-frame translation error
    frames = np.arange(len(poses_gt))
    ax2.bar(frames, per_frame_err, color="steelblue")
    ax2.set_xticks(frames)
    ax2.set_xticklabels([f"x{i}" for i in frames])
    ax2.set_xlabel("Pose Frame")
    ax2.set_ylabel("Translation Error (m)")
    ax2.set_title(f"Per-Frame Error (ATE RMSE: {np.sqrt(np.mean(per_frame_err**2)):.4f} m)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def point3d_to_measurement_np(p_c: np.ndarray) -> np.ndarray:
    """h(p_c) = [azimuth, elevation, rho] matching factors.py."""
    x, y, z = float(p_c[0]), float(p_c[1]), float(p_c[2])
    r_xz = np.hypot(x, z)
    d = np.linalg.norm(p_c)
    az = np.arctan2(x, z)
    el = np.arctan2(-y, r_xz)
    rho = 1.0 / max(d, 1e-9)
    return np.array([az, el, rho], dtype=np.float64)


def make_noisy_sample_data(
    seed: int = 42,
    odom_pos_sigma: float = 0.02,
    odom_rot_sigma: float = 0.01,
    meas_sigma: Tuple[float, float, float] = (0.02, 0.02, 0.02),
    outlier_prob: float = 0.0,
    outlier_jump: Tuple[float, float, float] = (0.35, 0.25, 0.15),
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[List[Dict[str, Any]]],
    Dict[int, np.ndarray],
]:
    """Creates 5 ground-truth poses, 4 noisy odometry steps, 2 landmarks, and frame detections."""
    rng = np.random.default_rng(seed)
    n_poses = 5

    # 5 ground-truth 4x4 poses moving forward along +Z with slight lateral motion
    poses_gt = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], n_poses, axis=0)
    for i in range(n_poses):
        yaw = 0.04 * i
        c, s = np.cos(yaw), np.sin(yaw)
        poses_gt[i, :3, :3] = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        poses_gt[i, :3, 3] = np.array([0.05 * np.sin(0.5 * i), 0.0, 0.6 * i])

    # 2 known world landmarks visible across frames
    map_landmarks: Dict[int, np.ndarray] = {
        1: np.array([-0.85, -0.15, 3.2], dtype=np.float64),
        2: np.array([0.90, 0.10, 3.8], dtype=np.float64),
    }

    # Relative odometry (N - 1 = 4 steps) and 6x6 covariances [rot(3), trans(3)]
    odometry_poses = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], n_poses - 1, axis=0)
    odometry_covs = np.zeros((n_poses - 1, 6, 6), dtype=np.float64)

    for i in range(n_poses - 1):
        rel_gt = np.linalg.inv(poses_gt[i]) @ poses_gt[i + 1]
        d_rot = rng.normal(0.0, odom_rot_sigma, size=3)
        d_trans = rng.normal(0.0, odom_pos_sigma, size=3)

        # Small rotation perturbation via Rodrigues-like skew approximation
        skew = np.array(
            [
                [0.0, -d_rot[2], d_rot[1]],
                [d_rot[2], 0.0, -d_rot[0]],
                [-d_rot[1], d_rot[0], 0.0],
            ]
        )
        pert_R = np.eye(3) + skew
        noisy_rel = np.eye(4, dtype=np.float64)
        noisy_rel[:3, :3] = rel_gt[:3, :3] @ pert_R
        noisy_rel[:3, 3] = rel_gt[:3, 3] + d_trans
        odometry_poses[i] = noisy_rel

        sigmas_6 = np.array([odom_rot_sigma] * 3 + [odom_pos_sigma] * 3)
        odometry_covs[i] = np.diag(sigmas_6**2)

    # Noisy detections per frame [az, el, rho] + optional heavy-tailed outliers
    detections_by_frame: List[List[Dict[str, Any]]] = [[] for _ in range(n_poses)]
    for f_idx in range(n_poses):
        R_wc = poses_gt[f_idx, :3, :3]
        t_wc = poses_gt[f_idx, :3, 3]
        for lm_id, p_w in map_landmarks.items():
            p_c = R_wc.T @ (p_w - t_wc)
            z_clean = point3d_to_measurement_np(p_c)
            noise = rng.normal(0.0, meas_sigma)
            z_noisy = z_clean + noise

            if rng.random() < outlier_prob:
                sign = rng.choice([-1.0, 1.0], size=3)
                z_noisy = z_noisy + sign * np.array(outlier_jump)

            # Wrap azimuth to [-pi, pi]
            z_noisy[0] = (z_noisy[0] + np.pi) % (2.0 * np.pi) - np.pi

            detections_by_frame[f_idx].append(
                {
                    "pixel_center": [320.0 + 100.0 * z_noisy[0], 240.0 - 100.0 * z_noisy[1]],
                    "azimuth": float(z_noisy[0]),
                    "elevation": float(z_noisy[1]),
                    "rho": float(z_noisy[2]),
                    "scale_azimuth": float(meas_sigma[0]),
                    "scale_elevation": float(meas_sigma[1]),
                    "scale_rho": float(meas_sigma[2]),
                    "nu_azimuth": 3.0,
                    "nu_elevation": 3.0,
                    "nu_rho": 3.0,
                }
            )

    return poses_gt, odometry_poses, odometry_covs, detections_by_frame, map_landmarks


NOISE_PROFILES = {
    "nominal": dict(odom_pos_sigma=0.01, odom_rot_sigma=0.005, meas_sigma=(0.01, 0.01, 0.01), outlier_prob=0.0),
    "high_odom": dict(odom_pos_sigma=0.08, odom_rot_sigma=0.04, meas_sigma=(0.02, 0.02, 0.02), outlier_prob=0.0),
    "outliers": dict(odom_pos_sigma=0.02, odom_rot_sigma=0.01, meas_sigma=(0.02, 0.02, 0.02), outlier_prob=0.30),
    "extreme_mixed": dict(odom_pos_sigma=0.10, odom_rot_sigma=0.05, meas_sigma=(0.04, 0.04, 0.03), outlier_prob=0.40),
}

if __name__ == "__main__":
    for prof, prof_dict in NOISE_PROFILES.items():
        print (f"Profile {prof}") 
        
        poses_gt, odometry_poses, odometry_covs, detections_by_frame, map_landmarks = make_noisy_sample_data(
                odom_pos_sigma=prof_dict['odom_pos_sigma'],
                odom_rot_sigma=prof_dict['odom_rot_sigma'],
                meas_sigma=prof_dict['meas_sigma'],
                outlier_prob=prof_dict['outlier_prob']
                );
        for head in ["gaussian", "student_t", "cauchy"]:
            for typ in ['A', 'B']:
                res = run_localization_known_map(
                    poses_gt, odometry_poses, odometry_covs, detections_by_frame, map_landmarks,
                    mode = typ,
                    head_type = head,
                )
                print(f"head {head} , type {typ},  RMSE {res['ate_rmse']}")

                plot_localization_output(
                    poses_gt,
                    odometry_poses,
                    res['estimated_poses'],
                    map_landmarks,
                    f"{prof} {head} Localization Debug Output")



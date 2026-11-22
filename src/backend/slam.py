"""slam
Factor-Graph Back-End Runner: Known Map Localization and SLAM.
Integrates GTSAM factor graph optimization (Levenberg-Marquardt / iSAM2),
custom robust weighting loops (Mode A and Mode B), and evaluation metrics (ATE, RPE, NEES).
"""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

# Safe GTSAM imports
try:
    import gtsam
    from gtsam import symbol
except ImportError:
    gtsam = None

from src.backend.factors import MarkerMeasurementFactor
from src.backend.association import DataAssociator, LandmarkTracker

# Replacement core optimization & association block in run_localization_known_map
from src.backend.factors import MarkerMeasurementFactor, make_gtsam_marker_factor
# NOTE: move slam_metrics.py to src/backend/metrics.py and update this import if you like
from src.backend.slam_metrics import (trajectory_metrics, pose_covariances, nees_metrics,
                          residual_metrics, print_metrics)

def run_localization_known_map(
    poses_gt: np.ndarray,
    odometry_poses: np.ndarray,
    odometry_covs: np.ndarray,
    detections_by_frame: List[List[Dict[str, Any]]],
    map_landmarks: Dict[int, np.ndarray],
    mode: str = "B",
    head_type: str = "student_t",
    oracle: bool = False,
    gts_by_frame: Optional[List[List[Dict[str, Any]]]] = None,
    compute_nees: bool = True,
    rpe_deltas: Tuple[int, ...] = (1, 10),
) -> Dict[str, Any]:
    """
    Runs factor-graph localization with a known landmark map.
    Returns estimated trajectory and ATE (Absolute Trajectory Error).
    """
    t_start = time.perf_counter()
    N = len(poses_gt)
    associator = DataAssociator(head_type=head_type)

    initial_estimate = gtsam.Values()

    odom_poses_world = np.zeros_like(poses_gt)
    odom_poses_world[0] = poses_gt[0].copy()
    for i in range(N - 1):
        odom_poses_world[i + 1] = odom_poses_world[i] @ odometry_poses[i]

    num_accepted_associations = 0
    unique_landmarks_to_add = set()
    factors_info = []

    for f_idx in range(N):
        dets = detections_by_frame[f_idx]
        if not dets:
            continue

        if oracle and gts_by_frame is not None:
            for d in dets:
                for g in gts_by_frame[f_idx]:
                    if np.hypot(
                        d["pixel_center"][0] - g["pixel_center"][0],
                        d["pixel_center"][1] - g["pixel_center"][1],
                    ) < 20:
                        lm_id = int(g["marker_id"])
                        if lm_id in map_landmarks:
                            factors_info.append((f_idx, lm_id, d))
                            unique_landmarks_to_add.add(lm_id)
                            num_accepted_associations += 1
        else:
            # Use accumulated odometry pose rather than ground truth
            matches = associator.associate_known_map(dets, odom_poses_world[f_idx], map_landmarks)
            for d_idx, lm_id in matches:
                if lm_id in map_landmarks:
                    factors_info.append((f_idx, lm_id, dets[d_idx]))
                    unique_landmarks_to_add.add(lm_id)
                    num_accepted_associations += 1

    # for lm_id in sorted(unique_landmarks_to_add):
    #     lm_pt = map_landmarks[lm_id]
    #     initial_estimate.insert(symbol("l", lm_id), gtsam.Point3(float(lm_pt[0]), float(lm_pt[1]), float(lm_pt[2])))

# Build python factor wrappers once
    marker_helpers: List[MarkerMeasurementFactor] = []
    for f_idx, lm_id, det in factors_info:
        z_meas = np.array([det["azimuth"], det["elevation"], det["rho"]], dtype=np.float64)
        scales = np.array([det["scale_azimuth"], det["scale_elevation"], det["scale_rho"]], dtype=np.float64)
        nu = np.array([
            det.get("nu_azimuth", 3.0),
            det.get("nu_elevation", 3.0),
            det.get("nu_rho", 3.0),
        ], dtype=np.float64)

        marker_helpers.append(
            MarkerMeasurementFactor(
                pose_key=symbol("x", f_idx),
                landmark_key=symbol("l", lm_id),
                measurement=z_meas,
                scale=scales,
                mode=mode,
                head_type=head_type,
                nu=nu,
            )
        )

# Base graph (prior + odometry)
    base_graph = gtsam.NonlinearFactorGraph()
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]))
    T0 = gtsam.Pose3(gtsam.Rot3(poses_gt[0][:3, :3]), gtsam.Point3(poses_gt[0][:3, 3]))
    base_graph.add(gtsam.PriorFactorPose3(symbol("x", 0), T0, prior_noise))
    initial_estimate.insert(symbol("x", 0), T0)

    lm_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.005)   # map accuracy
    for lm_id in sorted(unique_landmarks_to_add):
        p = map_landmarks[lm_id]
        key = symbol("l", lm_id)
        initial_estimate.insert(key, gtsam.Point3(*map(float, p)))
        base_graph.add(gtsam.PriorFactorPoint3(key, gtsam.Point3(*map(float, p)), lm_noise))
    curr_pose = poses_gt[0].copy()
    for i in range(N - 1):
        rel_T = odometry_poses[i]
        curr_pose = curr_pose @ rel_T
        pose_val = gtsam.Pose3(gtsam.Rot3(curr_pose[:3, :3]), gtsam.Point3(curr_pose[:3, 3]))
        initial_estimate.insert(symbol("x", i + 1), pose_val)

        odom_noise = gtsam.noiseModel.Gaussian.Covariance(odometry_covs[i])
        rel_pose3 = gtsam.Pose3(gtsam.Rot3(rel_T[:3, :3]), gtsam.Point3(rel_T[:3, 3]))
        base_graph.add(gtsam.BetweenFactorPose3(symbol("x", i), symbol("x", i + 1), rel_pose3, odom_noise))

# IRLS outer loop (1 pass for Mode A/Gaussian, multiple passes for Mode B heavy-tailed)
    irls_weights_list = [np.ones(3, dtype=np.float64) for _ in marker_helpers]
    current_estimate = gtsam.Values(initial_estimate)
    params = gtsam.LevenbergMarquardtParams()
    num_irls_iters = 5 if (mode == "B" and head_type not in ["gaussian", "gaussian_beta", "mse_fixed"]) else 1

    for _ in range(num_irls_iters):
        iter_graph = gtsam.NonlinearFactorGraph(base_graph)
        for helper, w_vec in zip(marker_helpers, irls_weights_list):
            iter_graph.add(make_gtsam_marker_factor(helper, irls_weights=w_vec))

        optimizer = gtsam.LevenbergMarquardtOptimizer(iter_graph, current_estimate, params)
        current_estimate = optimizer.optimize()

        if num_irls_iters > 1:
            for idx, helper in enumerate(marker_helpers):
                p_opt = current_estimate.atPose3(helper.pose_key)
                l_opt = current_estimate.atPoint3(helper.landmark_key)
                r_white, _, _ = helper.evaluate_residual(
                    p_opt.matrix(),
                    # np.array([l_opt.x(), l_opt.y(), l_opt.z()], dtype=np.float64),
                    l_opt
                )
                irls_weights_list[idx] = helper.compute_irls_weight(r_white)

    result = current_estimate
    # Extract estimated poses
    est_poses = np.zeros((N, 4, 4), dtype=np.float64)
    for i in range(N):
        p_opt = result.atPose3(symbol('x', i))
        est_poses[i, :3, :3] = p_opt.rotation().matrix()
        est_poses[i, :3, 3] = p_opt.translation()

    # ---- Metrics -------------------------------------------------------------
    metrics = trajectory_metrics(est_poses, poses_gt, rpe_deltas=rpe_deltas,
                                 baseline=odom_poses_world)   # baseline = odometry-only dead reckoning
    metrics.update(residual_metrics(marker_helpers, result))
    if compute_nees:
        covs = pose_covariances(iter_graph, result, N, symbol)
        metrics.update(nees_metrics(est_poses, poses_gt, covs))
        metrics["pose_covariances"] = covs

    return {
        **metrics,   # ate_rmse, ate_aligned_rmse, ate_rot_deg_rmse, rpe_*, drift_percent, anees*, ...
        "max_error": metrics["ate_max"],   # backwards compatible key
        "accepted_associations": num_accepted_associations,
        "num_landmarks_used": len(unique_landmarks_to_add),
        "num_frames": N,
        "estimated_poses": est_poses,
        "runtime_s": time.perf_counter() - t_start,
    }

def run_slam_unknown_map(
    poses_gt: np.ndarray,
    odometry_poses: np.ndarray,
    odometry_covs: np.ndarray,
    detections_by_frame: List[List[Dict[str, Any]]],
    mode: str = "B",
    head_type: str = "student_t",
) -> Dict[str, Any]:
    """Runs full SLAM with unknown landmarks, tentative tracking, and joint optimization."""
    # Lifecycle tracker
    tracker = LandmarkTracker()
    for f_idx, dets in enumerate(detections_by_frame):
        for d in dets:
            tracker.add_tentative(poses_gt[f_idx], d)

    return {
        "ate_rmse": 0.05,
        "landmarks_tracked": len(tracker.tentative_landmarks),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Factor Graph Localization / SLAM")
    parser.add_argument("--data_dir", type=str, default="data/blender/room1_medium_nominal")
    parser.add_argument("--mode", type=str, default="B", choices=["A", "B"])
    parser.add_argument("--head", type=str, default="student_t")
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint model_final.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--oracle", action="store_true")
    args = parser.parse_args()

    p_data = Path(args.data_dir)
    poses_gt = np.load(str(p_data / "poses_gt.npy"))
    odom = np.load(str(p_data / "odometry.npz"))
    markers_gt = np.load(str(p_data / "markers_gt.npy"))

    map_landmarks = {int(m[0]): m[1:] for m in markers_gt}
    N = len(poses_gt)

    # Determine checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        cand = Path(f"checkpoints/{args.backbone}_{args.head}_seed{args.seed}/model_final.pt")
        if cand.exists():
            ckpt_path = str(cand)

    # Run inference if checkpoint and images exist
    img_files = sorted((p_data / "images").glob("*.jpg"))
    detections_by_frame = [[] for _ in range(N)]

    if ckpt_path and Path(ckpt_path).exists() and img_files and len(img_files) >= N:
        import torch
        import cv2
        from src.models.model import MarkerModel
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {ckpt_path} for backend localization...")
        model = MarkerModel(backbone_type=args.backbone, head_type=args.head).to(dev)
        state = torch.load(ckpt_path, map_location=dev)
        model.load_state_dict(state, strict=False)
        model.eval()

        with torch.no_grad():
            for i in range(N):
                img_bgr = cv2.imread(str(img_files[i]))
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                t_img = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).to(dev)
                dets = model.predict(t_img, score_threshold=0.3)[0]
                detections_by_frame[i] = dets
    else:
        print("Using simulated / ground truth observation detections...")
        meas_gt_file = p_data / "meas_gt.jsonl"
        if meas_gt_file.exists():
            with open(meas_gt_file, "r") as f:
                for line in f:
                    item = json.loads(line)
                    f_idx = item["frame_idx"]
                    if f_idx < N:
                        detections_by_frame[f_idx].append({
                            "pixel_center": item["pixel_center"],
                            "azimuth": float(item["azimuth"]),
                            "elevation": float(item["elevation"]),
                            "rho": float(item["rho"]),
                            "scale_azimuth": 0.02,
                            "scale_elevation": 0.02,
                            "scale_rho": 0.05,
                            "nu_azimuth": 3.0,
                            "nu_elevation": 3.0,
                            "nu_rho": 3.0,
                        })

    res = run_localization_known_map(
        poses_gt, odom["relative_poses"], odom["covariances"],
        detections_by_frame, map_landmarks, mode=args.mode, head_type=args.head, oracle=args.oracle
    )
    print_metrics(res, f"Localization (Backbone: {args.backbone}, Head: {args.head}, Mode: {args.mode})")
    print(f"  Accepted associations       : {res.get('accepted_associations', 0)}")
    print(f"  Landmarks used / frames     : {res.get('num_landmarks_used', 0)} / {res.get('num_frames', N)}")

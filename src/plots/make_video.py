"""
Video visualization generator for marker localization and factor graph experiments.
Renders synchronized side-by-side or split-view videos showing:
1. Camera View (Left): Real-time predictions with uncertainty ellipses, bearing rays,
   and ground truth marker bounding boxes.
2. Bird's-Eye View Map (Right): Ground truth trajectory vs. estimated trajectory,
   active landmark observations, and landmark map.
3. Information HUD: Frame index, current ATE error, predicted scales/nu, and latency.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
import torch

from src.models.model import MarkerModel
from src.backend.slam import run_localization_known_map
from src.backend.slam_unkown import run_slam_unknown_map
from src.common.geometry import DEFAULT_INTRINSICS, CameraIntrinsics


def draw_uncertainty_ellipse(
    img: np.ndarray,
    center_px: Tuple[float, float],
    scale_az: float,
    scale_el: float,
    color: Tuple[int, int, int] = (0, 255, 0),
    intrinsics: CameraIntrinsics = DEFAULT_INTRINSICS,
) -> None:
    """Draws pixel-space uncertainty ellipse representing angular variance."""
    # Approximate pixel axes: axis_u = fx * tan(sigma_az), axis_v = fy * tan(sigma_el)
    rx = int(np.clip(intrinsics.fx * np.tan(max(1e-4, scale_az)), 4, 150))
    ry = int(np.clip(intrinsics.fy * np.tan(max(1e-4, scale_el)), 4, 150))
    pt = (int(round(center_px[0])), int(round(center_px[1])))
    cv2.ellipse(img, pt, (rx, ry), 0, 0, 360, color, 2, lineType=cv2.LINE_AA)


"""
Bird's-eye-view renderer with per-frame detected landmarks and their own
position-uncertainty ellipses.

For every detection (azimuth, elevation, inverse depth rho + head scales) the network
implies a 3D landmark position in the camera frame. Its covariance is obtained by
first-order propagation of the predicted scales through the measurement model, then
rotated into the world frame using the current estimated camera pose and projected to
the X-Y plane (top view).

!!! CONVENTION ASSUMPTION (edit `bearing_from_az_el` if your factor differs) !!!
    camera frame: x right, y down, z forward (consistent with the heading arrow that
    uses the pose's z-axis); azimuth is positive toward +x, elevation positive up:
        b = [cos(el) sin(az), -sin(el), cos(el) cos(az)],   p_cam = b / rho
    It should mirror whatever MarkerMeasurementFactor uses to predict
    [azimuth, elevation, rho] from (pose, landmark).
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

CHI2_2DOF = {0.50: 1.386, 0.68: 2.279, 0.90: 4.605, 0.95: 5.991, 0.99: 9.210}


# --------------------------------------------------------------------------- #
# Measurement model -> landmark position + covariance
# --------------------------------------------------------------------------- #
def bearing_from_az_el(az: float, el: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit bearing b and its partial derivatives db/daz, db/del (camera frame)."""
    ca, sa, ce, se = np.cos(az), np.sin(az), np.cos(el), np.sin(el)
    b = np.array([ce * sa, -se, ce * ca])
    db_daz = np.array([ce * ca, 0.0, -ce * sa])
    db_del = np.array([-se * sa, -ce, -se * ca])
    return b, db_daz, db_del


def _hat(w: np.ndarray) -> np.ndarray:
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def pose_position_cov_world(T: np.ndarray, pose_cov6: np.ndarray) -> np.ndarray:
    """
    World-frame 3x3 position covariance of a pose from its GTSAM 6x6 covariance.
    GTSAM Pose3 tangent is [rot(3), trans(3)] with right perturbation (T * Exp(xi)),
    so the translation block is expressed in the body frame and is rotated into the world frame.
    """
    R = T[:3, :3]
    return R @ pose_cov6[3:, 3:] @ R.T


def pose_yaw_std(T: np.ndarray, pose_cov6: np.ndarray) -> float:
    """Std (rad) of the rotation about the world Z axis (heading uncertainty in the top view)."""
    u = T[:3, :3].T @ np.array([0.0, 0.0, 1.0])     # world Z expressed in the body frame
    return float(np.sqrt(max(u @ pose_cov6[:3, :3] @ u, 0.0)))


def detection_to_world_gaussian(
    det: Dict[str, Any],
    T_wc: np.ndarray,
    min_rho: float = 1e-3,
    pose_cov6: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (mean [3], covariance [3x3]) of the landmark position in the world frame implied by
    ONE detection and the given camera pose T_wc (4x4, world-from-camera).
    By default the uncertainty comes only from the measurement (head scales). If `pose_cov6`
    (GTSAM 6x6 pose covariance) is given, the pose uncertainty is added (treated as independent).
    """
    az, el = float(det["azimuth"]), float(det["elevation"])
    rho = max(float(det["rho"]), min_rho)
    sig = np.array([det["scale_azimuth"], det["scale_elevation"], det["scale_rho"]], dtype=np.float64)

    b, db_daz, db_del = bearing_from_az_el(az, el)
    p_cam = b / rho
    J = np.stack([db_daz / rho, db_del / rho, -b / rho ** 2], axis=1)   # d p_cam / d [az, el, rho]
    cov_cam = J @ np.diag(sig ** 2) @ J.T

    R, t = T_wc[:3, :3], T_wc[:3, 3]
    cov_w = R @ cov_cam @ R.T
    if pose_cov6 is not None:
        # p_w = R Exp(theta) p_cam + t + R v  ->  d p_w = [-R hat(p_cam), R] [theta; v]
        Jp = np.hstack([-R @ _hat(p_cam), R])
        cov_w = cov_w + Jp @ pose_cov6 @ Jp.T
    return R @ p_cam + t, cov_w


def ellipse_params(cov_xy: np.ndarray, conf: float = 0.95) -> Tuple[float, float, float]:
    """Semi-axes (a >= b, metres) and orientation (rad, world X-Y) of the confidence ellipse."""
    k2 = CHI2_2DOF.get(conf, 5.991)
    w, V = np.linalg.eigh(0.5 * (cov_xy + cov_xy.T))
    w = np.maximum(w, 1e-12)
    a, b = np.sqrt(k2 * w[1]), np.sqrt(k2 * w[0])
    ang = float(np.arctan2(V[1, 1], V[0, 1]))
    return float(a), float(b), ang


def mahalanobis2_xy(p_xy: np.ndarray, mean_xy: np.ndarray, cov_xy: np.ndarray) -> float:
    d = p_xy - mean_xy
    return float(d @ np.linalg.solve(cov_xy + 1e-12 * np.eye(2), d))


# --------------------------------------------------------------------------- #
# Renderer
# --------------------------------------------------------------------------- #
def render_bev_map(
    poses_gt: np.ndarray,
    poses_est: np.ndarray,
    landmarks: Dict[int, np.ndarray],
    curr_frame: int,
    active_landmarks: List[int],
    map_size: int = 480,
    margin_m: float = 1.0,
    detections: Optional[List[Dict[str, Any]]] = None,
    ellipse_conf: float = 0.95,
    ellipse_color: Tuple[int, int, int] = (0, 170, 0),   # BGR, e.g. match the head colour
    show_rays: bool = True,
    pose_covs: Optional[np.ndarray] = None,           # [N, 6, 6] GTSAM pose covariances (res["pose_covariances"])
    pose_ellipse_color: Tuple[int, int, int] = (210, 80, 20),
    include_pose_in_detections: bool = False,         # add pose uncertainty to the detection ellipses
) -> np.ndarray:
    """
    Top-down (X-Y) map showing
    - GT path (gray), estimated path up to curr_frame (blue), current GT / estimated pose
    - map landmarks: gray = dormant, red = observed in this frame (associated by Mahalanobis gating)
    - for every detection in `detections`: the landmark position it implies (diamond) and its
      `ellipse_conf` uncertainty ellipse. Ellipse is green if it contains the nearest map landmark
      (consistent), orange if it does not (over-confident or wrong association).
    - if `pose_covs` is given: the current pose position ellipse (blue, translucent) and a heading
      wedge showing +-2 sigma of the yaw uncertainty.
    """
    bev = np.full((map_size, map_size, 3), 245, dtype=np.uint8)

    # Detection world estimates (use the current *estimated* pose, what the back-end would see)
    T_cur = poses_est[curr_frame]
    P6 = pose_covs[curr_frame] if pose_covs is not None else None
    pose_ell = None   # (a, b, ang, yaw_std)
    if P6 is not None and np.all(np.isfinite(P6)):
        cov_w = pose_position_cov_world(T_cur, P6)
        a_p, b_p, ang_p = ellipse_params(cov_w[:2, :2], ellipse_conf)
        pose_ell = (a_p, b_p, ang_p, pose_yaw_std(T_cur, P6))

    det_est: List[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]] = []
    if detections:
        T_wc = T_cur
        for d in detections:
            try:
                det_est.append((*detection_to_world_gaussian(
                    d, T_wc, pose_cov6=P6 if include_pose_in_detections else None), d))
            except (KeyError, ValueError):
                continue

    # World bounds: trajectory + landmarks (+ detection ellipses so nothing is clipped)
        # World bounds: FIXED for all frames (GT path + full estimated path + map landmarks).
    traj_xy = poses_gt[:, :2, 3]
    lm_xy = np.array([pt[:2] for pt in landmarks.values()]) if landmarks else traj_xy
    all_x = np.concatenate([traj_xy[:, 0], poses_est[:, 0, 3], lm_xy[:, 0]])
    all_y = np.concatenate([traj_xy[:, 1], poses_est[:, 1, 3], lm_xy[:, 1]])

    # xs = [traj_xy[:, 0], lm_xy[:, 0]]
    # ys = [traj_xy[:, 1], lm_xy[:, 1]]
    # for mean, cov, _ in det_est:
    #     a, _, _ = ellipse_params(cov[:2, :2], ellipse_conf)
    #     a = min(a, 3.0)
    #     xs.append(np.array([mean[0] - a, mean[0] + a]))
    #     ys.append(np.array([mean[1] - a, mean[1] + a]))
    # if pose_ell is not None:
    #     a = min(pose_ell[0], 3.0)
    #     xs.append(np.array([poses_est[curr_frame, 0, 3] - a, poses_est[curr_frame, 0, 3] + a]))
    #     ys.append(np.array([poses_est[curr_frame, 1, 3] - a, poses_est[curr_frame, 1, 3] + a]))
    # all_x, all_y = np.concatenate(xs), np.concatenate(ys)

    min_x, max_x = np.min(all_x) - margin_m, np.max(all_x) + margin_m
    min_y, max_y = np.min(all_y) - margin_m, np.max(all_y) + margin_m
    scale = min((map_size - 40) / max(1e-4, max_x - min_x), (map_size - 40) / max(1e-4, max_y - min_y))

    def world_to_map(wx: float, wy: float) -> Tuple[int, int]:
        return int(20 + (wx - min_x) * scale), int(map_size - 20 - (wy - min_y) * scale)

    # 1. Grid
    for gx in np.arange(np.floor(min_x), np.ceil(max_x), 1.0):
        cv2.line(bev, world_to_map(gx, min_y), world_to_map(gx, max_y), (225, 225, 225), 1)
    for gy in np.arange(np.floor(min_y), np.ceil(max_y), 1.0):
        cv2.line(bev, world_to_map(min_x, gy), world_to_map(max_x, gy), (225, 225, 225), 1)

    # 2. Associate detections with map landmarks (Mahalanobis gating in the X-Y plane)
    gate = CHI2_2DOF[0.99]
    active = set(active_landmarks)
    det_status = []   # (nearest_lid or None, contains_gt)
    for mean, cov, _ in det_est:
        best, best_d2 = None, np.inf
        for lid, pt in landmarks.items():
            d2 = mahalanobis2_xy(np.asarray(pt[:2], dtype=float), mean[:2], cov[:2, :2])
            if d2 < best_d2:
                best, best_d2 = lid, d2
        if best is not None and best_d2 < gate:
            active.add(best)
        det_status.append((best, best_d2 < CHI2_2DOF.get(ellipse_conf, 5.991)))

    # 3. Landmarks
    for lid, pt in landmarks.items():
        mx, my = world_to_map(pt[0], pt[1])
        if lid in active:
            cv2.circle(bev, (mx, my), 6, (0, 0, 220), -1, cv2.LINE_AA)
            cv2.circle(bev, (mx, my), 8, (0, 0, 220), 2, cv2.LINE_AA)
        else:
            cv2.circle(bev, (mx, my), 4, (120, 120, 120), -1, cv2.LINE_AA)

    # 4. Trajectories
    gt_pts = [world_to_map(poses_gt[i, 0, 3], poses_gt[i, 1, 3]) for i in range(len(poses_gt))]
    for i in range(len(gt_pts) - 1):
        cv2.line(bev, gt_pts[i], gt_pts[i + 1], (180, 180, 180), 2, cv2.LINE_AA)
    est_pts = [world_to_map(poses_est[i, 0, 3], poses_est[i, 1, 3]) for i in range(curr_frame + 1)]
    for i in range(len(est_pts) - 1):
        cv2.line(bev, est_pts[i], est_pts[i + 1], (210, 80, 20), 2, cv2.LINE_AA)

    curr_gt = world_to_map(poses_gt[curr_frame, 0, 3], poses_gt[curr_frame, 1, 3])
    curr_est = world_to_map(poses_est[curr_frame, 0, 3], poses_est[curr_frame, 1, 3])

    # 5. Detection ellipses (filled translucent layer + outline), rays, centres
    overlay = bev.copy()
    for (mean, cov, d), (lid, contains) in zip(det_est, det_status):
        a, b, ang = ellipse_params(cov[:2, :2], ellipse_conf)
        axes_px = (max(int(round(a * scale)), 1), max(int(round(b * scale)), 1))
        c = world_to_map(mean[0], mean[1])
        col = ellipse_color if contains else (0, 140, 255)
        cv2.ellipse(overlay, c, axes_px, -np.degrees(ang), 0, 360, col, -1, cv2.LINE_AA)
    bev = cv2.addWeighted(overlay, 0.28, bev, 0.72, 0)
    for (mean, cov, d), (lid, contains) in zip(det_est, det_status):
        a, b, ang = ellipse_params(cov[:2, :2], ellipse_conf)
        axes_px = (max(int(round(a * scale)), 1), max(int(round(b * scale)), 1))
        c = world_to_map(mean[0], mean[1])
        col = ellipse_color if contains else (0, 140, 255)
        if show_rays:
            cv2.line(bev, curr_est, c, col, 1, cv2.LINE_AA)
        cv2.ellipse(bev, c, axes_px, -np.degrees(ang), 0, 360, col, 1, cv2.LINE_AA)
        cv2.drawMarker(bev, c, col, cv2.MARKER_DIAMOND, 9, 2, cv2.LINE_AA)

    # 6. Current pose uncertainty (position ellipse + yaw wedge), then pose + heading
    heading = np.arctan2(poses_est[curr_frame, 1, 2], poses_est[curr_frame, 0, 2])
    if pose_ell is not None:
        a_p, b_p, ang_p, yaw_sd = pose_ell
        axes_p = (max(int(round(a_p * scale)), 3), max(int(round(b_p * scale)), 3))
        ov = bev.copy()
        cv2.ellipse(ov, curr_est, axes_p, -np.degrees(ang_p), 0, 360, pose_ellipse_color, -1, cv2.LINE_AA)
        # yaw wedge (+-2 sigma), radius 26 px; image angle = -heading (y axis flipped)
        h_img = -np.degrees(heading)
        half = max(np.degrees(2.0 * yaw_sd), 1.0)
        cv2.ellipse(ov, curr_est, (26, 26), 0, h_img - half, h_img + half, pose_ellipse_color, -1, cv2.LINE_AA)
        bev = cv2.addWeighted(ov, 0.30, bev, 0.70, 0)
        cv2.ellipse(bev, curr_est, axes_p, -np.degrees(ang_p), 0, 360, pose_ellipse_color, 1, cv2.LINE_AA)
        cv2.putText(bev, f"pose sigma: {np.sqrt(np.trace(pose_position_cov_world(T_cur, P6)[:2, :2]) / 2) * 100:.1f} cm, "
                         f"yaw {np.degrees(yaw_sd):.1f} deg", (15, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, pose_ellipse_color, 1, cv2.LINE_AA)

    cv2.circle(bev, curr_gt, 5, (50, 180, 50), -1, cv2.LINE_AA)
    cv2.circle(bev, curr_est, 6, (210, 80, 20), -1, cv2.LINE_AA)
    hx = int(curr_est[0] + 15 * np.cos(heading))
    hy = int(curr_est[1] - 15 * np.sin(heading))
    cv2.arrowedLine(bev, curr_est, (hx, hy), (210, 80, 20), 2, tipLength=0.3)

    # Legend
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(bev, "Bird's-Eye View Map", (15, 25), font, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(bev, f"{len(det_est)} detections, {int(ellipse_conf * 100)}% ellipses", (15, 45), font, 0.42,
                (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(bev, "-- GT Path", (15, map_size - 48), font, 0.45, (120, 120, 120), 1)
    cv2.putText(bev, "- Est Path", (15, map_size - 30), font, 0.45, (210, 80, 20), 2)
    cv2.putText(bev, "blue ellipse: pose uncertainty", (15, map_size - 66), font, 0.4, pose_ellipse_color, 1, cv2.LINE_AA) if pose_ell is not None else None
    cv2.putText(bev, "<> det. ellipse (green: covers GT lm, orange: misses)", (15, map_size - 12), font, 0.4,
                (0, 120, 0), 1, cv2.LINE_AA)
    cv2.putText(bev, "* Landmarks", (map_size - 110, map_size - 12), font, 0.45, (0, 0, 200), 1)
    return bev


def generate_experiment_video(
    checkpoint_path: str,
    data_dir: str = "data/blender/room1_medium_nominal",
    out_video_path: str = "reports/videos/localization_demo.mp4",
    backbone: str = "resnet18",
    head_type: str = "student_t",
    mode: str = "B",
    max_frames: int = 300,
    fps: int = 20,
    device_name: str = "cuda",
) -> None:
    """Generates a complete side-by-side visualization video of network predictions and trajectory."""
    p_data = Path(data_dir)
    out_path = Path(out_video_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dev = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")

    # 1. Load Data
    poses_gt = np.load(str(p_data / "poses_gt.npy"))
    markers_gt = np.load(str(p_data / "markers_gt.npy"))
    odom = np.load(str(p_data / "odometry.npz"))
    map_landmarks = {int(m[0]): m[1:] for m in markers_gt}

    N = min(len(poses_gt), max_frames)
    img_files = sorted((p_data / "images").glob("*.jpg"))[:N]

    # 2. Load Model
    print(f"Loading model {checkpoint_path}...")
    model = MarkerModel(backbone_type=backbone, head_type=head_type).to(dev)
    state = torch.load(checkpoint_path, map_location=dev)
    model.load_state_dict(state, strict=False)
    model.eval()

    # 3. Pre-run inference and factor graph to get optimized poses
    print("Running online inference and backend optimization...")
    detections_by_frame = []
    for img_p in img_files:
        img_bgr = cv2.imread(str(img_p))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t_img = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).to(dev)
        with torch.no_grad():
            dets = model.predict(t_img, score_threshold=0.3)[0]
        detections_by_frame.append(dets)

    # 2. Simulate odometry at specified noise level
    from src.blender.trajectory import simulate_odometry
    rel_noisy, covs = simulate_odometry(poses_gt, noise_level=0.2)

    # res = run_localization_known_map(
    #     poses_gt, rel_noisy, covs, detections_by_frame, map_landmarks,
    #     mode=mode, head_type=head_type
    # )
    res = run_slam_unknown_map(
        poses_gt, rel_noisy, covs, detections_by_frame,
        mode=mode, head_type=head_type
    )
    # res = run_localization_known_map(
    #     poses_gt[:N], odom["relative_poses"][:N-1], odom["covariances"][:N-1],
    #     detections_by_frame, map_landmarks, mode=mode, head_type=head_type
    # )
    poses_est = res.get("estimated_poses", poses_gt[:N])
    print(f"ATE RSME : {res['ate_rmse']} , maxerror {res['max_error']}")

    # 4. Set up Video Writer
    # Total canvas: Camera View (640x480) + BEV Map (480x480) -> 1120 x 480
    canvas_w = 640 + 480
    canvas_h = 480

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (canvas_w, canvas_h))

    print(f"Rendering {N} video frames to {out_path}...")
    font = cv2.FONT_HERSHEY_SIMPLEX

    for f_idx in range(N):
        frame_img = cv2.imread(str(img_files[f_idx]))
        dets = detections_by_frame[f_idx]

        # Draw detections and uncertainty ellipses on camera view
        active_lms = []
        for d in dets:
            u, v = d["pixel_center"]
            # Color based on head
            color = (0, 220, 30) if head_type in ["student_t", "student_t_fixed"] else ((220, 80, 20) if head_type == "cauchy" else (20, 150, 240))
            draw_uncertainty_ellipse(frame_img, (u, v), d["scale_azimuth"], d["scale_elevation"], color=color)

            # Center dot
            cv2.circle(frame_img, (int(round(u)), int(round(v))), 3, color, -1)

            # Text tag
            dof_str = f" nu={d['nu_azimuth']:.1f}" if "nu_azimuth" in d else ""
            txt = f"rho={d['rho']:.2f}{dof_str}"
            cv2.putText(frame_img, txt, (int(u) - 25, int(v) - 10), font, 0.4, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame_img, txt, (int(u) - 25, int(v) - 10), font, 0.4, color, 1, cv2.LINE_AA)

        # Current frame error
        cur_err = float(np.linalg.norm(poses_est[f_idx, :3, 3] - poses_gt[f_idx, :3, 3]))

        # Top HUD on Camera View
        cv2.rectangle(frame_img, (0, 0), (640, 45), (20, 20, 20), -1)
        hud1 = f"Frame {f_idx:03d}/{N} | Model: {backbone.upper()} + {head_type} (Mode {mode})"
        hud2 = f"Current ATE: {cur_err*100:.1f} cm | Detections: {len(dets)}"
        cv2.putText(frame_img, hud1, (10, 18), font, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame_img, hud2, (10, 38), font, 0.48, (80, 230, 80), 1, cv2.LINE_AA)

        # Render BEV Map
        # bev_map = render_bev_map(
        #     poses_gt[:N], poses_est, map_landmarks,
        #     curr_frame=f_idx, active_landmarks=active_lms, map_size=canvas_h
        # )
        pose_covs = res.get("pose_covariances")      # [N, 6, 6] or None

        bev_map = render_bev_map(
            poses_gt[:N], poses_est, map_landmarks,
            curr_frame=f_idx, active_landmarks=[], map_size=canvas_h,
            detections=dets, 
            pose_covs=pose_covs,                      # <- pose ellipse + yaw wedge
            include_pose_in_detections=True,        # optional: total landmark uncertainty
        ) 

        # Combine side-by-side
        canvas = np.hstack([frame_img, bev_map])
        writer.write(canvas)

    writer.release()
    print(f"Video saved successfully: {out_path} ({N} frames at {fps} FPS)")


def generate_all_experiment_videos(
    checkpoints_dir: str = "checkpoints",
    data_dir: str = "data/blender/room1_medium_nominal",
    out_dir: str = "reports/videos",
    max_frames: int = 250,
) -> None:
    """Batch generates visualization videos for all checkpoints found."""
    p_ckpts = Path(checkpoints_dir)
    p_out = Path(out_dir)
    p_out.mkdir(parents=True, exist_ok=True)

    models = sorted(p_ckpts.glob("*/model_final.pt"))
    print(f"Generating visualization videos for {len(models)} models...")

    for m in models:
        folder = m.parent.name
        tokens = folder.split("_")
        backbone = tokens[0]
        if "student_t_fixed" in folder:
            head = "student_t_fixed"
        elif "student_t" in folder:
            head = "student_t"
        elif "gaussian_beta" in folder:
            head = "gaussian_beta"
        elif "mse_fixed" in folder:
            head = "mse_fixed"
        elif "cauchy" in folder:
            head = "cauchy"
        else:
            head = "gaussian"

        out_vid = p_out / f"demo_{folder}.mp4"
        try:
            generate_experiment_video(
                checkpoint_path=str(m),
                data_dir=data_dir,
                out_video_path=str(out_vid),
                backbone=backbone,
                head_type=head,
                mode="B",
                max_frames=max_frames,
            )
        except Exception as e:
            print(f"Skipping {folder} due to: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Prediction & Localization Videos")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to specific model checkpoint")
    parser.add_argument("--all", action="store_true", help="Generate videos for all found checkpoints")
    parser.add_argument("--data_dir", type=str, default="data/blender/room1_medium_nominal")
    parser.add_argument("--out", type=str, default="reports/videos/demo_localization.mp4")
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--head", type=str, default="student_t")
    parser.add_argument("--mode", type=str, default="B", choices=["A", "B"])
    parser.add_argument("--frames", type=int, default=300)
    args = parser.parse_args()

    if args.all:
        generate_all_experiment_videos(data_dir=args.data_dir, max_frames=args.frames)
    else:
        ckpt = args.checkpoint
        if ckpt is None:
            ckpt = f"checkpoints/{args.backbone}_{args.head}_seed42/model_final.pt"
        generate_experiment_video(
            checkpoint_path=ckpt,
            data_dir=args.data_dir,
            out_video_path=args.out,
            backbone=args.backbone,
            head_type=args.head,
            mode=args.mode,
            max_frames=args.frames,
        )

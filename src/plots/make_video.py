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


def render_bev_map(
    poses_gt: np.ndarray,
    poses_est: np.ndarray,
    landmarks: Dict[int, np.ndarray],
    curr_frame: int,
    active_landmarks: List[int],
    map_size: int = 480,
    margin_m: float = 1.0,
) -> np.ndarray:
    """
    Renders top-down (X-Y plane) Bird's-Eye View map showing:
    - Trajectories (GT: Gray dotted, Est: Blue solid, Current: Red dot)
    - Landmarks: Gray circles (All), Red filled dots (Observed in current frame)
    """
    bev = np.full((map_size, map_size, 3), 245, dtype=np.uint8)

    # Determine world bounds across landmarks and trajectory
    traj_xy = poses_gt[:, :2, 3]
    lm_xy = np.array([pt[:2] for pt in landmarks.values()]) if landmarks else traj_xy

    all_x = np.concatenate([traj_xy[:, 0], lm_xy[:, 0]])
    all_y = np.concatenate([traj_xy[:, 1], lm_xy[:, 1]])

    min_x, max_x = np.min(all_x) - margin_m, np.max(all_x) + margin_m
    min_y, max_y = np.min(all_y) - margin_m, np.max(all_y) + margin_m

    scale_x = (map_size - 40) / max(1e-4, max_x - min_x)
    scale_y = (map_size - 40) / max(1e-4, max_y - min_y)
    scale = min(scale_x, scale_y)

    def world_to_map(wx: float, wy: float) -> Tuple[int, int]:
        mx = int(20 + (wx - min_x) * scale)
        # Flip Y so +Y is up
        my = int(map_size - 20 - (wy - min_y) * scale)
        return mx, my

    # 1. Draw Grid lines
    for gx in np.arange(np.floor(min_x), np.ceil(max_x), 1.0):
        p1 = world_to_map(gx, min_y)
        p2 = world_to_map(gx, max_y)
        cv2.line(bev, p1, p2, (225, 225, 225), 1)
    for gy in np.arange(np.floor(min_y), np.ceil(max_y), 1.0):
        p1 = world_to_map(min_x, gy)
        p2 = world_to_map(max_x, gy)
        cv2.line(bev, p1, p2, (225, 225, 225), 1)

    # 2. Draw Landmarks
    for lid, pt in landmarks.items():
        mx, my = world_to_map(pt[0], pt[1])
        if lid in active_landmarks:
            cv2.circle(bev, (mx, my), 6, (0, 0, 220), -1, cv2.LINE_AA)  # Active observed (Red)
            cv2.circle(bev, (mx, my), 8, (0, 0, 220), 2, cv2.LINE_AA)
        else:
            cv2.circle(bev, (mx, my), 4, (120, 120, 120), -1, cv2.LINE_AA)  # Dormant (Gray)

    # 3. Draw Full Ground Truth Trajectory (Gray)
    gt_pts = [world_to_map(poses_gt[i, 0, 3], poses_gt[i, 1, 3]) for i in range(len(poses_gt))]
    for i in range(len(gt_pts) - 1):
        cv2.line(bev, gt_pts[i], gt_pts[i+1], (180, 180, 180), 2, cv2.LINE_AA)

    # 4. Draw Estimated Trajectory up to curr_frame (Blue)
    est_pts = [world_to_map(poses_est[i, 0, 3], poses_est[i, 1, 3]) for i in range(curr_frame + 1)]
    for i in range(len(est_pts) - 1):
        cv2.line(bev, est_pts[i], est_pts[i+1], (210, 80, 20), 2, cv2.LINE_AA)

    # 5. Draw Current Camera Pose (GT: green circle, Est: blue triangle)
    curr_gt = world_to_map(poses_gt[curr_frame, 0, 3], poses_gt[curr_frame, 1, 3])
    curr_est = world_to_map(poses_est[curr_frame, 0, 3], poses_est[curr_frame, 1, 3])

    cv2.circle(bev, curr_gt, 5, (50, 180, 50), -1, cv2.LINE_AA)
    cv2.circle(bev, curr_est, 6, (210, 80, 20), -1, cv2.LINE_AA)

    # Heading arrow for estimated pose
    heading = np.arctan2(poses_est[curr_frame, 1, 2], poses_est[curr_frame, 0, 2])
    hx = int(curr_est[0] + 15 * np.cos(heading))
    hy = int(curr_est[1] - 15 * np.sin(heading))
    cv2.arrowedLine(bev, curr_est, (hx, hy), (210, 80, 20), 2, tipLength=0.3)

    # Legend & Title
    cv2.putText(bev, "Bird's-Eye View Map", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(bev, "-- GT Path", (15, map_size - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
    cv2.putText(bev, "- Est Path", (15, map_size - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 80, 20), 2)
    cv2.putText(bev, "* Landmarks", (map_size - 110, map_size - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1)

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
        bev_map = render_bev_map(
            poses_gt[:N], poses_est, map_landmarks,
            curr_frame=f_idx, active_landmarks=active_lms, map_size=canvas_h
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

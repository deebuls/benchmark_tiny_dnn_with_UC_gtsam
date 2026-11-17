"""
Blender Dataset Generator & Renderer.
Conforms to M4 specifications:
Produces images, poses_gt.npy, markers_gt.npy, meas_gt.jsonl, odometry.npz, and manifest.json.
Supports nominal and stress conditions (blur, low light, occlusion, distractors, distortion).
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np

from src.common.geometry import (
    DEFAULT_INTRINSICS,
    CameraIntrinsics,
    MARKER_SIZE_METERS,
    point3d_to_pixel,
    point3d_to_measurement,
)
from src.blender.trajectory import (
    ROOM_PRESETS,
    generate_marker_layout,
    generate_6dof_trajectory,
    simulate_odometry,
)
from src.synth.generate import compute_marker_homography, load_marker_panels
from src.synth.backgrounds import ProceduralBackgroundGenerator
from src.synth.augmentations import SyntheticAugmentor


def render_scene_trajectory(
    room_id: str,
    density: str = "medium",
    condition: str = "nominal",  # "nominal", "low_light", "motion_blur", "occlusion", "distractors"
    num_frames: int = 500,
    out_dir: str = "data/blender/room1_medium_nominal",
    seed: int = 42,
    intrinsics: CameraIntrinsics = DEFAULT_INTRINSICS,
) -> None:
    """
    Renders 6-DoF trajectory for a room condition, saving ground truth files
    and 640x480 RGB frames.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    img_dir = out_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Find room spec
    room = next((r for r in ROOM_PRESETS if r.room_id == room_id), ROOM_PRESETS[0])

    # 1. Generate Markers and 6-DoF Trajectory
    markers = generate_marker_layout(room, density=density, marker_size=MARKER_SIZE_METERS, seed=seed)
    poses_gt = generate_6dof_trajectory(room, num_frames=num_frames, with_loop_closure=True, seed=seed)

    # 2. Simulate Odometry at 1% drift
    rel_noisy, covs = simulate_odometry(poses_gt, noise_level=0.01, seed=seed)

    # 3. Load Marker assets
    can_panel, dist_panels = load_marker_panels("assets/logo")
    panel_px = can_panel.shape[0]

    # Save GT numpy arrays
    np.save(str(out_path / "poses_gt.npy"), poses_gt)
    # Save markers: array of [id, x, y, z]
    markers_array = np.array([[m["id"], m["position"][0], m["position"][1], m["position"][2]] for m in markers], dtype=np.float64)
    np.save(str(out_path / "markers_gt.npy"), markers_array)
    np.savez_compressed(str(out_path / "odometry.npz"), relative_poses=rel_noisy, covariances=covs)

    # Save intrinsics
    with open(out_path / "intrinsics.json", "w") as f:
        json.dump({
            "width": intrinsics.width, "height": intrinsics.height,
            "fx": intrinsics.fx, "fy": intrinsics.fy,
            "cx": intrinsics.cx, "cy": intrinsics.cy,
        }, f, indent=2)

    bg_gen = ProceduralBackgroundGenerator(is_validation=True, seed=seed)
    augmentor = SyntheticAugmentor(np.random.default_rng(seed))

    meas_gt_lines = []
    w, h = intrinsics.width, intrinsics.height

    print(f"Rendering {num_frames} frames for {room_id} ({density}, {condition})...")

    for f_idx in range(num_frames):
        T_w_c = poses_gt[f_idx]
        # Camera-to-world: R_w_c, t_w_c
        # World-to-camera: R_c_w = R_w_c^T, t_c_w = -R_c_w * t_w_c
        R_w_c = T_w_c[:3, :3]
        t_w_c = T_w_c[:3, 3]
        R_c_w = R_w_c.T
        t_c_w = -R_c_w @ t_w_c

        # Generate base room background
        img, _ = bg_gen.generate(width=w, height=h)

        visible_measurements = []
        frame_bboxes = []

        # Project markers into camera frame
        for m in markers:
            p_world = np.array(m["position"], dtype=np.float64)
            p_cam = R_c_w @ p_world + t_c_w

            # Check if in front of camera
            if p_cam[2] < 0.3 or p_cam[2] > 7.0:
                continue

            # Marker normal in camera frame
            n_world = np.array(m["normal"], dtype=np.float64)
            n_cam = R_c_w @ n_world
            view_ray = p_cam / np.linalg.norm(p_cam)
            cos_theta = np.dot(-n_cam, view_ray)
            view_angle_deg = float(np.rad2deg(np.arccos(np.clip(cos_theta, -1.0, 1.0))))

            if view_angle_deg > 75.0:
                continue

            # Compute marker orientation matrix in camera frame
            # Normal is n_cam. Construct orthogonal basis for marker plane
            up_hint = np.array([0.0, 0.0, 1.0])
            up_cam = R_c_w @ up_hint
            marker_x = np.cross(up_cam, n_cam)
            if np.linalg.norm(marker_x) < 1e-4:
                marker_x = np.array([1.0, 0.0, 0.0])
            else:
                marker_x = marker_x / np.linalg.norm(marker_x)
            marker_y = np.cross(n_cam, marker_x)
            R_marker_cam = np.column_stack([marker_x, marker_y, n_cam])

            H, corners, center_px, _ = compute_marker_homography(
                R_marker_cam, p_cam, MARKER_SIZE_METERS, panel_px, intrinsics
            )

            # Check if within image bounds
            u, v = center_px[0], center_px[1]
            if not (0 <= u < w and 0 <= v < h):
                continue

            # Warp panel onto canvas
            warped = cv2.warpPerspective(can_panel, H, (w, h), flags=cv2.INTER_LINEAR)
            mask_ones = np.ones((panel_px, panel_px), dtype=np.uint8) * 255
            warped_mask = cv2.warpPerspective(mask_ones, H, (w, h), flags=cv2.INTER_NEAREST) > 0
            img[warped_mask] = warped[warped_mask]

            # True measurement
            meas = point3d_to_measurement(p_cam)
            meas_entry = {
                "frame_idx": f_idx,
                "marker_id": int(m["id"]),
                "pos_cam": [float(p_cam[0]), float(p_cam[1]), float(p_cam[2])],
                "azimuth": float(meas[0]),
                "elevation": float(meas[1]),
                "rho": float(meas[2]),
                "pixel_center": [float(u), float(v)],
                "view_angle_deg": view_angle_deg,
                "in_frame": True,
            }
            visible_measurements.append(meas_entry)
            xmin, xmax = max(0, int(np.min(corners[:, 0]))), min(w - 1, int(np.max(corners[:, 0])))
            ymin, ymax = max(0, int(np.min(corners[:, 1]))), min(h - 1, int(np.max(corners[:, 1])))
            frame_bboxes.append((xmin, ymin, xmax, ymax))

        # Apply Condition-Specific Effects
        if condition == "low_light":
            img = (img.astype(np.float32) * 0.4).astype(np.uint8)
        elif condition == "motion_blur":
            img, _ = augmentor.apply_motion_blur(img, max_len=18)
        elif condition == "occlusion":
            img, _ = augmentor.apply_random_occlusions(img, frame_bboxes)
        elif condition == "distractors" and dist_panels:
            # Add distractor
            d_p = dist_panels[f_idx % len(dist_panels)]
            dH = np.eye(3)
            dH[0, 2] = (f_idx * 13) % (w - 120)
            dH[1, 2] = (f_idx * 17) % (h - 120)
            d_warp = cv2.warpPerspective(d_p, dH, (w, h))
            d_m = d_warp > 0
            img[d_m] = d_warp[d_m]

        # Save image
        img_name = f"{f_idx:06d}.jpg"
        cv2.imwrite(str(img_dir / img_name), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        for vm in visible_measurements:
            meas_gt_lines.append(vm)

    # Save meas_gt.jsonl
    with open(out_path / "meas_gt.jsonl", "w") as f:
        for line in meas_gt_lines:
            f.write(json.dumps(line) + "\n")

    # Save manifest.json
    manifest = {
        "room_id": room_id,
        "density": density,
        "condition": condition,
        "num_frames": num_frames,
        "num_markers": len(markers),
        "total_observations": len(meas_gt_lines),
        "seed": seed,
    }
    with open(out_path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Dataset saved to {out_dir}: {len(meas_gt_lines)} total marker observations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blender Validation Generator")
    parser.add_argument("--room", type=str, default="room1_small_office")
    parser.add_argument("--density", type=str, default="medium")
    parser.add_argument("--condition", type=str, default="nominal")
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--out_dir", type=str, default="data/blender/room1_medium_nominal")
    args = parser.parse_args()

    render_scene_trajectory(
        room_id=args.room,
        density=args.density,
        condition=args.condition,
        num_frames=args.frames,
        out_dir=args.out_dir,
    )

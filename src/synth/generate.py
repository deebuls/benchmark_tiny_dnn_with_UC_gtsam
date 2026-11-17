"""
M2: Synthetic Dataset Generator for identical marker localization and SLAM.
Generates 640x480 images with homography-warped panels, realistic variations,
and strict multi-format annotations (JSON lines + compact npz).
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import multiprocessing as mp
import cv2
import numpy as np

from src.common.geometry import (
    DEFAULT_INTRINSICS,
    CameraIntrinsics,
    MARKER_SIZE_METERS,
    point3d_to_measurement,
    point3d_to_pixel,
)
from src.synth.backgrounds import ProceduralBackgroundGenerator
from src.synth.augmentations import SyntheticAugmentor


def load_marker_panels(logo_dir: str = "assets/logo") -> Tuple[np.ndarray, List[np.ndarray]]:
    """Loads canonical panel and all distractor panels."""
    p_logo = Path(logo_dir)
    can_path = p_logo / "canonical_panel.png"
    if not can_path.exists():
        from src.logo.make_logo import generate_all_marker_assets
        generate_all_marker_assets(logo_dir)

    canonical_panel = cv2.imread(str(can_path))
    distractor_panels = []
    dist_dir = p_logo / "distractors"
    if dist_dir.exists():
        for f in sorted(dist_dir.glob("*_panel.png")):
            dist_img = cv2.imread(str(f))
            if dist_img is not None:
                distractor_panels.append(dist_img)
    return canonical_panel, distractor_panels


def compute_marker_homography(
    r_cam: np.ndarray,
    t_cam: np.ndarray,
    marker_size: float,
    panel_px_size: int,
    intrinsics: CameraIntrinsics,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Computes homography from panel image coordinates to camera image coordinates,
    and returns projected 4 corners, center pixel, and view angle (degrees).
    """
    half_s = marker_size / 2.0
    # Marker local coordinates: plane z=0, x in [-s/2, s/2], y in [-s/2, s/2]
    # Top-Left, Top-Right, Bottom-Right, Bottom-Left
    local_pts = np.array([
        [-half_s, -half_s, 0.0],
        [ half_s, -half_s, 0.0],
        [ half_s,  half_s, 0.0],
        [-half_s,  half_s, 0.0],
    ], dtype=np.float64)

    # Transform to camera frame: p_cam = R * p_loc + t
    cam_pts = (r_cam @ local_pts.T).T + t_cam

    # Normal of the marker in camera frame
    normal_loc = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    normal_cam = r_cam @ normal_loc

    # View ray from camera (origin) to marker center (t_cam)
    view_ray = t_cam / (np.linalg.norm(t_cam) + 1e-9)

    # Angle between marker normal and -view_ray (angle facing camera)
    cos_angle = np.clip(np.dot(-normal_cam, view_ray), -1.0, 1.0)
    view_angle_deg = float(np.rad2deg(np.arccos(cos_angle)))

    # Project to pixel coordinates
    img_pts = point3d_to_pixel(cam_pts, intrinsics)
    center_px = point3d_to_pixel(t_cam[None, :], intrinsics)[0]

    # Source points on panel texture
    src_pts = np.array([
        [0.0, 0.0],
        [panel_px_size - 1.0, 0.0],
        [panel_px_size - 1.0, panel_px_size - 1.0],
        [0.0, panel_px_size - 1.0],
    ], dtype=np.float32)

    dst_pts = img_pts.astype(np.float32)
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)

    return H, img_pts, center_px, view_angle_deg


def assign_difficulty(
    depth: float,
    view_angle: float,
    occ_fraction: float,
    motion_blur_len: int,
) -> str:
    """Documented difficulty rule per specs.md."""
    # Hard: extreme depth, steep view angle, high occlusion, or heavy motion blur
    if depth > 4.5 or view_angle > 60.0 or occ_fraction > 0.40 or motion_blur_len > 15:
        return "hard"
    elif depth > 2.5 or view_angle > 35.0 or occ_fraction > 0.15 or motion_blur_len > 6:
        return "medium"
    else:
        return "easy"


def generate_single_image(
    index: int,
    canonical_panel: np.ndarray,
    distractor_panels: List[np.ndarray],
    bg_generator: ProceduralBackgroundGenerator,
    augmentor: SyntheticAugmentor,
    intrinsics: CameraIntrinsics = DEFAULT_INTRINSICS,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generates one synthetic image with annotations."""
    rng = np.random.default_rng(seed)
    h, w = intrinsics.height, intrinsics.width
    panel_px = canonical_panel.shape[0]

    # 1. Base procedural background
    img, bg_type = bg_generator.generate(width=w, height=h)

    # 2. Sample number of canonical markers (0 to 6 markers, ~15% chance of 0)
    if rng.random() < 0.15:
        num_markers = 0
    else:
        num_markers = int(rng.integers(1, 7))

    marker_annotations = []
    marker_bboxes = []
    marker_masks = []

    # 3. Place canonical markers
    for m_idx in range(num_markers):
        # Sample inverse depth rho uniformly in [0.17, 2.0] (0.5m to 5.8m)
        rho = rng.uniform(0.17, 2.0)
        depth = 1.0 / rho

        # Sample camera-frame position within image FOV
        margin = 40.0
        target_u = rng.uniform(margin, w - margin)
        target_v = rng.uniform(margin, h - margin)
        # Unproject to 3D center
        tx = (target_u - intrinsics.cx) * depth / intrinsics.fx
        ty = (target_v - intrinsics.cy) * depth / intrinsics.fy
        tz = depth
        t_cam = np.array([tx, ty, tz], dtype=np.float64)

        # Marker orientation: sample Euler angles such that view angle <= 75 degrees
        for _ in range(20):
            pitch = rng.uniform(-np.deg2rad(70), np.deg2rad(70))
            yaw = rng.uniform(-np.deg2rad(70), np.deg2rad(70))
            roll = rng.uniform(-np.pi, np.pi)

            # Rotation matrix: Rx * Ry * Rz
            Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
            Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
            Rz = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
            R = Rz @ Ry @ Rx

            H, corners, center_px, view_angle = compute_marker_homography(
                R, t_cam, MARKER_SIZE_METERS, panel_px, intrinsics
            )
            if view_angle <= 75.0:
                break

        # Warp panel onto canvas with anti-aliasing
        warped = cv2.warpPerspective(canonical_panel, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        mask_ones = np.ones((panel_px, panel_px), dtype=np.uint8) * 255
        warped_mask = cv2.warpPerspective(mask_ones, H, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

        # Composite warped panel onto img
        fg_mask = warped_mask > 0
        img[fg_mask] = warped[fg_mask]

        # Bounding box of marker
        xmin, xmax = int(np.floor(np.min(corners[:, 0]))), int(np.ceil(np.max(corners[:, 0])))
        ymin, ymax = int(np.floor(np.min(corners[:, 1]))), int(np.ceil(np.max(corners[:, 1])))
        xmin_c = max(0, min(w - 1, xmin))
        xmax_c = max(0, min(w - 1, xmax))
        ymin_c = max(0, min(h - 1, ymin))
        ymax_c = max(0, min(h - 1, ymax))

        # Check in-frame and initial visible fraction (due to clipping)
        full_area = max(1.0, cv2.contourArea(corners.astype(np.float32)))
        clipped_area = float(np.sum(fg_mask))
        vis_fraction = min(1.0, clipped_area / full_area)
        in_frame = bool(0 <= center_px[0] < w and 0 <= center_px[1] < h)

        # True measurement
        meas = point3d_to_measurement(t_cam)
        azimuth, elevation, rho_true = float(meas[0]), float(meas[1]), float(meas[2])

        marker_annotations.append({
            "id": m_idx,
            "pos_cam": [float(tx), float(ty), float(tz)],
            "azimuth": azimuth,
            "elevation": elevation,
            "rho": rho_true,
            "pixel_center": [float(center_px[0]), float(center_px[1])],
            "corners": corners.tolist(),
            "depth": float(depth),
            "view_angle_deg": float(view_angle),
            "initial_vis_fraction": float(vis_fraction),
            "in_frame": in_frame,
            "bbox": [xmin_c, ymin_c, xmax_c, ymax_c],
        })
        marker_bboxes.append((xmin_c, ymin_c, xmax_c, ymax_c))
        marker_masks.append(fg_mask)

    # 4. Distractors: 0 to 4 distractor logos (in 30% of images)
    num_distractors = 0
    if distractor_panels and rng.random() < 0.30:
        num_distractors = int(rng.integers(1, 5))
        for _ in range(num_distractors):
            dist_panel = distractor_panels[rng.integers(0, len(distractor_panels))]
            d_rho = rng.uniform(0.2, 1.8)
            d_depth = 1.0 / d_rho
            d_u = rng.uniform(30, w - 30)
            d_v = rng.uniform(30, h - 30)
            d_tx = (d_u - intrinsics.cx) * d_depth / intrinsics.fx
            d_ty = (d_v - intrinsics.cy) * d_depth / intrinsics.fy
            d_tz = d_depth
            d_tcam = np.array([d_tx, d_ty, d_tz], dtype=np.float64)

            d_roll = rng.uniform(-np.pi, np.pi)
            dR = np.array([[np.cos(d_roll), -np.sin(d_roll), 0], [np.sin(d_roll), np.cos(d_roll), 0], [0, 0, 1]])
            dH, _, _, _ = compute_marker_homography(dR, d_tcam, MARKER_SIZE_METERS, dist_panel.shape[0], intrinsics)
            d_warped = cv2.warpPerspective(dist_panel, dH, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            d_mask = cv2.warpPerspective(np.ones((dist_panel.shape[0], dist_panel.shape[0]), dtype=np.uint8) * 255, dH, (w, h), flags=cv2.INTER_NEAREST) > 0
            img[d_mask] = d_warped[d_mask]

    # 5. Apply Occlusions
    img, occ_fractions = augmentor.apply_random_occlusions(img, marker_bboxes)

    # 6. Apply Lighting Variations
    img, light_params = augmentor.apply_lighting(img)

    # 7. Apply Motion & Defocus Blur
    img, motion_params = augmentor.apply_motion_blur(img, max_len=25)
    img, defocus_k = augmentor.apply_defocus_blur(img, max_k=7)

    # 8. Apply Sensor Noise & Vignetting & JPEG
    img, sensor_params = augmentor.apply_sensor_noise(img)

    # 9. Finalize marker tags and visible fractions
    final_markers = []
    for m_info, occ_frac in zip(marker_annotations, occ_fractions):
        final_vis = max(0.0, m_info["initial_vis_fraction"] * (1.0 - occ_frac))
        diff_tag = assign_difficulty(
            m_info["depth"],
            m_info["view_angle_deg"],
            occ_frac,
            motion_params["motion_blur_len"],
        )
        is_target = bool(final_vis >= 0.30 and m_info["in_frame"])

        m_info["visible_fraction"] = float(final_vis)
        m_info["occlusion_fraction"] = float(occ_frac)
        m_info["difficulty"] = diff_tag
        m_info["is_target"] = is_target
        final_markers.append(m_info)

    image_meta = {
        "image_index": index,
        "bg_type": bg_type,
        "num_canonical_markers": len(final_markers),
        "num_distractors": num_distractors,
        "lighting": light_params,
        "motion_blur": motion_params,
        "defocus_k": defocus_k,
        "sensor": sensor_params,
        "markers": final_markers,
    }

    return img, image_meta


def _worker_generate_chunk(args: Tuple) -> int:
    chunk_start, chunk_end, split_dir, is_val, base_seed, logo_dir = args
    bg_gen = ProceduralBackgroundGenerator(is_validation=is_val, seed=base_seed + chunk_start)
    augmentor = SyntheticAugmentor(np.random.default_rng(base_seed + chunk_start + 1))
    can_panel, dist_panels = load_marker_panels(logo_dir)

    img_dir = Path(split_dir) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = Path(split_dir) / f"annos_part_{chunk_start}_{chunk_end}.jsonl"

    with open(jsonl_path, "w") as f_jsonl:
        for idx in range(chunk_start, chunk_end):
            img_seed = base_seed + idx * 7 + 13
            img, meta = generate_single_image(
                idx, can_panel, dist_panels, bg_gen, augmentor, seed=img_seed
            )
            img_filename = f"{idx:07d}.jpg"
            cv2.imwrite(str(img_dir / img_filename), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            meta["file_name"] = img_filename
            f_jsonl.write(json.dumps(meta) + "\n")

    return chunk_end - chunk_start


def generate_dataset_split(
    split_dir: str,
    num_images: int,
    is_validation: bool = False,
    base_seed: int = 42,
    num_workers: int = 4,
    logo_dir: str = "assets/logo",
) -> None:
    """Generates a complete dataset split with multiprocessing."""
    p_dir = Path(split_dir)
    p_dir.mkdir(parents=True, exist_ok=True)

    chunk_size = max(1, num_images // num_workers)
    chunks = []
    curr = 0
    while curr < num_images:
        end = min(num_images, curr + chunk_size)
        chunks.append((curr, end, split_dir, is_validation, base_seed, logo_dir))
        curr = end

    print(f"Generating split {split_dir} ({num_images} images) across {len(chunks)} workers...")
    if num_workers > 1 and len(chunks) > 1:
        with mp.Pool(num_workers) as pool:
            pool.map(_worker_generate_chunk, chunks)
    else:
        for c in chunks:
            _worker_generate_chunk(c)

    # Merge JSON lines parts into unified annotations.jsonl
    unified_jsonl = p_dir / "annotations.jsonl"
    all_metas = []
    with open(unified_jsonl, "w") as out_f:
        for p_file in sorted(p_dir.glob("annos_part_*.jsonl")):
            with open(p_file, "r") as in_f:
                for line in in_f:
                    out_f.write(line)
                    all_metas.append(json.loads(line))
            p_file.unlink()

    # Create compact .npz file for fast tensor loading
    _build_compact_npz(all_metas, p_dir / "annotations_compact.npz")
    print(f"Split {split_dir} successfully created with {len(all_metas)} samples.")


def _build_compact_npz(annotations: List[Dict[str, Any]], out_path: Path) -> None:
    """Creates a fast, compact indexed numpy archive for PyTorch DataLoader."""
    img_indices = []
    target_counts = []
    
    # Per-target flat arrays
    t_img_idx = []
    t_meas = []        # [azimuth, elevation, rho]
    t_px = []          # [u, v]
    t_pos = []         # [x, y, z]
    t_vis = []         # visible fraction
    t_diff = []        # 0: easy, 1: medium, 2: hard

    diff_map = {"easy": 0, "medium": 1, "hard": 2}

    for meta in annotations:
        idx = meta["image_index"]
        img_indices.append(idx)
        count = 0
        for m in meta["markers"]:
            if m["is_target"]:
                count += 1
                t_img_idx.append(idx)
                t_meas.append([m["azimuth"], m["elevation"], m["rho"]])
                t_px.append(m["pixel_center"])
                t_pos.append(m["pos_cam"])
                t_vis.append(m["visible_fraction"])
                t_diff.append(diff_map.get(m["difficulty"], 1))
        target_counts.append(count)

    np.savez_compressed(
        out_path,
        image_indices=np.array(img_indices, dtype=np.int32),
        target_counts=np.array(target_counts, dtype=np.int16),
        target_image_idx=np.array(t_img_idx, dtype=np.int32),
        target_meas=np.array(t_meas, dtype=np.float32),
        target_px=np.array(t_px, dtype=np.float32),
        target_pos=np.array(t_pos, dtype=np.float32),
        target_vis=np.array(t_vis, dtype=np.float32),
        target_diff=np.array(t_diff, dtype=np.uint8),
    )


def draw_debug_annotations(
    img: np.ndarray, meta: Dict[str, Any], intrinsics: CameraIntrinsics = DEFAULT_INTRINSICS
) -> np.ndarray:
    """Draws ground truth marker boxes, centers, bearing, and rho labels for inspection."""
    vis = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for m in meta["markers"]:
        px_u, px_v = int(round(m["pixel_center"][0])), int(round(m["pixel_center"][1]))
        is_target = m.get("is_target", True)
        color = (0, 255, 0) if is_target else (0, 165, 255)  # Green if target, orange if ignored

        # Draw marker center dot
        cv2.circle(vis, (px_u, px_v), 4, color, -1)

        # Draw corners
        corners = np.array(m["corners"], dtype=np.int32)
        cv2.polylines(vis, [corners], isClosed=True, color=color, thickness=2)

        # Label text
        label = f"az:{np.rad2deg(m['azimuth']):.1f} el:{np.rad2deg(m['elevation']):.1f} r:{m['rho']:.2f}"
        cv2.putText(vis, label, (px_u - 40, px_v - 10), font, 0.4, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, label, (px_u - 40, px_v - 10), font, 0.4, color, 1, cv2.LINE_AA)

    return vis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Dataset Generator")
    parser.add_argument("--split", type=str, default="smoke", choices=["train", "val", "calib", "smoke"])
    parser.add_argument("--num_images", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default="data/synth/smoke")
    args = parser.parse_args()

    is_val = args.split in ["val", "calib"]
    generate_dataset_split(
        split_dir=args.out_dir,
        num_images=args.num_images,
        is_validation=is_val,
        num_workers=args.workers,
    )

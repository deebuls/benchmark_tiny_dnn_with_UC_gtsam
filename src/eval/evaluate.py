"""
Evaluator script running trained models on validation/test datasets.
Produces per-measurement errors, detection AP, calibration metrics, and saves results.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch
import cv2

from src.models.model import MarkerModel
from src.eval.metrics import (
    compute_detection_matches,
    evaluate_regression_metrics,
    compute_crps,
    compute_quantile_coverage,
    fit_temperature_recalibration,
    compute_model_compute_cost,
)
from src.common.geometry import DEFAULT_INTRINSICS


def evaluate_checkpoint_on_dataset(
    checkpoint_path: str,
    data_dir: str,
    backbone: str = "resnet18",
    head_type: str = "gaussian",
    device_name: str = "cuda",
) -> Dict[str, Any]:
    """Runs model inference over dataset frames and computes all M5 metrics."""
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    print(f"Loading {checkpoint_path} on {device}...")

    # Auto-detect head and backbone if checkpoint filename/path has standard naming
    detected_head = head_type
    detected_backbone = backbone
    p_str = str(checkpoint_path).lower()

    for cand_h in ["student_t_fixed", "student_t", "gaussian_beta", "gaussian", "cauchy", "mse_fixed"]:
        if cand_h in p_str:
            detected_head = cand_h
            break

    for cand_b in ["resnet50", "resnet18", "resnet9"]:
        if cand_b in p_str:
            detected_backbone = cand_b
            break

    # If user explicitly passed a head that disagrees or default was used, prefer the detected or loaded head
    print(f"Instantiating model with backbone='{detected_backbone}', head='{detected_head}'...")
    state = torch.load(checkpoint_path, map_location=device)

    # Inspect state dict to guarantee exact head type match
    has_nu_branch = any("head.nu_branch" in k for k in state.keys())
    has_unc_branch = any("head.unc_branch" in k for k in state.keys())

    if has_nu_branch:
        actual_head = "student_t"
    elif has_unc_branch:
        # Could be gaussian, gaussian_beta, cauchy, or student_t_fixed
        if detected_head in ["gaussian", "gaussian_beta", "cauchy", "student_t_fixed"]:
            actual_head = detected_head
        else:
            actual_head = "gaussian"
    else:
        actual_head = "mse_fixed"

    # Load Model
    model = MarkerModel(backbone_type=detected_backbone, head_type=actual_head).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Load Ground Truth
    p_data = Path(data_dir + "nominal")
    meas_gt_file = p_data / "meas_gt.jsonl"
    if not meas_gt_file.exists():
        meas_gt_file = p_data / "annotations.jsonl"

    gts_by_frame = {}
    with open(meas_gt_file, "r") as f:
        for line in f:
            item = json.loads(line)
            f_idx = item.get("frame_idx", item.get("image_index", 0))
            if f_idx not in gts_by_frame:
                gts_by_frame[f_idx] = []
            
            # Extract marker info
            if "markers" in item:
                for m in item["markers"]:
                    if m.get("is_target", True):
                        gts_by_frame[f_idx].append(m)
            else:
                gts_by_frame[f_idx].append(item)

    # Run inference on all frames
    img_files = sorted((p_data / "images").glob("*.jpg"))
    if not img_files:
        raise FileNotFoundError(f"No image files found in {p_data / 'images'}")

    preds_by_frame = []
    gts_list = []

    print(f"Running evaluation on {len(img_files)} frames...")
    for f_idx, img_p in enumerate(img_files):
        img_bgr = cv2.imread(str(img_p))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t_img = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)

        with torch.no_grad():
            dets = model.predict(t_img, score_threshold=0.3)[0]

        preds_by_frame.append(dets)
        gts_list.append(gts_by_frame.get(f_idx, []))

    # Match detections
    matches, total_fp, total_misses = compute_detection_matches(preds_by_frame, gts_list, pixel_gate=25.0)

    # Compute Regression Errors
    res_az = np.array([m["pred"]["azimuth"] - m["gt"]["azimuth"] for m in matches])
    # Wrap azimuth error to [-pi, pi)
    res_az = (res_az + np.pi) % (2.0 * np.pi) - np.pi
    res_el = np.array([m["pred"]["elevation"] - m["gt"]["elevation"] for m in matches])
    res_rho = np.array([m["pred"]["rho"] - m["gt"]["rho"] for m in matches])

    scales_az = np.array([m["pred"]["scale_azimuth"] for m in matches])
    scales_el = np.array([m["pred"]["scale_elevation"] for m in matches])
    scales_rho = np.array([m["pred"]["scale_rho"] for m in matches])

    has_nu = "nu_azimuth" in matches[0]["pred"] if matches else False
    nu_az = np.array([m["pred"]["nu_azimuth"] for m in matches]) if has_nu else None
    nu_el = np.array([m["pred"]["nu_elevation"] for m in matches]) if has_nu else None
    nu_rho = np.array([m["pred"]["nu_rho"] for m in matches]) if has_nu else None

    # Uncertainty & Regression Metrics
    metrics = {
        "num_frames": len(img_files),
        "total_matches": len(matches),
        "total_false_positives": total_fp,
        "total_misses": total_misses,
        "fp_per_frame": float(total_fp / max(1, len(img_files))),
        "precision": float(len(matches) / max(1, len(matches) + total_fp)),
        "recall": float(len(matches) / max(1, len(matches) + total_misses)),
        "azimuth": evaluate_regression_metrics(res_az),
        "elevation": evaluate_regression_metrics(res_el),
        "rho": evaluate_regression_metrics(res_rho),
        "crps": {
            "azimuth": compute_crps(res_az, scales_az, head_type, nu_az),
            "elevation": compute_crps(res_el, scales_el, head_type, nu_el),
            "rho": compute_crps(res_rho, scales_rho, head_type, nu_rho),
        },
        "coverage": {
            "azimuth": compute_quantile_coverage(res_az, scales_az, head_type, nu_az),
            "elevation": compute_quantile_coverage(res_el, scales_el, head_type, nu_el),
            "rho": compute_quantile_coverage(res_rho, scales_rho, head_type, nu_rho),
        },
        "temperature_recal": {
            "azimuth": fit_temperature_recalibration(res_az, scales_az, head_type, nu_az),
            "elevation": fit_temperature_recalibration(res_el, scales_el, head_type, nu_el),
            "rho": fit_temperature_recalibration(res_rho, scales_rho, head_type, nu_rho),
        },
        "compute": compute_model_compute_cost(model),
    }

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Marker Uncertainty Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/blender/room1_medium_nominal")
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--head", type=str, default="gaussian")
    parser.add_argument("--out", type=str, default="eval_results.json")
    args = parser.parse_args()

    results = evaluate_checkpoint_on_dataset(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        backbone=args.backbone,
        head_type=args.head,
    )

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Evaluation complete. Results saved to {args.out}")
    print(f"Precision: {results['precision']:.3f}, Recall: {results['recall']:.3f}")
    print(f"Azimuth MAE: {np.rad2deg(results['azimuth']['mae']):.2f} deg, Rho MAE: {results['rho']['mae']:.3f} m^-1")
    print(f"Inference Latency: {results['compute']['latency_ms']:.2f} ms ({results['compute']['fps']:.1f} FPS)")

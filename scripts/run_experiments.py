"""
M7: Experiment Matrix Runner.
Drives localization and SLAM experiments across heads, backbones, seeds, back-end modes,
odometry drift rates, and scenes.
Outputs a single tidy `results.parquet` file with caching and resumability.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import yaml
import numpy as np
import pandas as pd
import torch
import itertools

from src.models.model import MarkerModel
from src.backend.slam import run_localization_known_map 
from src.backend.slam_unkown import run_slam_unknown_map
from src.eval.metrics import evaluate_regression_metrics, compute_model_compute_cost
from src.eval.stats import tost_equivalence_test, compute_paired_bootstrap_ci


def run_experiment_trial(
    head: str,
    backbone: str,
    seed: int,
    mode: str,
    task: str,
    scene: str,
    condition: str,
    odometry_level: float,
    association: str,
    data_dir: str,
    checkpoint_dir: str = "checkpoints",
    device: str = "cuda",
) -> Dict[str, Any]:
    """Runs a single experimental configuration trial."""
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

    # 1. Load ground truth trajectory & data
    p_data = Path(data_dir+condition)
    poses_gt = np.load(str(p_data / "poses_gt.npy"))
    markers_gt = np.load(str(p_data / "markers_gt.npy"))
    map_landmarks = {int(m[0]): m[1:] for m in markers_gt}

    # 2. Simulate odometry at specified noise level
    from src.blender.trajectory import simulate_odometry
    rel_noisy, covs = simulate_odometry(poses_gt, noise_level=odometry_level, seed=seed)

    # 3. Load model if available
    ckpt_path = Path(checkpoint_dir) / f"{backbone}_{head}_seed{seed}" / "model_final.pt"
    if not ckpt_path.exists():
        # Fallback to seed 42 or smoke
        ckpt_path = Path(checkpoint_dir) / f"{backbone}_{head}_seed42" / "model_final.pt"
        print (f"checkppoint missing : {ckpt_path}");
        exit

    # Compute network detections or synthetic simulated observations
    N = len(poses_gt)
    # Check if real images exist to run inference
    img_files = sorted((p_data / "images").glob("*.jpg"))
    detections_by_frame = [[] for _ in range(N)]

    if ckpt_path.exists() and img_files and len(img_files) >= N:
        model = MarkerModel(backbone_type=backbone, head_type=head).to(dev)
        state = torch.load(str(ckpt_path), map_location=dev)
        model.load_state_dict(state, strict=False)
        model.eval()

        # Run inference
        for i in range(min(N, len(img_files))):
            import cv2
            img_bgr = cv2.imread(str(img_files[i]))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            t_img = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).to(dev)
            with torch.no_grad():
                dets = model.predict(t_img, score_threshold=0.3)[0]
            detections_by_frame[i] = dets
    else:
        print ("Missing model or image")
        exit()
        # # Fallback observation simulation from meas_gt.jsonl
        # meas_gt_file = p_data / "meas_gt.jsonl"
        # rng = np.random.default_rng(seed)
        # if meas_gt_file.exists():
        #     with open(meas_gt_file, "r") as f:
        #         for line in f:
        #             item = json.loads(line)
        #             f_idx = item["frame_idx"]
        #             if f_idx < N:
        #                 # Corrupt ground truth by head noise profile
        #                 sigma_mult = 1.0 if backbone == "resnet50" else (1.2 if backbone == "resnet18" else 1.5)
        #                 s_az = 0.02 * sigma_mult
        #                 s_el = 0.02 * sigma_mult
        #                 s_rho = 0.05 * sigma_mult
        #
        #                 d_az = rng.normal(0, s_az)
        #                 d_el = rng.normal(0, s_el)
        #                 d_rho = rng.normal(0, s_rho)
        #
        #                 detections_by_frame[f_idx].append({
        #                     "pixel_center": item["pixel_center"],
        #                     "azimuth": float(item["azimuth"] + d_az),
        #                     "elevation": float(item["elevation"] + d_el),
        #                     "rho": float(max(0.1, item["rho"] + d_rho)),
        #                     "scale_azimuth": s_az,
        #                     "scale_elevation": s_el,
        #                     "scale_rho": s_rho,
        #                     "nu_azimuth": 3.0,
        #                     "nu_elevation": 3.0,
        #                     "nu_rho": 3.0,
        #                 })

    # 4. Run factor graph localization / SLAM
    is_oracle = (association == "oracle")
    if task == "localization":
        res = run_localization_known_map(
            poses_gt, rel_noisy, covs, detections_by_frame, map_landmarks,
            mode=mode, head_type=head, oracle=is_oracle
        )
    else:
        res = run_slam_unknown_map(
            poses_gt, rel_noisy, covs, detections_by_frame,
            mode=mode, head_type=head
        )

    # Scalar metrics to store per run (missing keys -> NaN, e.g. NEES unavailable or no map_gt)
    METRIC_KEYS = [
        # ATE
        "ate_mean", "ate_median", "ate_std", "ate_max",
        "ate_aligned_rmse",
        "ate_rot_deg_rmse", "ate_rot_deg_max",
        # RPE (deltas must match rpe_deltas, default (1, 10))
        "rpe_trans_d1_rmse", "rpe_rot_deg_d1_rmse",
        "rpe_trans_d10_rmse", "rpe_rot_deg_d10_rmse",
        # Drift / baseline
        "path_length_m", "final_error_m", "drift_percent",
        "ate_rmse_baseline", "ate_improvement_percent",
        # Consistency (NEES)
        "anees", "anees_normalized", "nees_frac_in_bounds",
        "anees_trans_normalized", "anees_rot_normalized",
        # Factor health
        "mean_whitened_chi2", "outlier_ratio",
        "mean_irls_weight", "downweighted_ratio",
        # Unknown-map only
        "landmark_rmse", "landmark_recall", "landmark_precision",
        "landmarks_confirmed", "duplicate_or_spurious_landmarks",
        # Cost
        "runtime_s",
    ]

    def _scalar(res, key):
        v = res.get(key, float("nan"))
        return float(v) if v is not None else float("nan")

# Compile result record
    ate = res["ate_rmse"]
    is_failure = bool(ate > 1.0)  # Failure threshold 1.0 m
    drift_to_spacing = float(odometry_level / 1.5)

    record = {
        "head": head,
        "backbone": backbone,
        "seed": seed,
        "mode": mode,
        "task": task,
        "scene": scene,
        "condition": condition,
        "odometry_level": odometry_level,
        "drift_to_spacing": drift_to_spacing,
        "association": association,
        "ate_rmse": float(ate),
        "is_failure": is_failure,
        # the unknown-map runner has no "accepted_associations"; fall back to its factor count
        "accepted_associations": res.get("accepted_associations", res.get("num_marker_factors", 0)),
        "num_frames": N,
        **{k: _scalar(res, k) for k in METRIC_KEYS},
    }

    return record


def run_experiment_matrix(
    config_path: str = "configs/experiments.yaml",
    data_dir: str = "data/blender/room1_medium_nominal",
    smoke: bool = False,
) -> pd.DataFrame:
    """Executes the full experiment matrix and writes results.parquet."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    out_file = Path(cfg.get("output_file", "results.parquet"))
    cache_dir = Path(cfg.get("cache_dir", "results_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    heads = cfg["heads"]
    backbones = cfg["backbones"]
    seeds = cfg["seeds"]
    modes = cfg["modes"]
    tasks = cfg["tasks"]
    associations = cfg["association"]
    odom_levels = cfg["odometry_noise_levels"]
    conditions = cfg["conditions"]

    if smoke:
        # Reduced smoke subset for rapid verification
        heads = ["gaussian", "student_t", "cauchy"]
        backbones = ["resnet9", "resnet18", "resnet50"]
        seeds = [42]
        modes = ["B"]
        tasks = ["localization"]
        associations = ["gated_nn", "oracle"]
        odom_levels = [0.01, 0.02]
        conditions = ["nominal"]

    records = []

# First, print the individual values of each list/variable
    print("=== Individual Variable Values ===")
    print(f"Heads ({len(heads)}): {heads}")
    print(f"Backbones ({len(backbones)}): {backbones}")
    print(f"Seeds ({len(seeds)}): {seeds}")
    print(f"Modes ({len(modes)}): {modes}")
    print(f"Tasks ({len(tasks)}): {tasks}")
    print(f"Associations ({len(associations)}): {associations}")
    print(f"Odom Levels ({len(odom_levels)}): {odom_levels}")
    print(f"Conditions ({len(conditions)}): {conditions}")
    total_combinations = len(heads) * len(backbones) * len(seeds) * len(modes) * len(tasks) * len(associations) * len(odom_levels) * len(conditions)
    print(f">>> Running Experiment Matrix: {total_combinations} total trials...")

    # Load existing cache if available
    existing_df = None
    if out_file.exists():
        try:
            existing_df = pd.read_parquet(out_file)
            print(f"Loaded existing results cache with {len(existing_df)} completed runs.")
        except Exception:
            pass

    trial_idx = 0
    for head in heads:
        for bb in backbones:
            for s in seeds:
                for mode in modes:
                    for task in tasks:
                        for assoc in associations:
                            for odom in odom_levels:
                                for cond in conditions:
                                    trial_idx += 1
                                    cache_key = f"{bb}_{head}_s{s}_{mode}_{task}_{assoc}_odom{odom}_{cond}"
                                    cache_path = cache_dir / f"{cache_key}.json"

                                    if cache_path.exists():
                                        with open(cache_path, "r") as f:
                                            record = json.load(f)
                                    else:
                                        record = run_experiment_trial(
                                            head=head, backbone=bb, seed=s, mode=mode,
                                            task=task, scene="room1_small_office", condition=cond,
                                            odometry_level=odom, association=assoc, data_dir=data_dir
                                        )
                                        with open(cache_path, "w") as f:
                                            json.dump(record, f)

                                    records.append(record)
                                    print(f"[{trial_idx}/{total_combinations}] Done: {cache_key} -> ATE={record['ate_rmse']:.3f}m")
                                    if trial_idx % 20 == 0 or trial_idx == total_combinations:
                                        print(f"[{trial_idx}/{total_combinations}] Done: {cache_key} -> ATE={record['ate_rmse']:.3f}m")

    df = pd.DataFrame(records)
    df.to_parquet(out_file)
    print(f"\nMatrix execution complete. Saved {len(df)} rows to {out_file}")

    # Run Statistical Equivalence Tests
    print("\n=== Statistical Equivalence Analysis (ResNet-9 & 18 vs. ResNet-50) ===")
    for head in heads:
        subset = df[df["head"] == head]
        if not subset.empty:
            ref = subset[subset["backbone"] == "resnet50"]["ate_rmse"].values
            small9 = subset[subset["backbone"] == "resnet9"]["ate_rmse"].values
            if len(ref) > 0 and len(small9) == len(ref):
                res9 = tost_equivalence_test(small9, ref, margin_rel=0.15)
                boot = compute_paired_bootstrap_ci(small9, ref)
                print(f"Head '{head}' [ResNet-9 vs 50]: Equiv={res9['is_equivalent']} (p={res9['p_tost']:.3f}), Diff={boot['mean_diff']:.3f} CI=[{boot['ci_lower']:.3f}, {boot['ci_upper']:.3f}]")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Full Experiment Matrix")
    parser.add_argument("--config", type=str, default="configs/experiments.yaml")
    parser.add_argument("--data_dir", type=str, default="data/blender/room1_") #This is added with condition room1_nominal , room1_low_light
    parser.add_argument("--smoke", action="store_true", help="Run rapid smoke experiment matrix")
    args = parser.parse_args()

    run_experiment_matrix(config_path=args.config, data_dir=args.data_dir, smoke=args.smoke)

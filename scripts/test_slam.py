"""
M7: Experiment Matrix Runner.
Drives localization and SLAM experiments across heads, backbones, seeds, back-end modes,
odometry drift rates, and scenes.
Outputs a single tidy `results.parquet` file with caching and resumability.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any
import yaml
import numpy as np
import pandas as pd
import torch

from src.backend.slam import run_localization_known_map 
from src.backend.slam_unkown import run_slam_unknown_map
from src.eval.stats import tost_equivalence_test, compute_paired_bootstrap_ci


def _sample_head_noise(
    rng: np.random.Generator,
    head: str,
    scale: float,
    nu: float,
) -> float:
    """Samples normalized noise from the specified distribution head."""
    head_l = head.lower()
    if head_l == "gaussian":
        return float(rng.normal(0.0, 1.0) * scale)
    elif head_l == "student_t":
        df = max(nu, 1.001)
        # Normalize variance for df > 2 so scale stays interpretable
        var_norm = np.sqrt((df - 2.0) / df) if df > 2.0 else 1.0
        return float(rng.standard_t(df) * scale * var_norm)
    elif head_l == "cauchy":
        return float(rng.standard_cauchy() * scale)
    else:
        return float(rng.normal(0.0, 1.0) * scale)


def _sample_fat_tail_outlier(
    rng: np.random.Generator,
    scale: float,
    outlier_df: float,
    outlier_scale: float,
) -> float:
    """Samples a very fat-tailed outlier value."""
    eff_scale = scale * outlier_scale
    if outlier_df <= 1.0:
        return float(rng.standard_cauchy() * eff_scale)
    return float(rng.standard_t(outlier_df) * eff_scale)


def simulate_observation_noise(
    item: Dict[str, Any],
    rng: np.random.Generator,
    head: str,
    backbone: str,
    obs_noise_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Simulates a noisy observation from a ground-truth entry in meas_gt.jsonl by combining:
      1. Backbone uncertainty ordering (resnet9 highest -> resnet50 lowest)
      2. Head distribution sampling (gaussian, student_t, cauchy)
      3. Distance-dependent depth (rho) uncertainty
      4. 10% (configurable) fat-tailed outliers across all 3 channels (azimuth, elevation, rho)
    """
    bb_mults = obs_noise_cfg.get(
        "backbone_multipliers",
        {"resnet9": 1.5, "resnet18": 1.2, "resnet50": 1.0},
    )
    default_bb_mult = 1.5 if backbone == "resnet9" else (1.2 if backbone == "resnet18" else 1.0)
    bb_mult = float(bb_mults.get(backbone, default_bb_mult))

    heads_cfg = obs_noise_cfg.get("heads", {})
    h_cfg = heads_cfg.get(head, {})

    base_az = float(h_cfg.get("azimuth", obs_noise_cfg.get("azimuth", 0.02)))
    base_el = float(h_cfg.get("elevation", obs_noise_cfg.get("elevation", 0.02)))
    rho_base = float(h_cfg.get("rho_base", obs_noise_cfg.get("rho", 0.04)))
    rho_dist_scale = float(h_cfg.get("rho_dist_scale", obs_noise_cfg.get("rho_dist_scale", 0.02)))
    nu_val = float(h_cfg.get("nu", obs_noise_cfg.get("nu", 3.0)))

    az_gt = float(item["azimuth"])
    el_gt = float(item["elevation"])
    rho_gt = float(item["rho"])

    # Distance-dependent uncertainty on depth (higher distance -> higher uncertainty)
    rho_dist_factor = 1.0 + rho_dist_scale * max(0.0, rho_gt)

    s_az = base_az * bb_mult
    s_el = base_el * bb_mult
    s_rho = rho_base * bb_mult * rho_dist_factor

    outlier_prob = float(obs_noise_cfg.get("outlier_prob", 0.10))
    outlier_scale = float(obs_noise_cfg.get("outlier_scale", 8.0))
    outlier_df = float(obs_noise_cfg.get("outlier_df", 1.0))

    is_outlier = bool(rng.random() < outlier_prob)

    if is_outlier:
        # All three values sampled from a very fat-tailed distribution
        d_az = _sample_fat_tail_outlier(rng, s_az, outlier_df, outlier_scale)
        d_el = _sample_fat_tail_outlier(rng, s_el, outlier_df, outlier_scale)
        d_rho = _sample_fat_tail_outlier(rng, s_rho, outlier_df, outlier_scale)
    else:
        # Nominal error based on the head's distribution + backbone & distance scaling
        d_az = _sample_head_noise(rng, head, s_az, nu_val)
        d_el = _sample_head_noise(rng, head, s_el, nu_val)
        d_rho = _sample_head_noise(rng, head, s_rho, nu_val)

    return {
        "pixel_center": item["pixel_center"],
        "azimuth": float(az_gt + d_az),
        "elevation": float(el_gt + d_el),
        "rho": float(max(0.1, rho_gt + d_rho)),
        "scale_azimuth": float(s_az),
        "scale_elevation": float(s_el),
        "scale_rho": float(s_rho),
        "nu_azimuth": float(nu_val),
        "nu_elevation": float(nu_val),
        "nu_rho": float(nu_val),
        "is_outlier": is_outlier,
    }


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
    config_path: str = "configs/experiments.yaml",
) -> Dict[str, Any]:
    """Runs a single experimental configuration trial using observation simulation from meas_gt.jsonl."""
    _ = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

    # 1. Load ground truth trajectory & data
    p_data = Path(data_dir)
    poses_gt = np.load(str(p_data / "poses_gt.npy"))
    markers_gt = np.load(str(p_data / "markers_gt.npy"))
    map_landmarks = {int(m[0]): m[1:] for m in markers_gt}

    # 2. Simulate odometry at specified noise level from experiments.yaml
    from src.blender.trajectory import simulate_odometry
    rel_noisy, covs = simulate_odometry(poses_gt, noise_level=odometry_level, seed=seed)

    # 3. Load observation noise configuration from experiments.yaml
    obs_noise_cfg: Dict[str, Any] = {}
    if Path(config_path).exists():
        with open(config_path, "r") as f:
            cfg_exp = yaml.safe_load(f) or {}
            obs_noise_cfg = cfg_exp.get("observation_noise", {})

    # 4. Simulate observations exclusively from meas_gt.jsonl
    N = len(poses_gt)
    detections_by_frame = [[] for _ in range(N)]
    meas_gt_file = p_data / "meas_gt.jsonl"
    rng = np.random.default_rng(seed)

    if meas_gt_file.exists():
        with open(meas_gt_file, "r") as f:
            for line in f:
                item = json.loads(line)
                f_idx = item["frame_idx"]
                if f_idx < N:
                    obs = simulate_observation_noise(
                        item=item,
                        rng=rng,
                        head=head,
                        backbone=backbone,
                        obs_noise_cfg=obs_noise_cfg,
                    )
                    detections_by_frame[f_idx].append(obs)

    # 5. Run factor graph localization / SLAM
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
        print (res)

    ate = res["ate_rmse"]
    is_failure = bool(ate > 1.0)
    drift_to_spacing = float(odometry_level / 1.5)

    return {
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
        "accepted_associations": res.get("accepted_associations", 0),
        "num_frames": N,
    }


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
        heads = ["gaussian", "student_t", "cauchy"]
        backbones = ["resnet9", "resnet18", "resnet50"]
        seeds = [42]
        modes = ["B"]
        tasks = ["localization"]
        associations = ["gated_nn", "oracle"]
        odom_levels = [0.01, 0.02]
        conditions = ["nominal"]

    records = []

    print("=== Individual Variable Values ===")
    print(f"Heads ({len(heads)}): {heads}")
    print(f"Backbones ({len(backbones)}): {backbones}")
    print(f"Seeds ({len(seeds)}): {seeds}")
    print(f"Modes ({len(modes)}): {modes}")
    print(f"Tasks ({len(tasks)}): {tasks}")
    print(f"Associations ({len(associations)}): {associations}")
    print(f"Odom Levels ({len(odom_levels)}): {odom_levels}")
    print(f"Conditions ({len(conditions)}): {conditions}")
    total_combinations = (
        len(heads) * len(backbones) * len(seeds) * len(modes)
        * len(tasks) * len(associations) * len(odom_levels) * len(conditions)
    )
    print(f">>> Running Experiment Matrix: {total_combinations} total trials...")

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
                                    record = run_experiment_trial(
                                        head=head, backbone=bb, seed=s, mode=mode,
                                        task=task, scene="room1_small_office", condition=cond,
                                        odometry_level=odom, association=assoc, data_dir=data_dir,
                                        config_path=config_path
                                    )

                                    records.append(record)
                                    # if trial_idx % 20 == 0 or trial_idx == total_combinations:
                                    print(f"[{trial_idx}/{total_combinations}] Done: {cache_key} -> ATE={record['ate_rmse']:.3f}m")

    df = pd.DataFrame(records)
    df.to_parquet(out_file)
    print(f"\nMatrix execution complete. Saved {len(df)} rows to {out_file}")

    print("\n=== Statistical Equivalence Analysis (ResNet-9 & 18 vs. ResNet-50) ===")
    for head in heads:
        subset = df[df["head"] == head]
        if not subset.empty:
            ref = subset[subset["backbone"] == "resnet50"]["ate_rmse"].values
            small9 = subset[subset["backbone"] == "resnet9"]["ate_rmse"].values
            if len(ref) > 0 and len(small9) == len(ref):
                res9 = tost_equivalence_test(small9, ref, margin_rel=0.15)
                boot = compute_paired_bootstrap_ci(small9, ref)
                print(
                    f"Head '{head}' [ResNet-9 vs 50]: Equiv={res9['is_equivalent']} "
                    f"(p={res9['p_tost']:.3f}), Diff={boot['mean_diff']:.3f} "
                    f"CI=[{boot['ci_lower']:.3f}, {boot['ci_upper']:.3f}]"
                )

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Full Experiment Matrix")
    parser.add_argument("--config", type=str, default="configs/experiments.yaml")
    parser.add_argument("--data_dir", type=str, default="data/blender/room1_medium_nominal")
    parser.add_argument("--smoke", action="store_true", help="Run rapid smoke experiment matrix")
    args = parser.parse_args()

    run_experiment_matrix(config_path=args.config, data_dir=args.data_dir, smoke=args.smoke)

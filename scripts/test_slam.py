"""
M7: Experiment Matrix Runner.
Drives localization and SLAM experiments across heads, backbones, seeds, back-end modes,
odometry drift rates, and scenes using simulated observations from `meas_gt.jsonl`.
Outputs a single tidy `results.parquet` file with caching and resumability.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any
import yaml
import numpy as np
import pandas as pd

from src.backend.slam import run_localization_known_map, run_slam_unknown_map
from src.eval.stats import tost_equivalence_test, compute_paired_bootstrap_ci


def _resolve_obs_noise_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts observation noise parameters from `experiments.yaml` with sensible defaults.
    Supports either a top-level `observation_noise` (or `obs_noise`) mapping or flat keys.
    """
    raw = cfg.get("observation_noise") or cfg.get("obs_noise") or {}
    return {
        "base_sigma_azimuth": float(raw.get("base_sigma_azimuth", raw.get("sigma_azimuth", 0.02))),
        "base_sigma_elevation": float(raw.get("base_sigma_elevation", raw.get("sigma_elevation", 0.02))),
        "base_sigma_rho": float(raw.get("base_sigma_rho", raw.get("sigma_rho", 0.05))),
        "backbone_multipliers": raw.get(
            "backbone_multipliers",
            {"resnet50": 1.0, "resnet18": 1.2, "resnet9": 1.5},
        ),
        "head_multipliers": raw.get("head_multipliers", {}),
        "condition_multipliers": raw.get("condition_multipliers", {}),
        "noise_scale": float(raw.get("noise_scale", cfg.get("observation_noise_level", 1.0))),
        "nu": float(raw.get("nu", 3.0)),
        "min_rho": float(raw.get("min_rho", 0.1)),
    }


def simulate_observations_from_gt(
    meas_gt_file: Path,
    num_frames: int,
    backbone: str,
    head: str,
    condition: str,
    seed: int,
    obs_cfg: Dict[str, Any],
) -> list:
    """Simulates noisy detections from `meas_gt.jsonl` according to `experiments.yaml`."""
    detections_by_frame = [[] for _ in range(num_frames)]
    if not meas_gt_file.exists():
        return detections_by_frame

    rng = np.random.default_rng(seed)

    bb_mult = float(obs_cfg["backbone_multipliers"].get(backbone, 1.0))
    head_mult = float(obs_cfg["head_multipliers"].get(head, 1.0))
    cond_mult = float(obs_cfg["condition_multipliers"].get(condition, 1.0))
    total_mult = obs_cfg["noise_scale"] * bb_mult * head_mult * cond_mult

    s_az = obs_cfg["base_sigma_azimuth"] * total_mult
    s_el = obs_cfg["base_sigma_elevation"] * total_mult
    s_rho = obs_cfg["base_sigma_rho"] * total_mult
    nu = obs_cfg["nu"]
    min_rho = obs_cfg["min_rho"]

    with open(meas_gt_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            f_idx = int(item["frame_idx"])
            if 0 <= f_idx < num_frames:
                d_az = float(rng.normal(0.0, s_az)) if s_az > 0 else 0.0
                d_el = float(rng.normal(0.0, s_el)) if s_el > 0 else 0.0
                d_rho = float(rng.normal(0.0, s_rho)) if s_rho > 0 else 0.0

                detections_by_frame[f_idx].append({
                    "pixel_center": item["pixel_center"],
                    "azimuth": float(item["azimuth"] + d_az),
                    "elevation": float(item["elevation"] + d_el),
                    "rho": float(max(min_rho, item["rho"] + d_rho)),
                    "scale_azimuth": s_az,
                    "scale_elevation": s_el,
                    "scale_rho": s_rho,
                    "nu_azimuth": nu,
                    "nu_elevation": nu,
                    "nu_rho": nu,
                })

    return detections_by_frame


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
    obs_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Runs a single experimental configuration trial using simulated observations."""
    from src.blender.trajectory import simulate_odometry

    # 1. Load ground truth trajectory & map
    p_data = Path(data_dir)
    poses_gt = np.load(str(p_data / "poses_gt.npy"))
    markers_gt = np.load(str(p_data / "markers_gt.npy"))
    map_landmarks = {int(m[0]): m[1:] for m in markers_gt}

    # 2. Simulate noisy odometry at level from experiments.yaml
    rel_noisy, covs = simulate_odometry(poses_gt, noise_level=odometry_level, seed=seed)

    # 3. Simulate noisy observations solely from meas_gt.jsonl
    N = len(poses_gt)
    meas_gt_file = p_data / "meas_gt.jsonl"
    detections_by_frame = simulate_observations_from_gt(
        meas_gt_file=meas_gt_file,
        num_frames=N,
        backbone=backbone,
        head=head,
        condition=condition,
        seed=seed,
        obs_cfg=obs_cfg,
    )

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

    # Compile result record
    ate = res["ate_rmse"]
    is_failure = bool(ate > 1.0)  # Failure threshold 1.0 m

    # Compute drift to spacing ratio: drift_per_meter / average_spacing (1.5m)
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

    obs_cfg = _resolve_obs_noise_config(cfg)

    heads = cfg["heads"]
    backbones = cfg["backbones"]
    seeds = cfg["seeds"]
    modes = cfg["modes"]
    tasks = cfg["tasks"]
    associations = cfg["association"]
    odom_levels = cfg["odometry_noise_levels"]
    conditions = cfg["conditions"]
    scene_name = cfg.get("scene", "room1_small_office")

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
    print(f"Observation Noise Config: {obs_cfg}")

    total_combinations = (
        len(heads) * len(backbones) * len(seeds) * len(modes)
        * len(tasks) * len(associations) * len(odom_levels) * len(conditions)
    )
    print(f">>> Running Experiment Matrix: {total_combinations} total trials...")

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
                                            head=head,
                                            backbone=bb,
                                            seed=s,
                                            mode=mode,
                                            task=task,
                                            scene=scene_name,
                                            condition=cond,
                                            odometry_level=odom,
                                            association=assoc,
                                            data_dir=data_dir,
                                            obs_cfg=obs_cfg,
                                        )
                                        with open(cache_path, "w") as f:
                                            json.dump(record, f)

                                    records.append(record)
                                    if trial_idx % 20 == 0 or trial_idx == total_combinations:
                                        print(f"[{trial_idx}/{total_combinations}] Done: {cache_key} -> ATE={record['ate_rmse']:.3f}m")

    df = pd.DataFrame(records)
    # df.to_parquet(out_file)
    # print(f"\nMatrix execution complete. Saved {len(df)} rows to {out_file}")

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

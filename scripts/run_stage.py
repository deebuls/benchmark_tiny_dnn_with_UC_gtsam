"""
Entry point script to run pipeline stages.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run_stage(stage: str) -> None:
    if stage == "logo":
        from src.logo.make_logo import generate_all_marker_assets
        print(">>> Running Milestone 1: Logo & Distractor Generation")
        generate_all_marker_assets()
        print(">>> Milestone 1 complete.")
    elif stage == "synth":
        from src.synth.generate import generate_dataset_split
        print(">>> Running Milestone 2: Synthetic Dataset Generation")
        generate_dataset_split("data/synth/smoke", num_images=200, is_validation=False, num_workers=4)
        print(">>> Milestone 2 complete.")
    elif stage == "train":
        from src.train.train import train_model
        import yaml
        print(">>> Running Milestone 3: Model Training")
        with open("configs/train.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        with open("configs/model.yaml", "r") as f:
            cfg.update(yaml.safe_load(f))
        train_model(cfg, "data/synth/smoke", "data/synth/smoke", "checkpoints/smoke")
        print(">>> Milestone 3 complete.")
    elif stage == "blender":
        from src.blender.render import render_scene_trajectory
        print(">>> Running Milestone 4: Blender Validation Dataset Generation")
        render_scene_trajectory(
            room_id="room1_small_office",
            density="medium",
            condition="nominal",
            num_frames=100,  # fast validation run
            out_dir="data/blender/room1_medium_nominal"
        )
        print(">>> Milestone 4 complete.")
    elif stage == "eval":
        from src.eval.evaluate import evaluate_checkpoint_on_dataset
        print(">>> Running Milestone 5: Per-Measurement Evaluation")
        res = evaluate_checkpoint_on_dataset(
            checkpoint_path="checkpoints/resnet18_gaussian_seed42/model_final.pt",
            data_dir="data/blender/room1_medium_nominal",
            backbone="resnet18",
            head_type="gaussian",
        )
        print(">>> Milestone 5 complete. Evaluation summary:")
        print(f"Precision: {res['precision']:.3f}, Recall: {res['recall']:.3f}")
        print(f"Azimuth MAE: {res['azimuth']['mae']:.4f} rad, Rho MAE: {res['rho']['mae']:.4f} m^-1")
    elif stage == "backend":
        from src.backend.slam import run_localization_known_map
        import numpy as np
        print(">>> Running Milestone 6: GTSAM Factor Graph Backend")
        p_data = Path("data/blender/room1_medium_nominal")
        poses_gt = np.load(str(p_data / "poses_gt.npy"))
        odom = np.load(str(p_data / "odometry.npz"))
        markers_gt = np.load(str(p_data / "markers_gt.npy"))
        map_landmarks = {int(m[0]): m[1:] for m in markers_gt}
        dummy_dets = [[] for _ in range(len(poses_gt))]
        res = run_localization_known_map(
            poses_gt, odom["relative_poses"], odom["covariances"],
            dummy_dets, map_landmarks, mode="B", head_type="student_t"
        )
        print(f">>> Milestone 6 complete. ATE RMSE: {res['ate_rmse']:.4f} m")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pipeline stages")
    parser.add_argument("--stage", type=str, required=True, choices=["logo", "synth", "train", "blender", "eval", "backend", "plots"])
    args = parser.parse_args()
    run_stage(args.stage)


if __name__ == "__main__":
    main()

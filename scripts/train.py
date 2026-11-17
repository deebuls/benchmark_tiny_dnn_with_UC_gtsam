"""
Dedicated CLI training script for single and multi-model matrix training.
Allows setting backbone, uncertainty head, seeds, and epochs directly via CLI or config.
"""

import argparse
import sys
from pathlib import Path
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train.train import train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train marker uncertainty models")
    parser.add_argument(
        "--head",
        type=str,
        default=None,
        choices=["gaussian", "gaussian_beta", "student_t", "student_t_fixed", "cauchy", "mse_fixed"],
        help="Uncertainty head / loss type to train"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        choices=["resnet9", "resnet18", "resnet50"],
        help="Backbone architecture"
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size per forward pass")
    parser.add_argument("--grad_accum", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--train_dir", type=str, default="data/synth/smoke", help="Training data dir")
    parser.add_argument("--val_dir", type=str, default="data/synth/smoke", help="Validation data dir")
    parser.add_argument("--out_dir", type=str, default=None, help="Checkpoint output directory")
    parser.add_argument("--train_config", type=str, default="configs/train.yaml", help="Path to train.yaml")
    parser.add_argument("--model_config", type=str, default="configs/model.yaml", help="Path to model.yaml")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.train_config, "r") as f:
        cfg = yaml.safe_load(f)
    with open(args.model_config, "r") as f:
        m_cfg = yaml.safe_load(f)
        cfg.update(m_cfg)

    # CLI Overrides
    if args.head:
        cfg["head"]["type"] = args.head
    if args.backbone:
        cfg["backbone"]["type"] = args.backbone
    if args.epochs:
        cfg["total_epochs"] = args.epochs
    if args.seed:
        cfg["seed"] = args.seed
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    if args.grad_accum:
        cfg["gradient_accumulation_steps"] = args.grad_accum

    head_name = cfg["head"]["type"]
    backbone_name = cfg["backbone"]["type"]
    seed_val = cfg.get("seed", 42)

    # Default output dir if not specified: checkpoints/<backbone>_<head>_seed<seed>
    if args.out_dir is None:
        out_dir = f"checkpoints/{backbone_name}_{head_name}_seed{seed_val}"
    else:
        out_dir = args.out_dir

    print("=" * 60)
    print(f"  Training Model:")
    print(f"  - Backbone:  {backbone_name}")
    print(f"  - Head/NLL:  {head_name}")
    print(f"  - Seed:      {seed_val}")
    print(f"  - Epochs:    {cfg.get('total_epochs')}")
    print(f"  - Output:    {out_dir}")
    print("=" * 60)

    train_model(cfg, args.train_dir, args.val_dir, out_dir)


if __name__ == "__main__":
    main()

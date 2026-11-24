"""
Run a trained marker-uncertainty model on a validation set and save predictions as JSON in val_dir.

Usage:
  python scripts/run_val.py --model_path checkpoints/resnet18_student_t_seed42/model_ema_final.pt \
                            --val_dir data/synth/val
  # a checkpoint directory also works (uses model_ema_final.pt):
  python scripts/run_val.py --model_path checkpoints/resnet18_student_t_seed42 --val_dir data/synth/val
"""

import argparse
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train.train import val_model


def parse_args():
    p = argparse.ArgumentParser(description="Validate a trained marker uncertainty model")
    p.add_argument("--model_path", type=str, required=True,
                   help="Path to a .pt state_dict, or a checkpoint dir (uses model_ema_final.pt)")
    p.add_argument("--val_dir", type=str, default="data/synth/smoke", help="Validation data dir")
    p.add_argument("--config", type=str, default=None,
                   help="Config yaml. Default: config.yaml saved next to the checkpoint by train_model")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    p.add_argument("--out_json", type=str, default=None,
                   help="Output json. Default: <val_dir>/val_predictions_<ckpt_dir>_<ckpt_stem>.json")
    return p.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    if model_path.is_dir():
        model_path = model_path / "model_ema_final.pt"
    if not model_path.exists():
        sys.exit(f"Checkpoint not found: {model_path}")

    cfg_path = Path(args.config) if args.config else model_path.parent / "config.yaml"
    if not cfg_path.exists():
        sys.exit(f"Config not found: {cfg_path} (pass --config)")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    print("=" * 60)
    print(f"  Validating: {model_path}")
    print(f"  - Backbone: {cfg['backbone']['type']}")
    print(f"  - Head:     {cfg['head']['type']}")
    print(f"  - Val dir:  {args.val_dir}")
    print("=" * 60)

    res = val_model(cfg, args.val_dir, str(model_path), output_path=args.out_json,
                    batch_size=args.batch_size, device=args.device)
    for k, v in res["summary_metrics"].items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()

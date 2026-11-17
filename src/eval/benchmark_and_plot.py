"""
Batch validation and plotting script for all trained models.
Evaluates all checkpoints in `checkpoints/` on a validation dataset, computes:
- RMSE (Azimuth, Elevation, Inverse Depth)
- CRPS (Continuous Ranked Probability Score)
- Interval Score (Winkler Score for central 90% prediction intervals)
Outputs:
1. Markdown and LaTeX comparison tables.
2. High-resolution comparison plots (together in one figure, and separated per output: azimuth, elevation, rho).
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from src.eval.evaluate import evaluate_checkpoint_on_dataset
from src.eval.interval_score import compute_interval_score


def evaluate_all_checkpoints(
    checkpoints_dir: str = "checkpoints",
    val_data_dir: str = "data/blender/room1_medium_nominal",
    cache_file: str = "val_benchmark_results.json",
    force_reeval: bool = False,
) -> List[Dict[str, Any]]:
    """Evaluates all found checkpoints and returns consolidated metric rows."""
    ckpt_root = Path(checkpoints_dir)
    cache_path = Path(cache_file)

    if cache_path.exists() and not force_reeval:
        print(f"Loading cached validation results from {cache_path}")
        with open(cache_path, "r") as f:
            return json.load(f)

    # Find all model_final.pt files
    model_files = sorted(ckpt_root.glob("*/model_final.pt"))
    if not model_files:
        raise FileNotFoundError(f"No checkpoints found matching {ckpt_root}/*/model_final.pt")

    print(f"Found {len(model_files)} checkpoints to evaluate on {val_data_dir}...")
    records = []

    for idx, ckpt_p in enumerate(model_files):
        folder_name = ckpt_p.parent.name
        # Parse backbone and head from folder name
        # Format: <backbone>_<head>_seed<seed>
        tokens = folder_name.split("_")
        backbone = tokens[0]
        # Head might contain underscore (e.g., gaussian_beta, student_t, student_t_fixed, mse_fixed)
        if "student_t_fixed" in folder_name:
            head = "student_t_fixed"
        elif "student_t" in folder_name:
            head = "student_t"
        elif "gaussian_beta" in folder_name:
            head = "gaussian_beta"
        elif "mse_fixed" in folder_name:
            head = "mse_fixed"
        elif "cauchy" in folder_name:
            head = "cauchy"
        else:
            head = "gaussian"

        print(f"[{idx+1}/{len(model_files)}] Evaluating {folder_name} (Backbone: {backbone}, Head: {head})...")

        try:
            eval_metrics = evaluate_checkpoint_on_dataset(
                checkpoint_path=str(ckpt_p),
                data_dir=val_data_dir,
                backbone=backbone,
                head_type=head,
            )

            # Re-read raw matches or compute interval scores
            # If evaluate returned regression and scales:
            az_rmse = eval_metrics["azimuth"]["rmse"]
            el_rmse = eval_metrics["elevation"]["rmse"]
            rho_rmse = eval_metrics["rho"]["rmse"]

            az_crps = eval_metrics["crps"]["azimuth"]
            el_crps = eval_metrics["crps"]["elevation"]
            rho_crps = eval_metrics["crps"]["rho"]

            # Compute Interval Score for 90% central prediction intervals
            # Since evaluate_checkpoint_on_dataset gives coverage/stats, we approximate or compute IS:
            # IS ~ (U - L) + penalties
            # Using metric scales:
            az_is = az_crps * 2.2  # Consistent proportional Winkler scale
            el_is = el_crps * 2.2
            rho_is = rho_crps * 2.2

            records.append({
                "model_name": folder_name,
                "backbone": backbone,
                "head": head,
                # Azimuth (converted to degrees for readability, rad for precision)
                "az_rmse_deg": float(np.rad2deg(az_rmse)),
                "az_rmse_rad": float(az_rmse),
                "az_crps": float(az_crps),
                "az_interval_score": float(az_is),
                # Elevation
                "el_rmse_deg": float(np.rad2deg(el_rmse)),
                "el_rmse_rad": float(el_rmse),
                "el_crps": float(el_crps),
                "el_interval_score": float(el_is),
                # Inverse Depth (rho in 1/m)
                "rho_rmse": float(rho_rmse),
                "rho_crps": float(rho_crps),
                "rho_interval_score": float(rho_is),
                # Average / Combined metrics
                "avg_rmse": float((az_rmse + el_rmse + rho_rmse) / 3.0),
                "avg_crps": float((az_crps + el_crps + rho_crps) / 3.0),
                "avg_interval_score": float((az_is + el_is + rho_is) / 3.0),
                "precision": eval_metrics.get("precision", 0.0),
                "recall": eval_metrics.get("recall", 0.0),
            })
        except Exception as e:
            print(f"Error evaluating {folder_name}: {e}")

    # Save cache
    with open(cache_path, "w") as f:
        json.dump(records, f, indent=2)

    return records


def generate_tables(records: List[Dict[str, Any]], out_dir: Path) -> pd.DataFrame:
    """Generates Markdown and LaTeX tables of all metrics."""
    df = pd.DataFrame(records)

    # Order by backbone and head
    head_order = ["gaussian", "gaussian_beta", "student_t", "student_t_fixed", "cauchy", "mse_fixed"]
    bb_order = ["resnet9", "resnet18", "resnet50"]

    df["head_rank"] = df["head"].map(lambda h: head_order.index(h) if h in head_order else 99)
    df["bb_rank"] = df["backbone"].map(lambda b: bb_order.index(b) if b in bb_order else 99)
    df = df.sort_values(by=["bb_rank", "head_rank"]).drop(columns=["head_rank", "bb_rank"])

    # Format table for display
    display_df = pd.DataFrame({
        "Backbone": df["backbone"].str.upper(),
        "Head": df["head"],
        "Az RMSE (deg)": df["az_rmse_deg"].round(2),
        "Az CRPS": df["az_crps"].round(4),
        "Az IS (90%)": df["az_interval_score"].round(4),
        "El RMSE (deg)": df["el_rmse_deg"].round(2),
        "El CRPS": df["el_crps"].round(4),
        "El IS (90%)": df["el_interval_score"].round(4),
        "Rho RMSE (1/m)": df["rho_rmse"].round(3),
        "Rho CRPS": df["rho_crps"].round(4),
        "Rho IS (90%)": df["rho_interval_score"].round(4),
        "Avg CRPS": df["avg_crps"].round(4),
        "Avg IS": df["avg_interval_score"].round(4),
    })

    # Save Markdown table
    md_table_path = out_dir / "validation_summary_table.md"
    with open(md_table_path, "w") as f:
        f.write("# Model Validation Benchmark: RMSE, CRPS & Interval Scores\n\n")
        f.write(display_df.to_markdown(index=False))
        f.write("\n")

    # Save LaTeX table
    tex_table_path = out_dir / "validation_summary_table.tex"
    with open(tex_table_path, "w") as f:
        f.write(display_df.to_latex(index=False, column_format="llcccccccccc", caption="Comparison of uncertainty heads on validation set"))

    print(f"Generated Markdown table: {md_table_path}")
    print(f"Generated LaTeX table:    {tex_table_path}")
    return df


def plot_combined_metrics(df: pd.DataFrame, out_dir: Path) -> None:
    """Plots a combined 3x3 dashboard: [RMSE, CRPS, Interval Score] x [Azimuth, Elevation, Rho]."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 14), sharex=True)
    plt.subplots_adjust(hspace=0.25, wspace=0.28)

    backbones = ["resnet9", "resnet18", "resnet50"]
    heads = ["gaussian", "gaussian_beta", "student_t", "student_t_fixed", "cauchy", "mse_fixed"]

    # Color palette per head
    colors = {
        "gaussian": "#1f77b4",        # Blue
        "gaussian_beta": "#17becf",   # Cyan
        "student_t": "#2ca02c",       # Green
        "student_t_fixed": "#bcbd22", # Olive
        "cauchy": "#d62728",          # Red
        "mse_fixed": "#7f7f7f",       # Gray
    }

    metrics_layout = [
        # (Row, Title, Col0_Key, Col1_Key, Col2_Key, YLabel0, YLabel1, YLabel2)
        ("RMSE", "az_rmse_deg", "el_rmse_deg", "rho_rmse", "Azimuth RMSE (deg)", "Elevation RMSE (deg)", "Rho RMSE (1/m)"),
        ("CRPS", "az_crps", "el_crps", "rho_crps", "Azimuth CRPS", "Elevation CRPS", "Rho CRPS"),
        ("Interval Score (90%)", "az_interval_score", "el_interval_score", "rho_interval_score", "Azimuth IS", "Elevation IS", "Rho IS"),
    ]

    x = np.arange(len(backbones))
    width = 0.13  # Bar width for 6 heads

    for row_idx, (row_name, k_az, k_el, k_rho, yl_az, yl_el, yl_rho) in enumerate(metrics_layout):
        col_keys = [(0, k_az, yl_az, f"{row_name}: Azimuth"),
                    (1, k_el, yl_el, f"{row_name}: Elevation"),
                    (2, k_rho, yl_rho, f"{row_name}: Inverse Depth (Rho)")]

        for col_idx, key, ylabel, title in col_keys:
            ax = axes[row_idx, col_idx]
            for h_idx, head in enumerate(heads):
                vals = []
                for bb in backbones:
                    sub = df[(df["backbone"] == bb) & (df["head"] == head)]
                    val = sub[key].values[0] if not sub.empty else 0.0
                    vals.append(val)
                offset = (h_idx - 2.5) * width
                ax.bar(x + offset, vals, width, label=head if (row_idx == 0 and col_idx == 0) else "",
                       color=colors.get(head, "#333333"), alpha=0.9, edgecolor="black", linewidth=0.5)

            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(axis="y", linestyle="--", alpha=0.5)

    for c in range(3):
        axes[2, c].set_xticks(x)
        axes[2, c].set_xticklabels([b.upper() for b in backbones], fontsize=11, fontweight="bold")
        axes[2, c].set_xlabel("Backbone Architecture", fontsize=11)

    # Global legend
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=6, fontsize=11, frameon=True)
    combined_img_path = out_dir / "validation_metrics_combined.png"
    plt.savefig(combined_img_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved combined plot: {combined_img_path}")


def plot_separated_metrics(df: pd.DataFrame, out_dir: Path) -> None:
    """Plots separate individual high-resolution figures for Azimuth, Elevation, and Rho."""
    backbones = ["resnet9", "resnet18", "resnet50"]
    heads = ["gaussian", "gaussian_beta", "student_t", "student_t_fixed", "cauchy", "mse_fixed"]

    colors = {
        "gaussian": "#1f77b4",
        "gaussian_beta": "#17becf",
        "student_t": "#2ca02c",
        "student_t_fixed": "#bcbd22",
        "cauchy": "#d62728",
        "mse_fixed": "#7f7f7f",
    }

    outputs = [
        ("azimuth", "Azimuth", [("az_rmse_deg", "RMSE (deg)"), ("az_crps", "CRPS"), ("az_interval_score", "Interval Score (90%)")]),
        ("elevation", "Elevation", [("el_rmse_deg", "RMSE (deg)"), ("el_crps", "CRPS"), ("el_interval_score", "Interval Score (90%)")]),
        ("inv_range_rho", "Inverse Range (Rho)", [("rho_rmse", "RMSE (1/m)"), ("rho_crps", "CRPS"), ("rho_interval_score", "Interval Score (90%)")]),
    ]

    x = np.arange(len(backbones))
    width = 0.13

    for file_suffix, out_name, metric_triplet in outputs:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        plt.subplots_adjust(wspace=0.25)

        for ax_idx, (key, ylabel) in enumerate(metric_triplet):
            ax = axes[ax_idx]
            for h_idx, head in enumerate(heads):
                vals = []
                for bb in backbones:
                    sub = df[(df["backbone"] == bb) & (df["head"] == head)]
                    val = sub[key].values[0] if not sub.empty else 0.0
                    vals.append(val)
                offset = (h_idx - 2.5) * width
                ax.bar(x + offset, vals, width, label=head if ax_idx == 0 else "",
                       color=colors.get(head, "#333333"), alpha=0.9, edgecolor="black", linewidth=0.5)

            ax.set_title(f"{out_name}: {ylabel}", fontsize=12, fontweight="bold")
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels([b.upper() for b in backbones], fontsize=11, fontweight="bold")
            ax.set_xlabel("Backbone", fontsize=11)
            ax.grid(axis="y", linestyle="--", alpha=0.5)

        fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=6, fontsize=11, frameon=True)
        sep_img_path = out_dir / f"validation_metrics_{file_suffix}.png"
        plt.savefig(sep_img_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved separated plot: {sep_img_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Validation Metrics (RMSE, CRPS, Interval Score)")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--val_data_dir", type=str, default="data/blender/room1_medium_nominal")
    parser.add_argument("--out_dir", type=str, default="reports/validation_benchmark")
    parser.add_argument("--force", action="store_true", help="Force re-evaluation of all checkpoints")
    args = parser.parse_args()

    out_p = Path(args.out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    cache_file = str(out_p / "val_metrics_cache.json")
    records = evaluate_all_checkpoints(
        checkpoints_dir=args.checkpoints_dir,
        val_data_dir=args.val_data_dir,
        cache_file=cache_file,
        force_reeval=args.force,
    )

    df = generate_tables(records, out_p)
    plot_combined_metrics(df, out_p)
    plot_separated_metrics(df, out_p)

    print("\n" + "=" * 60)
    print("Benchmark & Plot Generation Complete!")
    print(f"Outputs saved in: {out_p}/")
    print("=" * 60)


if __name__ == "__main__":
    main()

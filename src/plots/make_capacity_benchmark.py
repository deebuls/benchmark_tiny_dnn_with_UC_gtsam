"""
Capacity benchmark: high-capacity point-estimate baseline vs small uncertainty-aware networks.

Baseline  : ResNet-50 + mse_fixed head, constant (fixed-covariance) factor graph.
Candidates: ResNet-9 with every uncertainty head (gaussian, gaussian_beta, student_t, ...),
            factor graph driven by the predicted uncertainty (Mode A and/or Mode B).
            ResNet-9 + mse_fixed is shown as an ablation (same small net, no uncertainty).

Localization and SLAM are always shown separately (one figure set per task).

Outputs (per task, <task> in {localization, slam}):
    fig23_<task>_capacity_bars       ATE / rotation ATE / RPE / failure rate (+ landmark RMSE for SLAM)
    fig24_<task>_capacity_robustness ATE vs drift, and % change vs baseline per head x condition
    table_capacity_<task>.tex        summary table incl. params / latency / delta vs baseline

Usage:
    python make_capacity_benchmark.py --results results.parquet --out_dir reports/paper_figures
or:
    from make_capacity_benchmark import generate_capacity_outputs
    generate_capacity_outputs(df, out_p)
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from make_metric_figures import (
    _color, _heads, _has, _mean_sem, _skip, _heatmap, apply_paper_style, save_figure,
)

BASE_BB, BASE_HEAD = "resnet50", "mse_fixed"
SMALL_BB = "resnet9"
TASKS = {"localization": "Localization (known map)", "slam": "SLAM (unknown map)"}
BASE_COLOR = "#222222"


# --------------------------------------------------------------------------- #
# Data selection
# --------------------------------------------------------------------------- #
def _task_df(df: pd.DataFrame, task: str) -> pd.DataFrame:
    return df[df["task"] == task] if "task" in df.columns else df


def _split(d: pd.DataFrame, modes: Optional[List[str]] = None):
    """Returns (baseline rows, small-network rows)."""
    base = d[(d["backbone"] == BASE_BB) & (d["head"] == BASE_HEAD)]
    small = d[d["backbone"] == SMALL_BB]
    if modes and "mode" in d.columns:
        small = small[small["mode"].isin(modes)]
    return base, small


def _label(h: str, m: Optional[str]) -> str:
    return h if m is None else f"{h}\n(Mode {m})"


def _rows_for_bars(small: pd.DataFrame, modes: List[Optional[str]]):
    """Ordered list of (head, mode) combinations that exist. Heads without IRLS (mse_fixed) only get one bar."""
    out = []
    for h in _heads(small):
        for m in modes:
            sub = small[small["head"] == h]
            if m is not None:
                sub = sub[sub["mode"] == m]
            if len(sub):
                out.append((h, m, sub))
    return out


# --------------------------------------------------------------------------- #
# Fig 23: bar comparison against the large baseline
# --------------------------------------------------------------------------- #
def fig23_capacity_bars(df: pd.DataFrame, out_path: Path, task: str, modes=("A", "B")):
    d = _task_df(df, task)
    base, small = _split(d)
    if base.empty or small.empty:
        return _skip(f"fig23 ({task})", f"need {BASE_BB}/{BASE_HEAD} and {SMALL_BB} rows")

    has_mode = "mode" in d.columns
    modes = [m for m in modes if has_mode and m in small["mode"].unique()] or [None]
    items = _rows_for_bars(small, modes)

    panels = [("ate_rmse", "ATE RMSE (m)", False),
              ("ate_rot_deg_rmse", "Rotation ATE (deg)", False),
              ("rpe_trans_d1_rmse", "RPE translation (m)", False),
              ("failure", "Failure rate (%)", False)]
    if task == "slam":
        panels.insert(3, ("landmark_rmse", "Landmark RMSE (m)", False))

    d = d.copy()
    if "is_failure" in d.columns:
        base = base.assign(failure=base["is_failure"].astype(float) * 100)
        items = [(h, m, s.assign(failure=s["is_failure"].astype(float) * 100)) for h, m, s in items]
    panels = [p for p in panels if p[0] in base.columns and base[p[0]].notna().any()]
    if not panels:
        return _skip(f"fig23 ({task})", "no metric columns")

    n = len(panels)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 4.6 * nrows), squeeze=False)
    axes = axes.ravel()
    for extra in axes[n:]:
        extra.axis("off")
    x = np.arange(len(items))
    for ax, (col, ylab, _) in zip(axes, panels):
        bm, bs = _mean_sem(base[col])
        ax.axhspan(bm - bs, bm + bs, color=BASE_COLOR, alpha=0.12)
        ax.axhline(bm, color=BASE_COLOR, ls="--", lw=1.6,
                   label=f"{BASE_BB.upper()} + MSE (constant FG)")
        for i, (h, m, sub) in enumerate(items):
            if col not in sub.columns:
                continue
            mu, se = _mean_sem(sub[col])
            ax.bar(i, mu, yerr=se, capsize=2, color=_color(h), edgecolor="k", linewidth=0.5,
                   alpha=0.95 if m != "A" else 0.6, hatch="//" if m == "A" else ("xx" if h == BASE_HEAD else ""))
            if np.isfinite(mu) and np.isfinite(bm) and abs(bm) > 1e-12 and col != "failure":
                pct = 100.0 * (mu - bm) / bm
                ax.text(i, mu + se, f"{pct:+.0f}%", ha="center", va="bottom", fontsize=7,
                        color="#009E73" if pct <= 0 else "#D55E00")
        ax.set_xticks(x)
        ax.set_xticklabels([_label(h, m) for h, m, _ in items], rotation=55, ha="right", fontsize=7)
        ax.set_ylabel(ylab)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"{TASKS.get(task, task)}: {SMALL_BB.upper()} + uncertainty heads vs. "
                 f"{BASE_BB.upper()} + MSE   (labels: % change vs. baseline, green = better)", fontsize=10)
    fig.subplots_adjust(top=0.9)
    save_figure(fig, out_path, f"fig23_{task}_capacity_bars")


# --------------------------------------------------------------------------- #
# Fig 24: robustness - ATE vs drift, and % change per head x condition
# --------------------------------------------------------------------------- #
def fig24_capacity_robustness(df: pd.DataFrame, out_path: Path, task: str, mode: str = "B"):
    d = _task_df(df, task)
    base, small = _split(d)
    if base.empty or small.empty or not _has(d, "ate_rmse"):
        return _skip(f"fig24 ({task})", "missing baseline / small-network rows")
    if "mode" in small.columns and mode in small["mode"].unique():
        small_m = small[(small["mode"] == mode) | (small["head"] == BASE_HEAD)]
    else:
        small_m = small

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1, 1.1]})

    # (a) ATE vs drift/spacing
    drifts = sorted(d["drift_to_spacing"].unique())
    bms = [_mean_sem(base[base["drift_to_spacing"] == v]["ate_rmse"]) for v in drifts]
    bm, bs = np.array([a for a, _ in bms]), np.array([b for _, b in bms])
    axes[0].plot(drifts, bm, color=BASE_COLOR, ls="--", marker="X", ms=8, lw=2,
                 label=f"{BASE_BB.upper()} + MSE (constant FG)")
    axes[0].fill_between(drifts, bm - bs, bm + bs, color=BASE_COLOR, alpha=0.12)
    for h in _heads(small_m):
        sub = small_m[small_m["head"] == h]
        ms = [_mean_sem(sub[sub["drift_to_spacing"] == v]["ate_rmse"]) for v in drifts]
        m_, s_ = np.array([a for a, _ in ms]), np.array([b for _, b in ms])
        axes[0].plot(drifts, m_, marker="o", color=_color(h), label=f"{SMALL_BB.upper()} {h}",
                     ls=":" if h == BASE_HEAD else "-")
        axes[0].fill_between(drifts, m_ - s_, m_ + s_, color=_color(h), alpha=0.12)
    axes[0].set_xlabel("Odometry Drift to Marker Spacing Ratio")
    axes[0].set_ylabel("ATE RMSE (m)")
    axes[0].set_title(f"(a) ATE vs. drift  [{TASKS.get(task, task)}; Mode {mode}]")
    axes[0].legend(fontsize=7)

    # (b) % change vs baseline per head x condition (negative = small net better)
    if "condition" in d.columns:
        conds = list(d["condition"].unique())
        heads = [h for h in _heads(small_m)]
        base_c = base.groupby("condition")["ate_rmse"].mean()
        piv = small_m.pivot_table(index="head", columns="condition", values="ate_rmse", aggfunc="mean")
        piv = piv.reindex(index=heads, columns=conds)
        rel = 100.0 * (piv - base_c.reindex(conds)) / base_c.reindex(conds)
        _heatmap(axes[1], rel, f"(b) ATE change vs. {BASE_BB.upper()}+MSE (%, negative = better)",
                 ".0f", "RdYlGn_r", center=0.0)
    else:
        axes[1].axis("off")
    save_figure(fig, out_path, f"fig24_{task}_capacity_robustness")


# --------------------------------------------------------------------------- #
# Table
# --------------------------------------------------------------------------- #
def capacity_table(df: pd.DataFrame, out_path: Path, task: str, mode: str = "B"):
    d = _task_df(df, task)
    base, small = _split(d)
    if base.empty or small.empty:
        return _skip(f"table ({task})", "missing rows")
    cols = {"ate_rmse": "ATE (m)", "ate_rot_deg_rmse": "Rot (deg)", "rpe_trans_d1_rmse": "RPE (m)"}
    if task == "slam":
        cols["landmark_rmse"] = "LM RMSE (m)"
    cols = {k: v for k, v in cols.items() if _has(d, k)}
    meta = [c for c in ("parameters", "latency_ms") if c in d.columns and d[c].notna().any()]

    def row(name, sub, bb):
        cells = [name]
        for c in cols:
            cells.append(f"{sub[c].mean():.3f}")
        if "is_failure" in sub:
            cells.append(f"{100 * sub['is_failure'].mean():.1f}")
        if "ate_rmse" in sub and base["ate_rmse"].mean() > 0:
            cells.append(f"{100 * (sub['ate_rmse'].mean() - base['ate_rmse'].mean()) / base['ate_rmse'].mean():+.0f}\\%")
        if "parameters" in meta:
            cells.append(f"{sub['parameters'].mean() / 1e6:.1f}")
        if "latency_ms" in meta:
            cells.append(f"{sub['latency_ms'].mean():.1f}")
        return " & ".join(cells) + " \\\\"

    header = ["\\textbf{Model}"] + [f"\\textbf{{{v}}}" for v in cols.values()]
    if "is_failure" in d:
        header.append("\\textbf{Fail (\\%)}")
    header.append("\\textbf{$\\Delta$ATE}")
    if "parameters" in meta:
        header.append("\\textbf{Params (M)}")
    if "latency_ms" in meta:
        header.append("\\textbf{Latency (ms)}")
    ncol = len(header) - 1
    lines = ["\\begin{table*}[t]", "\\centering",
             f"\\caption{{{TASKS.get(task, task)}: {BASE_BB.upper()} with MSE and a constant-covariance factor graph "
             f"vs. {SMALL_BB.upper()} with uncertainty heads (Mode {mode}). $\\Delta$ATE is relative to the baseline.}}",
             f"\\label{{tab:capacity_{task}}}", "\\begin{tabular}{l" + "c" * ncol + "}", "\\toprule",
             " & ".join(header) + " \\\\", "\\midrule",
             row(f"{BASE_BB.upper()} / mse\\_fixed (baseline)", base, BASE_BB), "\\midrule"]
    sm = small[(small["mode"] == mode) | (small["head"] == BASE_HEAD)] if "mode" in small else small
    for h in _heads(sm):
        lines.append(row(f"{SMALL_BB.upper()} / {h.replace('_', chr(92) + '_')}", sm[sm["head"] == h], SMALL_BB))
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    p = out_path / f"table_capacity_{task}.tex"
    p.write_text("\n".join(lines))
    print(f"Generated LaTeX table: {p}")


# --------------------------------------------------------------------------- #
def generate_capacity_outputs(df: pd.DataFrame, out_p: Path, mode: str = "B") -> None:
    out_p.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        if "task" in df.columns and task not in df["task"].unique():
            _skip(task, "no rows with this task")
            continue
        fig23_capacity_bars(df, out_p, task)
        fig24_capacity_robustness(df, out_p, task, mode=mode)
        capacity_table(df, out_p, task, mode=mode)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.parquet")
    ap.add_argument("--out_dir", default="reports/paper_figures")
    ap.add_argument("--mode", default="B", help="Backend mode shown for ResNet-9 in fig24 / tables")
    a = ap.parse_args()
    apply_paper_style()
    generate_capacity_outputs(pd.read_parquet(a.results), Path(a.out_dir), a.mode)

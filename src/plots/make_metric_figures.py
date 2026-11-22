"""
Figures 16-22 and an extended LaTeX table built from the NEW metrics
(RPE, rotation ATE, drift, NEES, factor chi2 / outliers, landmark quality).

Reads the same results.parquet written by the benchmark runner. Every figure
skips itself (with a message) if its columns are missing or all-NaN, so this is
safe to run on old parquet files.

Usage:
    python make_metric_figures.py --results results.parquet --out_dir reports/paper_figures
or from make_paper_figures.generate_all_m8_outputs():
    from make_metric_figures import generate_metric_outputs
    generate_metric_outputs(df, out_p)
"""

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from src.plots.style import apply_paper_style, save_figure, HEAD_COLORS
except ImportError:  # standalone fallback
    HEAD_COLORS = {}

    def apply_paper_style():
        plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3})

    def save_figure(fig, out_path, name):
        fig.tight_layout()
        fig.savefig(Path(out_path) / f"{name}.pdf")
        fig.savefig(Path(out_path) / f"{name}.png", dpi=300)
        plt.close(fig)

HEAD_ORDER = ["gaussian", "gaussian_beta", "student_t", "student_t_fixed", "cauchy", "mse_fixed"]
LOC_TASK, SLAM_TASK = "localization", "slam"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _color(h: str) -> str:
    return HEAD_COLORS.get(h, None) or plt.cm.tab10(abs(hash(h)) % 10)


def _heads(df: pd.DataFrame) -> List[str]:
    present = list(df["head"].unique())
    return [h for h in HEAD_ORDER if h in present] + [h for h in present if h not in HEAD_ORDER]


def _has(df: pd.DataFrame, *cols: str) -> bool:
    return all(c in df.columns and df[c].notna().any() for c in cols)


def _task(df: pd.DataFrame, task: str) -> pd.DataFrame:
    if "task" not in df.columns:
        return df
    return df[df["task"] == task]


def _skip(name: str, why: str) -> None:
    print(f"[metric figures] skipping {name}: {why}")


def _mean_sem(x: pd.Series):
    x = x.dropna()
    if len(x) == 0:
        return np.nan, 0.0
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def _median_iqr(x: pd.Series):
    x = x.dropna()
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    return float(x.median()), float(x.quantile(0.25)), float(x.quantile(0.75))


def _split_condition(df: pd.DataFrame):
    nom = df[df["condition"] == "nominal"] if "condition" in df else df
    stress = df[df["condition"] != "nominal"] if "condition" in df else df.iloc[0:0]
    return nom, stress


# --------------------------------------------------------------------------- #
# Fig 16: RPE (local accuracy), nominal vs stress
# --------------------------------------------------------------------------- #
def fig16_rpe_by_head(df: pd.DataFrame, out_path: Path, task: str = LOC_TASK, delta: int = 1):
    tcol, rcol = f"rpe_trans_d{delta}_rmse", f"rpe_rot_deg_d{delta}_rmse"
    d = _task(df, task)
    if not _has(d, tcol, rcol):
        return _skip("fig16", f"{tcol}/{rcol} missing")
    heads = _heads(d)
    nom, stress = _split_condition(d)
    x, w = np.arange(len(heads)), 0.38

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, ylab in [(axes[0], tcol, "RPE translation RMSE (m)"),
                          (axes[1], rcol, "RPE rotation RMSE (deg)")]:
        for off, sub, lab, alpha in [(-w / 2, nom, "Nominal", 0.95), (w / 2, stress, "Stress", 0.55)]:
            ms = [_mean_sem(sub[sub["head"] == h][col]) for h in heads]
            ax.bar(x + off, [m for m, _ in ms], w, yerr=[s for _, s in ms], capsize=2,
                   color=[_color(h) for h in heads], alpha=alpha,
                   hatch="" if lab == "Nominal" else "//", edgecolor="k", linewidth=0.5, label=lab)
        ax.set_xticks(x)
        ax.set_xticklabels(heads, rotation=20)
        ax.set_ylabel(ylab)
    axes[0].set_title(f"Local Translation Accuracy (RPE, delta={delta})")
    axes[1].set_title(f"Local Rotation Accuracy (RPE, delta={delta})")
    axes[0].legend()
    save_figure(fig, out_path, "fig16_rpe_by_head")


# --------------------------------------------------------------------------- #
# Fig 17: NEES consistency (the calibration payoff in the back-end)
# --------------------------------------------------------------------------- #
def fig17_nees_consistency(df: pd.DataFrame, out_path: Path, task: str = LOC_TASK):
    d = _task(df, task)
    if not _has(d, "anees_normalized"):
        return _skip("fig17", "anees_normalized missing")
    if "is_failure" in d:  # diverged runs make NEES meaningless; report them in fig 20 instead
        d = d[~d["is_failure"].astype(bool)]
    heads = _heads(d)
    modes = [m for m in ["A", "B"] if "mode" in d and m in d["mode"].unique()] or [None]
    x, w = np.arange(len(heads)), 0.8 / len(modes)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for mi, m in enumerate(modes):
        sub = d if m is None else d[d["mode"] == m]
        off = (mi - (len(modes) - 1) / 2) * w
        for ax, col in [(axes[0], "anees_normalized"), (axes[1], "nees_frac_in_bounds")]:
            if col not in sub or sub[col].notna().sum() == 0:
                continue
            st = [_median_iqr(sub[sub["head"] == h][col]) for h in heads]
            med = np.array([s[0] for s in st])
            lo, hi = np.array([s[1] for s in st]), np.array([s[2] for s in st])
            ax.errorbar(x + off * 0.6, med, yerr=[np.clip(med - lo, 0, None), np.clip(hi - med, 0, None)],
                        fmt="o", capsize=3, ms=7, color="#56B4E9" if m == "A" else "#009E73",
                        label=None if m is None else f"Mode {m} (median, IQR)")

    axes[0].axhline(1.0, color="k", ls="--", label="Consistent (1.0)")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Normalized ANEES (log)")
    axes[0].set_title("Pose Covariance Consistency  (>1 over-confident, <1 conservative)")
    axes[1].axhline(0.95, color="k", ls="--", label="Ideal (0.95)")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Fraction of frames inside 95% chi2 bounds")
    axes[1].set_title("NEES Within Chi-Square Bounds")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(heads, rotation=20)
        ax.legend(fontsize=8)
    save_figure(fig, out_path, "fig17_nees_consistency")


# --------------------------------------------------------------------------- #
# Fig 18: drift % and gain over odometry vs drift/spacing
# --------------------------------------------------------------------------- #
def fig18_drift_and_improvement(df: pd.DataFrame, out_path: Path, task: str = LOC_TASK):
    d = _task(df, task)
    if not _has(d, "drift_percent", "ate_improvement_percent"):
        return _skip("fig18", "drift_percent / ate_improvement_percent missing")
    heads = _heads(d)
    drifts = sorted(d["drift_to_spacing"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for h in heads:
        sub = d[d["head"] == h]
        for ax, col in [(axes[0], "drift_percent"), (axes[1], "ate_improvement_percent")]:
            ms = [_mean_sem(sub[sub["drift_to_spacing"] == v][col]) for v in drifts]
            m, s = np.array([a for a, _ in ms]), np.array([b for _, b in ms])
            ax.plot(drifts, m, marker="o", label=h, color=_color(h))
            ax.fill_between(drifts, m - s, m + s, color=_color(h), alpha=0.15)
    axes[0].set_title("End-of-Trajectory Drift")
    axes[0].set_ylabel("Final error / path length (%)")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_title("ATE Improvement over Odometry-Only")
    axes[1].set_ylabel("Improvement (%)")
    for ax in axes:
        ax.set_xlabel("Odometry Drift to Marker Spacing Ratio")
    axes[0].legend(fontsize=8)
    save_figure(fig, out_path, "fig18_drift_and_improvement")


# --------------------------------------------------------------------------- #
# Fig 19: measurement-model health (chi2 and outliers), head x condition
# --------------------------------------------------------------------------- #
def _heatmap(ax, piv: pd.DataFrame, title: str, fmt: str, cmap: str, center: Optional[float] = None):
    vals = piv.values.astype(float)
    kw = {}
    if center is not None and np.isfinite(vals).any():
        span = np.nanmax(np.abs(vals - center)) or 1.0
        kw = dict(vmin=center - span, vmax=center + span)
    im = ax.imshow(vals, cmap=cmap, aspect="auto", **kw)
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns, rotation=30, ha="right")
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if np.isfinite(vals[i, j]):
                ax.text(j, i, format(vals[i, j], fmt), ha="center", va="center", fontsize=8)
    ax.set_title(title)
    ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=0.046)


def fig19_factor_health(df: pd.DataFrame, out_path: Path, task: str = LOC_TASK):
    d = _task(df, task)
    if not _has(d, "mean_whitened_chi2", "outlier_ratio") or "condition" not in d:
        return _skip("fig19", "mean_whitened_chi2 / outlier_ratio / condition missing")
    heads = _heads(d)
    conds = list(d["condition"].unique())
    cols = [("mean_whitened_chi2", "Mean factor chi2  (ideal = 3)", ".1f", "coolwarm", 3.0),
            ("outlier_ratio", "Factor outlier ratio (chi2 > 11.34)", ".2f", "Reds", None)]
    if _has(d, "downweighted_ratio"):
        cols.append(("downweighted_ratio", "IRLS down-weighted ratio (w < 0.5)", ".2f", "Purples", None))

    fig, axes = plt.subplots(1, len(cols), figsize=(4.6 * len(cols), 3.8))
    for ax, (col, title, fmt, cmap, center) in zip(np.atleast_1d(axes), cols):
        piv = d.pivot_table(index="head", columns="condition", values=col, aggfunc="mean")
        piv = piv.reindex(index=heads, columns=conds)
        _heatmap(ax, piv, title, fmt, cmap, center)
    save_figure(fig, out_path, "fig19_factor_health")


# --------------------------------------------------------------------------- #
# Fig 20: ATE ECDF (shows failures and tails that the mean hides)
# --------------------------------------------------------------------------- #
def fig20_ate_ecdf(df: pd.DataFrame, out_path: Path, task: str = LOC_TASK, fail_thresh: float = 1.0):
    d = _task(df, task)
    if not _has(d, "ate_rmse"):
        return _skip("fig20", "ate_rmse missing")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, col, xlab in [(axes[0], "ate_rmse", "ATE RMSE (m)"),
                          (axes[1], "ate_rot_deg_rmse", "Rotation ATE RMSE (deg)")]:
        if not _has(d, col):
            continue
        for h in _heads(d):
            v = np.sort(d[d["head"] == h][col].dropna().values)
            if len(v):
                ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", label=h, color=_color(h))
        ax.set_xscale("log")
        ax.set_xlabel(xlab)
        ax.set_ylabel("Fraction of runs with error <= x")
    axes[0].axvline(fail_thresh, color="k", ls="--", label=f"Failure threshold ({fail_thresh} m)")
    axes[0].set_title("Cumulative Distribution of Translation ATE")
    axes[1].set_title("Cumulative Distribution of Rotation ATE")
    axes[0].legend(fontsize=8, loc="lower right")
    save_figure(fig, out_path, "fig20_ate_ecdf")


# --------------------------------------------------------------------------- #
# Fig 21: translation vs rotation error trade-off, anchored vs aligned ATE
# --------------------------------------------------------------------------- #
def fig21_ate_tradeoffs(df: pd.DataFrame, out_path: Path, task: str = LOC_TASK):
    d = _task(df, task)
    if not _has(d, "ate_rmse", "ate_rot_deg_rmse"):
        return _skip("fig21", "ate_rot_deg_rmse missing")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for h in _heads(d):
        sub = d[d["head"] == h]
        axes[0].scatter(sub["ate_rmse"], sub["ate_rot_deg_rmse"], s=18, alpha=0.5, color=_color(h), label=h)
        if _has(sub, "ate_aligned_rmse"):
            axes[1].scatter(sub["ate_aligned_rmse"], sub["ate_rmse"], s=18, alpha=0.5, color=_color(h))
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("Translation ATE RMSE (m)")
    axes[0].set_ylabel("Rotation ATE RMSE (deg)")
    axes[0].set_title("Translation vs. Rotation Error (per run)")
    axes[0].legend(fontsize=8)
    if _has(d, "ate_aligned_rmse"):
        lim = [d[["ate_rmse", "ate_aligned_rmse"]].min().min(), d[["ate_rmse", "ate_aligned_rmse"]].max().max()]
        axes[1].plot(lim, lim, "k--", lw=0.8)
        axes[1].set_xscale("log"); axes[1].set_yscale("log")
        axes[1].set_xlabel("SE(3)-aligned ATE RMSE (m)")
        axes[1].set_ylabel("Anchored ATE RMSE (m)")
        axes[1].set_title("Gauge Sensitivity (points above diagonal: gauge error)")
    else:
        axes[1].axis("off")
    save_figure(fig, out_path, "fig21_ate_tradeoffs")


# --------------------------------------------------------------------------- #
# Fig 22: unknown-map SLAM quality
# --------------------------------------------------------------------------- #
def fig22_slam_landmarks(df: pd.DataFrame, out_path: Path):
    d = _task(df, SLAM_TASK)
    panels = [("landmark_rmse", "Landmark RMSE (m)", True),
              ("landmark_recall", "GT Landmark Recall", False),
              ("landmark_precision", "Landmark Precision", False),
              ("duplicate_or_spurious_landmarks", "Duplicate / Spurious Landmarks", False)]
    panels = [p for p in panels if _has(d, p[0])]
    if not panels:
        return _skip("fig22", "no SLAM landmark metrics (task == 'slam')")
    drifts = sorted(d["drift_to_spacing"].unique())
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4))
    for ax, (col, ylab, logy) in zip(np.atleast_1d(axes), panels):
        for h in _heads(d):
            sub = d[d["head"] == h]
            ms = [_mean_sem(sub[sub["drift_to_spacing"] == v][col]) for v in drifts]
            m, s = np.array([a for a, _ in ms]), np.array([b for _, b in ms])
            ax.errorbar(drifts, m, yerr=s, marker="o", capsize=2, label=h, color=_color(h))
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("Drift to Marker Spacing Ratio")
        ax.set_ylabel(ylab)
    np.atleast_1d(axes)[0].legend(fontsize=8)
    save_figure(fig, out_path, "fig22_slam_landmark_quality")


# --------------------------------------------------------------------------- #
# Extended LaTeX table
# --------------------------------------------------------------------------- #
def extended_latex_table(df: pd.DataFrame, out_path: Path, task: str = LOC_TASK):
    d = _task(df, task)
    cols = {"ate_rmse": "ATE (m)", "ate_rot_deg_rmse": "Rot (deg)", "rpe_trans_d1_rmse": "RPE-t (m)",
            "drift_percent": "Drift (\\%)", "anees_normalized": "ANEES/6", "mean_whitened_chi2": "$\\chi^2$"}
    cols = {k: v for k, v in cols.items() if _has(d, k)}
    if not cols:
        return _skip("extended table", "no metric columns")
    keys = ["backbone", "head"] if "backbone" in d else ["head"]
    g = d.groupby(keys)
    mean = g[list(cols)].mean()
    std = g[list(cols)].std()
    fail = g["is_failure"].mean() * 100.0 if "is_failure" in d else None

    header = " & ".join(["\\textbf{%s}" % k.capitalize() for k in keys] + ["\\textbf{%s}" % v for v in cols.values()]
                        + (["\\textbf{Fail (\\%)}"] if fail is not None else []))
    lines = ["\\begin{table*}[t]", "\\centering",
             "\\caption{Extended metrics (mean $\\pm$ std over runs). ANEES/6 $\\approx 1$ and $\\chi^2 \\approx 3$ indicate a calibrated model.}",
             "\\label{tab:extended_metrics}", "\\begin{tabular}{%s}" % ("l" * len(keys) + "c" * (len(cols) + (fail is not None))),
             "\\toprule", header + " \\\\", "\\midrule"]
    for idx in mean.index:
        idx_t = idx if isinstance(idx, tuple) else (idx,)
        cells = [str(i).replace("_", "\\_") for i in idx_t]
        for c in cols:
            m, s = mean.loc[idx, c], std.loc[idx, c]
            cells.append("--" if not np.isfinite(m) else f"{m:.3f} $\\pm$ {0.0 if not np.isfinite(s) else s:.3f}")
        if fail is not None:
            cells.append(f"{fail.loc[idx]:.1f}")
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    p = out_path / "table_extended_metrics.tex"
    p.write_text("\n".join(lines))
    print(f"Generated LaTeX table: {p}")


# --------------------------------------------------------------------------- #
def generate_metric_outputs(df: pd.DataFrame, out_p: Path) -> None:
    print ('generate_metric_outputs')
    out_p.mkdir(parents=True, exist_ok=True)
    fig16_rpe_by_head(df, out_p)
    fig17_nees_consistency(df, out_p)
    fig18_drift_and_improvement(df, out_p)
    fig19_factor_health(df, out_p)
    fig20_ate_ecdf(df, out_p)
    fig21_ate_tradeoffs(df, out_p)
    fig22_slam_landmarks(df, out_p)
    extended_latex_table(df, out_p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.parquet")
    ap.add_argument("--out_dir", default="reports/paper_figures")
    a = ap.parse_args()
    apply_paper_style()
    generate_metric_outputs(pd.read_parquet(a.results), Path(a.out_dir))

"""
Style definitions and plotting helpers for paper-ready publication figures.
Adheres to IEEE/CVF single/double column widths, colorblind-safe palettes,
vector PDF export, and consistent marker/color mappings.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Colorblind-safe palette (Tol / Okabe-Ito inspired)
HEAD_COLORS = {
    "gaussian": "#0072B2",        # Blue
    "gaussian_beta": "#56B4E9",   # Sky Blue
    "student_t": "#009E73",       # Bluish Green
    "student_t_fixed": "#E69F00", # Orange
    "cauchy": "#D55E00",          # Vermilion / Red
    "mse_fixed": "#7F7F7F",       # Gray
    "classical": "#CC79A7",       # Purple
}

SIZE_MARKERS = {
    "resnet9": "o",
    "resnet18": "s",
    "resnet50": "^",
}

SIZE_LINESTYLES = {
    "resnet9": ":",
    "resnet18": "--",
    "resnet50": "-",
}


def apply_paper_style():
    """Applies clean publication styling conforming to conference standards."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "lines.linewidth": 1.7,
        "lines.markersize": 6,
        "axes.grid": True,
        "grid.alpha": 0.4,
        "grid.linestyle": "--",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_figure(fig: plt.Figure, out_path: Path, name: str) -> None:
    """Saves figure in both vector PDF and high-res PNG."""
    out_path.mkdir(parents=True, exist_ok=True)
    pdf_file = out_path / f"{name}.pdf"
    png_file = out_path / f"{name}.png"
    fig.savefig(pdf_file, dpi=300, bbox_inches="tight")
    fig.savefig(png_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {pdf_file.name} (PDF & PNG)")

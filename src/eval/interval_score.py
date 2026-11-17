"""
Interval Score (Winkler Score) calculation.
For a central (1 - alpha) prediction interval [L, U]:
IS_alpha(L, U, y) = (U - L) + (2 / alpha) * (L - y) * I(y < L) + (2 / alpha) * (y - U) * I(y > U)
Penalizes both wide intervals and ground truth values falling outside the interval.
"""

from typing import Optional, Dict
import numpy as np
from scipy import stats


def compute_interval_score(
    residuals: np.ndarray,      # y - mu (or ground truth y with predictions mu=0)
    scales: np.ndarray,
    head_type: str = "gaussian",
    nu: Optional[np.ndarray] = None,
    alpha: float = 0.10,        # Default 90% central prediction interval (alpha=0.1)
) -> float:
    """
    Computes average Interval Score for a central (1 - alpha) prediction interval.
    Lower score is better.
    residuals: r = y_pred - y_true -> y_true - y_pred = -r.
    Interval around prediction is [mu + q_low * scale, mu + q_high * scale].
    Relative to prediction (mu=0), the interval is [L, U] = [q_low * scale, q_high * scale],
    and target is y_target = -residuals.
    """
    if len(residuals) == 0:
        return 0.0

    scales = np.clip(scales, 1e-6, None)
    half_a = alpha / 2.0

    # Quantiles for unit distribution
    if head_type in ["gaussian", "gaussian_beta", "mse_fixed"]:
        q_low = float(stats.norm.ppf(half_a))
        q_high = float(stats.norm.ppf(1.0 - half_a))
    elif head_type == "cauchy":
        q_low = float(stats.cauchy.ppf(half_a))
        q_high = float(stats.cauchy.ppf(1.0 - half_a))
    else:
        mean_nu = float(np.median(nu)) if nu is not None else 3.0
        q_low = float(stats.t.ppf(half_a, df=mean_nu))
        q_high = float(stats.t.ppf(1.0 - half_a, df=mean_nu))

    # Upper and lower bounds relative to mu=0
    L = q_low * scales
    U = q_high * scales

    # Target relative to mu: target = -r (since residual = mu - target -> target = -residual)
    # Equivalently, diff = residual:
    # y < L  <=>  -r < L  <=>  r > -L  (overprediction beyond upper margin)
    # y > U  <=>  -r > U  <=>  r < -U  (underprediction beyond lower margin)
    y_rel = -residuals

    width = U - L
    pen_under = (2.0 / alpha) * (L - y_rel) * (y_rel < L)
    pen_over = (2.0 / alpha) * (y_rel - U) * (y_rel > U)

    scores = width + pen_under + pen_over
    return float(np.mean(scores))

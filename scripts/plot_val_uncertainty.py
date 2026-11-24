"""
Uncertainty-quality plots vs. imaging conditions (occlusion, blur, lighting, noise, ...).

Inputs : one or more prediction JSONs written by run_val.py (one per model) and the
         val_dir/annotations.jsonl that holds per-image / per-marker meta info.
Outputs: (in --out_dir, default <val_dir>/plots)
  conditions_<channel>.png   rows = condition factors, cols = RMSE | CRPS | Interval score | 90% coverage
  calibration_<channel>.png  reliability diagram (nominal vs empirical coverage) + predicted-sigma vs RMSE
  summary_overall.csv        overall metrics per model / channel
  metrics_by_condition.csv   all per-bin numbers behind the plots

Metrics (per channel, in native units; azimuth / elevation are converted to degrees):
  RMSE, MAE            point accuracy
  CRPS                 closed form for Gaussian and Student-t (nu>1). Undefined for Cauchy (infinite mean) -> NaN.
  Interval score (IS)  central (1-alpha) interval, alpha=0.1 (Gneiting & Raftery); works for every head.
  Coverage             fraction of targets inside the central 90% interval (nominal 0.9)
  Spearman(sigma,|e|)  does predicted scale rank the errors?
  mse_fixed has no uncertainty head -> evaluated with a constant Gaussian sigma (= its global RMSE), i.e.
  the "no uncertainty information" baseline.

Usage:
  python scripts/plot_val_uncertainty.py --val_dir data/synth/val \
      --preds data/synth/val/val_predictions_*.json
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse, Polygon
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import betaln
from scipy.spatial import ConvexHull

CHANNELS = ["az", "el", "logrho"]
CH_LABEL = {"az": "Azimuth [deg]", "el": "Elevation [deg]", "logrho": "log rho"}
CH_SCALE = {"az": np.rad2deg(1.0), "el": np.rad2deg(1.0), "logrho": 1.0}
ALPHA = 0.1

# (column, human label, level). Missing columns are skipped automatically.
FACTORS = [
    ("occlusion_fraction", "Occlusion fraction", "marker"),
    ("motion_blur_len", "Motion blur length [px]", "image"),
    ("defocus_k", "Defocus blur k", "image"),
    ("exposure", "Exposure", "image"),
    ("gamma", "Gamma", "image"),
    ("grad_strength", "Lighting gradient", "image"),
    ("has_glare", "Glare", "image"),
    ("gaussian_noise_sigma", "Sensor noise sigma", "image"),
    ("vignette_k", "Vignetting k", "image"),
    ("jpeg_quality", "JPEG quality", "image"),
    ("depth", "Marker depth [m]", "marker"),
    ("view_angle_deg", "View angle [deg]", "marker"),
    ("bg_type", "Background", "image"),
    ("num_distractors", "# distractors", "image"),
]
ZERO_BIN = {"occlusion_fraction", "motion_blur_len", "defocus_k"}


# ----------------------------------------------------------------------------- distributions
def dist_params(head, mu, log_scale, nu, fixed_nu, const_sigma):
    """Returns (family, scale, nu) for one channel's arrays."""
    if head in ("gaussian", "gaussian_beta"):
        return "normal", np.exp(log_scale), None
    if head in ("student_t", "student_t_fixed"):
        n = np.full_like(mu, fixed_nu) if nu is None else nu
        return "t", np.exp(log_scale), n
    if head == "cauchy":
        return "cauchy", np.exp(log_scale), None
    return "normal", np.full_like(mu, const_sigma), None  # mse_fixed baseline


def quantile(fam, p, nu):
    if fam == "normal":
        return stats.norm.ppf(p)
    if fam == "cauchy":
        return stats.cauchy.ppf(p)
    return stats.t.ppf(p, nu)


def crps(fam, y, mu, s, nu):
    z = (y - mu) / s
    if fam == "normal":
        return s * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    if fam == "t":
        nu = np.asarray(nu, dtype=float)
        out = np.full_like(z, np.nan)
        ok = nu > 1
        zz, n = z[ok], nu[ok]
        const = 2 * np.sqrt(n) / (n - 1) * np.exp(betaln(0.5, n - 0.5) - 2 * betaln(0.5, n / 2))
        out[ok] = s[ok] * (zz * (2 * stats.t.cdf(zz, n) - 1)
                           + 2 * stats.t.pdf(zz, n) * (n + zz ** 2) / (n - 1) - const)
        return out
    return np.full_like(z, np.nan)  # cauchy: CRPS infinite


def interval_score(fam, y, mu, s, nu, alpha=ALPHA):
    q = quantile(fam, 1 - alpha / 2, nu)
    lo, hi = mu - q * s, mu + q * s
    return (hi - lo) + (2 / alpha) * np.maximum(lo - y, 0) + (2 / alpha) * np.maximum(y - hi, 0), (y >= lo) & (y <= hi)


def central_coverage(fam, y, mu, s, nu, level):
    q = quantile(fam, 0.5 + level / 2, nu)
    return np.mean(np.abs(y - mu) <= q * s)


# ----------------------------------------------------------------------------- data loading
def load_annotations(val_dir):
    imgs = []
    with open(Path(val_dir) / "annotations.jsonl") as f:
        for line in f:
            if line.strip():
                imgs.append(json.loads(line))
    return imgs


def flatten_marker(img, m):
    row = {"image_index": img["image_index"], "file_name": img.get("file_name"), "marker_id": m["id"],
           "bg_type": img.get("bg_type"), "num_distractors": img.get("num_distractors"),
           "num_canonical_markers": img.get("num_canonical_markers"), "defocus_k": img.get("defocus_k")}
    for grp in ("lighting", "motion_blur", "sensor"):
        for k, v in (img.get(grp) or {}).items():
            row[k] = float(v) if isinstance(v, bool) else v
    for k in ("occlusion_fraction", "visible_fraction", "depth", "view_angle_deg", "rho", "difficulty",
              "pixel_center", "is_target", "in_frame", "azimuth", "elevation", "pos_cam"):
        row[k] = m.get(k)
    return row


def match_predictions(pred, imgs, stride_override=None):
    """Match every predicted center pixel to a GT marker of the right image. Returns DataFrame."""
    meta = pred["meta"]
    stride = np.array([meta["input_hw"][1] / meta["heatmap_hw"][1],
                       meta["input_hw"][0] / meta["heatmap_hw"][0]]) if stride_override is None \
        else np.array([stride_override] * 2)
    by_name = {im.get("file_name"): im for im in imgs}
    rows, dists = [], []
    for p in pred["predictions"]:
        im = by_name.get(p.get("file_name")) if p.get("file_name") else None
        if im is None:
            if p["image_index"] >= len(imgs):
                continue
            im = imgs[p["image_index"]]
        best, bd = None, 1e9
        for m in im["markers"]:
            if not m.get("is_target", True) or not m.get("in_frame", True):
                continue
            c = np.array(m["pixel_center"]) / stride
            d = np.hypot(c[0] - (p["x"] + 0.5), c[1] - (p["y"] + 0.5))
            if d < bd:
                best, bd = m, d
        if best is None or bd > 2.0:
            continue
        r = flatten_marker(im, best)
        r.update(pred_idx=len(rows), match_dist=bd, _p=p)
        rows.append(r)
        dists.append(bd)
    df = pd.DataFrame(rows)
    # one prediction per marker (closest cell)
    if len(df):
        df = df.sort_values("match_dist").drop_duplicates(["image_index", "marker_id"]).reset_index(drop=True)
    print(f"  matched {len(df)}/{len(pred['predictions'])} predictions; "
          f"median center distance {np.median(dists) if dists else float('nan'):.2f} cells "
          f"(should be << 1; large values mean stride / ordering mismatch)")
    return df


def channel_arrays(df, ch_i, head, fixed_nu, ch):
    sc = CH_SCALE[ch]
    P = df["_p"].tolist()
    mu = np.array([p["mu"][ch_i] for p in P]) * sc
    y = np.array([p["target"][ch_i] for p in P]) * sc
    ls = None if P[0]["log_scale"] is None else np.array([p["log_scale"][ch_i] for p in P])
    nu = None if P[0].get("nu") is None else np.array([p["nu"][min(ch_i, len(p["nu"]) - 1)] for p in P], float)
    return mu, y, ls, nu, sc


def per_sample_metrics(head, mu, y, ls, nu, sc, fixed_nu):
    const_sigma = float(np.sqrt(np.mean((mu - y) ** 2)))
    if ls is None:
        scale = np.full_like(mu, const_sigma)
        fam, nu_ = "normal", None
    else:
        fam, scale, nu_ = dist_params(head, mu, ls, nu, fixed_nu, const_sigma)
        scale = scale * sc  # exp(log_scale) is in native units; convert to degrees where needed
    if head == "mse_fixed":
        fam, scale, nu_ = "normal", np.full_like(mu, const_sigma), None
    is_, cov = interval_score(fam, y, mu, scale, nu_)
    return {
        "se": (mu - y) ** 2, "ae": np.abs(mu - y), "crps": crps(fam, y, mu, scale, nu_),
        "is": is_, "cover": cov.astype(float), "sigma": scale, "fam": fam, "nu": nu_,
    }


# ----------------------------------------------------------------------------- binning / stats
def bin_factor(series, name, n_bins=4):
    s = series.copy()
    if s.dtype == object or s.nunique() <= 6:
        codes, uniques = pd.factorize(s, sort=True)
        return pd.Series(codes, index=s.index), [str(u) for u in uniques]
    s = s.astype(float)
    labels = np.full(len(s), -1)
    names = []
    start = 0
    if name in ZERO_BIN and (s <= 1e-9).any():
        labels[(s <= 1e-9).values] = 0
        names.append("0")
        start = 1
    pos = s > 1e-9 if start else pd.Series(True, index=s.index)
    edges = np.unique(np.quantile(s[pos], np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        edges = np.array([s[pos].min(), s[pos].max() + 1e-9])
    idx = np.clip(np.digitize(s[pos], edges[1:-1], right=True), 0, len(edges) - 2)
    labels[pos.values] = idx + start
    names += [f"{edges[i]:.3g}-{edges[i + 1]:.3g}" for i in range(len(edges) - 1)]
    return pd.Series(labels, index=s.index), names


def boot_mean(x, rng, n_boot=200):
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    if len(x) == 1:
        return x[0], x[0], x[0]
    bs = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return x.mean(), *np.percentile(bs, [2.5, 97.5])


def summarize(ps, mask, rng):
    out = {"n": int(mask.sum())}
    se = ps["se"][mask]
    m, lo, hi = boot_mean(se, rng)
    out.update(rmse=np.sqrt(m), rmse_lo=np.sqrt(lo), rmse_hi=np.sqrt(hi), mae=np.mean(ps["ae"][mask]))
    for k in ("crps", "is", "cover"):
        out[k], out[k + "_lo"], out[k + "_hi"] = boot_mean(ps[k][mask], rng)
    sg, ae = ps["sigma"][mask], ps["ae"][mask]
    out["spearman"] = stats.spearmanr(sg, ae)[0] if mask.sum() > 3 and np.ptp(sg) > 0 else np.nan
    return out


# ----------------------------------------------------------------------------- plotting
METRICS_1D = [("rmse", "RMSE"), ("crps", "CRPS"),
              ("is", f"Interval score ({int((1 - ALPHA) * 100)}%)"),
              ("cover", f"Coverage ({int((1 - ALPHA) * 100)}% interval)")]


def plot_conditions(title, models, factors, out_path, min_n, metrics, nominal):
    """rows = condition factors, cols = metrics. Error bars = 95% bootstrap CI."""
    fig, axes = plt.subplots(len(factors), len(metrics), figsize=(4.25 * len(metrics), 2.9 * len(factors)),
                             squeeze=False)
    colors = plt.cm.tab10(np.arange(len(models)))
    for r, (_, flabel, bins_names, rows) in enumerate(factors):
        for c, (mk, mlabel) in enumerate(metrics):
            ax = axes[r, c]
            for mi, mname in enumerate(models):
                tab = rows[mname]
                x = np.arange(len(tab))
                ok = np.array([t is not None and t["n"] >= min_n for t in tab])
                v = np.array([t[mk] if t else np.nan for t in tab], float)
                lo = np.array([t[mk + "_lo"] if (t and mk + "_lo" in t) else np.nan for t in tab], float)
                hi = np.array([t[mk + "_hi"] if (t and mk + "_hi" in t) else np.nan for t in tab], float)
                v[~ok] = lo[~ok] = hi[~ok] = np.nan
                if np.all(np.isnan(v)):
                    continue
                ax.errorbar(x + 0.06 * (mi - len(models) / 2), v, yerr=[np.nan_to_num(v - lo), np.nan_to_num(hi - v)],
                            marker="o", ms=4, lw=1.4, capsize=2, color=colors[mi], label=mname)
            if mk.startswith("cov") or mk == "cover":
                ax.axhline(nominal, color="k", ls="--", lw=0.8)
                ax.set_ylim(0, 1.02)
            ax.set_xticks(np.arange(len(bins_names)))
            ax.set_xticklabels(bins_names, rotation=30, ha="right", fontsize=8)
            if r == 0:
                ax.set_title(mlabel, fontsize=11)
            if c == 0:
                ax.set_ylabel(flabel, fontsize=10)
            ax.grid(alpha=0.3)
    h, l = axes[0, 0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="lower center", ncol=min(len(l), 6), fontsize=9)
    fig.suptitle(f"{title}  (error bars: 95% bootstrap CI, bins with n<{min_n} hidden)", y=0.999, fontsize=11)
    fig.tight_layout(rect=(0, 0.015, 1, 0.985))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def summarize_arrays(arrs, mask, rng):
    out = {"n": int(mask.sum())}
    for k, v in arrs.items():
        out[k], out[k + "_lo"], out[k + "_hi"] = boot_mean(v[mask], rng)
    return out


def build_factor_tables(models, dfs, bins, summarizer, cond_rows, tag):
    factors = []
    for col, flabel, _ in FACTORS:
        ref = dfs[models[0]]
        if col not in ref.columns or ref[col].isna().all() or ref[col].nunique() < 2:
            continue
        rows, names = {}, None
        for name in models:
            lab, names_m = bin_factor(dfs[name][col], col, bins)
            names = names or names_m
            tab = []
            for b in range(len(names_m)):
                mask = (lab == b).values
                if mask.sum() == 0:
                    tab.append(None)
                    continue
                st = summarizer(name, mask)
                tab.append(st)
                cond_rows.append({"channel": tag, "model": name, "factor": col, "bin": names_m[b], **st})
            rows[name] = tab
        factors.append((col, flabel, names, rows))
    return factors

def plot_calibration(ch, models, ps_by_model, out_dir):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    levels = np.linspace(0.05, 0.95, 19)
    colors = plt.cm.tab10(np.arange(len(models)))
    for mi, name in enumerate(models):
        d = ps_by_model[name]
        ps, y, mu = d["ps"], d["y"], d["mu"]
        emp = [central_coverage(ps["fam"], y, mu, ps["sigma"], ps["nu"], lv) for lv in levels]
        a1.plot(levels, emp, marker="o", ms=3, color=colors[mi], label=name)
        order = np.argsort(ps["sigma"])
        for chunk in [np.array_split(order, 10)]:
            sg = [ps["sigma"][i].mean() for i in chunk]
            rm = [np.sqrt(ps["se"][i].mean()) for i in chunk]
            a2.plot(sg, rm, marker="o", ms=3, color=colors[mi], label=name)
    a1.plot([0, 1], [0, 1], "k--", lw=0.8)
    a1.set(xlabel="Nominal central-interval level", ylabel="Empirical coverage", title="Reliability (diagonal = calibrated)")
    lim = a2.get_xlim()
    a2.plot([0, max(lim[1], a2.get_ylim()[1])], [0, max(lim[1], a2.get_ylim()[1])], "k--", lw=0.8)
    a2.set(xlabel="Predicted scale sigma (decile mean)", ylabel="Empirical RMSE in decile",
           title="Predicted scale vs. actual error (dashed = ideal for Gaussian)")
    for a in (a1, a2):
        a.grid(alpha=0.3)
    a1.legend(fontsize=8)
    fig.suptitle(CH_LABEL[ch])
    fig.tight_layout()
    fig.savefig(out_dir / f"calibration_{ch}.png", dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- 3D points + ellipsoids
# Geometry (verified against annotations: pos_cam == sph_to_xyz(azimuth, elevation, rho) with rho = 1/range):
#   X = r cos(el) sin(az),  Y = -r sin(el)  (camera y points down),  Z = r cos(el) cos(az),  r = 1/rho = exp(-log_rho)
# Network channels: [delta_az, delta_el, log_rho]. The GT "base" of the delta is unknown here, but because the delta is
# additive, pred_abs = gt_abs + (mu - target) holds exactly for az/el (and for log_rho up to the same additive constant).
Z95 = stats.norm.ppf(0.95)


def sph_to_xyz(a):
    az, el, lr = a[:, 0], a[:, 1], a[:, 2]
    r = np.exp(-lr)
    return np.stack([r * np.cos(el) * np.sin(az), -r * np.sin(el), r * np.cos(el) * np.cos(az)], 1)


def jac_xyz(a):
    """d(X,Y,Z)/d(az, el, log_rho), shape [N,3,3]."""
    az, el, lr = a[:, 0], a[:, 1], a[:, 2]
    r = np.exp(-lr)
    P = sph_to_xyz(a)
    J = np.zeros((len(a), 3, 3))
    J[:, 0, 0] = r * np.cos(el) * np.cos(az)
    J[:, 0, 1] = -r * np.sin(el) * np.sin(az)
    J[:, 1, 1] = -r * np.cos(el)
    J[:, 2, 0] = -r * np.cos(el) * np.sin(az)
    J[:, 2, 1] = -r * np.sin(el) * np.cos(az)
    J[:, :, 2] = -P
    return J


def native_arrays(df):
    P = df["_p"].tolist()
    mu = np.array([p["mu"] for p in P], float)
    tgt = np.array([p["target"] for p in P], float)
    ls = None if P[0]["log_scale"] is None else np.array([p["log_scale"] for p in P], float)
    nu = None
    if P[0].get("nu") is not None:
        nu = np.array([p["nu"] * 3 if len(p["nu"]) == 1 else p["nu"] for p in P], float)
    return mu, tgt, ls, nu


def equiv_sigma(head, mu, tgt, ls, nu, fixed_nu):
    """Gaussian-equivalent sigma per channel whose 90% central interval equals the head's own 90% interval.
    (Student-t / Cauchy scales are not standard deviations; the ellipsoid therefore uses the matched Gaussian.)"""
    if head == "mse_fixed" or ls is None:
        return np.tile(np.sqrt(np.mean((mu - tgt) ** 2, 0)), (len(mu), 1))
    sc = np.exp(ls)
    if head in ("student_t", "student_t_fixed"):
        n = np.full_like(sc, fixed_nu) if nu is None else nu
        return sc * stats.t.ppf(0.95, n) / Z95
    if head == "cauchy":
        return sc * stats.cauchy.ppf(0.95) / Z95
    return sc


def compute_3d(df, head, fixed_nu, level):
    mu, tgt, ls, nu = native_arrays(df)
    gt = np.stack([df["azimuth"].values.astype(float), df["elevation"].values.astype(float),
                   np.log(df["rho"].values.astype(float))], 1)
    pr = gt + (mu - tgt)
    Pg, Pp = sph_to_xyz(gt), sph_to_xyz(pr)
    chk = np.abs(Pg - np.array(df["pos_cam"].tolist(), float)).max()
    if chk > 1e-3:
        print(f"  WARNING: sph_to_xyz(GT) differs from annotated pos_cam by up to {chk:.4f} m - check conventions")
    sig = equiv_sigma(head, mu, tgt, ls, nu, fixed_nu)
    J = jac_xyz(pr)
    S = np.einsum("nik,nk,njk->nij", J, sig ** 2, J)
    d = Pg - Pp
    d2 = np.einsum("ni,ni->n", d, np.linalg.solve(S + 1e-12 * np.eye(3), d[..., None])[..., 0])
    ix = [0, 2]  # top view = X-Z plane
    Sxz, dxz = S[:, ix][:, :, ix], d[:, ix]
    d2t = np.einsum("ni,ni->n", dxz, np.linalg.solve(Sxz + 1e-12 * np.eye(2), dxz[..., None])[..., 0])
    c3, c2 = stats.chi2.ppf(level, 3), stats.chi2.ppf(level, 2)
    vol = 4 / 3 * np.pi * np.sqrt(np.maximum(np.linalg.det(S), 1e-30)) * c3 ** 1.5
    return {"Pg": Pg, "Pp": Pp, "S": S, "sig": sig, "c3": c3, "c2": c2, "pred_abs": pr,
            "err3d": np.linalg.norm(d, axis=1), "d2_3d": d2, "d2_top": d2t,
            "cov3d": (d2 <= c3).astype(float), "covtop": (d2t <= c2).astype(float),
            "logvol": np.log10(vol)}


def fit_intrinsics(imgs):
    """Fit a pinhole camera (fx, fy, cx, cy) from annotated pos_cam <-> pixel_center pairs."""
    X, Y, Z, U, V = [], [], [], [], []
    for im in imgs:
        for m in im["markers"]:
            p = m["pos_cam"]
            if m.get("in_frame", True) and p[2] > 0.2:
                X.append(p[0]); Y.append(p[1]); Z.append(p[2]); U.append(m["pixel_center"][0]); V.append(m["pixel_center"][1])
    X, Y, Z, U, V = map(np.array, (X, Y, Z, U, V))
    Au = np.stack([X / Z, np.ones_like(X)], 1)
    Av = np.stack([Y / Z, np.ones_like(X)], 1)
    (fx, cx), (fy, cy) = np.linalg.lstsq(Au, U, rcond=None)[0], np.linalg.lstsq(Av, V, rcond=None)[0]
    rms = float(np.sqrt(np.mean((Au @ [fx, cx] - U) ** 2 + (Av @ [fy, cy] - V) ** 2)))
    print(f"Fitted camera: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f} (reprojection RMS {rms:.2f}px)")
    return dict(fx=fx, fy=fy, cx=cx, cy=cy)


def project(P, K):
    return np.stack([K["fx"] * P[:, 0] / P[:, 2] + K["cx"], K["fy"] * P[:, 1] / P[:, 2] + K["cy"]], 1)


def fib_sphere(n=200):
    i = np.arange(n) + 0.5
    phi, th = np.arccos(1 - 2 * i / n), np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


_IMG_INDEX = {}


def find_image(val_dir, file_name):
    val_dir = Path(val_dir)
    for sub in ("", "images", "imgs", "image", "img", "rgb"):
        p = val_dir / sub / file_name
        if p.exists():
            return p
    if not _IMG_INDEX:
        for p in val_dir.rglob("*"):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                _IMG_INDEX.setdefault(p.name, p)
    return _IMG_INDEX.get(Path(file_name).name)


def draw_image_panel(ax, img, df, r3, idx, K, c3):
    if img is None:
        img = np.full((480, 640, 3), 0.85)
    H, W = img.shape[:2]
    ax.imshow(img)
    sph = fib_sphere()
    for i in idx:
        inside = r3["cov3d"][i] > 0
        col = "lime" if inside else "red"
        gt = np.array(df["pixel_center"].iloc[i])
        pp = project(r3["Pp"][i:i + 1], K)[0]
        try:
            L = np.linalg.cholesky(r3["S"][i] + 1e-12 * np.eye(3))
            pts = r3["Pp"][i] + np.sqrt(c3) * sph @ L.T
            pts = pts[pts[:, 2] > 0.05]
            uv = np.clip(project(pts, K), -3 * W, 4 * W)
            if len(uv) >= 4:
                ax.add_patch(Polygon(uv[ConvexHull(uv).vertices], closed=True, fc=col, ec=col, alpha=0.18, lw=1))
        except (np.linalg.LinAlgError, Exception):
            pass
        ax.plot(*gt, "o", mfc="none", mec="lime", ms=9, mew=2)
        ax.plot(*pp, "x", color="red", ms=8, mew=2)
        ax.plot([gt[0], pp[0]], [gt[1], pp[1]], "-", color="yellow", lw=1)
        ax.text(gt[0] + 6, gt[1] - 6, str(int(df["marker_id"].iloc[i])), color="white", fontsize=8,
                bbox=dict(fc="black", alpha=0.5, pad=1, lw=0))
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_xticks([]); ax.set_yticks([])


def draw_topview_panel(ax, df, r3, idx, level):
    c2 = r3["c2"]
    cols = plt.cm.tab10(np.arange(10))
    pts = []
    for i in idx:
        c = cols[int(df["marker_id"].iloc[i]) % 10]
        Pg, Pp = r3["Pg"][i], r3["Pp"][i]
        Sx = r3["S"][i][[0, 2]][:, [0, 2]]
        w, V = np.linalg.eigh(Sx)
        w = np.maximum(w, 1e-12)
        v = V[:, 1]
        inside = r3["d2_top"][i] <= c2
        ax.add_patch(Ellipse((Pp[0], Pp[2]), 2 * np.sqrt(c2 * w[1]), 2 * np.sqrt(c2 * w[0]),
                             angle=np.degrees(np.arctan2(v[1], v[0])), fc=c, alpha=0.15 if inside else 0.05,
                             ec=c, lw=1.6, ls="-" if inside else "--"))
        ax.plot(Pg[0], Pg[2], "o", color=c, ms=7)
        ax.plot(Pp[0], Pp[2], "x", color=c, ms=8, mew=2)
        ax.plot([Pg[0], Pp[0]], [Pg[2], Pp[2]], ":", color=c, lw=1)
        ax.text(Pg[0], Pg[2], f" {int(df['marker_id'].iloc[i])}", fontsize=8, color=c)
        pts += [Pg[[0, 2]], Pp[[0, 2]]]
    pts = np.array(pts + [[0, 0]])
    lo, hi = pts.min(0), pts.max(0)
    mg = 0.25 * np.maximum(hi - lo, 0.5)
    ax.plot(0, 0, "k^", ms=10)
    ax.set_xlim(lo[0] - mg[0], hi[0] + mg[0]); ax.set_ylim(max(lo[1] - mg[1], -0.2), hi[1] + mg[1])
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("X right [m]"); ax.set_ylabel("Z forward [m]"); ax.grid(alpha=0.3)
    ax.set_title(f"Top view, {int(level * 100)}% ellipse (solid: GT inside, dashed: GT outside)", fontsize=9)


def plot_best_worst(name, df, r3, K, val_dir, out_dir, k, rank_by, level):
    score = {"error": r3["err3d"], "mahal": r3["d2_3d"], "miss": 1 - r3["cov3d"]}[rank_by]
    img_ids = df["image_index"].values
    per = pd.DataFrame({"i": img_ids, "s": score}).groupby("i")["s"].mean().sort_values()
    groups = {"best": list(per.index[:k]), "worst": list(per.index[::-1][:k])}
    for tag, ids in groups.items():
        fig, axes = plt.subplots(len(ids), 2, figsize=(14, 4.7 * len(ids)), squeeze=False,
                                 gridspec_kw={"width_ratios": [1.3, 1]})
        for r, iid in enumerate(ids):
            idx = np.nonzero(img_ids == iid)[0]
            fn = df["file_name"].iloc[idx[0]]
            path = find_image(val_dir, fn) if fn else None
            img = mpimg.imread(path) if path is not None else None
            if img is not None and img.dtype != np.uint8 and img.max() > 1.5:
                img = img / 255.0
            draw_image_panel(axes[r, 0], img, df, r3, idx, K, r3["c3"])
            draw_topview_panel(axes[r, 1], df, r3, idx, level)
            n_in = int(r3["cov3d"][idx].sum())
            axes[r, 0].set_title(f"{fn}  |  mean 3D err {r3['err3d'][idx].mean():.3f} m  |  "
                                 f"{n_in}/{len(idx)} GT inside {int(level * 100)}% 3D ellipsoid  "
                                 f"(green o=GT, red x=pred, hull=projected ellipsoid)", fontsize=8)
        fig.suptitle(f"{name}: {tag.upper()} {k} images (ranked by '{rank_by}')", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        fig.savefig(out_dir / f"bestworst_{name}_{tag}.png", dpi=110)
        plt.close(fig)


def plot_calibration_3d(models, r3s, out_dir):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    levels = np.linspace(0.05, 0.95, 19)
    colors = plt.cm.tab10(np.arange(len(models)))
    for mi, n in enumerate(models):
        r = r3s[n]
        a1.plot(levels, [np.mean(r["d2_3d"] <= stats.chi2.ppf(l, 3)) for l in levels], "o-", ms=3, color=colors[mi], label=n)
        a2.plot(levels, [np.mean(r["d2_top"] <= stats.chi2.ppf(l, 2)) for l in levels], "o-", ms=3, color=colors[mi], label=n)
    for a, t in ((a1, "3D ellipsoid (chi2, 3 dof)"), (a2, "Top-view ellipse (chi2, 2 dof)")):
        a.plot([0, 1], [0, 1], "k--", lw=0.8)
        a.set(xlabel="Nominal level", ylabel="Fraction of GT inside", title=t)
        a.grid(alpha=0.3)
    a1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "calibration_3d.png", dpi=130)
    plt.close(fig)

# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_dir", required=True, help="Dir with annotations.jsonl (+ images)")
    ap.add_argument("--preds", nargs="+", required=True, help="JSONs from run_val.py (one per model)")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--stride", type=float, default=None, help="Override input/heatmap stride")
    ap.add_argument("--min_n", type=int, default=20, help="Hide bins with fewer markers")
    ap.add_argument("--bins", type=int, default=4, help="Quantile bins for continuous factors")
    ap.add_argument("--k", type=int, default=5, help="# best / worst images per model")
    ap.add_argument("--rank_by", default="error", choices=["error", "mahal", "miss"],
                    help="image ranking: mean 3D error | mean Mahalanobis^2 | fraction of GT outside ellipsoid")
    ap.add_argument("--ellipse_level", type=float, default=0.9, help="Confidence level of ellipse/ellipsoid")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir or Path(args.val_dir) / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    imgs = load_annotations(args.val_dir)
    K = fit_intrinsics(imgs)

    models, dfs, preds = [], {}, {}
    for pth in args.preds:
        with open(pth) as f:
            pr = json.load(f)
        m = pr["meta"]
        name = f"{m['backbone']}_{m['head']}" + (f"_s{m['seed']}" if m.get("seed") is not None else "")
        if name in dfs:
            name = f"{name}_{Path(pth).stem[-12:]}"
        print(f"[{name}] {pth}")
        dfs[name] = match_predictions(pr, imgs, args.stride)
        preds[name] = pr
        models.append(name)

    # ---------------- per-channel (az / el / log rho) metrics
    overall, cond_rows = [], []
    for ch_i, ch in enumerate(CHANNELS):
        ps_by_model = {}
        for name in models:
            df, pr = dfs[name], preds[name]
            fn = pr["meta"].get("fixed_nu", 3.0)
            mu, y, ls, nu, sc = channel_arrays(df, ch_i, pr["meta"]["head"], fn, ch)
            ps = per_sample_metrics(pr["meta"]["head"], mu, y, ls, nu, sc, fn)
            ps_by_model[name] = {"ps": ps, "y": y, "mu": mu}
            o = summarize(ps, np.ones(len(y), bool), rng)
            o.update(model=name, channel=ch)
            overall.append(o)
        factors = build_factor_tables(models, dfs, args.bins,
                                      lambda n, mask: summarize(ps_by_model[n]["ps"], mask, rng), cond_rows, ch)
        plot_conditions(f"Uncertainty quality vs. conditions - {CH_LABEL[ch]}", models, factors,
                        out_dir / f"conditions_{ch}.png", args.min_n, METRICS_1D, 1 - ALPHA)
        plot_calibration(ch, models, ps_by_model, out_dir)

    pd.DataFrame(overall).to_csv(out_dir / "summary_overall.csv", index=False)
    print(pd.DataFrame(overall)[["model", "channel", "n", "rmse", "crps", "is", "cover", "spearman"]]
          .round(4).to_string(index=False))

    # ---------------- 3D points + uncertainty ellipsoids
    r3s, rows3 = {}, []
    lvl = args.ellipse_level
    for name in models:
        df, pr = dfs[name], preds[name]
        r3 = compute_3d(df, pr["meta"]["head"], pr["meta"].get("fixed_nu", 3.0), lvl)
        r3s[name] = r3
        chk = df[["image_index", "file_name", "marker_id"]].copy()
        for j, c in enumerate("XYZ"):
            chk[f"gt_{c}"], chk[f"pred_{c}"] = r3["Pg"][:, j], r3["Pp"][:, j]
        for j, c in enumerate(["az", "el", "logrho"]):
            chk[f"sigma_eq_{c}"] = r3["sig"][:, j]
        for k_ in ("err3d", "d2_3d", "d2_top", "cov3d", "covtop", "logvol"):
            chk[k_] = r3[k_]
        chk.rename(columns={"cov3d": "gt_inside_ellipsoid", "covtop": "gt_inside_topview_ellipse"}).to_csv(
            out_dir / f"ellipse_check_{name}.csv", index=False)
        rows3.append(dict(model=name, n=len(df), median_err3d_m=np.median(r3["err3d"]), mean_err3d_m=r3["err3d"].mean(),
                          inside_3d_ellipsoid=r3["cov3d"].mean(), inside_topview_ellipse=r3["covtop"].mean(),
                          nominal=lvl, mean_mahalanobis2_3d=r3["d2_3d"].mean(), expected_if_calibrated=3.0,
                          mean_log10_volume=r3["logvol"].mean()))
        plot_best_worst(name, df, r3, K, args.val_dir, out_dir, args.k, args.rank_by, lvl)
    sum3 = pd.DataFrame(rows3)
    sum3.to_csv(out_dir / "summary_3d.csv", index=False)
    print("\n3D ellipsoid check (GT inside the predicted ellipsoid):")
    print(sum3.round(4).to_string(index=False))

    arrs = {n: {"err3d": r3s[n]["err3d"], "cov3d": r3s[n]["cov3d"], "covtop": r3s[n]["covtop"],
                "logvol": r3s[n]["logvol"]} for n in models}
    m3 = [("err3d", "3D position error [m]"), ("cov3d", f"GT inside {int(lvl * 100)}% 3D ellipsoid"),
          ("covtop", f"GT inside {int(lvl * 100)}% top-view ellipse"), ("logvol", "log10 ellipsoid volume [m^3]")]
    factors = build_factor_tables(models, dfs, args.bins, lambda n, mask: summarize_arrays(arrs[n], mask, rng),
                                  cond_rows, "3d")
    plot_conditions("3D position uncertainty vs. conditions", models, factors, out_dir / "conditions_3d.png",
                    args.min_n, m3, lvl)
    plot_calibration_3d(models, r3s, out_dir)

    pd.DataFrame(cond_rows).to_csv(out_dir / "metrics_by_condition.csv", index=False)
    print(f"\nPlots and tables written to {out_dir}")


if __name__ == "__main__":
    main()

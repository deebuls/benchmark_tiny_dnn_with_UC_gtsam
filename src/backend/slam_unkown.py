"""
SLAM with an unknown map (visually identical landmarks, geometry-only association).

Drop-in replacement for the stub `run_slam_unknown_map` in slam.py.
Paste this whole file's contents into slam.py (replace the stub), or import from it.

Pipeline
--------
1. Poses are initialised by chaining odometry from the GT pose of frame 0.
2. Frame by frame: detections are associated (gated Mahalanobis + one-to-one
   Hungarian assignment) to confirmed + tentative landmarks using the CURRENT
   pose estimate. Unmatched detections with no landmark in the gate start a
   new tentative track. A track is promoted after `min_obs` observations.
3. Every `reopt_every` frames a batch factor graph (prior + odometry +
   marker factors of confirmed landmarks, with IRLS for Mode B) is solved.
   Poses/landmarks are updated, tentatives are re-triangulated, duplicates merged.
4. Optional final pass: re-associate ALL detections with the refined map and
   poses (tighter gate), then a last batch solve.
5. Metrics: anchored ATE, SE(3)-aligned ATE, landmark error vs map_gt (optional).

Note: odometry_covs must use GTSAM Pose3 ordering [rot(3), trans(3)].
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

try:
    import gtsam
    from gtsam import symbol
except ImportError:
    gtsam = None

from src.common.geometry import measurement_to_point3d
from src.backend.factors import MarkerMeasurementFactor, make_gtsam_marker_factor, compute_measurement_and_jacobians
from src.backend.association import LandmarkTracker

_NO_IRLS_HEADS = ["gaussian", "gaussian_beta", "mse_fixed"]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _det_arrays(det: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.array([det["azimuth"], det["elevation"], det["rho"]], dtype=np.float64)
    s = np.array([det["scale_azimuth"], det["scale_elevation"], det["scale_rho"]], dtype=np.float64)
    nu = np.array([det.get("nu_azimuth", 3.0), det.get("nu_elevation", 3.0), det.get("nu_rho", 3.0)],
                  dtype=np.float64)
    return z, s, nu


def _backproject(T_w_c: np.ndarray, det: Dict[str, Any]) -> np.ndarray:
    z, _, _ = _det_arrays(det)
    p_c = np.asarray(measurement_to_point3d(z), dtype=np.float64)
    return T_w_c[:3, :3] @ p_c + T_w_c[:3, 3]


def _umeyama_se3(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rigid alignment (no scale): dst ~= R @ src + t. src, dst: [N, 3]."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    H = (src - mu_s).T @ (dst - mu_d)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, mu_d - R @ mu_s


def _oracle_matches(dets: List[Dict[str, Any]], gts: List[Dict[str, Any]], px: float = 20.0) -> List[Tuple[int, int]]:
    """Nearest GT marker in pixel space (needs 'pixel_center' and GT 'marker_id')."""
    out = []
    for d_idx, d in enumerate(dets):
        best_id, best_dist = None, px
        for g in gts:
            dist = np.hypot(d["pixel_center"][0] - g["pixel_center"][0],
                            d["pixel_center"][1] - g["pixel_center"][1])
            if dist < best_dist:
                best_dist, best_id = dist, int(g["marker_id"])
        if best_id is not None:
            out.append((d_idx, best_id))
    return out


# --------------------------------------------------------------------------- #
# Association (geometry only)
# --------------------------------------------------------------------------- #
def _associate(
    dets: List[Dict[str, Any]],
    T_w_c: np.ndarray,
    lm_ids: List[int],
    lm_pts: np.ndarray,
    gate: float,
    floor: np.ndarray,
    ambiguity_ratio: Optional[float],
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """
    Returns (matches [(det_idx, lm_id)], free_dets [det_idx with NO landmark in the gate]).

    - Cost = Mahalanobis^2 on whitened residual, identical for every head so the
      gate is comparable across heads. Predicted scale is inflated by `floor`
      (pose-drift uncertainty) as sqrt(scale^2 + floor^2).
    - One-to-one assignment (Hungarian) -> two detections never take the same landmark.
    - ambiguity_ratio: if 2nd-best cost <= ratio * best cost the detection is dropped
      (neither matched nor used to spawn a new landmark). None disables the test.
    - Detections that had candidates but lost the assignment are NOT free, so they
      don't spawn duplicate landmarks.
    """
    D, M = len(dets), len(lm_ids)
    if D == 0:
        return [], []
    if M == 0:
        return [], list(range(D))

    H = np.array([compute_measurement_and_jacobians(T_w_c, p)[0] for p in lm_pts])  # [M, 3]
    cost = np.empty((D, M))
    for i, det in enumerate(dets):
        z, s, _ = _det_arrays(det)
        diff = z[None, :] - H
        diff[:, 0] = (diff[:, 0] + np.pi) % (2.0 * np.pi) - np.pi
        s_eff = np.sqrt(s ** 2 + floor ** 2)
        cost[i] = np.sum((diff / s_eff) ** 2, axis=1)

    in_gate = cost <= gate
    has_candidate = in_gate.any(axis=1)

    ambiguous = np.zeros(D, dtype=bool)
    if ambiguity_ratio is not None:
        for i in range(D):
            c = np.sort(cost[i][in_gate[i]])
            if len(c) >= 2 and c[1] <= ambiguity_ratio * c[0]:
                ambiguous[i] = True

    rows, cols = linear_sum_assignment(np.where(in_gate, cost, 1e9))
    matches = [(int(r), lm_ids[c]) for r, c in zip(rows, cols) if in_gate[r, c] and not ambiguous[r]]
    free = [i for i in range(D) if not has_candidate[i]]
    return matches, free


# --------------------------------------------------------------------------- #
# Landmark lifecycle helpers
# --------------------------------------------------------------------------- #
def _add_observation(
    tracker: LandmarkTracker,
    obs: Dict[int, List[Tuple[int, Dict[str, Any]]]],
    lm_id: int,
    f_idx: int,
    det: Dict[str, Any],
    poses_est: np.ndarray,
) -> None:
    if obs[lm_id] and obs[lm_id][-1][0] == f_idx:
        return  # one observation per landmark per frame
    obs[lm_id].append((f_idx, det))
    if lm_id in tracker.tentative_landmarks:
        # refresh tentative position (mean of back-projections) BEFORE a possible promotion
        pts = [_backproject(poses_est[f], d) for f, d in obs[lm_id]]
        tracker.tentative_landmarks[lm_id]["p_w"] = np.mean(pts, axis=0)
        tracker.update_observation(lm_id)


def _merge_confirmed(tracker: LandmarkTracker, obs: Dict[int, List[Tuple[int, Dict[str, Any]]]], dist: float) -> int:
    """Merge confirmed landmarks closer than `dist`; observations move to the surviving id."""
    ids = list(tracker.confirmed_landmarks.keys())
    merged = set()
    n_merges = 0
    for i, a in enumerate(ids):
        if a in merged:
            continue
        for b in ids[i + 1:]:
            if b in merged:
                continue
            pa, pb = tracker.confirmed_landmarks[a], tracker.confirmed_landmarks[b]
            if np.linalg.norm(pa - pb) <= dist:
                na, nb = len(obs[a]), len(obs[b])
                tracker.confirmed_landmarks[a] = (na * pa + nb * pb) / max(na + nb, 1)
                frames_a = {f for f, _ in obs[a]}
                obs[a].extend([(f, d) for f, d in obs[b] if f not in frames_a])
                obs[a].sort(key=lambda x: x[0])
                del tracker.confirmed_landmarks[b]
                del obs[b]
                merged.add(b)
                n_merges += 1
    return n_merges


# --------------------------------------------------------------------------- #
# Batch optimisation (prior + odometry + marker factors, IRLS for Mode B)
# --------------------------------------------------------------------------- #
def _optimize(
    poses_est: np.ndarray,
    upto: int,
    T0: np.ndarray,
    odometry_poses: np.ndarray,
    odometry_covs: np.ndarray,
    confirmed: Dict[int, np.ndarray],
    obs: Dict[int, List[Tuple[int, Dict[str, Any]]]],
    mode: str,
    head_type: str,
    pose0_sigma: float = 0.01,
    lm_prior_sigma: float = 50.0,
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    graph = gtsam.NonlinearFactorGraph()
    init = gtsam.Values()

    for i in range(upto + 1):
        init.insert(symbol("x", i), gtsam.Pose3(poses_est[i]))
    graph.add(gtsam.PriorFactorPose3(
        symbol("x", 0), gtsam.Pose3(T0), gtsam.noiseModel.Diagonal.Sigmas(np.full(6, pose0_sigma))))
    for i in range(upto):
        graph.add(gtsam.BetweenFactorPose3(
            symbol("x", i), symbol("x", i + 1), gtsam.Pose3(odometry_poses[i]),
            gtsam.noiseModel.Gaussian.Covariance(odometry_covs[i])))

    helpers: List[MarkerMeasurementFactor] = []
    used = set()
    for lid, olist in obs.items():
        if lid not in confirmed:
            continue
        for f, det in olist:
            if f > upto:
                continue
            z, s, nu = _det_arrays(det)
            helpers.append(MarkerMeasurementFactor(
                symbol("x", f), symbol("l", lid), z, s, mode=mode, head_type=head_type, nu=nu))
            used.add(lid)

    weak = gtsam.noiseModel.Isotropic.Sigma(3, lm_prior_sigma)  # only for numerical safety, not an anchor
    for lid in used:
        p = gtsam.Point3(*[float(v) for v in confirmed[lid]])
        init.insert(symbol("l", lid), p)
        graph.add(gtsam.PriorFactorPoint3(symbol("l", lid), p, weak))

    use_irls = (mode == "B" and head_type not in _NO_IRLS_HEADS)
    n_irls = 5 if use_irls else 1
    weights = [np.ones(3) for _ in helpers]
    estimate = gtsam.Values(init)
    params = gtsam.LevenbergMarquardtParams()

    for _ in range(n_irls):
        g = gtsam.NonlinearFactorGraph(graph)
        for h, w in zip(helpers, weights):
            g.add(make_gtsam_marker_factor(h, irls_weights=w))
        estimate = gtsam.LevenbergMarquardtOptimizer(g, estimate, params).optimize()
        if use_irls:
            for k, h in enumerate(helpers):
                r_white, _, _ = h.evaluate_residual(
                    estimate.atPose3(h.pose_key).matrix(), estimate.atPoint3(h.landmark_key))
                weights[k] = h.compute_irls_weight(r_white)

    poses_new = np.stack([estimate.atPose3(symbol("x", i)).matrix() for i in range(upto + 1)])
    lms_new = {lid: np.asarray(estimate.atPoint3(symbol("l", lid)), dtype=np.float64) for lid in used}
    return poses_new, lms_new


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def run_slam_unknown_map(
    poses_gt: np.ndarray,
    odometry_poses: np.ndarray,
    odometry_covs: np.ndarray,
    detections_by_frame: List[List[Dict[str, Any]]],
    mode: str = "B",
    head_type: str = "student_t",
    oracle: bool = False,
    gts_by_frame: Optional[List[List[Dict[str, Any]]]] = None,
    map_gt: Optional[Dict[int, np.ndarray]] = None,
    min_obs: int = 3,
    merge_dist: float = 0.35,
    gate: float = 11.34,
    pose_sigma_floor: Tuple[float, float, float] = (0.03, 0.03, 0.05),
    ambiguity_ratio: Optional[float] = None,
    reopt_every: int = 10,
    final_reassociate: bool = True,
    lm_prior_sigma: float = 50.0,
) -> Dict[str, Any]:
    """
    Full SLAM: unknown landmarks, tentative tracking, joint optimisation.

    oracle=True uses GT marker ids (needs gts_by_frame with 'marker_id' and
    'pixel_center'); use it to isolate the effect of the noise model from
    data association errors.
    map_gt (id -> p_world) is only used for evaluation, never by the estimator.
    """
    if gtsam is None:
        raise RuntimeError("GTSAM is required for run_slam_unknown_map.")
    if oracle and gts_by_frame is None:
        raise ValueError("oracle=True requires gts_by_frame.")

    N = len(poses_gt)
    floor = np.asarray(pose_sigma_floor, dtype=np.float64)
    T0 = poses_gt[0].copy()

    tracker = LandmarkTracker(min_observations_to_promote=min_obs, merge_distance_m=merge_dist)
    poses_est = np.zeros_like(poses_gt)
    poses_est[0] = T0
    obs: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {}
    oracle_map: Dict[int, int] = {}  # marker_id -> track id
    stats = {"merges": 0, "optimizations": 0, "tentatives_created": 0}

    def reoptimize(upto: int) -> None:
        if not tracker.confirmed_landmarks:
            return
        p_new, lms = _optimize(poses_est, upto, T0, odometry_poses, odometry_covs,
                               tracker.confirmed_landmarks, obs, mode, head_type,
                               lm_prior_sigma=lm_prior_sigma)
        poses_est[:upto + 1] = p_new
        for lid, p in lms.items():
            tracker.confirmed_landmarks[lid] = p
        for lid, tr in tracker.tentative_landmarks.items():  # re-triangulate with refined poses
            tr["p_w"] = np.mean([_backproject(poses_est[f], d) for f, d in obs[lid]], axis=0)
        if not oracle:
            stats["merges"] += _merge_confirmed(tracker, obs, merge_dist)
        stats["optimizations"] += 1

    # ---- sequential front-end ------------------------------------------------
    for f in range(N):
        if f > 0:
            poses_est[f] = poses_est[f - 1] @ odometry_poses[f - 1]
        dets = detections_by_frame[f]
        T = poses_est[f]

        if dets:
            if oracle:
                for d_idx, marker_id in _oracle_matches(dets, gts_by_frame[f]):
                    lid = oracle_map.get(marker_id)
                    if lid is None:
                        lid = tracker.add_tentative(T, dets[d_idx])
                        oracle_map[marker_id] = lid
                        obs[lid] = [(f, dets[d_idx])]
                        stats["tentatives_created"] += 1
                    else:
                        _add_observation(tracker, obs, lid, f, dets[d_idx], poses_est)
            else:
                lm_ids = list(tracker.confirmed_landmarks.keys()) + list(tracker.tentative_landmarks.keys())
                lm_pts = np.array([
                    tracker.confirmed_landmarks[i] if i in tracker.confirmed_landmarks
                    else tracker.tentative_landmarks[i]["p_w"] for i in lm_ids
                ]).reshape(-1, 3)
                matches, free = _associate(dets, T, lm_ids, lm_pts, gate, floor, ambiguity_ratio)
                for d_idx, lid in matches:
                    _add_observation(tracker, obs, lid, f, dets[d_idx], poses_est)
                for d_idx in free:
                    lid = tracker.add_tentative(T, dets[d_idx])
                    obs[lid] = [(f, dets[d_idx])]
                    stats["tentatives_created"] += 1

        if (f + 1) % reopt_every == 0 or f == N - 1:
            reoptimize(f)

    # ---- optional final re-association with refined map and poses ------------------
    if final_reassociate and not oracle and tracker.confirmed_landmarks:
        ids = list(tracker.confirmed_landmarks.keys())
        pts = np.array([tracker.confirmed_landmarks[i] for i in ids])
        new_obs: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {i: [] for i in ids}
        for f in range(N):
            matches, _ = _associate(detections_by_frame[f], poses_est[f], ids, pts,
                                    gate, 0.5 * floor, ambiguity_ratio)
            for d_idx, lid in matches:
                new_obs[lid].append((f, detections_by_frame[f][d_idx]))
        new_obs = {lid: o for lid, o in new_obs.items() if len(o) >= min_obs}
        confirmed = {lid: tracker.confirmed_landmarks[lid] for lid in new_obs}
        poses_est, lms = _optimize(poses_est, N - 1, T0, odometry_poses, odometry_covs,
                                   confirmed, new_obs, mode, head_type, lm_prior_sigma=lm_prior_sigma)
        confirmed.update(lms)
        tracker.confirmed_landmarks = confirmed
        obs = new_obs

    landmarks = {lid: p for lid, p in tracker.confirmed_landmarks.items() if obs.get(lid)}

    # ---- evaluation ----------------------------------------------------------
    est_t, gt_t = poses_est[:, :3, 3], poses_gt[:, :3, 3]
    errs = np.linalg.norm(est_t - gt_t, axis=-1)
    R, t = _umeyama_se3(est_t, gt_t)
    errs_aligned = np.linalg.norm((est_t @ R.T + t) - gt_t, axis=-1)

    result: Dict[str, Any] = {
        "ate_rmse": float(np.sqrt(np.mean(errs ** 2))),                 # anchored at GT frame 0
        "ate_rmse_aligned": float(np.sqrt(np.mean(errs_aligned ** 2))),  # after SE(3) alignment
        "max_error": float(errs.max()),
        "per_frame_error": errs,
        "num_frames": N,
        "landmarks_confirmed": len(landmarks),
        "landmarks_tentative_left": len(tracker.tentative_landmarks),
        "num_marker_factors": int(sum(len(o) for lid, o in obs.items() if lid in landmarks)),
        "estimated_poses": poses_est,
        "landmarks": landmarks,
        **stats,
    }

    if map_gt:
        gt_ids = list(map_gt.keys())
        gt_pts = np.array([map_gt[i] for i in gt_ids])
        if landmarks:
            est_pts = (np.array(list(landmarks.values())) @ R.T) + t   # same alignment as trajectory
            Dm = cdist(est_pts, gt_pts)
            nn = Dm.min(axis=1)
            recovered = int((Dm.min(axis=0) < 0.3).sum())
            result.update({
                "landmark_rmse": float(np.sqrt(np.mean(nn ** 2))),
                "landmark_median_err": float(np.median(nn)),
                "gt_landmarks_recovered": recovered,
                "gt_landmarks_total": len(gt_ids),
                "duplicate_or_spurious_landmarks": max(len(landmarks) - recovered, 0),
            })
    return result

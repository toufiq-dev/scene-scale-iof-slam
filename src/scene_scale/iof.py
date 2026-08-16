"""Pinhole Induced Optical Flow (IOF) geometry.

Implements the projection formula underlying the Princeton365 IOF metric
(Kayan et al., ICCV 2025): unproject sampled pixels to 3D using a depth map,
apply the relative pose error transformation, reproject, and average the
resulting pixel displacement.

Reference semantics (Princeton365):
    IOF = E_{t,d,u,v} || flow(t, d, u, v) ||_2
where the per-pixel flow is the displacement caused by transforming scene
points with the relative transformation between the estimated and the
ground-truth camera pose.

Pose convention (explicit, matches Princeton365 -- review fix, Critical
Issue 1).

    T_wc maps camera coordinates to world coordinates. The estimated pose
    T_hat = T_hat_wc and the ground-truth pose T_gt = T_gt_wc are both
    world-to-camera maps. A scene point P_c in the GT camera frame maps into
    the estimated camera frame by

        P'_c = (T_hat_wc)^{-1} T_gt_wc P_c = T_rel P_c,

    i.e.  T_rel = inv(T_est) @ T_gt  (``relative_error_pose``). The induced
    flow of a pixel is the reprojection displacement || pi(T_rel P) - p ||_2,
    and IOF is its expectation over frames, depth and pixels.

    Equivalently, T_rel is the SE(3) correction that carries the GT
    trajectory onto the estimated trajectory, expressed in the estimated
    frame -- the pose error AS SEEN BY THE CAMERA, which is what determines
    the visual consequence. Swapping est/gt yields inv(T_rel): the flow
    magnitudes are equal for pure rotations and near-equal for small errors
    (unit test ``test_relative_error_pose_inverse_swap`` pins this), but the
    convention is fixed -- it must match the official Princeton365 target
    generation (pilot gate P1-G1, rho > 0.98, Section 8.8 of the proposal).
"""

from __future__ import annotations

import numpy as np

# Default pinhole intrinsics used by the synthetic experiments (f = image size).
W = H = 256
FX = FY = float(W)
CX, CY = W / 2.0, H / 2.0


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix of a 3-vector."""
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float64,
    )


def exp_so3(w: np.ndarray) -> np.ndarray:
    """Rodrigues exponential map from an axis-angle vector to SO(3)."""
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = skew(w / theta)
    return (
        np.eye(3, dtype=np.float64)
        + np.sin(theta) * k
        + (1.0 - np.cos(theta)) * (k @ k)
    )


def se3_to_T(trans: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Build a 4x4 SE(3) matrix from a translation vector and axis-angle rotation."""
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.asarray(trans, dtype=np.float64)
    T[:3, :3] = exp_so3(np.asarray(rot, dtype=np.float64))
    return T


def log_so3(R: np.ndarray) -> np.ndarray:
    """Inverse Rodrigues: axis-angle vector from a rotation matrix."""
    R = np.asarray(R, dtype=np.float64)
    cos_a = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    theta = np.arccos(cos_a)
    if theta < 1e-9:
        return np.zeros(3, dtype=np.float64)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
                 dtype=np.float64)
    return (theta / (2.0 * np.sin(theta))) * w


def relative_error_pose(T_est: np.ndarray, T_gt: np.ndarray) -> np.ndarray:
    """Relative transformation that maps ground-truth points into the estimated frame.

    T_rel = inv(T_est) @ T_gt. Points transformed by T_rel are reprojected into
    the estimated view; their displacement w.r.t. the original pixel location is
    the induced optical flow.
    """
    return np.linalg.inv(np.asarray(T_est, dtype=np.float64)) @ np.asarray(
        T_gt, dtype=np.float64
    )


def compute_iof(
    T_err: np.ndarray,
    D: np.ndarray,
    num_samples: int = 200,
    seed: int = 0,
    fx: float = FX,
    fy: float = FY,
    cx: float = CX,
    cy: float = CY,
    return_flows: bool = False,
):
    """Vectorized pinhole IOF: mean per-pixel flow induced by pose error `T_err`.

    Args:
        T_err: 4x4 relative pose error (scene points transformed into the
            estimated frame), e.g. from :func:`relative_error_pose`.
        D: depth map (H, W) used to unproject sampled pixels.
        num_samples: number of uniformly sampled pixels.
        seed: RNG seed for pixel sampling (keeps the metric deterministic).
        fx, fy, cx, cy: pinhole intrinsics.

    Returns:
        Mean Euclidean pixel displacement over in-front samples (px), or
        ``(mean, flows)`` when ``return_flows=True`` where ``flows`` is a
        fixed-length ``(num_samples,)`` array with NaN entries for samples
        that are invalid (non-positive depth or behind the estimated camera).
    """
    rng = np.random.RandomState(seed)
    h, w = D.shape
    xs = rng.randint(0, w, num_samples).astype(np.float64)
    ys = rng.randint(0, h, num_samples).astype(np.float64)
    Z = D[ys.astype(int), xs.astype(int)].astype(np.float64)
    flows = np.full(num_samples, np.nan)
    ok = Z > 0
    if ok.any():
        xs_k, ys_k, Z_k = xs[ok], ys[ok], Z[ok]
        X = (xs_k - cx) * Z_k / fx
        Y = (ys_k - cy) * Z_k / fy
        P = np.stack([X, Y, Z_k, np.ones_like(Z_k)])
        Pp = np.asarray(T_err, dtype=np.float64) @ P
        zz = Pp[2]
        keep = zz > 0.1
        if keep.any():
            up = Pp[0][keep] * fx / zz[keep] + cx
            vp = Pp[1][keep] * fy / zz[keep] + cy
            flows[np.where(ok)[0][keep]] = np.sqrt(
                (xs_k[keep] - up) ** 2 + (ys_k[keep] - vp) ** 2
            )
    mean = float(np.nanmean(flows)) if np.isfinite(flows).any() else 0.0
    if return_flows:
        return mean, flows
    return mean


def iof_from_poses(T_est: np.ndarray, T_gt: np.ndarray, D: np.ndarray, **kw) -> float:
    """IOF directly from an estimated and a ground-truth pose plus a depth map."""
    return compute_iof(relative_error_pose(T_est, T_gt), D, **kw)


# ---------------------------------------------------------------------------
# Official-protocol machinery (round-3 review fixes C1 / C2 / M10).
#
# The official Princeton365 IOF is NOT the raw per-frame quantity
# ``iof_from_poses`` computes. Per the paper (evaluation protocol), the
# official metric (i) aligns the ESTIMATED trajectory to the GT trajectory
# with an Umeyama similarity (Sim(3)) alignment followed by a Kabsch SO(3)
# alignment, (ii) fits a per-sequence parametric depth distribution
# (Gaussian/Gamma mixture, BIC-selected), and (iii) numerically integrates
# the per-pixel flow over that depth distribution at every posed frame.
#
#   * ``align_trajectory``      -- Sim(3) + SO(3) trajectory alignment;
#   * ``fit_depth_distribution``-- BIC-selected Gaussian-mixture fit;
#   * ``compute_iof_official``  -- the official-protocol sequence metric.
#
# These exist (a) so pilot gate P1-G1 is a *reimplementation deliverable*
# rather than a correlation check of two different estimators, and (b) to pin
# the C2 paradox in code: trajectory alignment removes global scale and
# accumulated drift, so the official target cannot reproduce the raw
# per-frame target that motivates the thesis (10 cm at 30 cm depth). Both
# targets are reported; raw per-frame IOF is the PRIMARY (runtime-faithful)
# target, official IOF is the benchmark-comparable diagnostic.
# ---------------------------------------------------------------------------


def camera_centers(Ts: np.ndarray) -> np.ndarray:
    """World-frame camera centers from world-to-camera poses: c = -R^T t."""
    Ts = np.asarray(Ts, dtype=np.float64)
    R = Ts[:, :3, :3]
    t = Ts[:, :3, 3]
    return -np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), t)


def umeyama_sim3(src: np.ndarray, ref: np.ndarray):
    """Umeyama similarity alignment: (s, R, t) minimizing || s R src + t - ref ||.

    ``src``/``ref`` are (N, 3) point sets (camera centers). Returns the
    similarity transform mapping ``src`` onto ``ref``.
    """
    src = np.asarray(src, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    n = len(src)
    mu_s, mu_r = src.mean(0), ref.mean(0)
    S = src - mu_s
    Rc = ref - mu_r
    cov = S.T @ Rc / n
    U, D, Vt = np.linalg.svd(cov)
    if U.shape[0] > 1 and np.linalg.det(U) * np.linalg.det(Vt) < 0:
        U = U.copy()
        U[:, -1] *= -1.0
    R = U @ Vt
    var_s = float((S ** 2).sum() / n)
    s = float(D.sum() / var_s) if var_s > 1e-12 else 1.0
    t = mu_r - s * R @ mu_s
    return s, R, t


def kabsch_so3(src: np.ndarray, ref: np.ndarray):
    """Kabsch rotation alignment: (R, t) minimizing || R src + t - ref ||."""
    src = np.asarray(src, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    mu_s, mu_r = src.mean(0), ref.mean(0)
    S = src - mu_s
    Rc = ref - mu_r
    U, _, Vt = np.linalg.svd(S.T @ Rc)
    if U.shape[0] > 1 and np.linalg.det(U) * np.linalg.det(Vt) < 0:
        U = U.copy()
        U[:, -1] *= -1.0
    R = U @ Vt
    t = mu_r - R @ mu_s
    return R, t


def _apply_world_transform(Ts: np.ndarray, s: float, R_a: np.ndarray,
                           t_a: np.ndarray) -> np.ndarray:
    """Apply a world-frame similarity (s, R_a, t_a) to world-to-camera poses.

    Camera center c = -R^T t transforms as c' = s R_a c + t_a, giving
    R' = R R_a^T and t' = s t - R R_a^T t_a.
    """
    Ts = np.asarray(Ts, dtype=np.float64)
    R = Ts[:, :3, :3]
    t = Ts[:, :3, 3]
    Rp = R @ R_a.T
    tp = s * t - np.einsum("nij,j->ni", Rp, t_a)
    out = np.repeat(np.eye(4, dtype=np.float64)[None], len(Ts), axis=0)
    out[:, :3, :3] = Rp
    out[:, :3, 3] = tp
    return out


def align_trajectory(T_ests: np.ndarray, T_gts: np.ndarray):
    """Sim(3) + SO(3)-align the estimated trajectory onto the GT trajectory.

    Matches the official Princeton365 protocol (Umeyama in Sim(3), plus the
    SO(3) component) but estimated jointly from BOTH orientations and camera
    centers, which keeps the fit well-conditioned on near-linear trajectories
    (a center-only Umeyama is degenerate in the directions perpendicular to
    the motion axis and can return an arbitrary rotation that corrupts the
    per-frame IOF). Concretely:

      1. the common world rotation R_w that carries the estimated
         orientations onto the GT orientations: R_w = SO(3)-projection of
         mean_i (R_gt,i^T R_est,i);
      2. scale s and translation t_w from the camera centers with the
         rotation fixed:  s R_w c_est + t_w ~= c_gt  (least squares).

    Returns (aligned_estimated_poses, summary) where ``summary`` carries the
    fitted scale/rotation/translation -- the scale component is exactly the
    accumulated-drift/scale factor the official metric removes.
    """
    T_ests = np.asarray(T_ests, dtype=np.float64)
    T_gts = np.asarray(T_gts, dtype=np.float64)
    R_est, R_gt = T_ests[:, :3, :3], T_gts[:, :3, :3]
    # 1) common rotation: R_gt^T R_est ~= R_w for every frame
    A = np.einsum("nij,njk->nik", np.transpose(R_gt, (0, 2, 1)), R_est)
    Amean = A.mean(0)
    U, _, Vt = np.linalg.svd(Amean)
    R_w = U @ Vt
    if np.linalg.det(R_w) < 0:
        U = U.copy()
        U[:, -1] *= -1.0
        R_w = U @ Vt
    # 2) scale + translation on the centers, rotation fixed
    c_est, c_gt = camera_centers(T_ests), camera_centers(T_gts)
    rc = (R_w @ c_est.T).T
    mu_s, mu_r = rc.mean(0), c_gt.mean(0)
    num = float(((rc - mu_s) * (c_gt - mu_r)).sum())
    den = float(((rc - mu_s) ** 2).sum())
    s = num / den if den > 1e-12 else 1.0
    t_w = mu_r - s * mu_s
    aligned = _apply_world_transform(T_ests, s, R_w, t_w)
    return aligned, dict(scale=float(s), R1=R_w, t1=t_w)


def _gmm_em(x: np.ndarray, k: int, seed: int = 0, iters: int = 60,
            std_floor: float = 1e-2):
    """1-D Gaussian-mixture EM; returns (weights, means, stds)."""
    rng = np.random.RandomState(seed)
    n = len(x)
    qs = np.linspace(0.0, 1.0, k + 1)
    means = np.array([np.quantile(x, 0.5 * (qs[i] + qs[i + 1])) for i in range(k)])
    stds = np.full(k, max(float(np.std(x)), std_floor))
    weights = np.full(k, 1.0 / k)
    for _ in range(iters):
        logp = np.empty((n, k))
        for j in range(k):
            logp[:, j] = (
                np.log(weights[j] + 1e-300)
                - 0.5 * np.log(2.0 * np.pi * stds[j] ** 2)
                - (x - means[j]) ** 2 / (2.0 * stds[j] ** 2)
            )
        logp -= logp.max(axis=1, keepdims=True)
        r = np.exp(logp)
        r /= r.sum(axis=1, keepdims=True)
        nk = r.sum(axis=0)
        weights = nk / n
        means = (r * x[:, None]).sum(0) / nk
        var = (r * (x[:, None] - means[None, :]) ** 2).sum(0) / nk
        stds = np.sqrt(np.maximum(var, std_floor ** 2))
    return weights, means, stds


def _gmm_loglik(x: np.ndarray, w: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> float:
    n, k = len(x), len(w)
    logp = np.empty((n, k))
    for j in range(k):
        logp[:, j] = (
            np.log(w[j] + 1e-300)
            - 0.5 * np.log(2.0 * np.pi * sd[j] ** 2)
            - (x - mu[j]) ** 2 / (2.0 * sd[j] ** 2)
        )
    return float(np.logaddexp.reduce(logp, axis=1).sum())


def fit_depth_distribution(D: np.ndarray, max_components: int = 3, seed: int = 0):
    """BIC-selected Gaussian-mixture fit of a scene's depth distribution.

    The official protocol fits a per-sequence parametric depth distribution
    (mixture of Gaussians/Gamma, BIC-selected) and integrates the per-pixel
    flow against it. Returns ``(weights, means, stds)`` of the best mixture.
    """
    x = np.asarray(D, dtype=np.float64).ravel()
    x = x[np.isfinite(x) & (x > 0)]
    best = None
    for k in range(1, max_components + 1):
        w, mu, sd = _gmm_em(x, k, seed=seed)
        bic = -2.0 * _gmm_loglik(x, w, mu, sd) + (3 * k - 1) * np.log(len(x))
        if best is None or bic < best[0]:
            best = (bic, (w, mu, sd))
    return best[1]


def gmm_pdf(d: np.ndarray, w: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Gaussian-mixture density evaluated at depths ``d``."""
    d = np.asarray(d, dtype=np.float64)
    out = np.zeros_like(d)
    for j in range(len(w)):
        out += w[j] / (np.sqrt(2.0 * np.pi) * sd[j]) * np.exp(
            -0.5 * ((d - mu[j]) / sd[j]) ** 2
        )
    return out


def compute_iof_official(
    T_ests: np.ndarray,
    T_gts: np.ndarray,
    D_samples: np.ndarray,
    num_pixels: int = 64,
    num_depth_nodes: int = 32,
    seed: int = 0,
    fx: float = FX,
    fy: float = FY,
    cx: float = CX,
    cy: float = CY,
    depth_lo: float = 0.1,
    depth_hi: float = 60.0,
    align: bool = True,
    img_h: int = 256,
    img_w: int = 256,
):
    """Official-protocol IOF for a sequence (trajectory-aligned, marginal-depth).

    Implements the official Princeton365 recipe on top of the pinhole
    projector: (1) Sim(3)+SO(3)-align the estimated trajectory to GT
    (``align=True``); (2) fit the BIC-selected Gaussian mixture to the scene
    depth samples; (3) numerically integrate the per-pixel flow over the
    depth distribution on a log-spaced quadrature grid, averaged over pixels
    and frames.

    Args:
        T_ests: (T, 4, 4) estimated world-to-camera poses.
        T_gts: (T, 4, 4) ground-truth world-to-camera poses.
        D_samples: (T, S) or (S,) true scene-depth samples (per frame) used
            to fit the depth distribution (the marginal depth is what the
            official metric integrates over -- per-pixel depth maps are not
            required by the protocol).
        align: whether to apply trajectory alignment first (the official
            protocol does; ``align=False`` exposes how much of the raw IOF
            the alignment removes -- the C2 diagnostic).

    Returns ``(iof_per_frame, (w, mu, sd))`` with the per-frame official IOF
    and the fitted mixture parameters.
    """
    T_ests = np.asarray(T_ests, dtype=np.float64)
    T_gts = np.asarray(T_gts, dtype=np.float64)
    D_samples = np.asarray(D_samples, dtype=np.float64)
    if align:
        T_ests, _ = align_trajectory(T_ests, T_gts)
    rng = np.random.RandomState(seed)
    flat = D_samples.ravel()
    if len(flat) > 4000:
        flat = flat[rng.choice(len(flat), 4000, replace=False)]
    w, mu, sd = fit_depth_distribution(flat, seed=seed)
    nodes = np.geomspace(depth_lo, depth_hi, num_depth_nodes)
    widths = np.zeros(num_depth_nodes)
    widths[0] = nodes[1] - nodes[0]
    widths[-1] = nodes[-1] - nodes[-2]
    widths[1:-1] = (nodes[2:] - nodes[:-2]) / 2.0
    qw = gmm_pdf(nodes, w, mu, sd) * widths
    qw /= qw.sum()
    hh, ww = img_h, img_w
    iofs = np.zeros(len(T_ests))
    for i in range(len(T_ests)):
        T_rel = relative_error_pose(T_ests[i], T_gts[i])
        xs = rng.randint(0, ww, num_pixels).astype(np.float64)
        ys = rng.randint(0, hh, num_pixels).astype(np.float64)
        X = (xs[:, None] - cx) * nodes[None, :] / fx
        Y = (ys[:, None] - cy) * nodes[None, :] / fy
        Z = np.broadcast_to(nodes[None, :], X.shape)
        P = np.stack([X, Y, Z, np.ones_like(Z)], axis=-1)
        Pp = np.einsum("ij,pdj->pdi", T_rel, P)
        zz = Pp[..., 2]
        keep = zz > 0.1
        up = Pp[..., 0] * fx / np.maximum(zz, 1e-9) + cx
        vp = Pp[..., 1] * fy / np.maximum(zz, 1e-9) + cy
        flow = np.sqrt((xs[:, None] - up) ** 2 + (ys[:, None] - vp) ** 2)
        flow[~keep] = 0.0
        iofs[i] = float((flow * qw[None, :]).sum(axis=1).mean())
    return iofs, (w, mu, sd)

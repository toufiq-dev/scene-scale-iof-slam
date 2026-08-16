"""Pinhole Induced Optical Flow (IOF) geometry.

Implements the projection formula underlying the Princeton365 IOF metric
(Kayan et al., ICCV 2025): unproject sampled pixels to 3D using a depth map,
apply the relative pose error transformation, reproject, and average the
resulting pixel displacement.

Reference semantics (Princeton365):
    IOF = E_{t,d,u,v} || flow(t, d, u, v) ||_2
where the per-pixel flow is the displacement caused by transforming scene
points with the relative transformation between the estimated and the
ground-truth camera pose, T_rel = T_est^{-1} * T_gt.
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
) -> float:
    """Vectorized pinhole IOF: mean per-pixel flow induced by pose error `T_err`.

    Args:
        T_err: 4x4 relative pose error (scene points transformed into the
            estimated frame), e.g. from :func:`relative_error_pose`.
        D: depth map (H, W) used to unproject sampled pixels.
        num_samples: number of uniformly sampled pixels.
        seed: RNG seed for pixel sampling (keeps the metric deterministic).
        fx, fy, cx, cy: pinhole intrinsics.

    Returns:
        Mean Euclidean pixel displacement over in-front samples (px).
    """
    rng = np.random.RandomState(seed)
    h, w = D.shape
    xs = rng.randint(0, w, num_samples).astype(np.float64)
    ys = rng.randint(0, h, num_samples).astype(np.float64)
    Z = D[ys.astype(int), xs.astype(int)].astype(np.float64)
    ok = Z > 0
    if not ok.any():
        return 0.0
    xs, ys, Z = xs[ok], ys[ok], Z[ok]
    X = (xs - cx) * Z / fx
    Y = (ys - cy) * Z / fy
    P = np.stack([X, Y, Z, np.ones_like(Z)])
    Pp = np.asarray(T_err, dtype=np.float64) @ P
    zz = Pp[2]
    keep = zz > 0.1
    if not keep.any():
        return 0.0
    up = Pp[0][keep] * fx / zz[keep] + cx
    vp = Pp[1][keep] * fy / zz[keep] + cy
    flow = np.sqrt((xs[keep] - up) ** 2 + (ys[keep] - vp) ** 2)
    return float(flow.mean())


def iof_from_poses(T_est: np.ndarray, T_gt: np.ndarray, D: np.ndarray, **kw) -> float:
    """IOF directly from an estimated and a ground-truth pose plus a depth map."""
    return compute_iof(relative_error_pose(T_est, T_gt), D, **kw)

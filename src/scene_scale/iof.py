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

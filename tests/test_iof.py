"""Unit tests for the pinhole IOF geometry (pytest or `python tests/test_iof.py`)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scene_scale.iof import (  # noqa: E402
    align_trajectory,
    compute_iof,
    compute_iof_official,
    fit_depth_distribution,
    relative_error_pose,
    se3_to_T,
)

W = H = 256
F = float(W)


def test_identity_pose_error_gives_zero_iof():
    D = np.full((H, W), 1.5)
    T_id = se3_to_T(np.zeros(3), np.zeros(3))
    assert abs(compute_iof(T_id, D)) < 1e-9


def test_known_translation_matches_pinhole_formula():
    # Delta_u = f * dx / Z  =  256 * 0.01 / 1.5  =  1.7067 px
    D = np.full((H, W), 1.5)
    T = se3_to_T(np.array([0.01, 0.0, 0.0]), np.zeros(3))
    iof = compute_iof(T, D, seed=7)
    assert abs(iof - F * 0.01 / 1.5) < 0.05


def test_near_far_scaling_for_translation_only():
    # Same translation at 10x the depth -> ~10x smaller IOF (no rotation term).
    T = se3_to_T(np.array([0.01, 0.0, 0.0]), np.zeros(3))
    near = compute_iof(T, np.full((H, W), 1.5), seed=7)
    far = compute_iof(T, np.full((H, W), 15.0), seed=7)
    assert abs(near / far - 10.0) < 0.5


def test_rotation_flow_is_depth_invariant():
    # Pure rotation induces the same pixel displacement regardless of depth.
    T = se3_to_T(np.zeros(3), np.array([0.02, 0.01, 0.005]))
    near = compute_iof(T, np.full((H, W), 1.5), seed=7)
    far = compute_iof(T, np.full((H, W), 15.0), seed=7)
    assert abs(near - far) < 0.05 * max(near, 1e-6)


def test_relative_error_pose_identity():
    # T_est == T_gt  =>  T_rel = I  =>  IOF = 0
    T = se3_to_T(np.array([0.2, -0.1, 0.05]), np.array([0.03, -0.02, 0.01]))
    T_rel = relative_error_pose(T, T)
    assert np.allclose(T_rel, np.eye(4))
    assert abs(compute_iof(T_rel, np.full((H, W), 3.0))) < 1e-9


def test_relative_error_pose_composition():
    # T_gt = T_est * dT  =>  inv(T_est) @ T_gt  =  dT (up to numeric noise)
    T_est = se3_to_T(np.array([0.5, 0.2, -0.3]), np.array([0.1, 0.2, -0.05]))
    dT = se3_to_T(np.array([0.01, 0.0, 0.0]), np.zeros(3))
    T_gt = T_est @ dT
    assert np.allclose(relative_error_pose(T_est, T_gt), dT, atol=1e-9)


def test_relative_error_pose_inverse_swap():
    # Review fix (Critical Issue 1 -- pose convention). The convention is
    # fixed as T_rel = inv(T_est) @ T_gt (points mapped into the estimated
    # frame, matching Princeton365). Swapping est/gt inverts T_rel:
    #   relative_error_pose(T_gt, T_est) == inv(relative_error_pose(T_est, T_gt))
    # and the induced flow magnitudes are equal for pure rotations and
    # near-equal for small errors -- so a sign error in the convention would
    # not be caught by magnitude alone, which is exactly why the convention
    # must be pinned (and validated against the official implementation,
    # pilot gate P1-G1, rho > 0.98).
    T_est = se3_to_T(np.array([0.5, 0.2, -0.3]), np.array([0.1, 0.2, -0.05]))
    T_gt = se3_to_T(np.array([0.51, 0.19, -0.28]), np.array([0.11, 0.19, -0.06]))
    T_rel = relative_error_pose(T_est, T_gt)
    T_swap = relative_error_pose(T_gt, T_est)
    assert np.allclose(T_swap, np.linalg.inv(T_rel), atol=1e-9)
    D = np.full((H, W), 2.0)
    f1 = compute_iof(T_rel, D, seed=11)
    f2 = compute_iof(T_swap, D, seed=11)
    assert abs(f1 - f2) < 0.1 * max(f1, f2)
    # pure rotation: magnitudes agree to first order in the angle (the
    # O(theta^3) projection terms differ under the swap, so a tight relative
    # bound -- not exact equality -- is the correct property)
    T_est_r = se3_to_T(np.array([0.0, 0.0, 0.0]), np.array([0.2, -0.1, 0.05]))
    T_gt_r = se3_to_T(np.array([0.0, 0.0, 0.0]), np.array([0.21, -0.09, 0.04]))
    f_r1 = compute_iof(relative_error_pose(T_est_r, T_gt_r), D, seed=11)
    f_r2 = compute_iof(relative_error_pose(T_gt_r, T_est_r), D, seed=11)
    assert abs(f_r1 - f_r2) < 0.01 * max(f_r1, f_r2)


# --- round-3 review fixes (C1/C2/M10): official-protocol machinery -------

def _traj_trans_x(step: float = 0.02, n: int = 40):
    """GT world-to-camera trajectory: constant forward translation, no rotation."""
    return np.stack([se3_to_T(np.array([i * step, 0.0, 0.0]), np.zeros(3))
                     for i in range(n)])


def test_sim3_alignment_removes_global_scale():
    # A pure global-scale offset (camera positions at 0.5x GT) is exactly the
    # accumulated-drift component the official metric's Umeyama Sim(3)
    # alignment removes (review C1(a)/C2). Raw per-frame IOF grows with the
    # scale drift; after alignment the residual error is ~0.
    T_gt = _traj_trans_x(step=0.02, n=40)
    # estimated: world positions scaled by 0.5 -> camera-center scale 0.5
    T_est = np.stack([se3_to_T(np.array([i * 0.01, 0.0, 0.0]), np.zeros(3))
                      for i in range(40)])
    D = np.full((256, 256), 1.5)
    raw = np.array([compute_iof(relative_error_pose(T_est[i], T_gt[i]), D, seed=7)
                    for i in range(40)])
    assert raw.mean() > 10.0  # the 10-cm-at-30-cm-style accumulated misalignment
    aligned, summ = align_trajectory(T_est, T_gt)
    assert abs(summ["scale"] - 2.0) < 0.05  # 0.5x positions -> scale factor ~2
    off, _ = compute_iof_official(T_est, T_gt, np.full((40, 512), 1.5),
                                  align=True, seed=7)
    assert off.mean() < 1.0  # alignment removed the accumulated drift


def test_official_iof_without_alignment_matches_raw_for_constant_depth():
    # With a constant depth map the marginal depth distribution is a point
    # mass, so the official integration (align=False) must reproduce the raw
    # per-frame IOF up to pixel-sampling noise -- a consistency check of the
    # depth-distribution integration against the direct computation.
    T_gt = _traj_trans_x(step=0.02, n=40)
    T_est = np.stack([se3_to_T(np.array([i * 0.02 + 0.01, 0.001, -0.002]),
                               np.array([0.001, -0.002, 0.003])) for i in range(40)])
    D = np.full((256, 256), 1.5)
    raw = np.array([compute_iof(relative_error_pose(T_est[i], T_gt[i]), D, seed=7)
                    for i in range(40)])
    off, _ = compute_iof_official(T_est, T_gt, np.full((40, 512), 1.5),
                                  align=False, seed=7)
    ratio = off.mean() / max(raw.mean(), 1e-9)
    assert 0.7 < ratio < 1.3


def test_fit_depth_distribution_recovers_two_components():
    # BIC-selected Gaussian-mixture fit must recover the number of components
    # on a clean two-mode scene-depth distribution.
    rng = np.random.RandomState(0)
    x = np.concatenate([rng.normal(1.0, 0.1, 2000), rng.normal(8.0, 1.0, 2000)])
    w, mu, sd = fit_depth_distribution(x, max_components=3, seed=0)
    assert len(w) == 2
    assert abs(sorted(mu)[0] - 1.0) < 0.3 and abs(sorted(mu)[1] - 8.0) < 1.5


def test_compute_iof_return_flows_consistent_with_mean():
    T = se3_to_T(np.array([0.01, 0.0, 0.0]), np.array([0.01, 0.0, 0.0]))
    D = np.full((256, 256), 2.0)
    mean, flows = compute_iof(T, D, num_samples=100, seed=11, return_flows=True)
    assert flows.shape == (100,)
    assert abs(mean - np.nanmean(flows)) < 1e-9  # same sampling, same mean
    # an all-behind-camera error yields zero flow, not NaN
    T_far = se3_to_T(np.array([0.0, 0.0, -100.0]), np.zeros(3))
    mean2, flows2 = compute_iof(T_far, D, num_samples=100, seed=11, return_flows=True)
    assert mean2 == 0.0 and np.isnan(flows2).all()


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            fails += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - fails}/{len(fns)} passed")
    sys.exit(1 if fails else 0)

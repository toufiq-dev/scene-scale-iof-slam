"""Unit tests for the pinhole IOF geometry (pytest or `python tests/test_iof.py`)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scene_scale.iof import (  # noqa: E402
    compute_iof,
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

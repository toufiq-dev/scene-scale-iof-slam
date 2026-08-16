"""Unit tests for the synthetic generator's error models (G3 confound fix).

The decoupled error model must keep scene depth OUT of the pose-error
dynamics: the error magnitude may depend on motion, but not on depth. Depth
is allowed to influence only the IOF target (the geometric 1/Z projection
denominator). The legacy coupled model kept the confound that the G3 gate
detected; these tests pin both behaviors.

Run:  python tests/test_generator.py   (or pytest)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scene_scale.generator import generate_sequence  # noqa: E402


def _mean_err_mag(error_model, base_depth, seed):
    seq = generate_sequence(seq_len=60, base_depth=base_depth, seed=seed,
                            error_model=error_model)
    return float(np.mean(seq["trans_err_mag"]))


def test_decoupled_error_scale_is_depth_invariant():
    # Identical motion dynamics (same RNG stream) at 1.5 m vs 25 m scene
    # depth must produce the SAME error magnitude: depth must not enter the
    # error generation. Holds to numerical precision because the RNG stream
    # is independent of the depth value.
    near = _mean_err_mag("decoupled", 1.5, seed=3)
    far = _mean_err_mag("decoupled", 25.0, seed=3)
    assert abs(near - far) < 1e-6


def test_coupled_error_scale_grows_with_near_depth():
    # Legacy confounded model: the same motion regime produces LARGER error
    # in near scenes (error scale ~ motion / depth). This is the confound
    # removed by the default decoupled model.
    near = _mean_err_mag("coupled", 1.5, seed=3)
    far = _mean_err_mag("coupled", 25.0, seed=3)
    assert near > 1.25 * far


def test_decoupled_depth_enters_target_only():
    # With the decoupled model the error process is bit-identical across
    # depths, while the IOF target is larger at near depth (1/Z scaling).
    s_near = generate_sequence(seq_len=30, base_depth=1.0, seed=5,
                               error_model="decoupled")
    s_far = generate_sequence(seq_len=30, base_depth=10.0, seed=5,
                              error_model="decoupled")
    assert np.allclose(s_near["trans_err_mag"], s_far["trans_err_mag"], atol=1e-9)
    assert s_near["iof"].mean() > s_far["iof"].mean()


def test_invalid_error_model_raises():
    try:
        generate_sequence(seq_len=10, seed=0, error_model="nope")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown error_model")


# --- round-2 review: estimated motion, degraded reliability, jumps, depth ----

def test_estimated_motion_differs_from_true_and_stays_in_band():
    # The model must consume ESTIMATED motion (corrupted by the pose error),
    # not the true motion a running SLAM never sees.
    s_est = generate_sequence(seq_len=60, seed=1, motion_source="estimated")
    s_true = generate_sequence(seq_len=60, seed=1, motion_source="true")
    assert not np.allclose(s_est["motion"], s_true["motion"])
    # ...but the estimated motion remains a faithful, same-scale proxy of the
    # true motion (the corruption is relative to the error magnitude).
    assert np.allclose(s_est["motion"].mean(axis=0), s_true["motion"].mean(axis=0),
                       atol=0.05)


def test_estimated_motion_correlates_with_hidden_error():
    # Estimated motion inherits the pose error dynamics: M_hat =
    # inv(T_err[t-1]) . M_true . T_err[t], so the deviation from true motion
    # is driven by the error INNOVATION |d(err)|, not the error magnitude.
    # This pins that the corruption is not independent noise.
    s_est = generate_sequence(seq_len=80, seed=12, motion_source="estimated")
    s_true = generate_sequence(seq_len=80, seed=12, motion_source="true")
    dev = np.linalg.norm(s_est["motion"] - s_true["motion"], axis=1)
    d_err = np.abs(np.diff(np.concatenate([[0.0], s_est["trans_err_mag"]])))
    assert float(np.corrcoef(dev, d_err)[0, 1]) > 0.3


def test_reliability_masked_is_constant():
    s = generate_sequence(seq_len=40, seed=2, reliability_mode="masked")
    assert np.allclose(s["confidence"], 0.5)


def test_reliability_delayed_lags_clean():
    clean = generate_sequence(seq_len=60, seed=4, reliability_mode="clean")
    delayed = generate_sequence(seq_len=60, seed=4, reliability_mode="delayed")
    assert np.allclose(delayed["confidence"][3:], clean["confidence"][:-3], atol=1e-9)
    assert np.allclose(delayed["confidence"][:3], 0.5)


def test_reliability_miscalibrated_inverts_clean():
    clean = generate_sequence(seq_len=60, seed=6, reliability_mode="clean")
    mis = generate_sequence(seq_len=60, seed=6, reliability_mode="miscalibrated")
    assert np.allclose(mis["confidence"], np.clip(1.0 - clean["confidence"], 0.0, 1.0),
                       atol=1e-9)


def test_reliability_intermittent_is_sparse():
    s = generate_sequence(seq_len=60, seed=7, reliability_mode="intermittent")
    known = s["confidence"] != 0.5
    assert known.sum() == 12  # every 5th of 60 frames
    # the kept frames carry the real (non-trivial) signal
    assert not np.allclose(s["confidence"][::5], 0.5)


def test_jump_failures_produce_abrupt_spikes():
    s = generate_sequence(seq_len=300, seed=8, jump_prob=0.08, jump_scale=5.0)
    d = np.abs(np.diff(s["trans_err_mag"]))
    assert d.max() > 5.0 * np.median(d[d > 0])  # at least one jump dominates
    # without jumps the AR(1) process is smooth: no such spike
    s0 = generate_sequence(seq_len=300, seed=8, jump_prob=0.0)
    d0 = np.abs(np.diff(s0["trans_err_mag"]))
    assert d0.max() < 5.0 * np.median(d0[d0 > 0])


def test_realistic_depth_corruption_degrades_depth_stats():
    a = generate_sequence(seq_len=30, seed=10, depth_corruption="gaussian", base_depth=5.0)
    b = generate_sequence(seq_len=30, seed=10, depth_corruption="realistic", base_depth=5.0)
    # holes (far-plane defaults) and spikes inflate the std of the estimated
    # depth stats the model consumes
    assert b["depth_stats"][:, 3].mean() > a["depth_stats"][:, 3].mean()


def test_flow_samples_recorded_and_consistent_with_iof():
    # The per-frame per-pixel flow samples (the FlowAUC target, round-3 M5)
    # must be recorded, mostly finite, and average to the scalar IOF target.
    s = generate_sequence(seq_len=40, seed=9, num_samples=150)
    assert s["flow_samples"].shape == (40, 150)
    finite = np.isfinite(s["flow_samples"])
    assert finite.mean() > 0.8
    per_frame = np.nanmean(np.where(finite, s["flow_samples"], np.nan), axis=1)
    assert np.allclose(per_frame, s["iof"], atol=1e-9)
    # poses for the official-protocol alignment leg are recorded too
    assert s["T_gt"].shape == (40, 4, 4) and s["T_hat"].shape == (40, 4, 4)
    assert s["depth_samples"].shape == (40, 512)


def test_num_samples_changes_label_noise():
    # Fewer IOF pixel samples -> noisier targets (the M8 sensitivity the
    # sweep quantifies): the same sequence's IOF target differs across counts.
    a = generate_sequence(seq_len=60, seed=13, num_samples=100)
    b = generate_sequence(seq_len=60, seed=13, num_samples=1000)
    assert not np.allclose(a["iof"], b["iof"])
    assert np.corrcoef(a["iof"], b["iof"])[0, 1] > 0.9


def test_invalid_motion_source_raises():
    try:
        generate_sequence(seq_len=10, seed=0, motion_source="nope")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown motion_source")


def test_invalid_reliability_mode_raises():
    try:
        generate_sequence(seq_len=10, seed=0, reliability_mode="nope")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown reliability_mode")


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

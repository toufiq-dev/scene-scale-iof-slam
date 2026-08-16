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

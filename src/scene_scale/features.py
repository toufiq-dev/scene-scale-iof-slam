"""Feature construction for the IOF predictor.

The predictor consumes, per frame, only quantities available during SLAM
operation:

* a window of 6-DoF relative motion magnitudes  (motion[t-k+1 .. t]),
* a window of the reliability signal            (confidence[t-k+1 .. t]),
* current estimated-depth statistics            (median, q25, q75, std).

Targets are IOF at frame t+h (forecasting) and at frame t (current-error
estimation stage, h = 0), plus auxiliary train-time targets used by the
analytic two-stage baseline (future error magnitudes and median depth).
"""

from __future__ import annotations

import numpy as np


def build_windows(seq: dict, k: int = 5, h: int = 5):
    """Slide a window over one sequence.

    Returns (feats, y_h, y_0, y_prev, y_trans, y_rot, y_med, y_flow, y_flowp):
        feats     (T-k-h+1, 6k + k + 4)  runtime feature matrix
        y_h       IOF at t+h             (forecasting target)
        y_0       IOF at t               (current-error target)
        y_prev    IOF at t               (runtime-unavailable; naive-shift baseline)
        y_trans   log1p(trans_err_mag[t+h])   (train-time analytic stage-1)
        y_rot     log1p(rot_err_mag[t+h])     (train-time analytic stage-1)
        y_med     log(median_depth[t+h])      (train-time analytic stage-1)
        y_flow    per-pixel flow samples at t+h   (FlowAUC target, round-3 M5)
        y_flowp   per-pixel flow samples at t     (oracle FlowAUC persistence)
    """
    motion, depth, conf = seq["motion"], seq["depth_stats"], seq["confidence"]
    iof = seq["iof"]
    flows = seq.get("flow_samples")
    te, rot, med = seq["trans_err_mag"], seq["rot_err_mag"], seq["median"]
    T = len(iof)
    feats, yh, y0, yprev, yte, yrot, ymed = [], [], [], [], [], [], []
    yflow, yflowp = [], []
    for t in range(k, T - h):
        feat = np.concatenate(
            [motion[t - k + 1 : t + 1].ravel(), conf[t - k + 1 : t + 1].ravel(), depth[t]]
        )
        feats.append(feat)
        yh.append(iof[t + h])
        y0.append(iof[t])
        yprev.append(iof[t])
        yte.append(np.log1p(te[t + h]))
        yrot.append(np.log1p(rot[t + h]))
        ymed.append(np.log(med[t + h]))
        if flows is not None:
            yflow.append(flows[t + h])
            yflowp.append(flows[t])
    out = (np.array(feats), np.array(yh), np.array(y0), np.array(yprev),
           np.array(yte), np.array(yrot), np.array(ymed))
    if flows is not None:
        out = out + (np.array(yflow), np.array(yflowp))
    return out


def feature_masks(k: int = 5, d: int = 4):
    """Column indices for the FULL input-combination ablation matrix.

    Column layout of each window: [motion(6k) | reliability(k) | depth(d)].
    Returns a dict with all 7 combinations of {motion, reliability, depth}.
    The reviewer-requested table (inputs x RMSE / per-seq Spearman / AUROC /
    AP) uses exactly these masks, including the reliability-masked comparison
    (motion-only vs motion+depth, both WITHOUT reliability) that tests whether
    depth carries the scene-scale claim on its own.
    """
    m = 6 * k
    c = k
    full = np.arange(m + c + d)
    motion = np.arange(m)
    rel = np.arange(m, m + c)
    depth = np.arange(m + c, m + c + d)
    return dict(
        full=full,
        motion=motion,
        depth=depth,
        rel=rel,
        motion_depth=np.concatenate([motion, depth]),
        motion_rel=np.concatenate([motion, rel]),
        depth_rel=np.concatenate([depth, rel]),
    )


def build_dataset(n_train, n_val, n_test, seq_len, k, h, seed0, generator):
    """Build train/val/test splits from sequences generated with distinct seeds.

    Splitting is strictly by sequence (no temporal leakage across splits).
    Each split also carries ``median``: the per-sequence true median depth
    (scalar per sequence), used for the scene-scale evaluation split.
    """
    from collections import defaultdict

    splits = {}
    for name, n in [("train", n_train), ("val", n_val), ("test", n_test)]:
        cols = defaultdict(list)
        medians = []
        for j in range(n):
            seq = generator(seq_len=seq_len, seed=seed0 + j)
            medians.append(float(np.median(seq["median"])))
            win = build_windows(seq, k=k, h=h)
            keys = ("X", "yh", "y0", "yp", "yte", "yrot", "ymed")
            if len(win) > 7:  # flow samples present -> FlowAUC targets
                keys = keys + ("yflow", "yflowp")
            for key, arr in zip(keys, win):
                cols[key].append(arr)
            cols["seq"].extend([j] * len(win[1]))
        splits[name] = {k_: np.concatenate(v) if k_ != "seq" else np.array(v)
                        for k_, v in cols.items()}
        splits[name]["median"] = np.array(medians)
    return splits

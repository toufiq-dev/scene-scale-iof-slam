"""Evaluation metrics for the IOF prediction task.

Per-sequence aggregation is the honest way to report rank/classification
metrics here: within-sequence autocorrelation inflates pooled metrics (the
original feasibility study's pooled numbers flattered the naive-shift
baseline). Where a sequence has no positive/negative failure labels the
per-sequence metric is skipped (NaN).

Failure definitions (review fix, Critical Issue 3 -- no scene-scale leakage).
A global threshold alone makes failure mean "is this a near scene?", because
IOF scales with 1/Z: near scenes naturally produce higher IOF for the same
pose error. The primary early-warning metric is therefore the
**within-sequence AUROC under the per-sequence threshold**, which defines
failure relative to each sequence's own distribution and cannot be gamed by
detecting scene scale:

  * global threshold        tau   = Q_q(train IOF)          leaderboard-compatible
  * per-sequence threshold  tau_s = Q_q(IOF_s)              failure = degradation
                                                            vs. the sequence's own
                                                            normal behavior (PRIMARY)
  * robust z-score          z_t = (IOF_t - med_s)/(IQR_s+eps), z_t > z_thr
  * depth-normalized        IOF_t * Z_med_t > tau           removes the 1/Z trend so a
                                                            global threshold is not
                                                            "is this a near scene"
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


def pooled_rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def pooled_mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def per_seq(y, p, seq, fn):
    """Median of `fn` computed per sequence; NaN sequences are skipped."""
    y, p, seq = map(np.asarray, (y, p, seq))
    vals = []
    for s in np.unique(seq):
        m = seq == s
        v = fn(y[m], p[m])
        if not np.isnan(v):
            vals.append(v)
    return float(np.median(vals)) if vals else float("nan")


def spearman(y, p) -> float:
    if len(np.unique(y)) < 2 or len(np.unique(p)) < 2:
        return float("nan")
    return float(spearmanr(y, p).statistic)


def auroc(lab, p) -> float:
    """AUROC of binary labels from continuous predictions; NaN if degenerate."""
    lab, p = np.asarray(lab), np.asarray(p)
    if lab.sum() == 0 or lab.sum() == len(lab):
        return float("nan")
    return float(roc_auc_score(lab, p))


def ap(lab, p) -> float:
    lab, p = np.asarray(lab), np.asarray(p)
    if lab.sum() == 0 or lab.sum() == len(lab):
        return float("nan")
    return float(average_precision_score(lab, p))


# --- failure-label definitions -------------------------------------------------

def labels_global(y, tau):
    """Failure = IOF > tau, with a single global threshold (leaderboard-style)."""
    return (np.asarray(y) > tau).astype(int)


def per_seq_quantile(y, seq, q: float = 0.75) -> dict:
    """Per-sequence failure threshold tau_s = Q_q(IOF_s).

    Evaluation-time definition: the evaluator defines failure relative to
    each sequence's own IOF distribution, so the label cannot be gamed by
    detecting scene scale. The predictor never sees these labels.
    """
    y, seq = np.asarray(y), np.asarray(seq)
    return {s: float(np.quantile(y[seq == s], q)) for s in np.unique(seq)}


def labels_per_seq_threshold(y, seq, q: float = 0.75):
    y, seq = np.asarray(y), np.asarray(seq)
    thr = per_seq_quantile(y, seq, q)
    return np.array([int(y[i] > thr[s]) for i, s in enumerate(seq)], dtype=int)


def labels_robust_zscore(y, seq, z_thr: float = 1.5, eps: float = 1e-6):
    """Failure = (IOF_t - median_s) / (IQR_s + eps) > z_thr, per sequence."""
    y, seq = np.asarray(y), np.asarray(seq)
    lab = np.zeros(len(y), dtype=int)
    for s in np.unique(seq):
        m = seq == s
        ys = y[m]
        iqr = float(np.percentile(ys, 75) - np.percentile(ys, 25))
        z = (ys - np.median(ys)) / (iqr + eps)
        lab[m] = (z > z_thr).astype(int)
    return lab


def labels_depth_normalized(y, med, tau):
    """Failure = (IOF_t * Z_med_t) > tau.

    Multiplying by the scene's median depth removes the 1/Z trend, so a
    single global threshold on the normalized quantity is scene-scale fair.
    ``med`` is the per-frame estimated median depth (runtime-available).
    """
    return (np.asarray(y) * np.asarray(med) > tau).astype(int)


# --- per-sequence thresholded AUROC (the primary early-warning metric) ---------

def per_seq_auroc_with_labels(lab, p, seq) -> float:
    """Median per-sequence AUROC for a fixed label vector (NaN sequences skipped)."""
    lab, p, seq = np.asarray(lab), np.asarray(p), np.asarray(seq)
    vals = []
    for s in np.unique(seq):
        m = seq == s
        v = auroc(lab[m], p[m])
        if not np.isnan(v):
            vals.append(v)
    return float(np.median(vals)) if vals else float("nan")


def n_valid_sequences(lab, seq) -> int:
    """Number of sequences whose labels contain both classes (valid for AUROC)."""
    lab, seq = np.asarray(lab), np.asarray(seq)
    n = 0
    for s in np.unique(seq):
        m = lab[seq == s]
        if m.sum() not in (0, len(m)):
            n += 1
    return n


# --- legacy per-sequence helpers (global-threshold variants) -------------------

def make_auroc(tau: float):
    """AUROC of binary failure (y > tau), from continuous predictions."""

    def f(y, p) -> float:
        return auroc(labels_global(y, tau), p)

    return f


def make_ap(tau: float):
    def f(y, p) -> float:
        return ap(labels_global(y, tau), p)

    return f


def per_seq_nrmse(y, p, seq) -> float:
    """Median over sequences of RMSE_s / std_s (per-sequence normalized RMSE)."""
    y, p, seq = map(np.asarray, (y, p, seq))
    vals = []
    for s in np.unique(seq):
        m = seq == s
        sd = float(np.std(y[m]))
        if sd < 1e-12:
            continue
        vals.append(np.sqrt(np.mean((y[m] - p[m]) ** 2)) / sd)
    return float(np.median(vals)) if vals else float("nan")


def evaluate(name, y_test, p, seq, tau, q_per_seq: float = 0.75):
    """One row of the results table.

    ``tau`` is the global failure threshold (e.g. Q75 of the training IOF).
    Returns RMSE, per-sequence normalized RMSE, per-sequence Spearman, pooled
    and per-sequence AUROC/AP under the global threshold, and -- the PRIMARY
    early-warning metric -- within-sequence AUROC under the per-sequence
    threshold ``tau_s = Q_q(IOF_s)`` (pooled and median-per-sequence).
    """
    lab = labels_global(y_test, tau)
    lab_ps = labels_per_seq_threshold(y_test, seq, q_per_seq)
    return dict(
        name=name,
        rmse=pooled_rmse(y_test, p),
        nrmse=per_seq_nrmse(y_test, p, seq),
        med_spearman=per_seq(y_test, p, seq, spearman),
        auroc=auroc(lab, p),
        med_auroc=per_seq(y_test, p, seq, make_auroc(tau)),
        ap=ap(lab, p),
        # PRIMARY: within-sequence AUROC with per-sequence thresholds
        wauroc=auroc(lab_ps, p),
        med_wauroc=per_seq_auroc_with_labels(lab_ps, p, seq),
        n_valid=n_valid_sequences(lab_ps, seq),
    )


def expected_calibration_error(prob, y_fail, n_bins: int = 10) -> float:
    """ECE of a calibrated failure probability against binary failure labels."""
    prob, y_fail = np.asarray(prob, dtype=np.float64), np.asarray(y_fail, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (prob >= lo) & (prob < hi)
        if m.sum() == 0:
            continue
        total += (m.sum() / len(prob)) * abs(prob[m].mean() - y_fail[m].mean())
    return float(total)


def warning_lead_time(risk, fail_events, horizon: int = 30, tau_risk: float | None = None,
                      min_fail_dur: int = 3):
    """Frames of advance warning before each failure event (principled variant).

    Failure events are contiguous runs of ``fail`` of length >= ``min_fail_dur``
    (short blips are noise, not failures). A warning is a frame where
    ``risk > tau_risk`` (a FIXED risk threshold -- default: Q75 of the risk
    series -- rather than a per-event mean, which the review flagged as
    unprincipled). An event is detected if a warning occurs within ``horizon``
    frames before its onset; lead time is onset - first warning in that window.

    Returns dict(median_lead, detection_rate, n_events, false_alarms,
    precision) where false_alarms counts warning onsets not followed by an
    event onset within ``horizon``, and precision = detections /
    (detections + false_alarms).
    """
    risk, fail = np.asarray(risk, dtype=np.float64), np.asarray(fail_events, dtype=np.float64)
    if tau_risk is None:
        tau_risk = float(np.percentile(risk, 75))
    warned = risk > tau_risk
    # failure events = runs of >= min_fail_dur
    events, i, n = [], 0, len(fail)
    while i < n:
        if fail[i] == 1:
            j = i
            while j < n and fail[j] == 1:
                j += 1
            if j - i >= min_fail_dur:
                events.append(i)  # onset
            i = j
        else:
            i += 1
    leads, detected = [], 0
    for o in events:
        lo = max(0, o - horizon)
        w = warned[lo:o]
        hit = np.where(w)[0]
        if len(hit):
            detected += 1
            leads.append(o - (lo + hit[0]))
    # false alarms: warning onsets (start of a warning run) not within horizon
    # before any event onset
    warn_onsets = [i for i in range(1, n) if warned[i] and not warned[i - 1]]
    if n and warned[0]:
        warn_onsets = [0] + warn_onsets
    fa = sum(1 for w0 in warn_onsets if not any(w0 <= o < w0 + horizon for o in events))
    prec = detected / (detected + fa) if (detected + fa) else float("nan")
    return dict(
        median_lead=float(np.median(leads)) if leads else float("nan"),
        detection_rate=detected / len(events) if events else float("nan"),
        n_events=len(events),
        false_alarms=fa,
        precision=prec,
    )

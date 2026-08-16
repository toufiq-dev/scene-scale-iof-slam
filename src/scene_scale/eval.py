"""Evaluation metrics for the IOF prediction task.

Per-sequence aggregation is the honest way to report rank/classification
metrics here: within-sequence autocorrelation inflates pooled metrics (the
original feasibility study's pooled numbers flattered the naive-shift
baseline). Where a sequence has no positive/negative failure labels the
per-sequence metric is skipped (NaN).
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


def make_auroc(tau: float):
    """AUROC of binary failure (y > tau), from continuous predictions."""

    def f(y, p) -> float:
        lab = (y > tau).astype(int)
        if lab.sum() == 0 or lab.sum() == len(lab):
            return float("nan")
        return float(roc_auc_score(lab, p))

    return f


def make_ap(tau: float):
    def f(y, p) -> float:
        lab = (y > tau).astype(int)
        if lab.sum() == 0 or lab.sum() == len(lab):
            return float("nan")
        return float(average_precision_score(lab, p))

    return f


def evaluate(name, y_test, p, seq, tau):
    """One row of the results table: RMSE, per-seq Spearman, AUROC/AP (pooled + per-seq).

    `tau` is the failure threshold (e.g. 75th percentile of the training
    distribution); a frame is a positive if y_test > tau.
    """
    lab = (y_test > tau).astype(int)
    pooled_auroc = (
        float(roc_auc_score(lab, p)) if lab.sum() not in (0, len(lab)) else float("nan")
    )
    ap = (
        float(average_precision_score(lab, p))
        if lab.sum() not in (0, len(lab))
        else float("nan")
    )
    return dict(
        name=name,
        rmse=pooled_rmse(y_test, p),
        med_spearman=per_seq(y_test, p, seq, spearman),
        auroc=pooled_auroc,
        med_auroc=per_seq(y_test, p, seq, make_auroc(tau)),
        ap=ap,
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


def warning_lead_time(risk, fail_events, horizon: int = 30):
    """Frames of advance warning before each failure event.

    `risk` is a per-frame predicted failure probability / score series and
    `fail_events` a binary series marking true failures. For each failure
    onset, find the latest prior frame within `horizon` where risk exceeded a
    per-event threshold (mean risk over the preceding window), and report the
    median advance in frames.
    """
    risk, fail = np.asarray(risk, dtype=np.float64), np.asarray(fail_events)
    onsets = np.where(fail & (np.concatenate([[0], fail[:-1]]) == 0))[0]
    leads = []
    for o in onsets:
        lo = max(0, o - horizon)
        window = risk[lo:o]
        if len(window) == 0 or window.max() <= 0:
            continue
        thr = float(np.mean(window))
        warned = np.where(window > thr)[0]
        if len(warned) == 0:
            continue
        first = lo + warned[0]
        leads.append(o - first)
    return float(np.median(leads)) if leads else float("nan")

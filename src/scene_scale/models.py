"""Lightweight predictor models and training helpers.

The thesis targets a model small enough for a consumer GPU / Colab T4. The
feasibility stage uses a tiny MLP (the design allows swapping in a CNN/GRU
front-end for RGB windows later); the linear (ridge) model trained on the same
features is the fair baseline that a learned model must beat.

A small GRU is included as the temporal learned baseline (review fix): the
review asked why a sliding-window MLP should be the only learned model, when
an LSTM/GRU/TCN on the same features is a fairer test of the *task* claim
versus the *architecture* claim. The GRU consumes the k-frame window as a
sequence (per-frame motion + reliability) with the depth statistics as a
context vector.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Two-hidden-layer regressor (128 hidden units)."""

    def __init__(self, d: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def standardize(y_tr, y_va=None, y_te=None):
    """Standardize targets; returns (std_tr, std_va, std_te, mean, std)."""
    ym, ys = float(np.mean(y_tr)), float(np.std(y_tr)) + 1e-12
    out = [(y_tr - ym) / ys]
    for y in (y_va, y_te):
        out.append(None if y is None else (y - ym) / ys)
    return out[0], out[1], out[2], ym, ys


def train_mlp(Xtr, ytr, Xva, yva, seed=0, epochs=60, patience=8, lr=1e-3, batch=256):
    """Train the MLP with early stopping on validation RMSE; returns best model."""
    torch.manual_seed(seed)
    d = Xtr.shape[1]
    model = MLP(d)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    n = len(ytr)
    best, wait = None, 0
    Xtr_t = torch.from_numpy(np.ascontiguousarray(Xtr, dtype=np.float32))
    ytr_t = torch.from_numpy(np.ascontiguousarray(ytr, dtype=np.float32))
    Xva_t = torch.from_numpy(np.ascontiguousarray(Xva, dtype=np.float32))
    for _ in range(epochs):
        perm = torch.randperm(n)
        model.train()
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            loss = crit(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xva_t).numpy()
        rmse = float(np.sqrt(np.mean((pv - yva) ** 2)))
        if best is None or rmse < best[0]:
            best = (rmse, {k: v.clone() for k, v in model.state_dict().items()})
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best[1])
    return model


@torch.no_grad()
def predict_mlp(model, X):
    model.eval()
    return model(torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))).numpy()


class GRU(nn.Module):
    """Small temporal baseline: GRU over the k-frame sequence + depth context.

    Inputs: ``x_seq`` (B, k, d_seq) -- per-frame [motion(6), reliability(1)];
    ``x_ctx`` (B, d_ctx) -- depth statistics. Output: scalar prediction.
    """

    def __init__(self, d_seq: int = 7, d_ctx: int = 4, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(d_seq, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + d_ctx, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_seq, x_ctx):
        out, _ = self.gru(x_seq)
        h = out[:, -1]
        return self.head(torch.cat([h, x_ctx], dim=-1)).squeeze(-1)


def train_gru(Xseq_tr, Xctx_tr, ytr, Xseq_va, Xctx_va, yva, seed=0, epochs=150,
              patience=12, lr=3e-3, batch=256):
    """Train the GRU baseline with early stopping on validation RMSE.

    GRUs need a higher learning rate than the MLP on this task (lr=1e-3
    stalls around val-RMSE 0.63; lr=3e-3 reaches 0.30 and matches the MLP).
    """
    torch.manual_seed(seed)
    model = GRU(d_seq=Xseq_tr.shape[2], d_ctx=Xctx_tr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    n = len(ytr)
    best, wait = None, 0
    Xs_t = torch.from_numpy(np.ascontiguousarray(Xseq_tr, dtype=np.float32))
    Xc_t = torch.from_numpy(np.ascontiguousarray(Xctx_tr, dtype=np.float32))
    y_t = torch.from_numpy(np.ascontiguousarray(ytr, dtype=np.float32))
    Xs_v = torch.from_numpy(np.ascontiguousarray(Xseq_va, dtype=np.float32))
    Xc_v = torch.from_numpy(np.ascontiguousarray(Xctx_va, dtype=np.float32))
    for _ in range(epochs):
        perm = torch.randperm(n)
        model.train()
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            loss = crit(model(Xs_t[idx], Xc_t[idx]), y_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xs_v, Xc_v).numpy()
        rmse = float(np.sqrt(np.mean((pv - yva) ** 2)))
        if best is None or rmse < best[0]:
            best = (rmse, {k: v.clone() for k, v in model.state_dict().items()})
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best[1])
    return model


@torch.no_grad()
def predict_gru(model, Xseq, Xctx):
    model.eval()
    return model(
        torch.from_numpy(np.ascontiguousarray(Xseq, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(Xctx, dtype=np.float32)),
    ).numpy()


def split_gru_inputs(X, k: int = 5, d_seq: int = 7):
    """Split a flat feature matrix (N, 6k + k + d_ctx) into GRU inputs.

    Column layout (see features.build_windows): [motion(6k) | conf(k) | depth].
    Returns (X_seq (N, k, d_seq), X_ctx (N, d_ctx)).
    """
    seq = X[:, : 6 * k + k].reshape(X.shape[0], k, d_seq)
    ctx = X[:, 6 * k + k :]
    return seq, ctx


def fit_ridge(Xtr, ytr, Xva, yva, alphas=(0.1, 1.0, 10.0, 100.0)):
    """Ridge with alpha selection on validation; returns (best_rmse, model, scaler, ym, ys)."""
    sc = StandardScaler().fit(Xtr)
    Ztr, Zva = sc.transform(Xtr), sc.transform(Xva)
    ym, ys = float(np.mean(ytr)), float(np.std(ytr)) + 1e-12
    best = None
    for a in alphas:
        m = Ridge(alpha=a).fit(Ztr, (ytr - ym) / ys)
        pv = m.predict(Zva) * ys + ym
        rmse = float(np.sqrt(np.mean((pv - yva) ** 2)))
        if best is None or rmse < best[0]:
            best = (rmse, m, sc, ym, ys)
    return best

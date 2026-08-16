"""Lightweight predictor models and training helpers.

The thesis targets a model small enough for a consumer GPU / Colab T4. The
feasibility stage uses a tiny MLP (the design allows swapping in a CNN/GRU
front-end for RGB windows later); the linear (ridge) model trained on the same
features is the fair baseline that a learned model must beat.
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

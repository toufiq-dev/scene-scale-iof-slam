"""
Fixed synthetic feasibility study for the thesis
"Scene-Scale-Aware Prediction of Induced Optical-Flow Error for Visual SLAM".

Fixes applied relative to the original scene-scale.ipynb (whose own verdict was
PARTIAL/FAIL because a naive-shift baseline beat the learned model):

  1. FAIR BASELINE: a linear (ridge) forecast trained on the SAME input features
     as the learned model, instead of comparing against "persistence of the true
     IOF" (which is runtime-unavailable and information-privileged).
  2. RELIABILITY SIGNAL FED TO THE MODEL: the generator's `confidence` signal (a
     noisy observation of the hidden error state, like residuals/covariance in a
     real SLAM) is now an input feature. It is made noisy so it is a realistic
     observation, not a deterministic function of the true error (GIGO-circular).
  3. h=0 DECOMPOSITION: a "current-error estimation" stage (SEESys-style) is run
     alongside the h-step forecasting stage, separating estimation from forecasting.
  4. ANALYTIC TWO-STAGE BASELINE: linear forecasts of error magnitudes + depth,
     converted to IOF via the projection formula (physics-informed linear).
  5. INFORMATIVE MOTION: forward speed varies smoothly and rotation bursts occur;
     6-DoF motion magnitudes are used as features, matching the proposal.
  6. RIGOR: per-sequence Spearman/AUROC, multiple seeds, mean +/- std reporting.

Gate criteria (honest verdict):
  G1 learned > linear on the same features (RMSE and AUROC)
  G2 reliability signal adds value (full > motion+depth, RMSE and AUROC)
  G3 geometry/depth adds value (motion+depth > motion-only, RMSE and AUROC)
  G4 beats naive shift (contextual; not a gate - naive shift is runtime-unavailable)
  G5 current-error estimation works (h=0 beats constant by a wide margin)
"""

import json
import time

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

W = H = 256
FX = FY = float(W)
CX, CY = W / 2.0, H / 2.0

SEEDS = [0, 1, 2]
CFG = dict(n_train=60, n_val=15, n_test=15, seq_len=80, k=5, h=5, num_samples=200)


# --------------------------------------------------------------------------- #
# geometry + generator
# --------------------------------------------------------------------------- #

def se3_to_T(trans, rot):
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = trans
    theta = np.linalg.norm(rot)
    if theta > 1e-10:
        axis = rot / theta
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]], dtype=np.float64)
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        T[:3, :3] = R
    return T


def compute_iof(T_err, D, num_samples=200, seed=0):
    """Vectorized pinhole IOF: unproject -> apply relative pose error -> reproject."""
    rng = np.random.RandomState(seed)
    xs = rng.randint(0, W, num_samples).astype(np.float64)
    ys = rng.randint(0, H, num_samples).astype(np.float64)
    Z = D[ys.astype(int), xs.astype(int)].astype(np.float64)
    ok = Z > 0
    if not ok.any():
        return 0.0
    xs, ys, Z = xs[ok], ys[ok], Z[ok]
    X = (xs - CX) * Z / FX
    Y = (ys - CY) * Z / FY
    P = np.stack([X, Y, Z, np.ones_like(Z)])
    Pp = T_err @ P
    zz = Pp[2]
    keep = zz > 0.1
    if not keep.any():
        return 0.0
    up = Pp[0][keep] * FX / zz[keep] + CX
    vp = Pp[1][keep] * FY / zz[keep] + CY
    flow = np.sqrt((xs[keep] - up) ** 2 + (ys[keep] - vp) ** 2)
    return float(flow.mean())


def generate_sequence(seq_len=80, base_depth=None, seed=0):
    rng = np.random.RandomState(seed)
    if base_depth is None:
        base_depth = float(rng.choice([1.5, 3.0, 5.0, 8.0, 15.0, 25.0]))
    depth_corr, pose_corr = 0.95, 0.90
    trans_err = np.zeros(3, dtype=np.float64)
    rot_err = np.zeros(3, dtype=np.float64)
    depth_base = base_depth
    speed = 0.05
    xs = np.linspace(-1, 1, W)
    ys = np.linspace(-1, 1, H)
    Xg, Yg = np.meshgrid(xs, ys)
    rec = {'motion': [], 'depth_stats': [], 'confidence': [],
           'iof': [], 'trans_err_mag': [], 'rot_err_mag': [], 'median': []}
    for t in range(seq_len):
        # --- ground-truth scene depth (slowly drifting, spatially structured) ---
        depth_base = depth_corr * depth_base + (1 - depth_corr) * base_depth + 0.05 * rng.randn()
        depth_base = max(depth_base, 0.5)
        D = depth_base * (1.0 + 0.15 * np.sin(3.0 * Xg + 0.2 * t) + 0.10 * np.cos(2.0 * Yg - 0.3 * t))
        D = np.maximum(D + 0.02 * depth_base * rng.randn(H, W), 0.5).astype(np.float64)

        # --- informative camera motion (what SLAM estimates) ---
        speed = 0.90 * speed + 0.10 * (0.03 + 0.05 * rng.rand())
        trans_true = np.array([speed, 0.0, 0.0]) + 0.02 * rng.randn(3)
        if rng.rand() < 0.06:  # occasional rotation burst
            rot_true = rng.uniform(-0.12, 0.12, 3)
        else:
            rot_true = 0.01 * rng.randn(3)
        motion_mag = np.linalg.norm(trans_true) + 0.3 * np.linalg.norm(rot_true)

        # --- hidden SLAM error: AR(1) in signed components, scale grows with
        #     motion and with near depth (parallax/motion-blur intuition) ---
        depth_factor = 1.0 / (np.median(D) + 1e-3)
        error_scale = 0.005 + 0.05 * motion_mag * depth_factor
        trans_err = pose_corr * trans_err + error_scale * rng.randn(3)
        rot_err = pose_corr * rot_err + (error_scale * 0.5) * rng.randn(3)
        err_mag = np.linalg.norm(trans_err)

        # --- corrupted estimated depth (GIGO): input side degrades with error ---
        corruption = rng.randn(H, W) * (err_mag * 5.0)
        D_est = np.maximum(D + corruption, 0.5)

        # --- target: IOF from TRUE error + TRUE depth (offline GT) ---
        T_err = se3_to_T(trans_err, rot_err)
        iof = compute_iof(T_err, D, num_samples=200, seed=12345 + t)

        # --- runtime reliability signal: NOISY observation of the hidden error
        #     state (residuals/covariance in a real SLAM would be a noisy
        #     observation of the error, not a deterministic function of it) ---
        noisy_err = err_mag * (1.0 + 0.15 * rng.randn())
        confidence = float(np.exp(-5.0 * max(noisy_err, 0.0)))

        d = D_est.ravel()
        rec['motion'].append(np.concatenate([np.abs(trans_true), np.abs(rot_true)]))
        rec['depth_stats'].append([float(np.median(d)), float(np.percentile(d, 25)),
                                   float(np.percentile(d, 75)), float(np.std(d))])
        rec['confidence'].append(confidence)
        rec['iof'].append(iof)
        rec['trans_err_mag'].append(float(np.linalg.norm(trans_err)))
        rec['rot_err_mag'].append(float(np.linalg.norm(rot_err)))
        rec['median'].append(float(np.median(D)))

    return {k: np.array(v, dtype=np.float64) for k, v in rec.items()}


# --------------------------------------------------------------------------- #
# windows / dataset
# --------------------------------------------------------------------------- #

def build_windows(seq, k=5, h=5):
    motion, depth, conf, iof = seq['motion'], seq['depth_stats'], seq['confidence'], seq['iof']
    te, rot = seq['trans_err_mag'], seq['rot_err_mag']
    med = seq['median']
    T = len(iof)
    feats, yh, y0, yprev, yte, yrot, ymed = [], [], [], [], [], [], []
    for t in range(k, T - h):
        feat = np.concatenate([motion[t - k + 1:t + 1].ravel(),
                               conf[t - k + 1:t + 1].ravel(), depth[t]])
        feats.append(feat)
        yh.append(iof[t + h])
        y0.append(iof[t])
        yprev.append(iof[t])
        yte.append(np.log1p(te[t + h]))
        yrot.append(np.log1p(rot[t + h]))
        ymed.append(np.log(med[t + h]))
    return (np.array(feats), np.array(yh), np.array(y0), np.array(yprev),
            np.array(yte), np.array(yrot), np.array(ymed))


def build_dataset(n_train, n_val, n_test, seq_len, k, h, seed0):
    splits = {}
    for name, n in [('train', n_train), ('val', n_val), ('test', n_test)]:
        feats, yh, y0, yprev, yte, yrot, ymed, seq_ids = [], [], [], [], [], [], [], []
        for j in range(n):
            seq = generate_sequence(seq_len=seq_len, seed=seed0 + j)
            f, a, b, c, d, e, g = build_windows(seq, k=k, h=h)
            feats.append(f); yh.append(a); y0.append(b); yprev.append(c)
            yte.append(d); yrot.append(e); ymed.append(g)
            seq_ids.extend([j] * len(a))
        splits[name] = dict(X=np.concatenate(feats), yh=np.concatenate(yh),
                            y0=np.concatenate(y0), yp=np.concatenate(yprev),
                            yte=np.concatenate(yte), yrot=np.concatenate(yrot),
                            ymed=np.concatenate(ymed), seq=np.array(seq_ids))
    return splits


# feature column bookkeeping (k=window, d=4 depth stats)
def masks(k=5, d=4):
    m = 6 * k            # motion magnitudes
    c = k                # confidence window
    full = np.arange(m + c + d)
    motion_only = np.arange(m)
    motion_depth = np.concatenate([np.arange(m), np.arange(m + c, m + c + d)])
    return full, motion_only, motion_depth


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #

class MLP(nn.Module):
    def __init__(self, d, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(Xtr, ytr, Xva, yva, seed, epochs=60, patience=8, lr=1e-3, batch=256):
    torch.manual_seed(seed)
    d = Xtr.shape[1]
    model = MLP(d)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    n = len(ytr)
    best, wait = None, 0
    Xtr_t = torch.from_numpy(Xtr)
    ytr_t = torch.from_numpy(ytr.astype(np.float32))
    Xva_t = torch.from_numpy(Xva)
    for _ in range(epochs):
        perm = torch.randperm(n)
        model.train()
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = crit(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xva_t).numpy()
        rmse = float(np.sqrt(np.mean((pv - yva) ** 2)))
        if best is None or rmse < best[0]:
            best = (rmse, {k_: v.clone() for k_, v in model.state_dict().items()})
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best[1])
    return model


def predict_mlp(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X)).numpy()


def fit_ridge(Xtr, ytr, Xva, yva, alphas=(0.1, 1.0, 10.0, 100.0)):
    sc = StandardScaler().fit(Xtr)
    Ztr, Zva = sc.transform(Xtr), sc.transform(Xva)
    ym, ys = ytr.mean(), ytr.std() + 1e-12
    best = None
    for a in alphas:
        m = Ridge(alpha=a).fit(Ztr, (ytr - ym) / ys)
        pv = m.predict(Zva) * ys + ym
        rmse = float(np.sqrt(np.mean((pv - yva) ** 2)))
        if best is None or rmse < best[0]:
            best = (rmse, m, sc, ym, ys)
    return best


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def pooled_rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def per_seq(y, p, seq, fn):
    vals = []
    for s in np.unique(seq):
        m = seq == s
        v = fn(y[m], p[m])
        if not np.isnan(v):
            vals.append(v)
    return float(np.median(vals)) if vals else float('nan')


def sp(y, p):
    if len(np.unique(y)) < 2 or len(np.unique(p)) < 2:
        return float('nan')
    return float(spearmanr(y, p).statistic)


def auroc_fn(tau):
    def f(y, p):
        lab = (y > tau).astype(int)
        if lab.sum() == 0 or lab.sum() == len(lab):
            return float('nan')
        return float(roc_auc_score(lab, p))
    return f


def evaluate(name, y_test, p, seq, tau):
    lab = (y_test > tau).astype(int)
    pooled_auroc = float(roc_auc_score(lab, p)) if lab.sum() not in (0, len(lab)) else float('nan')
    ap = float(average_precision_score(lab, p)) if lab.sum() not in (0, len(lab)) else float('nan')
    return dict(name=name,
                rmse=pooled_rmse(y_test, p),
                med_spearman=per_seq(y_test, p, seq, sp),
                auroc=pooled_auroc,
                med_auroc=per_seq(y_test, p, seq, auroc_fn(tau)),
                ap=ap)


# --------------------------------------------------------------------------- #
# one full run (per seed)
# --------------------------------------------------------------------------- #

def run_seed(seed):
    rng = np.random.RandomState(seed)
    seed0 = 1000 + 100 * seed
    splits = build_dataset(CFG['n_train'], CFG['n_val'], CFG['n_test'],
                           CFG['seq_len'], CFG['k'], CFG['h'], seed0)
    tr, va, te = splits['train'], splits['val'], splits['test']
    full, motion_only, motion_depth = masks(CFG['k'])
    tau = float(np.percentile(tr['yh'], 75))

    results = []
    y_te = te['yh']
    seq_te = te['seq']

    # --- constant baseline ---
    c = float(tr['yh'].mean())
    results.append(evaluate('constant', y_te, np.full_like(y_te, c), seq_te, tau))

    # --- naive shift (runtime-unavailable; contextual reference) ---
    results.append(evaluate('naive-shift (context)', y_te, te['yp'], seq_te, tau))

    # --- linear (ridge) on the SAME features as the model (the fair baseline) ---
    def std_y(ytr, yva, yte_):
        ym, ys = ytr.mean(), ytr.std() + 1e-12
        return (ytr - ym) / ys, (yva - ym) / ys, (yte_ - ym) / ys, ym, ys

    ytr_s, yva_s, yte_s, ym, ys = std_y(tr['yh'], va['yh'], te['yh'])
    _, ridge_m, sc_f, _, _ = fit_ridge(tr['X'][:, full], tr['yh'], va['X'][:, full], va['yh'])
    p_ridge = ridge_m.predict(sc_f.transform(te['X'][:, full])) * ys + ym
    results.append(evaluate('ridge (same features)', y_te, p_ridge, seq_te, tau))

    # --- MLP ablations ---
    mlp_preds = {}
    for label, mask in [('mlp motion-only', motion_only),
                        ('mlp motion+depth', motion_depth),
                        ('mlp full', full)]:
        sc = StandardScaler().fit(tr['X'][:, mask])
        Ztr = sc.transform(tr['X'][:, mask]).astype(np.float32)
        Zva = sc.transform(va['X'][:, mask]).astype(np.float32)
        Zte = sc.transform(te['X'][:, mask]).astype(np.float32)
        m = train_mlp(Ztr, ytr_s, Zva, yva_s, seed=42 + seed)
        p = predict_mlp(m, Zte) * ys + ym
        results.append(evaluate(label, y_te, p, seq_te, tau))
        if label == 'mlp full':
            mlp_preds['full'] = p

    # --- analytic two-stage (FAIR version): nonlinear (MLP) forecast of the error
    #     components + depth, then explicit projection transform + linear calibration.
    #     This isolates whether the explicit geometry transform helps over the direct
    #     regressor. Stage-1 MLPs are trained on GT error magnitudes/depth (train-time
    #     only); inference uses only runtime features, exactly like the direct MLP.
    scx = StandardScaler().fit(tr['X'][:, full])
    Ztr = scx.transform(tr['X'][:, full]).astype(np.float32)
    Zva = scx.transform(va['X'][:, full]).astype(np.float32)
    Zte = scx.transform(te['X'][:, full]).astype(np.float32)

    def fit_stage1(target_tr, target_va):
        tm, ts = target_tr.mean(), target_tr.std() + 1e-12
        m = train_mlp(Ztr, (target_tr - tm) / ts, Zva, (target_va - tm) / ts, seed=42 + seed)
        return m, tm, ts

    m_tr, tm_tr, ts_tr = fit_stage1(tr['yte'], va['yte'])
    m_rot, tm_rot, ts_rot = fit_stage1(tr['yrot'], va['yrot'])
    m_med, tm_med, ts_med = fit_stage1(tr['ymed'], va['ymed'])

    def phys_feats(Z):
        t_hat = np.expm1(np.clip(predict_mlp(m_tr, Z) * ts_tr + tm_tr, 0, None))
        r_hat = np.expm1(np.clip(predict_mlp(m_rot, Z) * ts_rot + tm_rot, 0, None))
        z_hat = np.exp(np.clip(predict_mlp(m_med, Z) * ts_med + tm_med, np.log(0.5), None))
        return np.stack([FX * t_hat / z_hat, r_hat], axis=1)

    cal = LinearRegression().fit(phys_feats(Ztr), tr['yh'])   # calibrate on TRAIN
    p_analytic = cal.predict(phys_feats(Zte))                  # predict on TEST
    results.append(evaluate('analytic 2-stage (phys-explicit)', y_te, p_analytic, seq_te, tau))

    # --- h=0 current-error estimation stage (SEESys-style) ---
    y0tr_s, y0va_s, y0te_s, y0m, y0s = std_y(tr['y0'], va['y0'], te['y0'])
    m0 = train_mlp(Ztr.astype(np.float32), y0tr_s, Zva.astype(np.float32), y0va_s, seed=42 + seed)
    p0 = predict_mlp(m0, Zte.astype(np.float32)) * y0s + y0m
    results.append(evaluate('mlp full h=0 (current)', te['y0'], p0, seq_te, tau))

    # --- scene-scale split (test sequences by median true depth) ---
    med_te = np.array([np.median(generate_sequence(CFG['seq_len'], seed=seed0 + j)['median'])
                       for j in range(CFG['n_test'])])
    order = np.argsort(med_te)
    groups = np.array_split(order, 3)
    scale_rows = []
    p_mlp = mlp_preds['full']
    for gname, idxs in zip(['near', 'medium', 'far'], groups):
        sel = np.isin(seq_te, idxs)
        scale_rows.append(dict(group=gname,
                               med_depth=float(np.mean(med_te[idxs])),
                               ridge=pooled_rmse(y_te[sel], p_ridge[sel]),
                               mlp=pooled_rmse(y_te[sel], p_mlp[sel]),
                               naive=pooled_rmse(y_te[sel], te['yp'][sel])))

    return results, scale_rows, dict(tau=tau)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    print('================ UNIT TESTS (geometry) ================')
    # identity -> 0
    T_id = se3_to_T(np.zeros(3), np.zeros(3))
    D = np.full((H, W), 1.5)
    iof_id = compute_iof(T_id, D)
    print(f'identity T_err -> IOF {iof_id:.6f}  {"PASS" if abs(iof_id) < 1e-9 else "FAIL"}')
    # 1 cm translation at 1.5 m, f=256 -> theory 256*0.01/1.5 = 1.7067 px
    T_t = se3_to_T(np.array([0.01, 0, 0]), np.zeros(3))
    iof_near = compute_iof(T_t, np.full((H, W), 1.5), seed=7)
    iof_far = compute_iof(T_t, np.full((H, W), 15.0), seed=7)
    print(f'1cm at 1.5m -> IOF {iof_near:.4f} (theory 1.7067)  {"PASS" if abs(iof_near - 1.7067) < 0.05 else "FAIL"}')
    print(f'near/far ratio (translation-only) = {iof_near / iof_far:.2f} (theory ~10)  '
          f'{"PASS" if abs(iof_near / iof_far - 10.0) < 0.5 else "FAIL"}')

    print('\n================ FEASIBILITY (FIXED DESIGN) ================')
    t0 = time.time()
    per_seed = []
    scale_all = []
    for seed in SEEDS:
        res, scale_rows, info = run_seed(seed)
        per_seed.append(res)
        scale_all.append(scale_rows)
        print(f'seed {seed} done ({time.time() - t0:.0f}s)')

    names = [r['name'] for r in per_seed[0]]
    print('\nmodel                       RMSE(mean+-std)   medSpearman  AUROC      medAUROC   AP')
    agg = {}
    for i, name in enumerate(names):
        rows = [r[i] for r in per_seed]
        def m(k_):
            return f'{np.mean([r[k_] for r in rows]):.3f}+-{np.std([r[k_] for r in rows]):.3f}'
        agg[name] = rows
        print(f'{name:<27} {m("rmse")}   {m("med_spearman")}   {m("auroc")}   {m("med_auroc")}   {m("ap")}')

    # ---- scene-scale split (pooled over seeds) ----
    print('\n================ SCENE-SCALE SPLIT (test, by median depth) ================')
    print('group    med_depth    ridge_RMSE   mlp_RMSE   naive_RMSE')
    for g in range(3):
        row = {k: float(np.mean([scale_all[s][g][k] for s in range(len(SEEDS))]))
               for k in ('med_depth', 'ridge', 'mlp', 'naive')}
        print(f'{scale_all[0][g]["group"]:<9} {row["med_depth"]:9.2f}    {row["ridge"]:8.3f}   '
              f'{row["mlp"]:8.3f}   {row["naive"]:8.3f}')

    # ---- gates ----
    print('\n================ GATES (over 3 seeds) ================')
    def gate(cond):
        return 'PASS' if sum(cond(r) for r in per_seed) >= 2 else 'FAIL'

    def rmse_of(res, name):
        return next(r['rmse'] for r in res if r['name'] == name)

    def auroc_of(res, name):
        return next(r['auroc'] for r in res if r['name'] == name)

    g1 = gate(lambda r: rmse_of(r, 'mlp full') < rmse_of(r, 'ridge (same features)')
              and auroc_of(r, 'mlp full') > auroc_of(r, 'ridge (same features)'))
    g2 = gate(lambda r: rmse_of(r, 'mlp full') < rmse_of(r, 'mlp motion+depth')
              and auroc_of(r, 'mlp full') > auroc_of(r, 'mlp motion+depth'))
    g3 = gate(lambda r: rmse_of(r, 'mlp motion+depth') < rmse_of(r, 'mlp motion-only')
              and auroc_of(r, 'mlp motion+depth') > auroc_of(r, 'mlp motion-only'))
    g4 = gate(lambda r: rmse_of(r, 'mlp full') < rmse_of(r, 'naive-shift (context)')
              and auroc_of(r, 'mlp full') > auroc_of(r, 'naive-shift (context)'))
    g5 = gate(lambda r: rmse_of(r, 'mlp full h=0 (current)') < 0.5 * rmse_of(r, 'constant'))

    print(f'G1 learned > linear on same features      : {g1}')
    print(f'G2 reliability signal adds value          : {g2}')
    print(f'G3 geometry (depth) adds value            : {g3}')
    print(f'G4 beats naive shift (contextual)         : {g4}')
    print(f'G5 current-error estimation works (h=0)   : {g5}')

    overall = 'PASS' if all(g == 'PASS' for g in (g1, g2, g3, g5)) else 'PARTIAL/FAIL'
    print(f'\nOVERALL VERDICT (G1 & G2 & G3 & G5): {overall}')

    out = dict(cfg=CFG, gates=dict(g1=g1, g2=g2, g3=g3, g4=g4, g5=g5, overall=overall),
               per_seed=[{r['name']: {k: r[k] for k in ('rmse', 'med_spearman', 'auroc', 'med_auroc', 'ap')}
                          for r in res} for res in per_seed])
    with open('scene-scale-feasibility/results_fixed.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nresults written to scene-scale-feasibility/results_fixed.json ({time.time() - t0:.0f}s total)')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Corrected synthetic feasibility study (library version).

Rebuild of the original `scene-scale.ipynb` experiment (verdict: PARTIAL/FAIL,
a naive-shift baseline beat the learned model). This version fixes the
experimental design:

  1. fair baseline: ridge trained on the SAME features as the learned model
     (persistence of the true IOF is runtime-unavailable and privileged);
  2. the noisy reliability signal is part of the model input;
  3. an h=0 current-error estimation stage (SEESys-style) separates estimation
     from forecasting;
  4. an analytic two-stage baseline forecasts error magnitudes + depth with
     nonlinear MLPs and applies the explicit projection transform;
  5. informative motion (varying speed, rotation bursts), 6-DoF magnitudes;
  6. per-sequence Spearman/AUROC, multiple seeds, mean +- std;
  7. **depth-decoupled error model** (default): scene depth enters ONLY through
     the IOF 1/Z target, never through the pose-error dynamics -- the G3
     confound fix (see generator.py). `--error-model coupled` keeps the legacy
     confounded error dynamics (error scale ~ motion/depth) for comparison;
     note the archived pre-fix run (reports/feasibility_results_coupled.json)
     used the original generator constants (depth 1.5--25 m, rotation weight 0.5).

Gates (>=2 of 3 seeds):
  G1 learned > linear on same features (RMSE and AUROC)
  G2 reliability signal adds value (full > motion+depth)
  G3 geometry/depth adds value (motion+depth > motion-only on RMSE and pooled AUROC)
  G4 beats naive shift (contextual; not a gate)
  G5 current-error estimation works (h=0 << constant)

Run:  python scripts/run_feasibility.py [--error-model decoupled]
                                   [--out reports/feasibility_results.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scene_scale import eval as ev  # noqa: E402
from scene_scale.features import build_dataset, feature_masks  # noqa: E402
from scene_scale.generator import DEPTH_CHOICES, ROT_INNOV_WEIGHT, generate_sequence  # noqa: E402
from scene_scale.iof import compute_iof, se3_to_T  # noqa: E402
from scene_scale.models import (  # noqa: E402
    fit_ridge,
    predict_mlp,
    standardize,
    train_mlp,
)

SEEDS = [0, 1, 2]
CFG = dict(n_train=60, n_val=15, n_test=15, seq_len=80, k=5, h=5)
FX = 256.0


def unit_tests() -> bool:
    print("================ UNIT TESTS (geometry) ================")
    ok = True
    D = np.full((256, 256), 1.5)
    iof_id = compute_iof(se3_to_T(np.zeros(3), np.zeros(3)), D)
    t1 = abs(iof_id) < 1e-9
    iof_near = compute_iof(se3_to_T(np.array([0.01, 0, 0]), np.zeros(3)), D, seed=7)
    iof_far = compute_iof(
        se3_to_T(np.array([0.01, 0, 0]), np.zeros(3)),
        np.full((256, 256), 15.0), seed=7,
    )
    t2 = abs(iof_near - 256 * 0.01 / 1.5) < 0.05
    t3 = abs(iof_near / iof_far - 10.0) < 0.5
    print(f'identity T_err -> IOF {iof_id:.6f}  {"PASS" if t1 else "FAIL"}')
    print(f'1cm at 1.5m -> IOF {iof_near:.4f} (theory 1.7067)  {"PASS" if t2 else "FAIL"}')
    print(f'near/far ratio (translation-only) = {iof_near / iof_far:.2f} '
          f'{"PASS" if t3 else "FAIL"}')
    return t1 and t2 and t3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--error-model", choices=["coupled", "decoupled"],
                    default="decoupled",
                    help="generator error model: 'decoupled' (default; depth enters only via the "
                         "IOF 1/Z target, G3 confound fixed) or 'coupled' (legacy confounded "
                         "error scale ~ motion/depth, kept for comparison)")
    ap.add_argument("--out", default=str(ROOT / "reports" / "feasibility_results.json"))
    args = ap.parse_args()

    if not unit_tests():
        print("geometry unit tests failed; aborting")
        sys.exit(1)

    gen = partial(generate_sequence, error_model=args.error_model)
    print(f"\n================ FEASIBILITY (FIXED DESIGN, error_model={args.error_model}) "
          f"================")
    print(f"generator: DEPTH_CHOICES={DEPTH_CHOICES}, ROT_INNOV_WEIGHT={ROT_INNOV_WEIGHT}")
    t0 = time.time()
    per_seed, scale_all = [], []
    for seed in SEEDS:
        res, scale_rows = run_seed_full(seed, gen)
        per_seed.append(res)
        scale_all.append(scale_rows)
        print(f"seed {seed} done ({time.time() - t0:.0f}s)")

    names = [r["name"] for r in per_seed[0]]
    print("\nmodel                       RMSE(mean+-std)   medSpearman  AUROC      medAUROC   AP")
    for i, name in enumerate(names):
        rows = [r[i] for r in per_seed]
        cells = []
        for k_ in ("rmse", "med_spearman", "auroc", "med_auroc", "ap"):
            vals = [r[k_] for r in rows]
            cells.append(f"{np.nanmean(vals):.3f}+-{np.nanstd(vals):.3f}")
        print(f"{name:<27} {'   '.join(cells)}")

    print("\n================ SCENE-SCALE SPLIT (test, by median depth) ================")
    print("group    med_depth    ridge_RMSE   mlp_RMSE   naive_RMSE")
    for g in range(3):
        row = {k_: float(np.nanmean([scale_all[s][g][k_] for s in range(len(SEEDS))]))
               for k_ in ("med_depth", "ridge", "mlp", "naive")}
        print(f"{scale_all[0][g]['group']:<9} {row['med_depth']:9.2f}    "
              f"{row['ridge']:8.3f}   {row['mlp']:8.3f}   {row['naive']:8.3f}")

    def rmse_of(res, name):
        return next(r["rmse"] for r in res if r["name"] == name)

    def auroc_of(res, name):
        return next(r["auroc"] for r in res if r["name"] == name)

    def gate(cond):
        return "PASS" if sum(cond(r) for r in per_seed) >= 2 else "FAIL"

    g1 = gate(lambda r: rmse_of(r, "mlp full") < rmse_of(r, "ridge (same features)")
              and auroc_of(r, "mlp full") > auroc_of(r, "ridge (same features)"))
    g2 = gate(lambda r: rmse_of(r, "mlp full") < rmse_of(r, "mlp motion+depth")
              and auroc_of(r, "mlp full") > auroc_of(r, "mlp motion+depth"))
    g3 = gate(lambda r: rmse_of(r, "mlp motion+depth") < rmse_of(r, "mlp motion-only")
              and auroc_of(r, "mlp motion+depth") > auroc_of(r, "mlp motion-only"))
    g4 = gate(lambda r: rmse_of(r, "mlp full") < rmse_of(r, "naive-shift (context)")
              and auroc_of(r, "mlp full") > auroc_of(r, "naive-shift (context)"))
    g5 = gate(lambda r: rmse_of(r, "mlp full h=0 (current)")
              < 0.5 * rmse_of(r, "constant"))

    print("\n================ GATES (over 3 seeds) ================")
    for label, g in [("G1 learned > linear on same features", g1),
                     ("G2 reliability signal adds value", g2),
                     ("G3 geometry (depth) adds value", g3),
                     ("G4 beats naive shift (contextual)", g4),
                     ("G5 current-error estimation works (h=0)", g5)]:
        print(f"{label:<40}: {g}")

    overall = "PASS" if all(g == "PASS" for g in (g1, g2, g3, g5)) else "PARTIAL/FAIL"
    print(f"\nOVERALL VERDICT (G1 & G2 & G3 & G5): {overall}")

    out = dict(cfg=CFG, error_model=args.error_model,
               generator=dict(depth_choices=list(DEPTH_CHOICES),
                              rot_innov_weight=ROT_INNOV_WEIGHT),
               gates=dict(g1=g1, g2=g2, g3=g3, g4=g4, g5=g5, overall=overall),
               per_seed=[{r["name"]: {k_: r[k_] for k_ in
                                      ("rmse", "med_spearman", "auroc", "med_auroc", "ap")}
                          for r in res} for res in per_seed])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nresults -> {args.out} ({time.time() - t0:.0f}s total)")


def run_seed_full(seed, gen):
    """Full per-seed experiment; returns (results list, scene-scale rows)."""
    seed0 = 1000 + 100 * seed
    splits = build_dataset(CFG["n_train"], CFG["n_val"], CFG["n_test"],
                           CFG["seq_len"], CFG["k"], CFG["h"], seed0,
                           generator=gen)
    tr, va, te = splits["train"], splits["val"], splits["test"]
    full, motion_only, motion_depth = feature_masks(CFG["k"])
    tau = float(np.percentile(tr["yh"], 75))
    y_te, seq_te = te["yh"], te["seq"]
    results = []

    # --- constant ---
    c = float(tr["yh"].mean())
    results.append(ev.evaluate("constant", y_te, np.full_like(y_te, c), seq_te, tau))

    # --- naive shift (runtime-unavailable; contextual reference) ---
    results.append(ev.evaluate("naive-shift (context)", y_te, te["yp"], seq_te, tau))

    # --- ridge on the SAME features as the model (the fair linear baseline) ---
    ytr_s, yva_s, yte_s, ym, ys = standardize(tr["yh"], va["yh"], te["yh"])
    _, ridge_m, sc_f, _, _ = fit_ridge(tr["X"][:, full], tr["yh"], va["X"][:, full], va["yh"])
    p_ridge = ridge_m.predict(sc_f.transform(te["X"][:, full])) * ys + ym
    results.append(ev.evaluate("ridge (same features)", y_te, p_ridge, seq_te, tau))

    # --- MLP ablations ---
    mlp_preds = {}
    for label, mask in [("mlp motion-only", motion_only),
                        ("mlp motion+depth", motion_depth),
                        ("mlp full", full)]:
        sc = StandardScaler().fit(tr["X"][:, mask])
        Ztr = sc.transform(tr["X"][:, mask]).astype(np.float32)
        Zva = sc.transform(va["X"][:, mask]).astype(np.float32)
        Zte = sc.transform(te["X"][:, mask]).astype(np.float32)
        m = train_mlp(Ztr, ytr_s, Zva, yva_s, seed=42 + seed)
        p = predict_mlp(m, Zte) * ys + ym
        results.append(ev.evaluate(label, y_te, p, seq_te, tau))
        if label == "mlp full":
            mlp_preds["full"] = p

    # --- analytic two-stage: MLP forecast of error magnitudes + depth, then the
    #     explicit projection transform + linear calibration (physics-explicit) ---
    from sklearn.linear_model import LinearRegression

    scx = StandardScaler().fit(tr["X"][:, full])
    Ztr = scx.transform(tr["X"][:, full]).astype(np.float32)
    Zva = scx.transform(va["X"][:, full]).astype(np.float32)
    Zte = scx.transform(te["X"][:, full]).astype(np.float32)

    def fit_stage1(t_tr, t_va):
        tm, ts = t_tr.mean(), t_tr.std() + 1e-12
        m = train_mlp(Ztr, (t_tr - tm) / ts, Zva, (t_va - tm) / ts, seed=42 + seed)
        return m, tm, ts

    m_tr, tm_tr, ts_tr = fit_stage1(tr["yte"], va["yte"])
    m_rot, tm_rot, ts_rot = fit_stage1(tr["yrot"], va["yrot"])
    m_med, tm_med, ts_med = fit_stage1(tr["ymed"], va["ymed"])

    def phys_feats(Z):
        t_hat = np.expm1(np.clip(predict_mlp(m_tr, Z) * ts_tr + tm_tr, 0, None))
        r_hat = np.expm1(np.clip(predict_mlp(m_rot, Z) * ts_rot + tm_rot, 0, None))
        z_hat = np.exp(np.clip(predict_mlp(m_med, Z) * ts_med + tm_med, np.log(0.5), None))
        return np.stack([FX * t_hat / z_hat, r_hat], axis=1)

    cal = LinearRegression().fit(phys_feats(Ztr), tr["yh"])
    p_analytic = cal.predict(phys_feats(Zte))
    results.append(ev.evaluate("analytic 2-stage (phys-explicit)", y_te, p_analytic, seq_te, tau))

    # --- h=0 current-error estimation stage ---
    y0tr_s, y0va_s, y0te_s, y0m, y0s = standardize(tr["y0"], va["y0"], te["y0"])
    m0 = train_mlp(Ztr, y0tr_s, Zva, y0va_s, seed=42 + seed)
    p0 = predict_mlp(m0, Zte) * y0s + y0m
    results.append(ev.evaluate("mlp full h=0 (current)", te["y0"], p0, seq_te, tau))

    # --- scene-scale split: test sequences grouped by true median depth ---
    med_te = te["median"]
    order = np.argsort(med_te)
    scale_rows = []
    for gname, idxs in zip(["near", "medium", "far"], np.array_split(order, 3)):
        sel = np.isin(seq_te, idxs)
        scale_rows.append(dict(group=gname, med_depth=float(np.mean(med_te[idxs])),
                               ridge=ev.pooled_rmse(y_te[sel], p_ridge[sel]),
                               mlp=ev.pooled_rmse(y_te[sel], mlp_preds["full"][sel]),
                               naive=ev.pooled_rmse(y_te[sel], te["yp"][sel])))
    return results, scale_rows


if __name__ == "__main__":
    main()

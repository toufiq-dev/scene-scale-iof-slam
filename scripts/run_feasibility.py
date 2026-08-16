#!/usr/bin/env python3
"""Corrected synthetic feasibility study (library version).

Rebuild of the original `scene-scale.ipynb` experiment (verdict: PARTIAL/FAIL,
a naive-shift baseline beat the learned model). This version fixes the
experimental design:

  1. fair baseline: ridge trained on the SAME features as the learned model
     (persistence of the true IOF is runtime-unavailable and privileged --
     it is the ORACLE persistence diagnostic, "persistence (oracle)");
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

Round-2 review fixes (this version):

  * **estimated motion** by default (generator ``motion_source="estimated"``):
    the model consumes the relative motion derived from the accumulated
    estimated trajectory, corrupted by the pose error, as a real SLAM exposes;
  * **oracle vs runtime-available baselines**: "naive-shift (context)" is
    renamed "persistence (oracle)" and the table now separates the two
    families; a GRU temporal baseline and a classical covariance-propagation
    baseline are added;
  * **within-sequence AUROC as the PRIMARY early-warning metric** (labels from
    per-sequence thresholds tau_s = Q75(IOF_s)); global/per-seq/z-score/
    depth-normalized failure definitions are all reported (no scene-scale
    leakage via a single global threshold);
  * **full input-combination ablation matrix** (motion/depth/reliability x 7),
    including the reliability-masked depth test (motion+depth vs motion-only,
    both without reliability) that G3 gates on;
  * **stress mode** (`--stress`): re-runs the gates under degraded reliability
    (delayed/noisy/miscalibrated/masked/intermittent), jump failures, and
    realistic depth corruption.

Round-3 review fixes (this version):

  * **official-protocol IOF machinery** (C1/C2/M10): the proposal's P1-G1 gate
    is a *reimplementation deliverable* -- the repo now ships the official
    protocol's trajectory alignment (Umeyama Sim(3) + Kabsch SO(3)) and
    BIC-selected depth-distribution integration (`iof.py`). The runner
    reports a raw-vs-official target comparison leg on synthetic sequences:
    correlation (the check that both measure the same quantity) AND scale
    ratio (the C2 evidence that alignment removes the accumulated-drift
    component the raw target is built on).
  * **FlowAUC target with a distributional head** (M5): the generator records
    per-pixel flow samples and the runner trains a histogram head
    (`models.DistributionalMLP`) that predicts the flow distribution at
    t+h; Flow AUC(tau) follows from the predicted cumulative histogram and
    is reported against oracle persistence and constant baselines.
  * **label-noise sensitivity sweep** (`--sweep-samples`): the reviewer-
    requested 100/200/500/1000 per-frame IOF sample counts (M8).

Gates (>=2 of 3 seeds; with fewer seeds all must pass):
  G1 learned > linear on same features (RMSE and AUROC)
  G2 reliability signal adds value (full > motion+depth)
  G3 depth adds value with reliability MASKED (motion+depth > motion-only
     on RMSE and pooled AUROC) -- the make-or-break scene-scale claim
  G4 beats oracle persistence (contextual; the strongest simple forecast)
  G5 current-error estimation works (h=0 << constant)

Run:  python scripts/run_feasibility.py [--error-model decoupled]
                                   [--seeds 0,1,2]
                                   [--out reports/feasibility_results.json]
                                   [--fail-on-gates]
                                   [--stress]

`--fail-on-gates` exits non-zero unless gates G1, G2, G3 and G5 all pass,
so CI (`.github/workflows/ci.yml`) fails on a regression of the feasibility
verdict. `--seeds` selects the RNG seeds (default 0,1,2); CI uses a single
seed (e.g. `--seeds 0`) for the fast 1-seed smoke on pull requests and the
full 3-seed run on main. With fewer than 3 seeds the gate rule is strict
(all seeds must pass); with 3 seeds it is the majority rule (>=2 of 3).

`--stress` additionally runs a 1-seed matrix over generator stress variants
(degraded reliability, jump failures, realistic depth corruption, true vs
estimated motion) and writes reports/stress_results.json.
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
from scene_scale.generator import (  # noqa: E402
    POSE_CORR,
    ROT_INNOV_WEIGHT,
    DEPTH_CHOICES,
    generate_sequence,
)
from scene_scale.iof import (  # noqa: E402
    compute_iof,
    compute_iof_official,
    se3_to_T,
)
from scene_scale.models import (  # noqa: E402
    fit_ridge,
    flow_auc_from_probs,
    predict_distributional,
    predict_gru,
    predict_mlp,
    split_gru_inputs,
    standardize,
    train_distributional,
    train_gru,
    train_mlp,
)

SEEDS = [0, 1, 2]
CFG = dict(n_train=60, n_val=15, n_test=15, seq_len=80, k=5, h=5)
FX = 256.0

# --- generator stress variants (round-2 review: realism of the inputs) --------
STRESS_VARIANTS = {
    "baseline (estimated motion)": dict(),
    "true motion (pre-fix input)": dict(motion_source="true"),
    "reliability: delayed": dict(reliability_mode="delayed"),
    "reliability: noisy": dict(reliability_mode="noisy"),
    "reliability: miscalibrated": dict(reliability_mode="miscalibrated"),
    "reliability: masked": dict(reliability_mode="masked"),
    "reliability: intermittent": dict(reliability_mode="intermittent"),
    "jump failures": dict(jump_prob=0.05, jump_scale=3.0),
    "realistic depth corruption": dict(depth_corruption="realistic"),
}


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
    ap.add_argument("--fail-on-gates", action="store_true",
                    help="exit non-zero unless gates G1, G2, G3, G5 all pass (CI mode)")
    ap.add_argument("--seeds", default="0,1,2",
                    help="comma-separated RNG seeds (default 0,1,2; use e.g. 0 for a "
                         "1-seed CI smoke)")
    ap.add_argument("--stress", action="store_true",
                    help="also run the 1-seed generator stress matrix (degraded reliability, "
                         "jump failures, realistic depth) -> reports/stress_results.json")
    ap.add_argument("--samples", type=int, default=200,
                    help="per-frame IOF pixel samples (default 200; the M8 label-noise "
                         "sweep uses 100/200/500/1000)")
    ap.add_argument("--sweep-samples", action="store_true",
                    help="run the label-noise sensitivity sweep over per-frame IOF sample "
                         "counts 100/200/500/1000 (1 seed each) -> reports/samples_sweep.json "
                         "and exit")
    args = ap.parse_args()
    if args.sweep_samples:
        run_sweep(args.error_model)
        return
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        ap.error("--seeds must be a non-empty comma-separated list of integers")

    if not unit_tests():
        print("geometry unit tests failed; aborting")
        sys.exit(1)

    gen = partial(generate_sequence, error_model=args.error_model,
                  num_samples=args.samples)
    print(f"\n================ FEASIBILITY (FIXED DESIGN, error_model={args.error_model}) "
          f"================\n"
          f"generator: DEPTH_CHOICES={DEPTH_CHOICES}, ROT_INNOV_WEIGHT={ROT_INNOV_WEIGHT}, "
          f"motion_source=estimated (default), reliability_mode=clean, "
          f"num_samples={args.samples}")
    t0 = time.time()
    per_seed, scale_all, fail_defs, fa_all, official_all = [], [], [], [], []
    for seed in seeds:
        res, scale_rows, fdef, fa, official = run_seed_full(seed, gen)
        per_seed.append(res)
        scale_all.append(scale_rows)
        fail_defs.append(fdef)
        fa_all.append(fa)
        official_all.append(official)
        print(f"seed {seed} done ({time.time() - t0:.0f}s)")

    names = [r["name"] for r in per_seed[0]]
    print("\nmodel                       RMSE(mean+-std)  nRMSE  medSp   AUROC     wAUROC     AP")
    for i, name in enumerate(names):
        rows = [r[i] for r in per_seed]
        cells = []
        for k_ in ("rmse", "nrmse", "med_spearman", "auroc", "wauroc", "ap"):
            vals = np.array([r[k_] for r in rows], dtype=float)
            if np.isnan(vals).all():
                cells.append("   ---   ")
            else:
                cells.append(f"{np.nanmean(vals):.3f}+-{np.nanstd(vals):.3f}")
        print(f"{name:<27} {'   '.join(cells)}")

    print("\n===== FAILURE-DEFINITION COMPARISON (mlp full; AUROC, mean over seeds) =====")
    print("global-tau   per-seq-tau(wAUROC)   robust-zscore   depth-normalized")
    for k_ in ("global_tau", "per_seq", "zscore", "depthnorm"):
        vals = [f[k_] for f in fail_defs]
        print(f"  {k_:<12} {np.nanmean(vals):.3f}")

    print("\n================ SCENE-SCALE SPLIT (test, by median depth) ================")
    print("group    med_depth    ridge_RMSE   mlp_RMSE   naive_RMSE")
    for g in range(3):
        row = {k_: float(np.nanmean([scale_all[s][g][k_] for s in range(len(seeds))]))
               for k_ in ("med_depth", "ridge", "mlp", "naive")}
        print(f"{scale_all[0][g]['group']:<9} {row['med_depth']:9.2f}    "
              f"{row['ridge']:8.3f}   {row['mlp']:8.3f}   {row['naive']:8.3f}")

    print("\n===== FLOWAUC LEG (distributional head vs baselines; RMSE, mean over seeds) =====")
    print("tau=1px    tau=5px    tau=20px   officialAUC(0-100)")
    for label, key in (("model (histogram head)", "model"),
                       ("persistence (oracle)", "persist"),
                       ("constant", "constant")):
        cells = [f"{np.nanmean([f_[key][k] for f_ in fa_all]):.4f}"
                 for k in ("fa1", "fa5", "fa20", "fa_auc")]
        print(f"  {label:<24} {'   '.join(cells)}")

    print("\n===== OFFICIAL-PROTOCOL LEG (raw vs official IOF on synthetic sequences) =====")
    print("    corr(raw,official)   scale-ratio(official/raw)   GMM components")
    o_al = np.nanmean([o["corr_aligned"] for o in official_all])
    o_unal = np.nanmean([o["corr_unaligned"] for o in official_all])
    o_ratio = np.nanmean([o["ratio_aligned"] for o in official_all])
    o_ratio_unal = np.nanmean([o["ratio_unaligned"] for o in official_all])
    o_k = int(round(np.nanmean([o["gmm_components"] for o in official_all])))
    print(f"  aligned   {o_al:.3f}   ratio {o_ratio:.3f}   (unaligned corr {o_unal:.3f}, "
          f"ratio {o_ratio_unal:.3f})  k={o_k}")

    def rmse_of(res, name):
        return next(r["rmse"] for r in res if r["name"] == name)

    def auroc_of(res, name):
        return next(r["auroc"] for r in res if r["name"] == name)

    need = min(2, len(seeds))  # majority rule for 3 seeds; strict below that

    def gate(cond):
        return "PASS" if sum(cond(r) for r in per_seed) >= need else "FAIL"

    g1 = gate(lambda r: rmse_of(r, "mlp full") < rmse_of(r, "ridge (same features)")
              and auroc_of(r, "mlp full") > auroc_of(r, "ridge (same features)"))
    g2 = gate(lambda r: rmse_of(r, "mlp full") < rmse_of(r, "mlp motion+depth")
              and auroc_of(r, "mlp full") > auroc_of(r, "mlp motion+depth"))
    # G3: depth adds value with the reliability signal MASKED (both legs exclude
    # reliability) -- the make-or-break scene-scale test.
    g3 = gate(lambda r: rmse_of(r, "mlp motion+depth") < rmse_of(r, "mlp motion-only")
              and auroc_of(r, "mlp motion+depth") > auroc_of(r, "mlp motion-only"))
    g4 = gate(lambda r: rmse_of(r, "mlp full") < rmse_of(r, "persistence (oracle)")
              and auroc_of(r, "mlp full") > auroc_of(r, "persistence (oracle)"))
    g5 = gate(lambda r: rmse_of(r, "mlp full h=0 (current)")
              < 0.5 * rmse_of(r, "constant"))
    # G6 (reported, not gating): depth adds value with reliability PRESENT
    g6 = gate(lambda r: rmse_of(r, "mlp full") < rmse_of(r, "mlp motion+reliability")
              and auroc_of(r, "mlp full") > auroc_of(r, "mlp motion+reliability"))

    print(f"\n================ GATES (over {len(seeds)} seed(s)) ================")
    for label, g in [("G1 learned > linear on same features", g1),
                     ("G2 reliability signal adds value", g2),
                     ("G3 depth adds value (reliability MASKED)", g3),
                     ("G4 beats oracle persistence (contextual)", g4),
                     ("G5 current-error estimation works (h=0)", g5),
                     ("G6 depth adds value (reliability PRESENT, reported)", g6)]:
        print(f"{label:<52}: {g}")

    overall = "PASS" if all(g == "PASS" for g in (g1, g2, g3, g5)) else "PARTIAL/FAIL"
    print(f"\nOVERALL VERDICT (G1 & G2 & G3 & G5): {overall}")

    out = dict(cfg=CFG, seeds=seeds, error_model=args.error_model,
               generator=dict(depth_choices=list(DEPTH_CHOICES),
                              rot_innov_weight=ROT_INNOV_WEIGHT,
                              motion_source="estimated", reliability_mode="clean",
                              num_samples=args.samples),
               gates=dict(g1=g1, g2=g2, g3=g3, g4=g4, g5=g5, g6=g6, overall=overall),
               failure_definitions=dict(
                   global_tau=[f["global_tau"] for f in fail_defs],
                   per_seq=[f["per_seq"] for f in fail_defs],
                   zscore=[f["zscore"] for f in fail_defs],
                   depthnorm=[f["depthnorm"] for f in fail_defs]),
               flowauc=dict(model=[{k: f_["model"][k] for k in ("fa1", "fa5", "fa20",
                                                                 "fa_auc")} for f_ in fa_all],
                            persist=[{k: f_["persist"][k] for k in ("fa1", "fa5",
                                                                     "fa20", "fa_auc")} for f_ in fa_all],
                            constant=[{k: f_["constant"][k] for k in ("fa1", "fa5",
                                                                       "fa20", "fa_auc")} for f_ in fa_all]),
               official_iof=dict(corr_aligned=[o["corr_aligned"] for o in official_all],
                                 corr_unaligned=[o["corr_unaligned"] for o in official_all],
                                 ratio_aligned=[o["ratio_aligned"] for o in official_all],
                                 ratio_unaligned=[o["ratio_unaligned"] for o in official_all],
                                 gmm_components=[o["gmm_components"] for o in official_all]),
               per_seed=[{r["name"]: {k_: r[k_] for k_ in
                                      ("rmse", "nrmse", "med_spearman", "auroc",
                                       "med_auroc", "wauroc", "med_wauroc", "ap")}
                          for r in res} for res in per_seed])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nresults -> {args.out} ({time.time() - t0:.0f}s total)")

    if args.stress:
        run_stress(args.error_model, args.samples)

    if args.fail_on_gates and overall != "PASS":
        print(f"GATES NOT ALL PASS ({overall}) -> exiting 1 (CI gate)")
        sys.exit(1)


def run_stress(error_model: str, num_samples: int = 200):
    """1-seed gate matrix over the generator stress variants."""
    print("\n================ STRESS MATRIX (1 seed each) ================")
    t0 = time.time()
    results = {}
    for label, kwargs in STRESS_VARIANTS.items():
        gen = partial(generate_sequence, error_model=error_model,
                      num_samples=num_samples, **kwargs)
        res, _, _ = run_seed_full(0, gen)[:3]
        rmse = lambda n: next(r["rmse"] for r in res if r["name"] == n)  # noqa: E731
        auroc = lambda n: next(r["auroc"] for r in res if r["name"] == n)  # noqa: E731
        wauroc = lambda n: next(r["wauroc"] for r in res if r["name"] == n)  # noqa: E731
        g = dict(
            g1=rmse("mlp full") < rmse("ridge (same features)")
            and auroc("mlp full") > auroc("ridge (same features)"),
            g2=rmse("mlp full") < rmse("mlp motion+depth")
            and auroc("mlp full") > auroc("mlp motion+depth"),
            g3=rmse("mlp motion+depth") < rmse("mlp motion-only")
            and auroc("mlp motion+depth") > auroc("mlp motion-only"),
            g5=rmse("mlp full h=0 (current)") < 0.5 * rmse("constant"),
        )
        overall = all(g.values())
        results[label] = dict(gates={k: ("PASS" if v else "FAIL") for k, v in g.items()},
                              overall="PASS" if overall else "FAIL",
                              full_rmse=float(rmse("mlp full")),
                              full_auroc=float(auroc("mlp full")),
                              full_wauroc=float(wauroc("mlp full")),
                              persistence_rmse=float(rmse("persistence (oracle)")),
                              persistence_auroc=float(auroc("persistence (oracle)")))
        print(f"{label:<38} full RMSE {results[label]['full_rmse']:.3f} "
              f"wAUROC {results[label]['full_wauroc']:.3f} "
              f"gates {results[label]['gates']} -> {results[label]['overall']}")
    path = ROOT / "reports" / "stress_results.json"
    with open(path, "w") as f:
        json.dump(dict(error_model=error_model, variants=results), f, indent=2)
    print(f"\nstress results -> {path} ({time.time() - t0:.0f}s)")


def run_seed_full(seed, gen):
    """Full per-seed experiment.

    Returns (results list, scene-scale rows, failure-definition comparison
    dict, flowauc dict, official-iof leg dict).
    """
    seed0 = 1000 + 100 * seed
    splits = build_dataset(CFG["n_train"], CFG["n_val"], CFG["n_test"],
                           CFG["seq_len"], CFG["k"], CFG["h"], seed0,
                           generator=gen)
    tr, va, te = splits["train"], splits["val"], splits["test"]
    masks = feature_masks(CFG["k"])
    tau = float(np.percentile(tr["yh"], 75))
    y_te, seq_te = te["yh"], te["seq"]
    results = []

    # --- constant ---
    c = float(tr["yh"].mean())
    results.append(ev.evaluate("constant", y_te, np.full_like(y_te, c), seq_te, tau))

    # --- persistence (ORACLE: uses the true IOF at t, which requires GT pose;
    #     the strongest simple forecast and the bar to clear, not a deployable
    #     baseline -- the runtime-available persistence proxy is B6) ---
    results.append(ev.evaluate("persistence (oracle)", y_te, te["yp"], seq_te, tau))

    # --- ridge on the SAME features as the model (the fair linear baseline) ---
    ytr_s, yva_s, yte_s, ym, ys = standardize(tr["yh"], va["yh"], te["yh"])
    _, ridge_m, sc_f, _, _ = fit_ridge(tr["X"][:, masks["full"]], tr["yh"],
                                       va["X"][:, masks["full"]], va["yh"])
    p_ridge = ridge_m.predict(sc_f.transform(te["X"][:, masks["full"]])) * ys + ym
    results.append(ev.evaluate("ridge (same features)", y_te, p_ridge, seq_te, tau))

    # --- classical covariance propagation: AR(1) stationary forecast covariance
    #     pushed through the analytic IOF Jacobian, using ONLY runtime quantities
    #     (estimated motion -> innovation scale; estimated median depth -> 1/Z;
    #     model knowledge of the AR coefficient). A blind forecast: it never
    #     observes the error realization, so it is the classical competitor. ---
    p_classical = classical_covariance_forecast(te["X"], CFG["k"], CFG["h"])
    results.append(ev.evaluate("classical cov-prop (blind)", y_te, p_classical, seq_te, tau))

    # --- full input-combination ablation matrix (7 masks) ---
    mlp_preds = {}
    labels = {"motion": "mlp motion-only", "depth": "mlp depth-only",
              "rel": "mlp reliability-only", "motion_depth": "mlp motion+depth",
              "motion_rel": "mlp motion+reliability",
              "depth_rel": "mlp depth+reliability", "full": "mlp full"}
    for key, mask in masks.items():
        sc = StandardScaler().fit(tr["X"][:, mask])
        Ztr = sc.transform(tr["X"][:, mask]).astype(np.float32)
        Zva = sc.transform(va["X"][:, mask]).astype(np.float32)
        Zte = sc.transform(te["X"][:, mask]).astype(np.float32)
        m = train_mlp(Ztr, ytr_s, Zva, yva_s, seed=42 + seed)
        p = predict_mlp(m, Zte) * ys + ym
        results.append(ev.evaluate(labels[key], y_te, p, seq_te, tau))
        if key in ("full", "motion_depth", "motion_only"):
            mlp_preds[key] = p

    # --- GRU temporal baseline on the same full features (review: a learned
    #     temporal model must be the fairer comparison than an MLP alone) ---
    Xs_tr, Xc_tr = split_gru_inputs(tr["X"], CFG["k"])
    Xs_va, Xc_va = split_gru_inputs(va["X"], CFG["k"])
    Xs_te, Xc_te = split_gru_inputs(te["X"], CFG["k"])
    scs, scc = StandardScaler().fit(Xs_tr.reshape(-1, Xs_tr.shape[2])), \
        StandardScaler().fit(Xc_tr)
    Zs_tr = scs.transform(Xs_tr.reshape(-1, Xs_tr.shape[2])).reshape(Xs_tr.shape).astype(np.float32)
    Zs_va = scs.transform(Xs_va.reshape(-1, Xs_va.shape[2])).reshape(Xs_va.shape).astype(np.float32)
    Zs_te = scs.transform(Xs_te.reshape(-1, Xs_te.shape[2])).reshape(Xs_te.shape).astype(np.float32)
    Zc_tr = scc.transform(Xc_tr).astype(np.float32)
    Zc_va = scc.transform(Xc_va).astype(np.float32)
    Zc_te = scc.transform(Xc_te).astype(np.float32)
    mg = train_gru(Zs_tr, Zc_tr, ytr_s, Zs_va, Zc_va, yva_s, seed=42 + seed)
    p_gru = predict_gru(mg, Zs_te, Zc_te) * ys + ym
    results.append(ev.evaluate("gru (temporal, same features)", y_te, p_gru, seq_te, tau))

    # --- analytic two-stage: MLP forecast of error magnitudes + depth, then the
    #     explicit projection transform + linear calibration (physics-explicit) ---
    from sklearn.linear_model import LinearRegression

    scx = StandardScaler().fit(tr["X"][:, masks["full"]])
    Ztr = scx.transform(tr["X"][:, masks["full"]]).astype(np.float32)
    Zva = scx.transform(va["X"][:, masks["full"]]).astype(np.float32)
    Zte = scx.transform(te["X"][:, masks["full"]]).astype(np.float32)

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

    # --- failure-definition comparison for the full model (primary metric) ---
    p_full = mlp_preds["full"]
    med_te = te["X"][:, 6 * CFG["k"] + CFG["k"]]  # first depth stat = median est. depth
    z_lab = ev.labels_robust_zscore(y_te, seq_te, z_thr=1.5)
    dn_tau = float(np.percentile(y_te * med_te, 75))
    dn_lab = ev.labels_depth_normalized(y_te, med_te, dn_tau)
    fail_defs = dict(
        global_tau=ev.auroc(ev.labels_global(y_te, tau), p_full),
        per_seq=ev.auroc(ev.labels_per_seq_threshold(y_te, seq_te), p_full),
        zscore=ev.auroc(z_lab, p_full),
        depthnorm=ev.auroc(dn_lab, p_full),
    )

    # --- scene-scale split: test sequences grouped by true median depth ---
    med_te_true = te["median"]
    order = np.argsort(med_te_true)
    scale_rows = []
    for gname, idxs in zip(["near", "medium", "far"], np.array_split(order, 3)):
        sel = np.isin(seq_te, idxs)
        scale_rows.append(dict(group=gname, med_depth=float(np.mean(med_te_true[idxs])),
                               ridge=ev.pooled_rmse(y_te[sel], p_ridge[sel]),
                               mlp=ev.pooled_rmse(y_te[sel], mlp_preds["full"][sel]),
                               naive=ev.pooled_rmse(y_te[sel], te["yp"][sel])))

    # --- FlowAUC leg (round-3 M5): distributional histogram head vs baselines.
    #     Target: the per-pixel flow distribution at t+h; FlowAUC(tau) is the
    #     fraction of pixels with flow < tau, reported at 1/5/20 px plus the
    #     official 0-100 px AUC (mean CDF, rescaled to 0-100).
    fa = flowauc_leg(tr, va, te, masks, seed)

    # --- official-protocol leg (round-3 C1/C2): raw vs official IOF on a few
    #     fresh synthetic sequences -- correlation (same quantity?) and scale
    #     ratio (how much of the raw target the trajectory alignment removes).
    official = official_iof_leg(gen, seed)

    return results, scale_rows, fail_defs, fa, official


def flowauc_leg(tr, va, te, masks, seed: int):
    """Train the distributional FlowAUC head and compare against baselines.

    Returns dict(model/persist/constant) each with RMSE of the predicted
    FlowAUC(tau) at tau in {1, 5, 20} px and of the official 0--100 AUC.
    Persistence is the oracle variant (true per-pixel flows at time t); the
    constant predicts the training-set mean FlowAUC vector.
    """
    sc = StandardScaler().fit(tr["X"][:, masks["full"]])
    Ztr = sc.transform(tr["X"][:, masks["full"]]).astype(np.float32)
    Zva = sc.transform(va["X"][:, masks["full"]]).astype(np.float32)
    Zte = sc.transform(te["X"][:, masks["full"]]).astype(np.float32)
    thr = (1.0, 5.0, 20.0)
    true_te = flow_auc_from_probs(flow_hist(te["yflow"]), thresholds=thr)
    true_persist = flow_auc_from_probs(flow_hist(te["yflowp"]), thresholds=thr)
    auc_te = flow_auc_auc(te["yflow"])
    auc_persist = flow_auc_auc(te["yflowp"])
    # constant: training-set mean FlowAUC at each threshold
    true_tr = flow_auc_from_probs(flow_hist(tr["yflow"]), thresholds=thr)
    c_vec = np.concatenate([true_tr.mean(0), [flow_auc_auc(tr["yflow"]).mean()]])

    def rmse_vec(pred):
        err = pred - np.concatenate([true_te, auc_te[:, None]], axis=1)
        return np.sqrt(np.mean(err ** 2, axis=0))

    mg = train_distributional(Ztr, tr["yflow"], Zva, va["yflow"], seed=42 + seed)
    probs = predict_distributional(mg, Zte)
    pred_model = np.concatenate(
        [flow_auc_from_probs(probs, thresholds=thr), flow_auc_auc_from_probs(probs)[:, None]],
        axis=1,
    )
    pred_persist = np.concatenate([true_persist, auc_persist[:, None]], axis=1)
    pred_const = np.broadcast_to(c_vec[None, :], pred_model.shape)
    r_model, r_persist, r_const = (rmse_vec(p) for p in (pred_model, pred_persist, pred_const))
    return dict(
        model=dict(fa1=float(r_model[0]), fa5=float(r_model[1]),
                   fa20=float(r_model[2]), fa_auc=float(r_model[3])),
        persist=dict(fa1=float(r_persist[0]), fa5=float(r_persist[1]),
                     fa20=float(r_persist[2]), fa_auc=float(r_persist[3])),
        constant=dict(fa1=float(r_const[0]), fa5=float(r_const[1]),
                      fa20=float(r_const[2]), fa_auc=float(r_const[3])),
    )


def flow_hist(flows):
    """(N, S) per-pixel flow samples -> (N, K) normalized bin counts."""
    from scene_scale.models import flow_histograms

    return flow_histograms(flows)


def flow_auc_auc(flows, grid=(0.0, 100.0, 41)):
    """Official-style Flow AUC: mean CDF over the 0--100 px threshold grid.

    Returns (N,) values on the 0--1 scale (multiply by 100 for the official
    score). The CDF at each grid threshold comes from the empirical
    per-pixel flow distribution.
    """
    from scene_scale.models import FLOW_BIN_EDGES

    probs = flow_hist(flows)
    taus = np.linspace(grid[0], grid[1], grid[2])
    cdf = np.cumsum(probs, axis=1)
    upper = FLOW_BIN_EDGES[1:]
    out = np.zeros((len(probs), len(taus)))
    for j, tau in enumerate(taus):
        idx = int(np.searchsorted(upper, tau))
        if idx >= len(upper):
            out[:, j] = 1.0
            continue
        below = cdf[:, idx - 1] if idx > 0 else 0.0
        lo = upper[idx - 1] if idx > 0 else 0.0
        frac = (tau - lo) / max(upper[idx] - lo, 1e-9)
        out[:, j] = below + probs[:, idx] * frac
    return out.mean(axis=1)


def flow_auc_auc_from_probs(probs):
    """Official-style Flow AUC from predicted bin probabilities (mean CDF)."""
    from scene_scale.models import FLOW_BIN_EDGES

    cdf = np.cumsum(probs, axis=1)
    upper = FLOW_BIN_EDGES[1:]
    taus = np.linspace(0.0, 100.0, 41)
    out = np.zeros((len(probs), len(taus)))
    for j, tau in enumerate(taus):
        idx = int(np.searchsorted(upper, tau))
        if idx >= len(upper):
            out[:, j] = 1.0
            continue
        below = cdf[:, idx - 1] if idx > 0 else 0.0
        lo = upper[idx - 1] if idx > 0 else 0.0
        frac = (tau - lo) / max(upper[idx] - lo, 1e-9)
        out[:, j] = below + probs[:, idx] * frac
    return out.mean(axis=1)


def official_iof_leg(gen, seed: int, n_seq: int = 3, seq_len: int = 80):
    """Raw vs official-protocol IOF on fresh synthetic sequences.

    Computes per-frame raw IOF (the primary target) and official IOF (with
    and without trajectory alignment) for ``n_seq`` generated sequences and
    returns per-sequence correlations and scale ratios -- the C1/C2 evidence:
    correlation asks whether both measure the same quantity, the scale ratio
    quantifies how much of the raw target the alignment removes.
    """
    from scipy.stats import pearsonr

    cors_al, cors_unal, ratio_al, ratio_unal, ks = [], [], [], [], []
    for j in range(n_seq):
        seq = gen(seq_len=seq_len, seed=2000 + seed * 100 + j)
        raw = seq["iof"]
        off_al, (w, mu, sd) = compute_iof_official(
            seq["T_hat"], seq["T_gt"], seq["depth_samples"], align=True, seed=seed
        )
        off_unal, _ = compute_iof_official(
            seq["T_hat"], seq["T_gt"], seq["depth_samples"], align=False, seed=seed
        )
        cors_al.append(float(pearsonr(raw, off_al).statistic))
        cors_unal.append(float(pearsonr(raw, off_unal).statistic))
        ratio_al.append(float(np.mean(off_al) / max(np.mean(raw), 1e-9)))
        ratio_unal.append(float(np.mean(off_unal) / max(np.mean(raw), 1e-9)))
        ks.append(int(len(w)))
    return dict(
        corr_aligned=float(np.mean(cors_al)),
        corr_unaligned=float(np.mean(cors_unal)),
        ratio_aligned=float(np.mean(ratio_al)),
        ratio_unaligned=float(np.mean(ratio_unal)),
        gmm_components=float(np.mean(ks)),
    )


def run_sweep(error_model: str, samples=(100, 200, 500, 1000)):
    """Label-noise sensitivity sweep over per-frame IOF sample counts (M8).

    One seed per count; reports the mlp-full / ridge / persistence metrics
    plus the gate verdict at each count -> reports/samples_sweep.json.
    """
    print("\n================ LABEL-NOISE SENSITIVITY SWEEP (M8) ================")
    t0 = time.time()
    out = {}
    for ns in samples:
        gen = partial(generate_sequence, error_model=error_model, num_samples=ns)
        res, _, _, _, _ = run_seed_full(0, gen)
        rmse = lambda n: next(r["rmse"] for r in res if r["name"] == n)  # noqa: E731
        auroc = lambda n: next(r["auroc"] for r in res if r["name"] == n)  # noqa: E731
        wauroc = lambda n: next(r["wauroc"] for r in res if r["name"] == n)  # noqa: E731
        g1 = rmse("mlp full") < rmse("ridge (same features)") and \
            auroc("mlp full") > auroc("ridge (same features)")
        g3 = rmse("mlp motion+depth") < rmse("mlp motion-only") and \
            auroc("mlp motion+depth") > auroc("mlp motion-only")
        out[str(ns)] = dict(
            full_rmse=float(rmse("mlp full")),
            full_auroc=float(auroc("mlp full")),
            full_wauroc=float(wauroc("mlp full")),
            ridge_rmse=float(rmse("ridge (same features)")),
            persistence_rmse=float(rmse("persistence (oracle)")),
            g1="PASS" if g1 else "FAIL",
            g3="PASS" if g3 else "FAIL",
        )
        print(f"num_samples={ns:<5} full RMSE {out[str(ns)]['full_rmse']:.3f} "
              f"wAUROC {out[str(ns)]['full_wauroc']:.3f} "
              f"G1 {out[str(ns)]['g1']}  G3 {out[str(ns)]['g3']}")
    path = ROOT / "reports" / "samples_sweep.json"
    with open(path, "w") as f:
        json.dump(dict(error_model=error_model, samples=list(samples), rows=out), f, indent=2)
    print(f"\nsamples sweep -> {path} ({time.time() - t0:.0f}s)")


def classical_covariance_forecast(X, k: int = 5, h: int = 5) -> np.ndarray:
    """Classical blind forecast: AR(1) covariance -> IOF Jacobian -> image space.

    Uses ONLY runtime quantities (the reviewer's requirement):
      * estimated motion magnitudes -> per-frame innovation scale
        sigma_t = 0.005 + 0.05 * motion_mag  (the generator's known model);
      * estimated median depth Z_hat -> the 1/Z translation Jacobian;
      * model knowledge: AR(1) coefficient rho and rotation weight w.

    The h-step forecast covariance of each pose-error component is
        var_i(h) = sigma_i^2 * (1 - rho^{2h}) / (1 - rho^2),
    and the predicted RMS image-space displacement is
        pred = sqrt( sum_i J_i^2 * var_i(h) )
    with the analytic pinhole Jacobian J:  d(flow)/dt ~= f/Z_hat (translation,
    with f/(2Z_hat) for the optical-axis component), d(flow)/d(omega) ~= f.
    This is the classical competitor: it never observes the error realization,
    so it cannot anticipate error spikes -- the learned model's job.
    """
    m = 6 * k
    last = X[:, m - 6 : m]  # last-frame estimated motion (|t|, |r|)
    motion_mag = np.linalg.norm(last[:, :3], axis=1) + 0.3 * np.linalg.norm(last[:, 3:], axis=1)
    sigma_t = 0.005 + 0.05 * motion_mag
    z_hat = np.clip(X[:, m + k], 1e-3, None)  # first depth stat: median est. depth
    decay = (1.0 - POSE_CORR ** (2 * h)) / (1.0 - POSE_CORR ** 2)
    var_t = (sigma_t ** 2) * decay          # per translation component
    var_r = (sigma_t * ROT_INNOV_WEIGHT) ** 2 * decay  # per rotation component
    j_t = FX / z_hat                        # d flow / d t_x, t_y
    j_tz = 0.5 * j_t                        # optical-axis component (radial)
    pred2 = (j_t ** 2 + j_t ** 2 + j_tz ** 2) * var_t + 3.0 * (FX ** 2) * var_r
    return np.sqrt(np.clip(pred2, 0.0, None))


if __name__ == "__main__":
    main()

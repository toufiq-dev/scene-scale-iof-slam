# Scene-Scale-Aware Prediction of Induced Optical-Flow Error for Visual SLAM

[![CI](https://github.com/toufiq-dev/scene-scale-iof-slam/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/toufiq-dev/scene-scale-iof-slam/actions/workflows/ci.yml)

Thesis repository. Predicts the **future, scene-conditioned visual consequence of SLAM uncertainty**
(Induced Optical Flow / Flow AUC, Princeton365, ICCV 2025) from runtime-available SLAM information,
as a lightweight reliability layer around existing SLAM systems.

Thesis proposal: `docs/thesis_proposal.md` · Annotated related work: `docs/related_work.md` ·
Feasibility report: `reports/feasibility.md`.

## Status

- ✅ Corrected synthetic feasibility gate (this repo): **all gates G1–G6 PASS** under the round-2
  generator — depth-decoupled error model, **estimated (corrupted) motion by default**, oracle vs
  runtime-available baseline split (incl. GRU and classical covariance propagation), and
  **within-sequence AUROC as the primary early-warning metric**. The stress matrix confirms the
  scene-scale claim (G3) survives degraded reliability, jump failures, and realistic depth
  corruption.
- ⏳ Phase 1 (real-data pilot on Princeton365 with DROID-SLAM, governed by pre-registered gates
  P1-G1…P1-G7 incl. official-IOF reproduction ρ > 0.98) — next.

## Repository layout

```
configs/feasibility.json        experiment configuration
docs/thesis_proposal.md         refactored proposal (post round-2 review)
docs/related_work.md            annotated literature review (Aug 2026)
reports/feasibility.md          feasibility report + honest verdict
reports/feasibility_results.json per-seed results (generated; decoupled, estimated motion)
reports/feasibility_results_coupled.json pre-fix (coupled) run, archived
reports/feasibility_results_true_motion.json round-1 (decoupled, true motion) run, archived
reports/stress_results.json     generator stress matrix (generated)
src/scene_scale/                library
  iof.py                        pinhole IOF geometry (se3, vectorized IOF; explicit pose convention)
  generator.py                  synthetic generator; decoupled error model, estimated motion,
                                reliability/jump/depth stress variants
  features.py                   windowing, full input-combination masks, dataset builder
  models.py                     MLP + GRU + ridge, training helpers
  eval.py                       per-seq metrics, 4 failure definitions (within-seq PRIMARY),
                                ECE, principled warning lead time
scripts/run_feasibility.py      corrected feasibility experiment (3 seeds, gates, --stress)
tests/test_iof.py               geometry unit tests (incl. inverse-swap convention test)
tests/test_generator.py         error-model + stress-variant unit tests
experiments/feasibility_fixed/  archived original corrected study (standalone)
```

## Quickstart

```bash
pip install -r requirements.txt        # numpy scipy scikit-learn torch

# geometry unit tests
python tests/test_iof.py               # 6/6 pass

# corrected feasibility experiment (3 seeds; ~3 min on CPU)
python scripts/run_feasibility.py                 # default: decoupled error model, estimated motion
python scripts/run_feasibility.py --error-model coupled \
    --out reports/feasibility_results_coupled.json # legacy confounded generator (reproduction)
python scripts/run_feasibility.py --stress        # 1-seed stress matrix -> reports/stress_results.json
# results -> reports/feasibility_results.json
```

## CI

`.github/workflows/ci.yml` runs on every push and pull request:

- **unit-tests** — geometry (7, incl. inverse-swap pose-convention test) and generator (14,
  incl. stress-variant tests) unit tests;
- **feasibility-smoke** — 1-seed gate on pull requests and non-main branches for fast feedback
  (`--seeds 0 --fail-on-gates`);
- **feasibility-gate** — the full 3-seed experiment on `main` only, with `--fail-on-gates`, so
  the workflow fails if any of gates G1, G2, G3, G5 regress from **PASS** (torch is installed
  CPU-only to keep the runner fast).

## Key numbers (feasibility, 3 seeds, unseen test sequences, decoupled + estimated-motion generator)

| model | RMSE | AUROC (pooled) | wAUROC (within-seq) |
|---|---|---|---|
| constant | 4.209 | 0.500 | 0.500 |
| persistence (oracle) | 3.446 | 0.886 | 0.549 |
| ridge (same features) | 3.626 | 0.880 | 0.519 |
| classical cov-prop (blind) | 2.826 | 0.918 | 0.509 |
| gru (temporal, same features) | 1.323 | 0.981 | 0.638 |
| mlp motion-only | 1.729 | 0.971 | 0.721 |
| mlp motion+depth | 1.124 | 0.988 | 0.672 |
| **mlp full (motion+depth+reliability)** | **0.948** | **0.992** | **0.688** |

Gates: G1 (learned > linear) PASS · G2 (reliability matters) PASS · G3 (depth matters, reliability
**masked**) **PASS** · G4 (beats oracle persistence) PASS · G5 (h=0 estimation) PASS · G6 (depth with
reliability present) PASS — **overall PASS**. The pre-fix G3 confound (depth coupled into the error
scale, plus a depth range too narrow for the rotation-flow floor to admit a material depth signal) is
resolved, and the stress matrix shows G3 survives degraded reliability, jump failures, and realistic
depth corruption (see `reports/feasibility.md` §3, §6).

## Roadmap

1. Princeton365 subset pipeline + IOF target generation; **official-IOF reproduction gate (ρ > 0.98)**;
   DROID-SLAM instrumentation with ORB-SLAM3 fallback; failure-event counts within posed segments;
   pre-registered pilot gates P1-G1…P1-G7.
2. Predictor training; mandatory baselines B1–B9 (oracle persistence, oracle pose-error, oracle IOF,
   constant, ridge, SEESys-style h=0, classical covariance propagation, analytic two-stage, GRU).
3. Ablations (full input-combination matrix; horizons; targets; frozen VGGT features).
4. Generalization: cross-category and cross-backbone (ORB-SLAM3, MASt3R-SLAM).
5. Adaptive-SLAM demonstration with oracle-triggered upper bound; official test evaluation;
   final systematic literature re-check.

# Scene-Scale-Aware Prediction of Induced Optical-Flow Error for Visual SLAM

[![CI](https://github.com/toufiq-dev/scene-scale-iof-slam/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/toufiq-dev/scene-scale-iof-slam/actions/workflows/ci.yml)

Thesis repository. Predicts the **future, scene-conditioned visual consequence of SLAM uncertainty**
(Induced Optical Flow / Flow AUC, Princeton365, ICCV 2025) from runtime-available SLAM information,
as a lightweight reliability layer around existing SLAM systems.

Thesis proposal: `docs/thesis_proposal.md` · Annotated related work: `docs/related_work.md` ·
Feasibility report: `reports/feasibility.md`.

## Status

- ✅ Corrected synthetic feasibility gate (this repo): **all gates G1–G5 PASS** with the
  depth-decoupled error model — learned predictor beats persistence and a fair linear baseline,
  scene-scale-aware, reliability signal and scene geometry each add value (the pre-fix G3 confound is
  resolved: depth now enters only through the IOF 1/Z target, mirroring reality).
- ⏳ Phase 1 (real-data pilot on Princeton365 with DROID-SLAM) — next.

## Repository layout

```
configs/feasibility.json        experiment configuration
docs/thesis_proposal.md         refactored proposal (v2.0, post-review)
docs/related_work.md            annotated literature review (Aug 2026)
reports/feasibility.md          feasibility report + honest verdict
reports/feasibility_results.json per-seed results (generated, decoupled model)
reports/feasibility_results_coupled.json pre-fix (coupled) run, archived
src/scene_scale/                library
  iof.py                        pinhole IOF geometry (se3, vectorized IOF)
  generator.py                  synthetic generator; depth-decoupled error model (G3 fix)
  features.py                   windowing, ablation masks, dataset builder
  models.py                     MLP + ridge baselines, training helpers
  eval.py                       per-sequence metrics, ECE, warning lead time
scripts/run_feasibility.py      corrected feasibility experiment (3 seeds, gates)
tests/test_iof.py               geometry unit tests
tests/test_generator.py         error-model unit tests (G3 confound)
experiments/feasibility_fixed/  archived original corrected study (standalone)
```

## Quickstart

```bash
pip install -r requirements.txt        # numpy scipy scikit-learn torch

# geometry unit tests
python tests/test_iof.py               # 6/6 pass

# corrected feasibility experiment (3 seeds; ~6 min on CPU)
python scripts/run_feasibility.py                 # default: depth-decoupled error model
python scripts/run_feasibility.py --error-model coupled \
    --out reports/feasibility_results_coupled.json # legacy confounded generator (reproduction)
# results -> reports/feasibility_results.json
```

## CI

`.github/workflows/ci.yml` runs on every push and pull request:

- **unit-tests** — geometry (6) and error-model (4) unit tests;
- **feasibility-smoke** — 1-seed gate on pull requests and non-main branches for fast feedback
  (`--seeds 0 --fail-on-gates`);
- **feasibility-gate** — the full 3-seed experiment on `main` only, with `--fail-on-gates`, so
  the workflow fails if any of gates G1, G2, G3, G5 regress from **PASS** (torch is installed
  CPU-only to keep the runner fast).

## Key numbers (feasibility, 3 seeds, unseen test sequences, decoupled generator)

| model | RMSE | AUROC |
|---|---|---|
| constant | 4.209 | 0.500 |
| naive-shift (context) | 3.446 | 0.886 |
| ridge (same features) | 3.624 | 0.881 |
| mlp motion-only | 1.731 | 0.972 |
| mlp motion+depth | 1.130 | 0.988 |
| **mlp full (motion+depth+reliability)** | **0.993** | **0.991** |

Gates: G1 (learned > linear) PASS · G2 (reliability matters) PASS · G3 (depth matters) **PASS** ·
G4 (beats persistence) PASS · G5 (h=0 estimation) PASS — **overall PASS** (the pre-fix G3 confound —
depth coupled into the error scale, plus a depth range too narrow for the rotation-flow floor to
admit a material depth signal — is resolved; see `reports/feasibility.md` §3).

## Roadmap

1. Princeton365 subset pipeline + IOF target generation + DROID-SLAM instrumentation (cached outputs).
2. Predictor training; mandatory baselines (constant, persistence, ridge-on-features, SEESys-style
   h=0, error-propagation, analytic two-stage, oracle).
3. Ablations (image/motion/depth/reliability; horizons; targets; frozen VGGT features).
4. Generalization: cross-category and cross-backbone (ORB-SLAM3, MASt3R-SLAM).
5. Adaptive-SLAM demonstration with oracle-triggered upper bound; official test evaluation;
   final systematic literature re-check.

# Fixed synthetic feasibility study

Rebuild of the `scene-scale.ipynb` feasibility experiment after senior-review feedback.
Original verdict: **PARTIAL/FAIL** (a naive-shift baseline beat the learned model on AUROC).
This version fixes the experimental design, not just the model.

## Fixes applied

1. **Fair baseline**: a ridge forecast trained on the *same input features* as the learned
   model, instead of "persistence of the true IOF" (runtime-unavailable, privileged).
2. **Reliability signal fed to the model**: the generator's `confidence` signal (a *noisy*
   observation of the hidden error state, like residuals/covariance in real SLAM) is now an
   input. It is noisy so it is not a deterministic function of the true error (GIGO-circular).
3. **h=0 decomposition**: a current-error estimation stage (SEESys-style) is run alongside
   the h-step forecasting stage.
4. **Analytic two-stage baseline**: nonlinear (MLP) forecasts of error magnitudes + depth,
   converted to IOF via the explicit projection formula, then linearly calibrated.
5. **Informative motion**: varying forward speed, rotation bursts, 6-DoF magnitudes.
6. **Rigor**: per-sequence Spearman/AUROC, 3 seeds, mean ± std.

## Results (test on unseen sequences, 3 seeds, mean ± std)

| model | RMSE | medSpearman | AUROC | medAUROC | AP |
|---|---|---|---|---|---|
| constant | 1.810 ± 0.128 | — | 0.500 | 0.500 | 0.280 |
| naive-shift (context) | 1.807 ± 0.022 | 0.184 | 0.742 | 0.604 | 0.564 |
| ridge (same features) | 1.660 ± 0.065 | 0.099 | 0.734 | 0.538 | 0.513 |
| mlp motion-only | 0.728 ± 0.021 | 0.864 | 0.964 | 0.963 | 0.923 |
| mlp motion+depth | 0.654 ± 0.035 | 0.844 | 0.954 | 0.953 | 0.906 |
| **mlp full** | **0.569 ± 0.043** | **0.897** | **0.967** | **0.963** | **0.933** |
| analytic 2-stage (phys-explicit) | 1.187 ± 0.062 | 0.726 | 0.888 | 0.882 | 0.715 |
| mlp full h=0 (current) | 0.588 ± 0.058 | 0.872 | 0.967 | 0.964 | 0.932 |

Scene-scale split (test, by median depth): near (2.1 m) RMSE 0.473 vs ridge 2.225 / naive 2.441;
medium (8.1 m) 0.576 vs 1.341 / 1.490; far (21.4 m) 0.643 vs 1.213 / 1.261.

## Gates (≥2 of 3 seeds)

- G1 learned > linear on same features: **PASS**
- G2 reliability signal adds value: **PASS**
- G3 geometry (depth) adds value: **PARTIAL** — RMSE improves in every seed (0.728→0.654),
  but AUROC does not (0.964→0.954). The generator confounds motion and depth (the error
  scale couples them multiplicatively), so motion-only already saturates the ranking.
- G4 beats naive shift: **PASS** (the failure that sank the original study)
- G5 current-error estimation works (h=0): **PASS**

## Verdict: conditional pass — proceed to a small real-data pilot

The failure mode of the original study is resolved: with runtime-available features that carry
error-state information (motion + noisy reliability), the learned predictor beats the
persistence baseline on RMSE *and* AUROC, and beats the fair linear baseline on the same
features by a wide margin. Scene-scale awareness is demonstrated (near-scene RMSE is far
below the baselines').

Caveats to carry into the real-data stage:
- The generator still makes IOF strongly determined by runtime features; the synthetic task
  is comparatively easy (AUROC ceiling ~0.96). Real DROID-SLAM error dynamics will be harder.
- Depth's marginal *rank* benefit (G3) is not demonstrable in-silico because of the
  motion/depth confound; it must be re-tested on Princeton365, where depth enters the IOF
  denominator independently of motion.
- The explicit physics two-stage is *worse* than direct regression in this regime — a genuine,
  reportable negative for the "differentiable projection layer" hypothesis; revisit on real data.
- The naive-shift (persistence) baseline must appear in every real-data table, with per-sequence
  metrics (pooled metrics flattered the naive baseline in the original study).

## Files

- `fixed_feasibility.py` — the experiment (run with any numpy/scipy/sklearn/torch env).
- `results_fixed.json` — per-seed metrics for all models and gates.

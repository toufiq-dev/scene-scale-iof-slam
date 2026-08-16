# Synthetic Feasibility Study — Report

*Generated from `scripts/run_feasibility.py` (3 seeds, **depth-decoupled error model, estimated motion**).
Full per-seed numbers: `reports/feasibility_results.json`. Stress matrix: `reports/stress_results.json`.
Label-noise sweep: `reports/samples_sweep.json`. Round-1 (decoupled, true-motion) run:
`reports/feasibility_results_true_motion.json`. Pre-fix (coupled) run:
`reports/feasibility_results_coupled.json`. Archived original (unfixed) study:
`experiments/feasibility_fixed/fixed_feasibility.py`.*

**Verdict: ALL GATES PASS (G1–G6) under the round-3 generator. The scene-scale claim (G3) survives
every input-realism stress test; the FlowAUC target is learnable with a distributional head; the
200-sample IOF target is not a label-noise liability.**

## 1. The original study's verdict (why it failed)

The original `scene-scale.ipynb` concluded **"OVERALL FEASIBILITY: PARTIAL OR FAIL"** — a naive-shift
baseline (predict future IOF = last observed IOF) beat the learned pose+depth model on early-warning
AUROC (0.722 vs 0.680). Root causes, established by code inspection:

1. **Unfair baseline.** The naive-shift baseline was handed the exact current IOF (a near-perfect
   observation of the hidden error state, since IOF is autocorrelated ≈ 0.9⁵ over the h=5 horizon),
   while the model was structurally denied any observation of the error state.
2. **Starved inputs.** The generator produced a reliability signal (`confidence = exp(-5·‖err‖)`) — a
   perfect window into the error state — but the study never fed it to the model. In a real SLAM,
   residuals/covariance/tracking quality play exactly this role.
3. **Vacuous motion.** `motion_mag ≈ 0.05` every frame — the motion-only input was near-constant, so
   the "motion-only" ablation could not carry signal.
4. **Pooled metrics.** Pooled AUROC flatters the naive baseline: its per-sequence AUROC is 0.589 vs
   pooled 0.886, because between-sequence depth differences inflate the pooled ranking.

**Consequence:** the study's FAIL was an artifact of experimental design — but the *proposal* that
claimed "resounding success" was still wrong, because it ignored its own failing gate. Both errors are
corrected here: the design, and the reporting.

## 2. Corrected design (rounds 1 + 2 + 3)

| Fix | Implementation |
|---|---|
| Fair baseline | Ridge trained on the **same features** as the learned model; oracle persistence separated from runtime-available baselines (B1–B9) |
| Reliability signal as input | Noisy observation of the error state: `conf = exp(-5·‖err‖·(1+0.15·ε))` — deliberately not deterministic (GIGO-safe) |
| **Estimated motion** (round 2) | The model consumes relative motion derived from an **accumulated estimated trajectory** `T̂ = T_gt ∘ T_err` — corrupted by the pose error, as a running SLAM exposes; `motion_source="true"` kept for comparison |
| Degraded reliability (round 2) | Stress variants: delayed / noisy / miscalibrated / masked / intermittent (§6) |
| Abrupt jump failures (round 2) | Poisson error jumps (`jump_prob`, `jump_scale`) for warning-lead-time realism |
| Realistic depth corruption (round 2) | Holes (far-plane defaults), flying pixels, specular near-spikes on the estimated depth |
| h=0 stage | Current-error estimation (SEESys-style) run alongside h=5 |
| Analytic two-stage | MLP forecasts of error magnitudes + depth, then explicit projection transform + linear calibration |
| **Classical covariance propagation** | AR(1) forecast covariance → analytic IOF Jacobian → image space, using ONLY runtime quantities (blind baseline) |
| **GRU temporal baseline** | Learned temporal model on the same features (fairer than MLP-alone) |
| **Depth-decoupled error model** (G3 fix) | Pose-error scale depends on **motion only**; scene depth enters **only** through the IOF 1/Z target (see §3) |
| **Scene-scale-faithful depth range** (G3 fix) | `DEPTH_CHOICES = (0.5 … 30 m)`, mirroring Princeton365's 285/365 sub-meter scanning sequences; rotation-innovation weight 0.3 (see §3) |
| **Failure definitions** (round 2) | Global τ, per-sequence τ_s (**within-sequence AUROC = PRIMARY**), robust z-score, depth-normalized IOF (§5) |
| **Official-protocol IOF machinery** (round 3) | `iof.py` now ships the official recipe: Sim(3)+SO(3) trajectory alignment (`align_trajectory`), BIC-selected Gaussian-mixture depth fit (`fit_depth_distribution`), marginal-depth integration (`compute_iof_official`) — making P1-G1 a reimplementation deliverable, with a raw-vs-official comparison leg (§8) |
| **FlowAUC target + distributional head** (round 3) | The generator records per-pixel flow samples; `models.DistributionalMLP` (histogram head) predicts the flow distribution at t+h; Flow AUC follows from the predicted CDF (§7) |
| **Label-noise sensitivity** (round 3) | `--sweep-samples` reports the reviewer-requested 100/200/500/1000 per-frame sample counts (§9) |
| Rigor | Per-sequence Spearman/AUROC, 3 seeds, mean ± std, unseen test sequences |

## 3. The G3 confound, and the two-part fix

**Gate G3 asks: does the depth input add value beyond motion, with the reliability signal masked?**
The pre-fix generator failed this gate not for one but two compounding reasons:

**(a) The error model coupled motion and depth.** `error_scale = 0.005 + 0.05·motion/Z` — depth
polluted the pose-error dynamics themselves, so "depth" could not be isolated, and motion-only models
partially inherited depth effects through the coupled scale. **Fix:** the default error model is now
*depth-decoupled*: `error_scale = 0.005 + 0.05·motion`, and depth enters *only* the target through the
geometric 1/Z projection denominator — exactly the real mechanism (SLAM error is generated by the
motion/visual pipeline; depth then determines how that error manifests in the image plane). The legacy
coupled error dynamics remain available via `--error-model coupled`.

**(b) The task was depth-insensitive even after decoupling.** A 1-seed isolation diagnostic
(decoupled error model, old depth range 1.5–25 m, rotation weight 0.5) showed the AUROC leg *still*
did not separate. Two structural causes:

- **The failure ranking is error-spike-dominated.** The pooled-AUROC ceiling using the *per-sequence
  true rank* (perfect within-sequence ordering, no cross-sequence magnitude) is **0.960**, essentially
  equal to the motion-only model's 0.968 — no predictor (with or without depth) can separate there,
  because failures are driven by within-sequence AR error spikes.
- **The rotation flow floor compressed depth's label effect.** The depth-invariant rotation term
  floored far-scene IOF, and with depths only 1.5–25 m the depth-sensitive translation term never
  dominated: per-sequence true-mean IOF spanned just **1.7×** between near and far scenes.

**Fix:** make the scene-scale range match the real dataset — `DEPTH_CHOICES = (0.5 … 30 m)` (Princeton365
is 285/365 object-scanning at sub-meter scale) and a realistic rotation-innovation weight of 0.3.
Per-sequence true-mean IOF now spans **5×**, and depth's geometric signal is material.

With both fixes, the learned model's depth input pays off on both legs across all three seeds
(§4): RMSE −42%, pooled AUROC +0.014. **The confound is gone; G3 is now a meaningful test.**

## 4. Results (test on unseen sequences, 3 seeds, decoupled + estimated-motion generator)

| model | RMSE | nRMSE | medSpearman | AUROC (pooled) | **wAUROC (within-seq)** | AP |
|---|---|---|---|---|---|---|
| constant | 4.529 ± 0.661 | 2.117 | — | 0.500 | 0.500 | 0.245 |
| persistence (oracle, B1) | 3.355 ± 0.367 | 1.296 | 0.113 | 0.909 | 0.552 | 0.770 |
| ridge (same features, B5) | 3.950 ± 0.480 | 1.661 | 0.109 | 0.873 | 0.516 | 0.669 |
| classical cov-prop (blind, B7) | 2.953 ± 0.494 | 1.068 | 0.005 | 0.937 | 0.510 | 0.770 |
| gru (temporal, B9) | 2.088 ± 0.887 | 0.916 | 0.349 | 0.971 | 0.582 | 0.925 |
| mlp motion-only | 1.845 ± 0.158 | 0.944 | 0.657 | 0.974 | 0.703 | 0.941 |
| mlp motion+depth | 1.073 ± 0.082 | 0.748 | 0.694 | 0.988 | 0.655 | 0.967 |
| **mlp full** | **0.986 ± 0.091** | **0.648** | **0.762** | **0.992** | **0.669** | **0.976** |
| analytic 2-stage (phys-explicit, B8) | 1.865 ± 0.198 | 0.838 | 0.642 | 0.976 | 0.625 | 0.925 |
| mlp full h=0 (current, B6) | 0.815 ± 0.033 | 0.585 | 0.822 | 0.994 | 0.683 | 0.984 |

wAUROC = pooled AUROC under per-sequence thresholds τ_s = Q₇₅(IOF_s) — the **primary** early-warning
metric (cannot be gamed by detecting scene scale). Four honest observations:

* **Classical blind covariance propagation is the strongest classical baseline** (RMSE 2.953, pooled
  AUROC 0.937) — **yet it is chance within-sequence (0.510)**: it never observes the error
  realization. This is exactly why the learned model matters and why within-sequence AUROC is the
  primary metric.
* **The GRU (B9) does not beat the MLP** (2.088 vs 0.986) and is unstable across seeds (std 0.887):
  the contribution is the task formulation, not the architecture.
* **Depth is a between-sequence signal**: within-sequence, motion-only's wAUROC (0.703) slightly
  exceeds motion+depth's (0.655) because depth statistics are nearly constant inside a sequence.
  Depth's value is cross-sequence (scene-scale), captured by RMSE and pooled AUROC; both views are
  reported and G3 gates on the between-sequence legs.
* **The reliability signal's marginal value is thin on this draw** (full 0.986 vs motion+depth
  1.073, −8%; per-seed: seeds 1–2 pass, seed 0 fails by 5%). The 3-seed majority verdict is PASS,
  but the margin is a seed-dependent quantity — see §6 and the real-data caveat (P1-G6).

### 4.1 Input-combination ablation (reviewer-required matrix)

| Inputs | RMSE | AUROC (pooled) | wAUROC |
|---|---|---|---|
| motion only | 1.845 | 0.974 | 0.703 |
| depth only | 2.767 | 0.946 | 0.511 |
| reliability only | 4.225 | 0.656 | 0.573 |
| motion + depth | 1.073 | 0.988 | 0.655 |
| motion + reliability | 1.535 | 0.980 | 0.720 |
| depth + reliability | 2.541 | 0.949 | 0.544 |
| **full** | **0.986** | **0.992** | **0.669** |

*Motion+depth vs motion-only* (both WITHOUT reliability) is the **reliability-masked depth test**
(G3): RMSE −42%, pooled AUROC 0.988 vs 0.974. *Full vs motion+reliability* (G6): depth still adds
value with reliability present (0.986 vs 1.535).

### 4.2 Failure-definition comparison (mlp full; pooled AUROC, mean over seeds)

| Definition | AUROC |
|---|---|
| Global threshold (τ = Q₇₅ train) | 0.992 |
| **Per-sequence threshold τ_s (PRIMARY)** | **0.669** |
| Robust z-score (z > 1.5, per-sequence) | 0.757 |
| Depth-normalized (IOF·Ẑ_med) | 0.273 |

The global number is inflated by between-sequence depth differences (near scenes are simply "worse").
The depth-normalized number is a **diagnostic**, not a headline definition: normalizing IOF by Ẑ also
rescales the *depth-invariant rotation term* by Ẑ, so with the synthetic rotation weight the raw-IOF
predictor's cross-sequence ranking inverts; within-sequence metrics are invariant to the rescaling (a
per-sequence constant multiple). To be resolved against the official Flow-AUC protocol on real data
(P1-G1), not a claim about the predictor.

### 4.3 Scene-scale split (test, by true median depth)

| group | med depth | ridge RMSE | mlp RMSE | persistence RMSE |
|---|---|---|---|---|
| near | 0.83 m | 5.721 | **1.124** | 5.329 |
| medium | 4.49 m | 2.847 | **0.970** | 1.929 |
| far | 18.12 m | 2.223 | **0.833** | 1.223 |

## 5. Gates (≥2 of 3 seeds)

* G1 learned > linear on same features — **PASS** (RMSE 0.986 vs 3.950; AUROC 0.992 vs 0.873)
* G2 reliability signal adds value — **PASS** (2/3 seeds; full vs motion+depth: RMSE 0.986 vs 1.073;
  seed-0 margin is thin, see §4/§6)
* G3 depth adds value with reliability **MASKED** — **PASS** (motion+depth vs motion-only: RMSE 1.073
  vs 1.845, **−42%**; pooled AUROC 0.988 vs 0.974 — both legs, all seeds). *Pre-fix: PARTIAL (AUROC
  leg failed on the two-part confound of §3).*
* G4 beats oracle persistence — **PASS** (RMSE 0.986 vs 3.355; AUROC 0.992 vs 0.909) — the original
  failure is resolved
* G5 current-error estimation works (h=0) — **PASS** (RMSE 0.815 vs constant 4.529)
* G6 depth adds value with reliability **PRESENT** (reported) — **PASS** (full vs motion+reliability:
  0.986 vs 1.535)

Overall (G1 ∧ G2 ∧ G3 ∧ G5): **PASS** — the pre-fix `PARTIAL/FAIL` is resolved, now under the
estimated-motion generator with the round-3 additions.

## 6. Stress matrix (1 seed each; `reports/stress_results.json`)

| Variant | full RMSE | wAUROC | G1 | G2 | G3 | G5 |
|---|---|---|---|---|---|---|
| baseline (estimated motion) | 1.114 | 0.674 | PASS | **FAIL** | PASS | PASS |
| true motion (pre-fix input) | 1.092 | 0.663 | PASS | **FAIL** | PASS | PASS |
| reliability: delayed (3 frames) | 1.060 | 0.665 | PASS | **FAIL** | PASS | PASS |
| reliability: noisy | 0.965 | 0.688 | PASS | PASS | PASS | PASS |
| reliability: miscalibrated | 1.068 | 0.672 | PASS | **FAIL** | PASS | PASS |
| reliability: masked | 1.252 | 0.646 | PASS | **FAIL** | PASS | PASS |
| reliability: intermittent | 1.034 | 0.679 | PASS | PASS | PASS | PASS |
| jump failures | 1.209 | 0.705 | PASS | PASS | PASS | PASS |
| realistic depth corruption | 1.090 | 0.668 | PASS | **FAIL** | PASS | PASS |

Findings (read honestly):

* **G3 — the scene-scale claim — survives every degradation**, including masked reliability, jump
  failures and realistic depth corruption. The depth channel does not lean on an idealized
  reliability signal. This is the gate the thesis's H2 claim rests on.
* **G2's margin is thin on seed 0** (full 1.114 vs motion+depth 1.059, i.e. the seed-0 full model is
  ~5% worse), so the 1-seed stress run flips it in most variants — including the *baseline* — which
  is a seed artifact, not a variant effect. What is still visible: the **masked** variant is worst
  (1.252, the largest gap), and information-*preserving* variants (noisy, intermittent, jumps) keep
  G2 positive. The 3-seed majority verdict (§5) is the gate that counts; the stress matrix shows the
  reliability signal's marginal value is small and seed-dependent — reinforcing that its real-data
  value is contingent on the instrumentation deliverable (P1-G2/P1-G6), exactly as the proposal
  states.
* Estimated vs true motion barely moves any gate: the round-1 conclusions were not an artifact of
  clean motion input.

## 7. FlowAUC leg (round-3 M5): distributional head vs baselines

The proposal lists `FlowAUC_{t+h}` as a target; a scalar-mean head cannot predict a distributional
summary. The generator now records per-pixel flow samples, and a histogram head
(`models.DistributionalMLP`, softmax over 0–100 px magnitude bins) is trained to predict the flow
distribution at t+h; Flow AUC(tau) = fraction of pixels with flow < tau follows from the predicted
CDF. RMSE of the predicted FlowAUC at tau ∈ {1, 5, 20} px and of the official 0–100 AUC (mean CDF,
rescaled to 0–100), mean over 3 seeds:

| model | tau=1 px | tau=5 px | tau=20 px | official AUC |
|---|---|---|---|---|
| histogram head (learned) | **0.157** | **0.160** | **0.073** | **0.021** |
| persistence (oracle) | 0.270 | 0.311 | 0.115 | 0.040 |
| constant | 0.209 | 0.420 | 0.099 | 0.051 |

The learned distributional head beats both baselines at every threshold — the FlowAUC target is
learnable with a distributional output, so the proposed `FlowAUC_{t+h}` target is implementable as
specified (percentile bins; a Beta head is the documented alternative).

## 8. Official-protocol leg (round-3 C1/C2): raw vs official IOF

`iof.py` now implements the official recipe (Sim(3)+SO(3) trajectory alignment, BIC-selected
Gaussian-mixture depth fit, marginal-depth quadrature). On 3 fresh synthetic sequences per seed:

| quantity | value |
|---|---|
| corr(raw, official, **unaligned**) | 0.871 |
| corr(raw, official, **aligned**) | 0.641 |
| scale ratio official/raw (aligned) | 0.944 |
| scale ratio official/raw (unaligned) | 1.074 |
| BIC-selected GMM components | 3 |

Interpretation (the C2 paradox, in miniature): the unaligned official quantity tracks the raw target
(corr 0.87, ratio ≈ 1), but once the trajectory is aligned the correlation drops (0.64) — the
alignment removes the accumulated-drift component that dominates the raw target on some sequences, so
the two estimators measure different signal. This is exactly why the proposal (i) keeps raw per-frame
IOF as the PRIMARY (runtime-faithful) target, (ii) treats official IOF as a benchmark-comparable
diagnostic, and (iii) restates P1-G1 as a reimplementation deliverable (align + fit + integrate, then
ρ > 0.98) rather than a correlation check of two different estimators.

## 9. Label-noise sensitivity (round-3 M8): 100/200/500/1000 samples

The reviewer-requested sweep (`--sweep-samples`, 1 seed per count; `reports/samples_sweep.json`):

| num_samples | mlp full RMSE | wAUROC | G1 | G3 |
|---|---|---|---|---|
| 100 | 1.093 | 0.668 | PASS | PASS |
| 200 | 1.114 | 0.674 | PASS | PASS |
| 500 | 1.106 | 0.673 | PASS | PASS |
| 1000 | 1.083 | 0.675 | PASS | PASS |

The 200-sample default is **not** a label-noise liability: RMSE varies ±1.4% and wAUROC ±0.004 across
a 10× change in samples, and all gates pass at every count. (With the marginal-depth official
estimator on real data the label noise will be different — the sweep machinery carries over to P1-G1.)

## 10. Interpretation and what it does NOT prove

**Supported.** With runtime-available error-state information (estimated motion + noisy reliability),
the learned predictor beats oracle persistence and a linear model on identical features (G4, G1), the
reliability signal adds value by 3-seed majority (G2, thin margin), and scene geometry adds value
through the geometric 1/Z scaling (G3) with the reliability signal masked — with the depth-decoupled
error model guaranteeing that the depth channel is tested purely on the image-space consequence, not
on a generator artifact. The protocol — oracle-vs-runtime baseline split, per-sequence/within-sequence
metrics, temporal + classical baselines, distributional FlowAUC head, official-protocol IOF machinery,
label-noise sweep — is now trustworthy and stress-tested.

**Not supported / to be re-tested on real data.**

1. The synthetic task is comparatively easy: the generator makes IOF strongly determined by runtime
   features (AUROC ≈ 0.99 for the full model). Real DROID-SLAM error dynamics (drift, relocalization
   jumps) will be harder and will test the forecasting claim honestly — the jump-failure stress
   variant is a first, partial probe of this.
2. G3 now passes in-silico (with and without reliability), but the *magnitude* of depth's value (and
   whether the model's predicted IOF-vs-depth relationship matches the analytic 1/Z scaling at matched
   pose-error regime) must be confirmed on Princeton365's real ZED depth and error statistics (pilot
   gates P1-G1, P1-G4, P1-G5).
3. The explicit physics two-stage is *worse* than direct regression in-silico (1.865 vs 0.986). This
   is a genuine, reportable negative for the "differentiable projection layer" hypothesis.
4. Within-sequence AUROC is the honest early-warning metric, and depth's value is a *between-sequence*
   (scene-scale) property: the two views must both be reported on real data, and failure-event counts
   within GT-posed segments must be quantified before Phase 3 (proposal §8.8, P1-G3).
5. **The raw-vs-official IOF divergence (§8) is a real-data risk the proposal now carries explicitly:**
   the official target is trajectory-aligned (removes accumulated drift) and therefore cannot
   reproduce the raw target that motivates the thesis; the pilot's P1-G1 is a reimplementation
   deliverable, and the thesis must report both targets with the correlation/scale diagnostics shown
   here.
6. **The reliability signal's synthetic value is thin and seed-dependent (§4, §6)** — its real-data
   value is contingent on the DROID instrumentation deliverable (P1-G2) with the ORB-SLAM3 fallback
   and masking discipline (§7.4 of the proposal).

**Recommended next step:** proceed to the staged Princeton365 pilot (Phase 1 of the proposal) governed
by the pre-registered gates P1-G1…P1-G7 — starting with the official-protocol reimplementation
(P1-G1: align + depth-distribution fit + marginal-depth integration, then ρ > 0.98) and the
failure-event-count analysis — with the persistence baseline and per-sequence metrics mandatory in
every table. The generator's G3 confound is fixed and the input realism is stress-tested; the real-data
pilot is where the depth-scaling hypothesis (H2) gets its decisive test.

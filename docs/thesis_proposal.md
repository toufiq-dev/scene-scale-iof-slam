# Scene-Scale-Aware Prediction of Induced Optical-Flow Error for Visual SLAM

**Thesis proposal (revised)**

*Working title: "Beyond Pose Error: Predicting Scene-Conditioned Visual Degradation for
Proactive SLAM Failure Prevention"*
Author: *(to be filled)*
Institution: *(to be filled)* — Date: August 2026 — Version: 2.0 (post-review)

---

## Abstract

Visual Simultaneous Localization and Mapping (SLAM) is conventionally evaluated with geometric
trajectory metrics such as Absolute Trajectory Error (ATE), which do not describe the visual
consequence of a pose error: a 10 cm translation can displace hundreds of pixels when scanning a
nearby object but less than a pixel when observing a distant scene. Princeton365 (ICCV 2025) addresses
this with a scene-scale-aware metric — Induced Optical Flow (IOF) and its Flow AUC variant — that
converts pose error into the image-plane displacement it induces, given the scene's depth.

This thesis asks the next question: **can the scene-conditioned visual consequence of SLAM uncertainty
be predicted online, ahead of time, using only information a running SLAM system possesses?** We propose
a lightweight, external "reliability layer" that maps recent RGB frames, estimated camera motion,
estimated scene geometry, and SLAM-internal reliability signals to a forecast of future IOF / Flow AUC
degradation, several frames ahead of the current one. The work repurposes Princeton365's evaluation
quantity (IOF) into a supervised training target — a task for which, to the best of our knowledge, no
published work exists as of August 2026 — and is validated on the mm-accurate, scene-scale-diverse
Princeton365 dataset, with per-sequence metrics, mandatory persistence and classical error-propagation
baselines, and an explicit oracle-vs-runtime-available split.

A corrected synthetic feasibility study (this repository) validates the experimental protocol: when the
predictor receives the same runtime-available error-state information a real SLAM exposes (estimated
motion + noisy reliability signal), the learned model beats both the oracle-persistence baseline (pooled
AUROC 0.992 vs 0.886) and a linear model on identical features (RMSE 0.948 vs 3.626), with the
reliability signal and scene geometry each adding measurable value — including when the reliability
signal is masked (the G3 gate). All five feasibility gates pass under the new, more realistic generator
(estimated motion, depth-decoupled error model), and a stress matrix confirms the scene-scale claim
survives degraded reliability, jump failures, and realistic depth corruption. The thesis proceeds to a
staged real-data validation on Princeton365 with DROID-SLAM as the primary backbone, culminating in an
adaptive-SLAM demonstration in which predicted degradation triggers relocalization or stronger
optimization.

---

## 1. Introduction and Motivation

Visual SLAM estimates a camera's 6-DoF trajectory and a 3D scene representation from sequential images.
For decades, SLAM systems have been ranked by geometric trajectory accuracy: ATE (mean positional error
after trajectory alignment) and rotation error. These metrics are necessary but insufficient. A fixed
geometric error produces wildly different image-space effects depending on scene scale and depth:

* A 10 cm translation error while scanning an object 30 cm away induces on the order of hundreds of
  pixels of image misalignment (with typical wide-FOV intrinsics, ≈ 300 px at f ≈ 1000).
* The same 10 cm error observing a building 50 m away displaces the image by ≈ 2 px.

Pose error therefore cannot be equated with its practical, visual consequence — the quantity that
matters for augmented-reality registration, visual servoing, obstacle avoidance, and any task that
relies on image alignment. Princeton365 (Kayan et al., ICCV 2025) was introduced precisely to close this
gap, proposing **Induced Optical Flow (IOF)**: the expected per-pixel displacement obtained by
transforming scene points according to the relative pose error and reprojecting them, and its accuracy
variant **Flow AUC** (0–100), which measures the percentage of pixels whose induced flow falls under
increasing thresholds. Because IOF is computed with a depth distribution, it is inherently
scene-scale-aware and enables comparison of SLAM performance across trajectories of different scales —
something ATE cannot do.

The published Princeton365 leaderboard illustrates the disconnect between the two metric families. For
example (Table 6 of the paper):

* **DPVO** achieves ATE 0.01 m on easy scanning sequences (Flow AUC 84.97) yet collapses to **Flow AUC
  0.01 on hard indoor sequences** despite an ATE of only 0.44 m.
* **ORB-SLAM3** reports a moderate ATE of 1.04 m on medium indoor sequences while its Flow AUC falls to
  0.03 (coverage 17.8%), i.e., the trajectory error looks acceptable while the visual consequence is
  catastrophic.
* Across hard indoor/outdoor sequences, every evaluated method except ORB-SLAM3's outdoor regime drops
  below Flow AUC 1.0 while ATE stays between 0.4 and 1.6 m.

The message is unambiguous: ATE is an incomplete characterization of a SLAM system's practical utility,
and failure modes are driven by the *interaction* of pose error with scene geometry, texture, and motion
dynamics. This motivates the research question at the heart of this thesis:

> **Can we predict the scene-conditioned visual consequence of SLAM uncertainty — the future IOF — in
> real time, before it becomes catastrophic, using only the information a running SLAM system has?**

## 2. Background

### 2.1 Visual SLAM

Visual SLAM takes image sequences from monocular, stereo, or RGB-D cameras and jointly estimates camera
poses {T̂_t} ⊂ SE(3) and a scene representation (point cloud, depth, or neural field). Modern systems
combine learned representations with geometric optimization. This thesis uses **DROID-SLAM** (Teed &
Deng, NeurIPS 2021) as the primary backbone — a recurrent dense-bundle-adjustment system supporting
monocular/stereo/RGB-D — with **ORB-SLAM3** (Campos et al., T-RO 2021) and **MASt3R-SLAM** (Murai et al.,
CVPR 2025) as cross-backbone generalization targets.

### 2.2 Evaluation metrics and their limits

| Metric | Definition | Limitation |
|---|---|---|
| ATE | Mean positional error after Sim(3) alignment | Scale-blind; identical error values across scenes with very different visual impact |
| Rotation error | Angular discrepancy | Same limitation |
| IOF | E[‖flow(t,d,u,v)‖₂] over frames, depth distribution, pixels | Requires GT pose + depth; not comparable to ATE |
| Flow AUC | AUC of % pixels with induced flow < thresholds (0–100) | Interpretable, scale-aware accuracy score |
| Composite | Combines Flow AUC and tracking coverage | Used by Princeton365 for ranking |

The central observation: **visual consequence = f(pose error, scene geometry, camera intrinsics)**. The
same pose error is benign or catastrophic depending on depth.

### 2.3 Princeton365

Princeton365 is a dataset of 365 videos (285 object-scanning, 40 indoor, 40 outdoor; ≈ 2.13M frames,
≈ 10 h) with **mm-accurate ground-truth pose** (validated against MoCap: avg ATE 2.88 mm vs 11.4 mm for
COLMAP) and full 6-DoF motion. It provides monocular and stereo RGB plus IMU, and per-pixel depth from a
ZED X stereo camera used for the official IOF computation. Critically for this thesis, **only 56.1% of
frames have ground-truth pose** (scanning 98.5%, indoor 42.0%, outdoor 18.8%), and indoor/outdoor
sequences carry GT only near their beginnings and ends. This coverage structure constrains where
"warning before failure" can be measured (see §8.7, §9).

## 3. Problem Statement and Research Questions

Let T̂_t be the estimated pose, T^gt_t the ground-truth pose (unavailable at inference), D_t the scene
depth, and K the intrinsics. The visual consequence is IOF_t = g(T̂_t, T^gt_t, D_t, K). At runtime, T^gt_t
is unknown, so we ask whether

**Pose convention (explicit).** T̂_t and T^gt_t are world-to-camera maps, T̂_t = T̂^{wc}_t and T^gt_t =
T^{wc,gt}_t, matching the target-generation code (`iof.py`) and the Princeton365 definition. A scene
point P_c in the ground-truth camera frame is mapped into the estimated camera frame by

    T_rel = (T̂_t)^{-1} T^gt_t,

and IOF is the expected reprojection displacement induced by T_rel. Equivalently, T_rel is the SE(3)
correction that carries the GT trajectory onto the estimated trajectory, expressed in the estimated
frame — the pose error *as seen by the camera*, which is what determines the visual consequence. The
convention is pinned by a unit test (`relative_error_pose_inverse_swap`: swapping est/gt inverts T_rel;
flow magnitudes are equal for pure rotation and near-equal for small errors) and will be validated
against the official Princeton365 IOF implementation in pilot gate P1-G1 (ρ > 0.98, §8.8).

> E_visual = IOF_{t+h} (or its per-pixel distribution, or a failure probability) can be predicted from
> X_t = (I_{t−k:t}, T̂_{t−k:t}, D̂_t, R_t, K),

where I is recent RGB, T̂ the estimated trajectory (as relative transformations), D̂ the estimated depth
(summarized by scene-scale statistics), R_t SLAM-internal reliability signals, and K intrinsics.

**Research questions**

* RQ1 — Can future induced optical-flow error be predicted from SLAM-available information?
* RQ2 — Does scene geometry (depth statistics) improve prediction beyond image/motion-only models?
* RQ3 — Does predicting induced *visual* error provide information that conventional pose-error
  prediction does not?
* RQ4 — Can the predictor generalize across scene categories (scanning/indoor/outdoor) and across SLAM
  backbones (DROID → ORB-SLAM3 / MASt3R-SLAM)?
* RQ5 — Can the predicted signal identify impending visual degradation ahead of time (warning lead time),
  with calibrated probabilities?
* RQ6 — Can the predicted signal improve SLAM robustness through adaptive computation (relocalization /
  stronger optimization)?

## 4. Related Work

A systematic review (performed August 2026; see `docs/related_work.md` for the annotated bibliography)
identifies the following clusters. Each is close to — but distinct from — this thesis.

1. **Uncertainty-aware VO/SLAM.** Learning-based VO with uncertainty outputs: Costante & Mancini
   (T-RO 2020), MDN-VO (IROS 2021), D3VO (CVPR 2020), TartanVO (CoRL 2020), and dense neural SLAM with
   learned uncertainty: UncLe-SLAM (ICCVW 2023), Uni-SLAM, conformalized VO (Stutts et al., 2023).
   These quantify *pose* uncertainty, not its scene-conditioned *visual consequence*, and none predicts
   h-steps ahead.
2. **Introspective perception / perception-performance prediction.** Daftry et al. (IROS 2016) learn a
   CNN to predict failure probability of vision systems from images; the Oxford line (Gurau, Rao, Tong,
   Posner — IJRR 2018; Dequaire et al., ICRA 2016; "Know Your Limits") predicts localisation performance
   and its evolution. These predict *current or near-future success of a fixed task*, not image-space
   consequence, and mostly target teach-and-repeat / MAV navigation rather than SLAM error dynamics.
3. **SLAM tracking-failure avoidance.** Introspective-SLAM (Naveed et al., Autonomous Robots 2022) and
   Deeper Introspective SLAM (Naveed et al., IEEE RA-L 2024) use RL / video transformers to avoid
   tracking failure in ORB-SLAM; "Learning to Prevent Monocular SLAM Failure using RL" (arXiv 2016).
   These act on the *camera trajectory* (active view planning) rather than emitting a continuous
   image-space reliability signal for a fixed trajectory.
4. **Online pose-error and runtime-risk estimation.** **SEESys** (Hu et al., SenSys 2024) is the first
   online deep-learning pose-error estimator for SLAM; **SUPER** (Gaus et al., arXiv:2512.14189; ICRA
   2026) provides sensitivity-based runtime warning signals for VIO; **VERF** (Maggio et al., 2023)
   performs runtime assurance of pose estimation with NeRFs. These are the closest competitors: they
   predict *current* error or risk, in pose or generic risk terms, not future scene-scale-aware
   image-space consequence, and none uses IOF.
5. **Degeneracy / uninformative-geometry detection.** Degeneracy-aware LiDAR SLAM (Zhang & Singh 2016;
   probabilistic degeneracy detection 2024; DALI-SLAM) and visual degeneracy detection detect
   uninformative geometry causing estimator collapse — a geometric precursor of scene-conditioned
   degradation, but reactive and pose-domain.
6. **Classical uncertainty propagation.** EKF/CRLB/Fisher-information prediction of *future* pose
   covariance (e.g., "Visual Active SLAM Method Considering Measurement and State Uncertainty," 2025;
   "Principled Uncertainty Propagation for Stereo VO," 2026) is the classical analogue of forecasting
   future error. It predicts covariance under a motion model, not realized error of a learned system,
   and requires explicit models.

**Generative / foundation-model geometry.** VGGT (CVPR 2025), MASt3R-SLAM (CVPR 2025), and DUSt3R-family
models predict dense geometry from images. This thesis treats them as *optional frozen feature
extractors* for the depth-statistics input — a scoped experiment (§8.5), not the core contribution.

## 5. Research Gap and Novelty Claim

Based on the August 2026 systematic search, **to the best of our knowledge, no published work** (as of
August 2026):

1. uses Princeton365's IOF / Flow AUC as a supervised *learning target*;
2. predicts *future* (h-step) image-space visual degradation of SLAM;
3. does so with inference restricted to SLAM-available quantities (no GT at inference);
4. compares directly against classical error propagation and persistence baselines for this task; and
5. demonstrates the downstream use of such a signal to trigger adaptive SLAM behavior.

The narrow claim (1)–(3) is the defensible core of the thesis. Two honesty notes, stated explicitly:

* **The gap is time-limited.** Princeton365 was released at ICCV 2025 (arXiv v2 June 2026); the field has
  had months, not years. A final systematic literature search must be re-run immediately before
  submission — this is a stated milestone (§12), not an assumption.
* **The broader framing is crowded.** "Predicting SLAM uncertainty" is *not* novel (SEESys, SUPER,
  introspective perception, uncertainty-aware VO all exist). The contribution is specifically: *future,
  scene-scale-aware, image-space* consequence prediction with SLAM-available inputs, benchmarked against
  the correct baselines (persistence, error propagation, linear-on-features).

## 6. Hypotheses

* **H1 (Predictability):** A learned predictor of future IOF beats a non-learned baseline (constant and
  persistence) on unseen sequences, on RMSE and rank metrics.
* **H2 (Geometry):** Including scene-geometry statistics improves visual-error prediction over
  image/motion-only models.
* **H3 (Visual vs pose):** Predicting induced visual error characterizes future image-space degradation
  better than predicting pose error alone, when both are evaluated in image space (i.e., pose-error
  predictions transformed through the IOF equation).
* **H4 (Generalization):** Models trained across scene categories retain predictive performance on
  unseen categories and unseen backbones.
* **H5 (Utility):** Using the predicted signal to trigger adaptive SLAM computation improves the
  Flow AUC / coverage trade-off without prohibitive cost.

## 7. Proposed Method

### 7.1 Inputs (runtime-available only)

| Input | Representation | Role |
|---|---|---|
| Recent RGB | I_{t−k:t}, stacked frames | Motion blur, texture, dynamic objects, illumination |
| Estimated motion | relative transforms ΔT̂_t (or 6-DoF magnitudes) | Motion regime (translation vs rotation, speed) |
| Estimated geometry | depth statistics: median, q25, q50, q75, q90, std, 1/median (optionally a downsampled depth map) | Scene scale — the core hypothesis |
| Reliability signals | tracked correspondences, reprojection residuals, tracking confidence, BA residual | Error-state observation (like SEESys's inputs) |

The feasibility study (§10) established that the reliability signal — a *noisy observation of the
error state* — is the single most informative input, and that omitting it (as the original feasibility
study did) makes the comparison against persistence structurally unfair.

### 7.2 Targets (definition decision)

The thesis will compute **two target variants** and report both, to remain comparable with the official
leaderboard while supporting the model's geometry-conditioned design:

1. **Official variant** — IOF / Flow AUC computed with the *ZED depth distribution* as in Princeton365
   (required for leaderboard comparability).
2. **Model variant** — IOF computed with the *SLAM system's estimated depth* D̂_t (the quantity available
   at inference); used to check internal consistency of the geometry-conditioned claim.

Future targets: y_t = IOF_{t+h} (h ∈ {5, 10, 20, 30}), optionally per-pixel flow quantiles (q50, q90,
q95) to avoid scalar-mean masking of local misalignment, Flow AUC_{t+h}, and a binary failure label
F_{t+h} = 1[IOF_{t+h} > τ], τ from the training distribution.

### 7.3 Architecture

A deliberately lightweight model (< 25 k parameters target; feasibility-stage baseline is a 2×128 MLP):

```
RGB history ──► CNN / small ViT encoder ─┐
Pose history ──► GRU / MLP ──────────────┼──► Fusion MLP ──► heads
Depth stats ──► MLP ─────────────────────┤      │
Reliability ──► MLP ─────────────────────┘   IOF_{t+h} / FlowAUC_{t+h} / P_fail
```

Multi-task losses: Smooth-L1 (IOF, Flow AUC) + BCE (failure), with a calibrated failure probability
(ECE evaluated, §8.7). An optional **differentiable geometric projection layer** (hypothesize pose
perturbation, project through estimated depth, regress the residual) is an architectural variant —
the synthetic study shows this explicit two-stage is *worse* than direct regression in-silico
(§10), so it is treated as a hypothesis to test on real data, not a default.

### 7.4 Backbone instrumentation (DROID-SLAM, with ORB-SLAM3 fallback)

DROID-SLAM exposes estimated poses and dense depth but not off-the-shelf tracking statistics — its
recurrent update is not designed as a modular feature extractor (reviewer-confirmed engineering risk).
The instrumentation plan, staged as a Phase-1 milestone with an explicit fallback:

1. **Fork-and-instrument the recurrent update** (a narrow, stable patch): capture per-frame dense flow
   residuals and bundle-adjustment residuals from the update iterations, the number of active
   keyframes/factors, and optimization convergence metrics (iteration counts, residual norms).
   Validate on a handful of sequences against manually induced failures (blur, textureless walls)
   before scaling to the pilot subset.
2. **Fallback — ORB-SLAM3 as an instrumentable backbone.** ORB-SLAM3 natively exposes tracking-quality
   signals (tracked feature count, median reprojection error, number of matches, tracking state
   transitions) with no forking. If DROID instrumentation proves fragile or unstable across sequence
   lengths, the primary backbone switches to ORB-SLAM3 (stereo/RGB-D, consistent depth source) and
   DROID-SLAM becomes the generalization target instead. The decision is made at the end of Phase 1
   against the P1-G2 gate (§8.8).
3. **Signal-masking discipline regardless of backbone.** If a reliability signal cannot be faithfully
   obtained, the model is trained with the signal *masked* and evaluated both ways — an explicit
   ablation (§8.2), because a reliability head that depends on unavailable signals would not
   generalize. The synthetic stress matrix (§10.4) already demonstrates what happens when the signal
   is absent (G2 fails by construction; G3 — the depth claim — survives).

## 8. Experimental Plan

### 8.1 Mandatory baselines (the table that makes or breaks the claims)

The baselines are split into **oracle (diagnostic)** and **runtime-available** families. Oracle baselines
use quantities a deployed system cannot observe (true IOF, GT pose); they are upper bounds and
calibration references, not competitors. Runtime-available baselines are the practical bar: a learned
model must beat them to claim deployable value. The synthetic study implements B1–B9 (see §10); the
persistence and error-propagation rows must appear in every results table.

| # | Baseline | Family | Purpose |
|---|---|---|---|
| B1 | **Oracle IOF persistence** (IOF_t → IOF_{t+h}) | Oracle | Strongest simple forecast; the bar to clear (uses true IOF at t, which needs GT pose) |
| B2 | Oracle pose-error persistence → IOF | Oracle | Isolates forecasting from estimation in pose terms |
| B3 | Oracle current IOF (GT pose error → IOF) | Oracle | Upper bound / calibration reference |
| B4 | Constant (train mean) | Runtime | Floor |
| B5 | **Linear (ridge) on the same features as the model** | Runtime | Isolates the value of nonlinear learning |
| B6 | SEESys-style current-error estimator (h = 0) + persistence of its output | Runtime | Separates estimation from forecasting; the runtime-available persistence proxy |
| B7 | **Classical covariance propagation** (EKF/AR pose covariance → image space via the IOF Jacobian) | Runtime | The classical baseline; directly tests H3 |
| B8 | **Analytic two-stage**: forecast pose error & depth, apply IOF formula | Runtime | Physics-explicit decomposition |
| B9 | **GRU temporal baseline** on the same features | Runtime | Fairer learned baseline (a sliding-window MLP alone could understate the temporal signal) |

**H3 protocol:** the pose-error predictor's output must be *transformed through the IOF equation* before
comparison — comparing pose error in meters against IOF in pixels is invalid. The pose-error baseline
must predict enough structure to approximate IOF (at minimum the 6-DoF relative pose error, ideally a
distribution p(ΔT) propagated through the projection with estimated depth), not merely a scalar ATE
magnitude. This is the single most important methodological fix relative to naive formulations.

### 8.2 Ablations

* **Full input-combination matrix** — all 7 combinations of {motion, depth, reliability}, each vs. ridge
  on the same features (the reviewer-required table):

| Inputs | Question |
|---|---|
| Motion only | Motion-regime baseline |
| Depth only | Does scene scale alone carry signal? |
| Reliability only | Is the error-state observation sufficient alone? |
| Motion + Depth | **Depth value with reliability MASKED** (the G3 test) |
| Motion + Reliability | Depth value with reliability present (G6, reported) |
| Depth + Reliability | Geometry + error-state without motion |
| Full (M + D + R) | The model |

* Target: scalar IOF vs per-pixel quantiles vs Flow AUC.
* Horizon: h = 0 (current) vs h ∈ {5, 10, 20, 30}.
* Reliability degradation (generator stress): delayed / noisy / miscalibrated / masked / intermittent
  reliability, jump-failure error processes, realistic depth corruption — the synthetic claim must
  survive at least some of these before the real-data pilot is justified (§10.4).
* Frozen VGGT/DUSt3R depth features vs raw depth statistics (scoped experiment).

**Reliability-masked depth test.** G3 is deliberately defined with the reliability signal excluded from
both compared models (motion+depth vs motion-only): it asks whether *depth alone* carries the
scene-scale claim. G6 (full vs motion+reliability) asks the complementary question with reliability
present. The stress matrix (§10.4) re-runs both under every reliability degradation.

### 8.3 Decomposition: estimation vs forecasting vs utility

Report three layers separately: (i) current-error estimation quality (h = 0), (ii) h-step forecasting
quality, (iii) downstream utility. A model that only matches persistence at h = 5 is an *estimator*, not
a *forecaster*; the thesis's contribution claim depends on (ii) and (iii).

### 8.4 Scene-scale evaluation

Partition test sequences by median depth (near / medium / far terciles) and report per-group error and
rank metrics. **Refined test:** the naive "does the model output differ for near vs far" check is
trivially passed by any model that reads its depth input; the meaningful test is whether the model's
predicted IOF-vs-depth relationship matches the analytic 1/Z scaling *given the same pose-error regime*,
i.e., whether depth improves prediction *beyond* what pose + persistence alone provide (this is H2 and
the G3 gate of the synthetic study).

### 8.5 Generalization

* Cross-category: train on scanning + indoor, test on outdoor (and all permutations).
* Cross-backbone: train on DROID-SLAM outputs, test on ORB-SLAM3 and MASt3R-SLAM outputs of the same
  sequences — the strongest test of the "learned physics of degradation" claim.

### 8.6 Adaptive SLAM demonstration

When predicted risk crosses a threshold, trigger a predefined intervention (relocalization, tighter
optimization, descriptor switch) with cost accounting. **Control conditions:** (A) baseline SLAM;
(B) SLAM + pose-error predictor; (C) SLAM + visual-error predictor; (D) C + adaptive response; and an
**oracle-triggered upper bound** (GT-triggered interventions) to bound achievable gain and detect
evaluation circularity (the intervention improving pose is not evidence the *predictor* is good).

### 8.8 Phase-1 real-data pilot gates (go/no-go), with pre-registered stop criterion

The synthetic pass justifies proceeding to the small real-data pilot (§9) — nothing more. The pilot
(10–20 sequences, cached SLAM outputs) is governed by the following pre-registered gates, each with a
binary decision:

| Gate | Requirement | Decision if FAIL |
|---|---|---|
| P1-G1 | **Official IOF reproduction:** our IOF/Flow AUC implementation vs the official Princeton365 code on one sequence: per-frame correlation ρ > 0.98 | Fix implementation; do not proceed until reproduced |
| P1-G2 | **Cached SLAM outputs + reliability signals** for 10–20 sequences (DROID instrumented, or ORB-SLAM3 fallback per §7.4) | Switch backbone / fix instrumentation |
| P1-G3 | **Failure-event counts within GT-posed segments**, per category (see below): enough positives for early-warning AUROC | Redesign evaluation (scanning-first; pseudo-GT auxiliary) |
| P1-G4 | **Depth improves motion-only** under sequence-level metrics with reliability masked (H2, the G3 analogue) | Pause; re-examine the scene-scale claim |
| P1-G5 | **Full model beats the runtime-available persistence proxy** (B6) by ≥ 0.05 within-sequence AUROC on the pilot subset — the **pre-registered stop criterion** | Pause and re-evaluate the thesis framing (negative result is still publishable) |
| P1-G6 | **Failure AUROC remains meaningful with reliability masked** (the model does not depend on unavailable signals) | Drop the reliability input; document |
| P1-G7 | **ECE and warning lead time stable across seeds** on the pilot subset | Revisit calibration/targets before Phase 2 |

**Pre-registered stop criterion (explicit):** if, on the 10–20 sequence pilot, the full model does not
beat the runtime-available persistence proxy (B6) by at least **0.05 within-sequence AUROC**, the
project pauses and re-evaluates — the forecasting claim would not be supported, and the thesis would
re-scope to a negative/descriptive result rather than quietly proceeding.

**Failure-event counts (required before Phase 3):** report, per scene category (scanning/indoor/
outdoor), the number of high-IOF failure events that fall *within GT-posed segments* (recall: only 56.1%
of frames are posed; indoor 42.0%, outdoor 18.8%, and GT sits at sequence ends). If failures occur
primarily in unposed segments — plausible, since tracking is hardest where GT is absent — the counts
will be small and the evaluation must be redesigned (scanning-first, or pseudo-GT for auxiliary
analysis). This quantification is a Phase-1 deliverable, not an afterthought.

### 8.7 Metrics and statistical rigor

* Regression: RMSE/MAE and **per-sequence normalized RMSE** (RMSE_s / std_s, reported as the median
  over sequences); **per-sequence** Spearman (pooled rank metrics are inflated by between-sequence
  depth differences — demonstrated in §10).
* **Failure definitions (no scene-scale leakage).** A single global threshold makes failure mean "is
  this a near scene?", because IOF scales with 1/Z. Four definitions are computed and reported:
  1. *Global threshold* τ = Q₇₅(train IOF) — leaderboard-compatible;
  2. *Per-sequence threshold* τ_s = Q₇₅(IOF_s) — failure = degradation relative to the sequence's own
     normal behavior — **the PRIMARY early-warning metric (within-sequence AUROC)**, which cannot be
     gamed by detecting scene scale;
  3. *Robust z-score* z_t = (IOF_t − med_s)/(IQR_s + ε), failure = z_t > z_thr;
  4. *Depth-normalized* IOF·Ẑ_med > τ — removes the 1/Z trend so a global threshold is not a scene
     classifier (note: rescaling also rescales the depth-invariant rotation term by Ẑ; reported with
     this caveat).
  Report all four, plus the number of sequences with valid positive/negative labels and AUROC
  stratified by near/medium/far scene scale. **The primary early-warning metric is within-sequence,
  not pooled.**
* **ECE** for the calibrated failure probability (plus Brier score and a reliability diagram in the
  final study); **warning lead time** in frames with a *fixed* risk threshold chosen on validation,
  a minimum failure duration (short blips are noise), detection rate, false alarms and precision at
  fixed lead time (measured only on GT-posed segments).
* Multiple seeds (≥ 3) with bootstrap CIs; strict sequence-level train/val/test splits; report
  per-category results; no frame-level shuffling across sequences.

## 9. Data, Compute, and Feasibility

* **GT coverage constraint:** warning-lead-time and failure-detection evaluation is only possible where
  GT pose exists (scanning 98.5%, indoor 42.0%, outdoor 18.8%; indoor/outdoor GT at sequence ends).
  The plan restricts these metrics to posed segments and reports the effective evaluation coverage per
  category. Unposed frames can still provide auxiliary self-supervised targets (photometric/flow
  consistency) — a scoped extension.
* **Pipeline:** run a pretrained SLAM system once on a Princeton365 subset, cache poses/depth/statistics
  (the predictor must *not* re-run SLAM during training). Staged: 10–20 sequences, governed by the
  pre-registered Phase-1 gates P1-G1…P1-G7 including official-IOF reproduction (ρ > 0.98) and the
  failure-event-count analysis (§8.8) → 50–80 (learns anything?) → full training split → official
  test sequences.
* **Compute:** consumer GPU / Colab T4 target; frozen pretrained components + small trainable predictor;
  ~$100 budget envelope (see companion cost plan).

## 10. Synthetic Feasibility Study (honest report)

The repository (`scripts/run_feasibility.py`) contains the corrected synthetic study. **The original
study's verdict was PARTIAL/FAIL and is reported as such here**: a persistence baseline (predict future
IOF = last observed IOF) beat the learned model on AUROC (0.722 vs 0.680), because (a) the model was
starved of the reliability signal while the baseline was given perfect error-state information, and
(b) motion input was near-constant. The corrected design feeds the model a *noisy* reliability signal,
uses informative motion, adds a fair linear baseline on identical features, an h=0 stage, analytic and
classical baselines, per-sequence metrics, and — critically — a **depth-decoupled error model**: the
pose-error scale depends on motion only, and scene depth enters *only* through the IOF 1/Z target, with
a scene-scale range (0.5–30 m) mirroring Princeton365's near-scanning emphasis. This removes the G3
confound (see the feasibility report, §3). Round-2 review fixes applied in this version: **estimated
motion by default** (relative motion derived from an accumulated estimated trajectory corrupted by the
pose error — what a running SLAM actually exposes), the oracle-vs-runtime-available baseline split
(B1–B9, §8.1), **within-sequence AUROC as the primary early-warning metric**, the full input-combination
ablation matrix, and a **generator stress matrix** (§10.4). Results (test on unseen sequences, 3 seeds,
estimated-motion generator):

### 10.1 Main results

| model | RMSE | medSpearman | AUROC (pooled) | **wAUROC (within-seq)** | AP |
|---|---|---|---|---|---|
| constant | 4.209 | — | 0.500 | 0.500 | 0.263 |
| persistence (oracle, B1) | 3.446 | 0.116 | 0.886 | 0.549 | 0.718 |
| ridge (same features, B5) | 3.626 | 0.125 | 0.880 | 0.519 | 0.648 |
| classical cov-prop (blind, B7) | 2.826 | 0.032 | 0.918 | 0.509 | 0.761 |
| gru (temporal, B9) | 1.323 | 0.566 | 0.981 | 0.638 | 0.957 |
| mlp motion-only | 1.729 | 0.666 | 0.971 | 0.721 | 0.939 |
| mlp motion+depth | 1.124 | 0.683 | 0.988 | 0.672 | 0.970 |
| **mlp full** | **0.948** | **0.792** | **0.992** | **0.688** | **0.981** |
| analytic 2-stage (phys-explicit, B8) | 1.914 | 0.676 | 0.966 | 0.637 | 0.903 |
| mlp full h=0 (current, B6) | 0.798 | 0.826 | 0.995 | 0.688 | 0.988 |

wAUROC = pooled AUROC under the *per-sequence* failure threshold τ_s = Q₇₅(IOF_s) — the primary
early-warning metric (cannot be gamed by detecting scene scale). Three honest observations:

* **The classical baseline is non-trivial but blind.** Blind covariance propagation (AR forecast
  covariance through the analytic IOF Jacobian, using only estimated motion and estimated depth)
  *beats* ridge and persistence on RMSE (2.826) and pooled AUROC (0.918) — it is scene-scale-aware —
  yet is chance within-sequence (wAUROC 0.509): it never observes the error realization, so it cannot
  anticipate within-sequence degradation. This is exactly why the learned model matters and why
  within-sequence is the primary metric.
* **The GRU (B9) does not beat the MLP.** The fair temporal baseline (RMSE 1.323 vs 0.948) confirms
  the contribution is the *task formulation* (features carry the signal), not the architecture.
* **Depth is a between-sequence signal.** Within-sequence, motion-only's wAUROC (0.721) slightly
  exceeds motion+depth's (0.672): depth statistics are nearly constant inside a sequence, so their
  value is cross-sequence (scene-scale), captured by RMSE and pooled AUROC — not by within-sequence
  AUROC. Both views are reported; G3 gates on the between-sequence legs.

### 10.2 Input-combination ablation (the reviewer-required matrix)

| Inputs | RMSE | AUROC (pooled) | wAUROC (within-seq) |
|---|---|---|---|
| motion only | 1.729 | 0.971 | 0.721 |
| depth only | 2.701 | 0.924 | 0.510 |
| reliability only | 4.075 | 0.633 | 0.565 |
| motion + depth | 1.124 | 0.988 | 0.672 |
| motion + reliability | 1.468 | 0.982 | 0.717 |
| depth + reliability | 2.568 | 0.934 | 0.531 |
| **full (motion+depth+reliability)** | **0.948** | **0.992** | **0.688** |

*Motion+depth vs motion-only* (both WITHOUT reliability) is the **reliability-masked depth test** — the
G3 gate: RMSE −35%, pooled AUROC 0.988 vs 0.971, both legs in every seed. *Full vs motion+reliability*
(G6, reported) asks whether depth still adds value when reliability is present: RMSE −35% again
(0.948 vs 1.468). The depth channel carries the scene-scale claim with and without the reliability
signal.

### 10.3 Failure-definition comparison (mlp full; pooled AUROC, mean over seeds)

| Definition | AUROC |
|---|---|
| Global threshold (τ = Q₇₅ train) | 0.992 |
| **Per-sequence threshold τ_s (PRIMARY)** | **0.688** |
| Robust z-score (z > 1.5, per-sequence) | 0.810 |
| Depth-normalized (IOF·Ẑ_med) | 0.251 |

The global number is inflated by between-sequence depth differences (near scenes are simply "worse").
The per-sequence number is the honest early-warning metric. The depth-normalized number is a diagnostic
caveat: normalizing IOF by Ẑ also rescales the *depth-invariant rotation term* by Ẑ, so with the
synthetic rotation weight the raw-IOF predictor's cross-sequence ranking inverts; within-sequence
metrics are invariant to this rescaling (a per-sequence constant multiple). This is a definitional
artifact to resolve against the official Flow-AUC protocol on real data (P1-G1), not a claim about the
predictor.

Scene-scale split (test, by true median depth): near (0.81 m) MLP RMSE **1.079** vs ridge 5.214 /
persistence 5.367; medium (4.44 m) 0.951 vs 2.782 / 2.275; far (18.15 m) 0.791 vs 2.083 / 1.188 — the
model tracks near-scene degradation (where Princeton365 concentrates its object-scanning sequences) far
better than the baselines.

**Gates (≥2 of 3 seeds):** G1 learned > linear on same features — PASS (RMSE 0.948 vs 3.626; AUROC 0.992
vs 0.880); G2 reliability signal adds value — PASS (full vs motion+depth: RMSE 0.948 vs 1.124);
G3 depth adds value with reliability **MASKED** — **PASS** (motion+depth vs motion-only: RMSE 1.124 vs
1.729, −35%, pooled AUROC 0.988 vs 0.971, both legs in every seed; the pre-fix PARTIAL was a two-part
generator artifact — depth coupled into the error scale, plus a rotation-flow floor compressing depth's
label effect over a 1.5–25 m range — resolved by the decoupled error model and the scene-scale-faithful
range); G4 beats oracle persistence — PASS (0.948 vs 3.446; the original failure is resolved); G5
current-error estimation works — PASS (h=0 RMSE 0.798 vs constant 4.209); G6 depth with reliability
present (reported) — PASS. Overall: **PASS** (was PARTIAL/FAIL).

### 10.4 Generator stress matrix (1 seed each; full results in `reports/stress_results.json`)

| Variant | full RMSE | wAUROC | G1 | G2 | G3 | G5 |
|---|---|---|---|---|---|---|
| baseline (estimated motion) | 1.029 | 0.715 | PASS | PASS | PASS | PASS |
| true motion (pre-fix input) | 1.095 | 0.704 | PASS | PASS | PASS | PASS |
| reliability: delayed (3 frames) | 1.053 | 0.707 | PASS | PASS | PASS | PASS |
| reliability: noisy | 0.965 | 0.711 | PASS | PASS | PASS | PASS |
| reliability: miscalibrated | 1.005 | 0.710 | PASS | PASS | PASS | PASS |
| reliability: masked | 1.232 | 0.693 | PASS | **FAIL** | PASS | PASS |
| reliability: intermittent | 1.052 | 0.704 | PASS | PASS | PASS | PASS |
| jump failures | 1.275 | 0.701 | PASS | **FAIL** | PASS | PASS |
| realistic depth corruption | 1.013 | 0.680 | PASS | PASS | PASS | PASS |

Two findings, both honest:

* **G3 — the scene-scale claim — survives every degradation**, including masked reliability and
  realistic depth corruption. The depth channel does not lean on an idealized reliability signal.
* **G2 fails exactly where the reliability signal is genuinely unusable**: masked (no information by
  construction) and jump failures (a smooth `confidence ≈ exp(−5·‖err‖)` observation cannot anticipate
  an abrupt within-horizon jump). A model can only extract value from a signal that carries it; the
  real-data analogue is that reliability helps predict smooth drift, not sudden relocalization jumps —
  which is precisely why the drift-vs-jump decomposition and lead-time evaluation (§8.7, R2) are
  mandatory on real data.

**What this study does and does not prove.** It proves the experimental protocol works: with
runtime-available error-state information, the learned predictor beats the oracle-persistence and linear
baselines, is scene-scale-aware, and — with the confound fixed and the input realism stress-tested —
the depth channel adds measurable value through the geometric 1/Z scaling (G3) with and without the
reliability signal. It does *not* prove the same on real SLAM error dynamics — the synthetic generator
makes IOF strongly determined by runtime features, and real DROID-SLAM errors involve drift, jumps, and
relocalization events with different autocorrelation structure. The gate therefore justifies proceeding
to the small real-data pilot (§8.8, §9), with the persistence baseline mandatory in every table and the
depth-scaling hypothesis (H2) re-tested against real ZED depth and error statistics under the
pre-registered stop criterion (P1-G5).

## 11. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Novelty window closes (others use IOF for prediction) | Medium | Publish early (workshop tier); re-run systematic search before submission |
| Persistence dominates on real data (error process ~ random walk) | Medium | Explicit drift-vs-jump decomposition; lead-time evaluation; adaptive-SLAM contribution independent of forecasting win |
| GT coverage limits lead-time evaluation | High | Restrict to posed segments; report effective coverage; auxiliary self-supervised targets |
| Reliability signals unavailable / DROID instrumentation fragile | Medium | ORB-SLAM3 fallback backbone (native tracking signals); signal masking ablation; stress matrix shows G3 survives masked reliability |
| ZED depth noise in targets | Medium | Denoise targets; report official-variant numbers; P1-G1 validates our IOF against the official implementation |
| Data scale / compute (435 GB dataset) | High | Staged subset pipeline; cached SLAM outputs; frozen pretrained components |

## 12. Timeline and Deliverables

* **Phase 0 (done):** synthetic feasibility gate (this repository).
* **Phase 1 (Months 1–2):** Princeton365 subset pipeline; IOF target generation; **official-IOF
  reproduction gate (P1-G1, ρ > 0.98)**; DROID-SLAM instrumentation with ORB-SLAM3 fallback decision;
  failure-event counts within posed segments; cached outputs — all governed by the pre-registered
  pilot gates P1-G1…P1-G7 (§8.8).
* **Phase 2 (Months 3–5):** predictor training on 50–80 sequences; ablations; baseline table (B1–B9);
  h=0 vs h decomposition.
* **Phase 3 (Months 6–8):** full-split training; cross-category and cross-backbone generalization;
  per-sequence metrics, ECE, lead time.
* **Phase 4 (Months 9–10):** adaptive SLAM demonstration with oracle bound; official test evaluation;
  **final systematic literature re-check**.
* **Phase 5 (Month 11):** writing. Deliverables: dataset pipeline, target-generation code, benchmark,
  lightweight model, ablations, generalization study, adaptive-SLAM demo — all open-source.

**Publication strategy:** Tier 1 — workshop paper (scene-scale-aware visual-error prediction);
Tier 2 — mid-tier conference (prediction + generalization + ablations + lead time + downstream);
Tier 3 — journal (multiple backbones/datasets, calibration, failure taxonomy, adaptive controller).

## 13. References

1. Kayan, K., Alexandropoulos, S., Jain, R., Zuo, Y., Liang, E., Deng, J. *Princeton365: A Diverse
   Dataset with Accurate Camera Pose.* ICCV 2025. arXiv:2506.09035.
2. Teed, Z., Deng, J. *DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras.*
   NeurIPS 2021.
3. Campos, C., Elvira, R., Rodríguez, J. J. G., Montiel, J. M. M., Tardós, J. D. *ORB-SLAM3.*
   IEEE T-RO 37(6), 2021.
4. Murai, R., Dexheimer, E., Davison, A. J. *MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction
   Priors.* CVPR 2025. arXiv:2412.12392.
5. Wang, J., Chen, M., Karaev, N., Vedaldi, A., Rupprecht, C., Novotny, D. *VGGT: Visual Geometry
   Grounded Transformer.* CVPR 2025.
6. Costante, G., Mancini, M. *Uncertainty Estimation for Data-Driven Visual Odometry.* IEEE T-RO
   36(6):1738–1757, 2020.
7. Kaygusuz, N., Mendez, O., Bowden, R. *MDN-VO: Estimating Visual Odometry with Confidence.* IROS 2021.
8. Wang, W., Hu, Y., Scherer, S. *TartanVO: A Generalizable Learning-based Visual Odometry.* CoRL 2020.
9. Daftry, S., Zeng, S., Bagnell, J. A., Hebert, M. *Introspective Perception: Learning to Predict
   Failures in Vision Systems.* IROS 2016.
10. Gurau, C., Rao, D., Tong, C. H., Posner, I. *Learn from Experience: Probabilistic Prediction of
    Perception Performance to Avoid Failure.* IJRR 37(9):981–995, 2018.
11. Dequaire, J., Tong, C. H., Churchill, W., Posner, I. *Off the Beaten Track: Predicting Localisation
    Performance in Visual Teach and Repeat.* ICRA 2016.
12. Naveed, K., Anjum, M. L., Hussain, W., Lee, D. *Deep Introspective SLAM: Deep Reinforcement
    Learning Based Approach to Avoid Tracking Failure in Visual SLAM.* Autonomous Robots 46:705–720, 2022.
13. Naveed, K., Anjum, M. L., Lee, D. *Deeper Introspective SLAM: How to Avoid Tracking Failures Over
    Longer Routes?* IEEE RA-L 9(3), 2024.
14. Hu, T., Scargill, T., Yang, F., Chen, Y., Lan, G., Gorlatova, M. *SEESys: Online Pose Error
    Estimation System for Visual SLAM.* SenSys 2024:322–335.
15. Gaus, J. A., Häufle, D., Baek, W.-J. *SUPER: A Framework for Sensitivity-based Uncertainty-aware
    Performance and Risk Assessment in Visual Inertial Odometry.* arXiv:2512.14189, 2025 (ICRA 2026).
16. Maggio, D., et al. *VERF: Runtime Monitoring of Pose Estimation with Neural Radiance Fields.*
    arXiv:2308.05939, 2023.
17. Sandström, E., Li, Y., Van Gool, L., Oswald, M. R. *UncLe-SLAM: Uncertainty Learning for Dense
    Neural SLAM.* ICCVW 2023.
18. Stutts, A. C., Erricolo, D., Tulabandhula, T., Trivedi, A. R. *Lightweight, Uncertainty-Aware
    Conformalized Visual Odometry.* 2023.
19. Zhang, J., Singh, S. *On the Degeneracy of Localization with Vision and LiDAR.* 2016.
20. *Probabilistic Degeneracy Detection for Point-to-Plane Error Minimization.* arXiv:2410.10784, 2024.
21. Teed, Z., Deng, J. *RAFT-3D: Scene Flow Using Rigid-Motion Embeddings.* CVPR 2021.
22. *Principled Uncertainty Propagation for Stereo Visual Odometry.* 2026.
23. *Visual Active SLAM Method Considering Measurement and State Uncertainty.* 2025 (CRLB/FIM
    prediction of future pose uncertainty).

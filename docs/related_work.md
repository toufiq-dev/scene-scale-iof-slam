# Related Work — Annotated Review

*Compiled via live systematic literature search, August 2026. Status: every cluster below contains
work that partially overlaps the thesis framing; the *narrow* novelty claim (future, scene-scale-aware,
image-space IOF prediction with SLAM-available inputs) was not found in any published work as of this
date. This document must be refreshed immediately before submission.*

---

## 1. Uncertainty-aware visual odometry and SLAM

**Learned pose uncertainty.** Costante & Mancini, "Uncertainty Estimation for Data-Driven Visual
Odometry," IEEE T-RO 36(6):1738–1757, 2020 — aleatoric covariance outputs from deep VO. Kaygusuz,
Mendez, Bowden, "MDN-VO: Estimating Visual Odometry with Confidence," IROS 2021 — mixture-density
confidence. Wang, Hu, Scherer, "TartanVO," CoRL 2020 — cross-dataset generalizable learned VO with
intrinsics conditioning. D3VO (CVPR 2020) — deep depth, pose, uncertainty for monocular VO.

**Dense neural SLAM with uncertainty.** Sandström, Li, Van Gool, Oswald, "UncLe-SLAM: Uncertainty
Learning for Dense Neural SLAM," ICCVW 2023; Uni-SLAM; Stutts, Erricolo, Tulabandhula, Trivedi,
"Lightweight, Uncertainty-Aware Conformalized Visual Odometry," 2023 (distribution-free uncertainty
with coverage guarantees); "Combining Projected Uncertainty for Self-Supervised Visual Odometry"
(IJCV 2026) — *projects* depth/pose uncertainty into the image plane, the closest "image-space"
flavor in this cluster.

**Assessment / positioning.** This cluster quantifies *pose or depth* uncertainty. It does not predict
*future* realized error, does not express consequence in scene-scale-aware pixel displacement (IOF),
and is not framed as a forward-looking reliability signal for downstream action. Our H3 (visual vs pose
prediction, evaluated in image space) is precisely the differentiating experiment.

## 2. Introspective perception and perception-performance prediction

Daftry, Zeng, Bagnell, Hebert, "Introspective Perception: Learning to Predict Failures in Vision
Systems," IROS 2016 — a two-stack CNN predicts failure probability of a vision system from its inputs;
the direct ancestor of "learn to predict your own failure." Oxford line: Gurau, Rao, Tong, Posner,
"Learn from Experience: Probabilistic Prediction of Perception Performance to Avoid Failure,"
IJRR 37(9):981–995, 2018; "Fit for Purpose? Predicting Perception Performance" (IROS 2016); Dequaire,
Tong, Churchill, Posner, "Off the Beaten Track: Predicting Localisation Performance in Visual Teach
and Repeat," ICRA 2016; Churchill et al., "Know Your Limits." Hu et al., "Introspective Evaluation of
Perception Performance for Parameter Selection without Ground Truth," RSS 2017.

**Assessment / positioning.** These predict success/failure of a *fixed* perception task (localisation
match, MAV obstacle avoidance) — mostly current or near-term, task-level, and trained for specific
pipelines. None targets the *image-space magnitude of visual degradation* of a running SLAM system, and
none uses a mm-accurate, scene-scale-diverse benchmark as supervision. We must cite and position against
this line explicitly — it is the strongest "predicting failure" precedent.

## 3. SLAM tracking-failure avoidance

Naveed, Anjum, Hussain, Lee, "Deep Introspective SLAM: Deep Reinforcement Learning Based Approach to
Avoid Tracking Failure in Visual SLAM," Autonomous Robots 46:705–720, 2022 (RL policy evaluates
navigation-step safety wrt tracking failure). Naveed, Anjum, Lee, "Deeper Introspective SLAM: How to
Avoid Tracking Failures Over Longer Routes?," IEEE RA-L, 2024 (video-transformer introspection).
Prasad, Yadav, Saurabh, et al., "Learning to Prevent Monocular SLAM Failure using Reinforcement
Learning," arXiv:1607.07558 (2016; ICCPS 2019).

**Assessment / positioning.** These *avoid* failure by acting on the camera trajectory (active view
planning) inside a specific SLAM (ORB-SLAM). Ours emits a continuous, forward-looking image-space
reliability signal for a *fixed* trajectory and is backbone-agnostic by design. Complementary, not
competing — and a natural comparison target for the adaptive-SLAM demonstration (H5).

## 4. Online pose-error and runtime-risk estimation (closest competitors)

Hu, Scargill, Yang, Chen, Lan, Gorlatova, "SEESys: Online Pose Error Estimation System for Visual
SLAM," SenSys 2024:322–335 — *the first online, deep-learning-based SLAM pose-error estimator*;
predicts current error magnitude from SLAM-internal signals; embedded deployment. Gaus, Häufle, Baek,
"SUPER: Sensitivity-based Uncertainty-aware Performance and Risk Assessment in Visual Inertial
Odometry," arXiv:2512.14189 (Dec 2025; ICRA 2026) — lightweight runtime warning signals via
sensitivity analysis, evaluated on long-horizon SLAM mapping. Maggio et al., "VERF: Runtime Monitoring
of Pose Estimation with Neural Radiance Fields," arXiv:2308.05939 (2023) — runtime assurance on pose
correctness.

**Assessment / positioning.** These are the works a reviewer will (correctly) demand we engage. Our
differences: (i) target — *future* (h-step) scene-scale-aware *image-space* degradation (IOF) rather
than current pose error magnitude or generic risk; (ii) supervision — a scene-scale-aware metric with
mm-accurate GT rather than heuristic error proxies; (iii) validation — per-sequence metrics against
persistence, linear, error-propagation, and oracle baselines. SEESys is also the blueprint for our
h=0 "current-error estimation" stage and a mandatory comparison baseline (B4 in §8.1 of the proposal).

## 5. Degeneracy / uninformative-geometry detection

Zhang & Singh, "On the Degeneracy of Localization with Vision and LiDAR," 2016; "Probabilistic
Degeneracy Detection for Point-to-Plane Error Minimization," arXiv:2410.10784 (2024); DALI-SLAM
(2025); LODESTAR (degeneracy-aware LiDAR-inertial odometry); uncertainty-aware LiDAR-inertial-visual
SLAM with adaptive fusion (2026).

**Assessment / positioning.** Degeneracy detection is *reactive* (detect estimator collapse from
Jacobian rank) and mostly LiDAR. Ours predicts the *visual consequence* of error evolution and is
supervised end-to-end on image-space targets; degeneracy detection is a useful prior/feature, not a
competitor. Worth one paragraph in the final related-work section.

## 6. Classical uncertainty propagation (the baseline that must not be omitted)

EKF covariance propagation and Cramér–Rao / Fisher-information prediction of *future* pose
uncertainty: e.g., "Visual Active SLAM Method Considering Measurement and State Uncertainty" (2025,
uses FIM/CRLB for pose-uncertainty prediction in stereo visual SLAM); "Principled Uncertainty
Propagation for Stereo Visual Odometry" (2026); "Development of an error propagation model for Visual
Odometry." Classical photogrammetry/BA already maps pose covariance to reprojection-error statistics.

**Assessment / positioning.** This is the classical answer to "predict the image-space consequence of
pose uncertainty": propagate the covariance through the projection Jacobian. Our B5 baseline makes the
comparison explicit and fair (pose covariance → image space via the IOF Jacobian, then forecast the
covariance h steps ahead under a motion model). If B5 matches our learned forecaster, the thesis's
contribution collapses to time-series forecasting of SLAM error — so B5 must be in the table, not
assumed away.

## 7. Generative / foundation-model geometry (adjacent, scoped)

VGGT (Wang et al., CVPR 2025, best paper) — feed-forward camera/depth/point-map prediction; MASt3R-SLAM
(Leroy et al., CVPR 2025) — real-time dense monocular SLAM from a pretrained 3D prior; DUSt3R family.

**Assessment / positioning.** These make geometry prediction almost free, which (a) strengthens the
feasibility of our depth-statistics input and (b) invites the question "why not just use VGGT
features?" We scope this as one explicit experiment (frozen VGGT/DUSt3R features vs raw depth
statistics, §8.2), not the thesis itself — the research question remains *predicting the visual
consequence of SLAM uncertainty*, not engineering better geometry.

---

## Gap summary

| Criterion | Found in literature? |
|---|---|
| IOF / Flow AUC used as a supervised learning target | **No** (as of Aug 2026) |
| h-step-ahead prediction of future visual degradation | **No** (SEESys/SUPER predict current risk; classical work predicts future *covariance*) |
| Inference restricted to SLAM-available quantities | Partially (SEESys, SUPER) — but with different targets |
| Explicit comparison vs persistence + error propagation + oracle | **No** |
| Downstream adaptive-SLAM use of an image-space reliability signal | Partially (Introspective-SLAM line — trajectory-level, RL) |

**Bottom line:** the narrow gap (first two rows) is real but young (Princeton365 is from ICCV 2025).
The defensible thesis formulation is: *future, scene-scale-aware, image-space visual-consequence
prediction for SLAM, trained on mm-accurate GT, evaluated with the correct baselines.* Everything
broader is prior art and must be cited.

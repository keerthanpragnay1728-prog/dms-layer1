# Milestone 3 notes: Haar front-end on WFLW — report material

**Date:** 2026-08-21 · **Dataset:** WFLW test split · **Detector:**
`haarcascade_frontalface_default` (vendored), histogram equalisation on.
Detector parameters were re-decided by the pose sweep below; the shipped
operating point is **scale_factor 1.05, min_neighbors 2, min_size_frac
0.08** (the table in the next section was measured at the original
1.10 / 5 / 0.08 settings and is kept as the sweep's baseline).

## Detection rate at the ORIGINAL settings (1.10, 5, 0.08), IoU ≥ 0.3

| population   | rate    |
|--------------|--------:|
| overall      | 60.04%  |
| no flags     | 82.21%  |
| pose         | **17.79%** |
| expression   | 53.82%  |
| illumination | 63.90%  |
| makeup       | 56.31%  |
| occlusion    | 46.60%  |
| blur         | 51.62%  |

3,843 unmatched Haar boxes (1.81/image) — not a false-positive rate, since
WFLW images contain unannotated faces, but usable as a relative cost signal.

## Calibration decision: containment beats median fit

Measured median fit of raw Haar boxes to the model crop box: box_scale
1.121 (IQR 1.056–1.205), box_shift_y 0.130 (IQR 0.108–0.155). Landmark
containment in the Haar-derived crop: **98.47% with the config values
(1.45, 0.10)** vs **84.34% with the measured medians (1.12, 0.13)**.

Decision: keep `box_scale: 1.45`. A landmark outside the crop is a landmark
the model cannot predict, so containment outranks crop tightness; the
median-fit box loses the distribution's tails. Consequence:
`verify_haar_pipeline.py` now recommends by containment-maximising grid
search instead of median fit, and reports a candidate table that includes
(1.45, 0.13) — shifting the crop down to the measured median shift is
adopted only if its measured containment stays ≥ 98%. Until that number
comes back from a Kaggle run, the config stays at (1.45, 0.10).

## Timing and environment facts

* Haar detection timing on Kaggle CPU: measurements of **identical settings
  varied 2× across sessions** (160 ms/image in one session, 83 ms in
  another; the sweep session measured 157 ms for the adopted settings and
  84 ms for the old baseline). Kaggle CPU allocation is not stable, so no
  single ms figure is citable as absolute: report timings only as
  within-session relative comparisons (and later, ours-vs-MediaPipe on
  identical hardware in the same session). Deployment latency is a separate
  question anyway — single camera frame, optional downscale, every-Nth-frame
  detection with box persistence.
* Coordinate round trip frame → crop space → frame: exact (0.000000000 px).
* **OpenCV 5.x removed the Haar `CascadeClassifier` API and stopped shipping
  the cascade data files** (observed directly on the opencv 5.0 wheel:
  neither is present). The project therefore pins `opencv-python>=4.8,<5`
  and vendors `haarcascade_frontalface_default.xml` and
  `haarcascade_profileface.xml` in `assets/` (extracted from the OpenCV
  4.10 PyPI wheel, original license headers retained). Worth one sentence in
  the report: the classical detector this pipeline deliberately uses is
  being retired from its home library.

## The pose problem (17.79%): sweep results and the decision

Detecting turned heads is the inattentiveness case, so this number was a
finding to address, not a footnote. `scripts/sweep_haar_pose.py` measured a
parameter grid (scale_factor {1.05, 1.10} × min_neighbors {2, 3, 5} ×
min_size_frac {0.05, 0.08}) and the profile-face cascade as a fallback, on
every pose-subset face with a 250-image no-flag reference sample. Key rows
(full grid in the sweep run's `sweep_results.yaml`):

| combo (sf, mn, msf)        | pose    | no-flag | unmatched/img | ms/img |
|----------------------------|--------:|--------:|--------------:|-------:|
| 1.10, 5, 0.08 (baseline)   | 17.8%   | 79.8%   | 1.80          | 84     |
| 1.05, 2, 0.05 (best pose)  | 34.0%   | 87.1%   | 7.32          | 272    |
| **1.05, 2, 0.08 (adopted)**| **33.7%** | **86.8%** | **5.00**  | **157** |
| 1.05, 2, 0.05 + profile    | 35.0%   | —       | —             | 664    |

(The no-flag rates here are on the sweep's sampled reference population and
differ slightly from the full-split 82.21% above.)

**Tuning beats the profile fallback.** The profile cascade lifts most where
the frontal cascade is strict (baseline: 17.8% → 27.9%), but never beats
simply loosening the frontal cascade, and it roughly triples latency
everywhere (best row: +1.0 point for 272 → 664 ms). Decision:
`profile_fallback` stays **off** — tried and rejected on evidence, not
skipped.

**Adopted operating point: (1.05, 2, 0.08)** — within 0.3 points of the
best pose rate at 40% of its latency, and it also lifts the no-flag
population 79.8% → 86.8%. The accepted cost is unmatched boxes rising
1.80 → 5.00 per image: acceptable because deployment takes the largest box
in a one-face cabin, and WFLW's unmatched boxes are largely unannotated
faces. Doubling of pose detection (17.8% → 33.7%) does not make Haar a
turned-head detector; it moves the measured ceiling, which the framing
below reports as measured.

## Containment at the adopted detector settings, and the calibration decision

Re-run of the verifier at (1.05, 2, 0.08):

| calibration candidate         | containment |
|-------------------------------|------------:|
| (1.45, 0.10) previous config  | 98.27%      |
| (1.45, 0.13) shift candidate  | 98.47%      |
| (1.12, 0.13) measured medians | 82.81%      |
| **(1.75, 0.08) grid best**    | **99.73%**  |

**Adopted: (1.75, 0.08).** Containment outranks crop tightness (the same
argument that kept 1.45 over the median fit), and the grid best converts a
1.5% landmark-loss rate into 0.27%.

**Flagged cost — less face per pixel at deploy time.** The deploy crop is
1.75× the Haar side; the ground-truth crop (which training uses) measures
~1.12× the Haar side at the median, so the deploy crop is ~1.56× wider than
the training crop. The face spans ~77% of a training crop (1/1.3 expand)
but only ~49% of a deploy crop — equivalent to a scale-augmentation factor
of ~0.64, *outside* the configured [0.85, 1.15] range, and it roughly
shrinks the inter-ocular distance at the 112 px input from ~35 px to
~22 px. Plan: train the milestone-4 baseline as configured (the NME
protocol evaluates on ground-truth boxes, unaffected); milestone 6 measures
NME through the real Haar pipeline vs ground-truth boxes, which prices this
mismatch directly. If the gap is material, the counter-measures are, in
order of preference: widen the scale augmentation's lower bound (~0.6) and
retrain, or fall back to the tighter (1.45, 0.13) at 98.47% containment.
No silent changes to the training recipe before that number exists.

The full detection-rate table at the shipped settings comes from the same
re-run (milestone-3 notebook); record it above when it lands.

Honest framing for the report:

0. The ceiling was measured, not assumed: a 12-point parameter grid and the
   profile-cascade fallback were both evaluated on the pose subset with
   their frontal-side and latency costs, and the operating point was chosen
   from that table. The profile cascade in particular was tried and
   rejected on evidence (never beat loosening the frontal cascade; ~3×
   latency), not skipped.
1. WFLW's pose subset is dominated by extreme yaw approaching profile —
   harsher than the in-cabin envelope, where a driver-facing camera sees a
   near-frontal face most of the time and mirror/shoulder glances are brief
   excursions. Report the detection envelope per subset rather than one
   number.
2. The deployed system is temporal; the perception layer is per-frame. A
   Haar miss during a head turn does not erase the event: Layer 3 can hold
   the last box briefly, and a sustained loss of the frontal face is itself
   evidence for "eyes off road" (loss-of-lock as signal, to be validated in
   Layer 3, not assumed).
3. The scientific comparison stays clean by construction: landmark-model
   NME (ours vs MediaPipe) is evaluated on ground-truth boxes — the
   standard WFLW protocol — so landmark quality is not confounded by the
   detector. The Haar envelope is reported separately as a property of the
   deployment front-end. MediaPipe brings its own detector (BlazeFace), so
   the full-pipeline ablation compares complete stacks, and the detector
   difference is named explicitly as part of what differs.
4. If the profile fallback earns its keep, it becomes a config-on feature;
   if not, the limitation is stated with the sweep table as evidence that
   tuning within the Haar family does not fix it — which itself supports
   the report's argument for why modern in-cabin systems moved to learned
   detectors.

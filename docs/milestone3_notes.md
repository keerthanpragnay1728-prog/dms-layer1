# Milestone 3 notes: Haar front-end on WFLW — report material

**Date:** 2026-08-21 · **Dataset:** WFLW test split · **Detector:**
`haarcascade_frontalface_default` (vendored), scale_factor 1.1,
min_neighbors 5, min_size_frac 0.08, histogram equalisation on.

## Detection rate (Haar box matches the GT crop box at IoU ≥ 0.3)

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

* Haar detection: **median 160 ms/image on the Kaggle CPU** (WFLW scene
  images, not face crops). Relative comparison only — not representative of
  deployment hardware, where the input is a single camera frame and the
  detector can run on a downscaled frame and/or every Nth frame with box
  persistence in between.
* Coordinate round trip frame → crop space → frame: exact (0.000000000 px).
* **OpenCV 5.x removed the Haar `CascadeClassifier` API and stopped shipping
  the cascade data files** (observed directly on the opencv 5.0 wheel:
  neither is present). The project therefore pins `opencv-python>=4.8,<5`
  and vendors `haarcascade_frontalface_default.xml` and
  `haarcascade_profileface.xml` in `assets/` (extracted from the OpenCV
  4.10 PyPI wheel, original license headers retained). Worth one sentence in
  the report: the classical detector this pipeline deliberately uses is
  being retired from its home library.

## The pose problem (17.79%) and what is being done

Detecting turned heads is the inattentiveness case, so this number is a
finding to address, not a footnote. `scripts/sweep_haar_pose.py` measures,
on the pose subset with a no-flag reference sample:

* (a) a parameter sweep — scale_factor {1.05, 1.10} × min_neighbors
  {2, 3, 5} × min_size_frac {0.05, 0.08} — reporting pose and frontal
  detection, unmatched boxes/image, and runtime per combo;
* (b) the profile-face cascade as a fallback (image + mirrored pass, only
  when the frontal cascade finds nothing), same costs reported, plus a
  separate box calibration for profile detections.

Honest framing if neither lifts pose detection to a usable level (to be
finalised with the sweep numbers):

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

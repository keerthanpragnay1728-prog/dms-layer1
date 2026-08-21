# dms-layer1 — Facial Landmark Perception

Layer 1 of a three-layer driver monitoring system: face detection (Haar
cascade) followed by a 24-point landmark model. The headline experiment is an
ablation — our small from-scratch landmark model vs MediaPipe Face Mesh,
feeding an **identical** downstream pipeline — so the landmark source is
swappable behind one interface and everything downstream is blind to which
front-end produced the coordinates.

* Layer 1 (this repo): face detection + 24 landmarks per frame
* Layer 2 (not here): EAR, MAR, head pose, gaze, eye-state CNN
* Layer 3 (not here): PERCLOS, microsleep timing, glance duration, state machine

## The 24-point schema

Defined once in [`configs/landmarks_24.yaml`](configs/landmarks_24.yaml) — the
single source of truth for point identities, the WFLW 98→24 index mapping, and
the horizontal-flip permutation. Groups:

| group   | ours  | purpose                    | WFLW sources                         |
|---------|-------|----------------------------|--------------------------------------|
| eyelids | 0–11  | eye aspect ratio           | 60,61,63,64,65,67 / 68,69,71,72,73,75 |
| pupils  | 12–13 | gaze direction             | 96 / 97                              |
| mouth   | 14–17 | mouth aspect ratio, yawn   | 76, 82, 79, 85 (outer lip)           |
| axis    | 18–19 | head pitch                 | 54 (nose tip), 16 (chin)             |
| contour | 20–23 | head yaw                   | 4, 8, 28, 24 (symmetric pairs)       |

Conventions (documented in the schema file, enforced by tests):

* **left/right are image space** — "left" = viewer's left = subject's right.
  (WFLW's own readme names index 96 "right pupil" in the subject frame; that
  is the same physical point as our `left_pupil`.)
* Each eye is the canonical 6-point EAR arrangement
  `[corner, upper, upper, corner, lower, lower]`:
  `EAR = (|p1−p5| + |p2−p4|) / (2·|p0−p3|)` on local offsets, identical for
  both eyes. `MAR = |ours16−ours17| / |ours14−ours15|`.
* NME is normalised by inter-ocular distance (outer eye corners, ours 0 and 9).

## Milestones

1. **(closed)** Scaffolding, config system, WFLW loader, 98→24 mapping,
   verification overlays. Mapping confirmed correct against the real
   annotation file — see [docs/milestone1_verdict.md](docs/milestone1_verdict.md).
2. **(closed)** Crop cache (one-time preprocess to a uint8 array +
   crop-space labels). Verified on real WFLW: 7,500 + 2,500 faces, label
   round-trip 0.00005 px, previews confirmed. The cache is attached to
   training notebooks read-only via `cache.dir` in the config.
3. **(closed)** Haar face detection wrapper + crop/resize/coordinate
   round-trip (with test). Verified on real WFLW; findings and the pose
   detection problem in [docs/milestone3_notes.md](docs/milestone3_notes.md);
   `scripts/sweep_haar_pose.py` measures the tuning/profile-fallback options.
4. **(current)** Model, training loop, augmentation (flip-index unit test),
   checkpointing, epoch-level resume, CSV metrics.
5. Evaluation: NME overall / per group / per WFLW subset, failure rate @10%,
   model size, CPU inference time.
6. `LandmarkDetector` interface + our implementation + MediaPipe mapped to the
   same 24 semantics.
7. Frame-to-frame stability harness comparing both detectors (landmark and
   EAR standard deviation on a still face).

## Milestone 1 verification (run on Kaggle, where WFLW is attached)

Open `notebooks/kaggle_milestone1.ipynb` in Kaggle with the WFLW dataset
attached, or run in any environment with the dataset:

```bash
pip install -r requirements.txt
python tests/run_tests.py                                        # unit tests
python scripts/verify_layout.py --config configs/layer1_base.yaml --split test
python scripts/visualize_mapping.py --config configs/layer1_base.yaml --split test
```

`verify_layout.py` checks the assumed WFLW index layout statistically against
the real annotation file (e.g. point 96 must lie inside the 60–67 eye polygon
on ~100% of faces). `visualize_mapping.py` writes labelled overlays — main
panel plus zoomed eye/mouth insets with a crosshair through each pupil point —
for frontal, large-pose and occluded samples, and a contact sheet. The
checklist to confirm by eye is in that script's docstring.

The dataset root lives in `configs/layer1_base.yaml` under `dataset.root`
(default: the Kaggle mount). The loader fails with a listing of what it found
if the path is wrong — it never guesses silently.

## Milestone 2: crop cache

`notebooks/kaggle_milestone2.ipynb`, or directly:

```bash
python scripts/build_crop_cache.py --config configs/layer1_base.yaml --split both
```

One-time preprocess (every image decoded exactly once): square grayscale
crops around each face's 98-point extent (`preprocess.crop_expand`), resized
to `preprocess.cache_size`, stored per split as parallel `.npy` arrays with
[0,1] crop-space labels for our 24 points, the crop boxes (for mapping back
to frame coordinates), and the attribute flags (for per-subset evaluation).
The script verifies its own output: read-back plus a label round-trip against
a fresh parse of the annotations (float32 rounding only), and renders preview
grids for an eyeball check. The cache is attached to later notebooks as a
read-only input (`cache.dir` in the config points at the mount — a published
dataset or a committed notebook's output); training (milestone 4) loads it
fully into RAM and augments on the fly.

## Milestone 3: Haar face-detection front-end

`notebooks/kaggle_milestone3.ipynb`, or directly:

```bash
python scripts/verify_haar_pipeline.py --config configs/layer1_base.yaml --split test
```

`dms_layer1/detect/haar.py` wraps the OpenCV cascade (the XML is vendored in
`assets/` — OpenCV 5.x wheels dropped both the cascade API and the data
files, so `opencv-python` is pinned `<5` and the file is pinned in-repo) and
maps a raw Haar box to the model's crop box via two calibrated config values
(`face_detector.box_scale`, `box_shift_y`). The verification script measures
that calibration against ground truth on real WFLW, reports detection rates
overall and per subset, landmark containment, the exact coordinate
round-trip, and CPU timing, and renders matched/missed previews. Crop
extraction and coordinate mapping reuse `data/crops.py`, so the detector and
the training cache cannot disagree on the transform (unit-tested, including
an image-content round-trip within one pixel).

## Milestone 4: training

`notebooks/kaggle_milestone4_train.ipynb`, or directly:

```bash
python train.py --config configs/layer1_base.yaml           # fresh run
python train.py --config configs/layer1_base.yaml --resume  # continue one
```

`LandmarkNet` (`dms_layer1/model/net.py`): a small conv-BN-ReLU stack,
trained from random initialisation (no pretrained weights, by design),
~0.59 M params / ~2.4 MB fp32 at width 32 — well under the 5 MB budget.
Output is 24 (x, y) pairs in [0, 1] crop coordinates from a single linear
layer, bias-initialised to the crop centre. Loss is selectable in the config
(`train.loss: l2 | wing`). Augmentation runs on the fly over the RAM cache —
flip (with the landmark index remap, unit-tested), rotation, scale,
translation, brightness/contrast, blur — as one affine shared by image and
labels, with per-sample RNG seeded from (seed, epoch, index) so runs are
reproducible by construction.

Session survival: a kill-safe checkpoint every epoch carrying optimiser,
scheduler, epoch counter, early-stop state and python/numpy/torch RNG;
metrics append to a CSV after every epoch; loss/NME curves re-render to a
PNG each epoch (the "TensorBoard or equivalent" — no extra dependency).
`tests/test_resume.py` proves a stopped-and-resumed run reproduces an
uninterrupted one row-for-row, and that a death between the CSV write and
the checkpoint save cannot duplicate rows. `train.stop_after_epochs` gives a
clean stop ahead of Kaggle's session cap. Validation is a seeded 10% split
of the train cache (WFLW has no subject IDs, so a random face split is the
only option; the subject-independence concern applies to the later in-cabin
recordings, not WFLW); early stopping tracks val NME (inter-ocular).

Local smoke test without the dataset (schematic faces, code-path check only,
loudly labelled as such): add `--synthetic` to either script.

## Repo layout

```
configs/            layer1_base.yaml (all knobs), landmarks_24.yaml (schema)
dms_layer1/         package: config, landmarks/schema, data/wflw, viz/overlay
scripts/            verify_layout.py, visualize_mapping.py
tests/              run_tests.py (no pytest needed; pytest-compatible files)
notebooks/          thin Kaggle notebooks (clone repo, run scripts)
```

## Reproducibility

* Global seed in the config (`seed: 42`), used for sampling everywhere;
  torch seeding joins in milestone 4.
* Every script that writes outputs also writes `config_used.yaml` next to
  them, so any reported number can be reconstructed.
* No pretrained weights for the landmark model — training from random
  initialisation is a deliberate part of the contribution.
* Dependencies are pinned to the agreed set in `requirements.txt`; nothing
  gets added without discussion.

# dms-layer1

Facial landmark perception for a driver monitoring system. This repo is
Layer 1 of a three layer design: a Haar cascade finds the face, a small CNN
predicts 24 landmarks inside the face box. Layer 2 (eye aspect ratio, mouth
aspect ratio, head pose, gaze) and Layer 3 (PERCLOS, microsleep timing,
glance duration) are separate and not part of this repo.

The main experiment of the project is an ablation. We train our own landmark
model and compare it against MediaPipe Face Mesh, with both feeding the same
downstream pipeline. For that to be a fair comparison the landmark source
has to be swappable behind one interface, and nothing downstream is allowed
to know which detector produced the coordinates. The code is structured
around that requirement.

## The 24 point schema

The point set is defined once, in
[`configs/landmarks_24.yaml`](configs/landmarks_24.yaml). That file is the
single source of truth for point identities, the WFLW 98 to 24 index
mapping, and the horizontal flip permutation.

| group   | ours  | used for                 | WFLW sources                          |
|---------|-------|--------------------------|---------------------------------------|
| eyelids | 0-11  | eye aspect ratio         | 60,61,63,64,65,67 / 68,69,71,72,73,75 |
| pupils  | 12-13 | gaze direction           | 96 / 97                               |
| mouth   | 14-17 | mouth aspect ratio, yawn | 76, 82, 79, 85 (outer lip)            |
| axis    | 18-19 | head pitch               | 54 (nose tip), 16 (chin)              |
| contour | 20-23 | head yaw                 | 4, 8, 28, 24 (symmetric pairs)        |

Conventions, all enforced by tests:

* "left" and "right" mean image space, so our left is the viewer's left and
  the subject's right. WFLW's readme names index 96 the "right pupil" in the
  subject frame; that is the same physical point as our `left_pupil`.
* Each eye is stored in the standard six point arrangement for the eye
  aspect ratio: corner, upper, upper, corner, lower, lower. With local
  offsets p0..p5, EAR = (|p1-p5| + |p2-p4|) / (2*|p0-p3|), and the formula
  is the same for both eyes. MAR = |ours16-ours17| / |ours14-ours15|.
* NME is normalised by the distance between the outer eye corners, our
  points 0 and 9.

## Milestones

1. Done. Repo scaffolding, config system, WFLW loader, the 98 to 24
   mapping, and the verification overlays. The mapping was checked
   statistically against the real annotation file and confirmed by eye.
   See [docs/milestone1_verdict.md](docs/milestone1_verdict.md).
2. Done. Crop cache. Verified on real WFLW: 7,500 train and 2,500 test
   faces, label round trip error 0.00005 px, 167 MB total.
3. Done. Haar face detection with a calibrated mapping from the raw Haar
   box to the model's crop box. Findings, the pose detection problem, and
   the tuning that followed are in
   [docs/milestone3_notes.md](docs/milestone3_notes.md).
4. Done. Model, training loop, augmentation, checkpointing and resume.
   The Wing loss run is the baseline: best val NME 5.386% at epoch 113.
   See [docs/milestone4_notes.md](docs/milestone4_notes.md).
5. Done. Test set evaluation. Baseline locked: Wing width 32, 2.37 MB,
   7.346% overall test NME, 4.089% on no-flag faces. All three runs and
   the reading of the numbers are in
   [docs/milestone5_results.md](docs/milestone5_results.md).
6. In progress. The `LandmarkDetector` interface with two implementations,
   ours and MediaPipe mapped to the same 24 points. The first full-frame run
   exposed a framing problem and two proxy inversions in earlier decisions;
   the diagnosis and the fix sequence are in
   [docs/milestone6_findings.md](docs/milestone6_findings.md).
7. Planned. The frame to frame stability harness comparing both.

## Running things

Everything runs from configs, no hardcoded paths. All the heavy work
happens on Kaggle where the WFLW dataset and the crop cache are attached as
inputs; the notebooks in `notebooks/` are thin wrappers that clone this
repo, print the commit they are on, and call the scripts. Development and
tests run locally on CPU.

Milestone 1, mapping verification:

```bash
python scripts/verify_layout.py --config configs/layer1_base.yaml --split test
python scripts/visualize_mapping.py --config configs/layer1_base.yaml --split test
```

Milestone 2, build the crop cache (decodes every image once, writes uint8
crops plus crop space labels, checks its own output, renders previews):

```bash
python scripts/build_crop_cache.py --config configs/layer1_base.yaml --split both
```

Milestone 3, detector verification and the pose sweep:

```bash
python scripts/verify_haar_pipeline.py --config configs/layer1_base.yaml --split test
python scripts/sweep_haar_pose.py --config configs/layer1_base.yaml --split test
```

Milestone 4, training. Checkpoints are written every epoch and carry
optimiser, scheduler and RNG state, so `--resume` continues a killed run
exactly where it stopped (`tests/test_resume.py` proves the resumed metrics
match an uninterrupted run). Metrics append to a CSV each epoch and the
loss curves render to a PNG, which stands in for TensorBoard without adding
a dependency.

```bash
python train.py --config configs/layer1_base.yaml
python train.py --config configs/layer1_base.yaml --resume
```

Milestone 5, evaluation on the test set, plus the export step that strips a
checkpoint down to deployable weights:

```bash
python scripts/evaluate.py --config configs/layer1_base.yaml --checkpoint <path>/best.pth
python scripts/export_weights.py --config configs/layer1_base.yaml \
    --checkpoint <path>/best.pth --out landmarks24.pt
```

Milestone 6, the swappable detectors and the ablation run:

```bash
python scripts/compare_detectors.py --config configs/layer1_base.yaml --split test
```

`dms_layer1/detect/interface.py` defines `LandmarkDetector`: a full frame
in, 24 (x, y) points in frame coordinates out (schema order), or None when
no face. `OurLandmarkDetector` is the deployment path from milestone 3
(largest Haar face, calibrated crop box, our model, coordinates mapped
back). `MediaPipeLandmarkDetector` wraps the Tasks API `FaceLandmarker` and
selects the 24 indices defined in `configs/landmarks_24.yaml`, so both
implementations emit identical semantics. The comparison script renders
mapping-verification overlays for MediaPipe (checked by eye before the
mapping is trusted), then reports detection rates, NME on matched and on
jointly matched faces, a per-point cross-detector offset table, the
GT-box versus Haar-box price for our model, same-machine timing, and
footprints.

MediaPipe's legacy `solutions` Face Mesh, which this wrapper originally
used, no longer exists in the package, and the last version that has it
cannot be installed alongside Kaggle's tensorflow. The Tasks API replaces
it. Its model is a `.task` bundle rather than files inside the wheel:
`assets/face_landmarker.task` is vendored, and
`detector.mediapipe.model_path` takes an explicit path or `auto`, which
searches the same places the crop cache and the weights are searched for.
The pupil indices are iris points, which exist only in the 478-point
bundle; the wrapper checks and says so rather than failing obscurely.
`scripts/verify_mediapipe_mapping.py` re-verifies the 24 indices against
ground truth, per point, and also reports whether a different mesh index
would have been closer. `docs/dependency_notes.md` records the version
history alongside the same finding for OpenCV 5.

Most scripts also take `--synthetic`, which runs them on generated
schematic faces. That exists so the code paths can be exercised on a
machine without the dataset. It is clearly labelled in the output and is
not a substitute for the real runs.

## Reproducibility notes

The global seed lives in the config (`seed: 42`) and feeds sampling,
augmentation and training. Every script that writes results also writes a
`config_used.yaml` next to them, so any number can be traced back to the
settings that produced it. The notebooks print the git commit they run
from. The landmark model trains from random initialisation; not using
pretrained weights is part of the project's argument, not an oversight.

Two practical notes. WFLW has no subject IDs, so the validation split is a
seeded random split of faces; the subject independence concern applies to
the later in-cabin recordings, not to WFLW. And `opencv-python` is pinned
below version 5 because OpenCV 5 removed the Haar cascade API this project
depends on; the cascade XML files are vendored in `assets/` since even 4.x
wheels do not always ship them.

## Layout

```
configs/      layer1_base.yaml (all settings), landmarks_24.yaml (point schema)
dms_layer1/   the package: data loading, crops, cache, detector, model,
              training, evaluation, visualisation
scripts/      one script per verification or build step
tests/        python tests/run_tests.py (plain python, pytest compatible)
notebooks/    thin Kaggle notebooks, one per milestone
assets/       vendored OpenCV cascade files
docs/         milestone records with the numbers cited in the report
```

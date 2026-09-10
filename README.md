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

## Open defect: the eye aspect ratio does not respond to eye closure

Read this before the numbers below. On live webcam frames our EAR moves less
than 4% between open and closed eyes, where the formula halves when the lids
halve. Per eye the picture is worse than the mean: closing takes the left eye
from 0.196 to 0.214 and the right from 0.246 to 0.213, so the left reads
HIGHER closed than open and both converge on one value. That is a prediction
collapsing onto a fixed eyelid shape rather than a response that is merely
too small.

The leading hypothesis is a dataset gap. WFLW is web photography and people
photograph themselves with their eyes open, so the training labels may
contain almost no closed eyes.
[`scripts/diagnose_ear_response.py`](scripts/diagnose_ear_response.py) tests
that directly and separates it from a model failure, by reporting the ground
truth EAR distribution alongside the slope of predicted against true EAR for
our model and for MediaPipe on the same faces.

Two consequences for what is claimed elsewhere in this repo, both recorded in
[docs/milestone7_design.md](docs/milestone7_design.md). The milestone 7
"zero false crossings" result and its margin figure are suspended, because a
signal that does not move cannot cross a threshold and a constant has
infinite margin; the jitter, box, dropout and refinement results do not
depend on the EAR moving and stand. And milestone 6's reading that the
landmark gap does not reach the decision rests on the same crossing result,
so it is unresolved rather than established.

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
5. Done. Test set evaluation. Loss comparison locked on Wing at width 32,
   2.37 MB: 7.346% overall test NME, 4.089% on no-flag faces. Those figures
   belong to the PRE-FRAMING checkpoint, which is not what ships. Milestone 6
   found that model could not tolerate deploy framing and retrained it with a
   wider framing envelope, and the full-split milestone 5 protocol has not
   been re-run on the retrained weights, so no directly comparable number
   exists yet. What is measured for the shipped model is in milestone 6: on
   the 410 test faces every path found, 6.388% on ground truth boxes and
   7.421% end to end. Re-running `scripts/evaluate.py` with the framing aware
   weights would close the gap in one command. All three loss runs are in
   [docs/milestone5_results.md](docs/milestone5_results.md).
6. Done. The `LandmarkDetector` interface with two implementations, ours and
   MediaPipe mapped to the same 24 points. The first full-frame run exposed a
   framing problem and two proxy inversions in earlier decisions; the
   diagnosis and the fix sequence are in
   [docs/milestone6_findings.md](docs/milestone6_findings.md), and the final
   comparison with its per-group table is in
   [docs/milestone6_results.md](docs/milestone6_results.md). Headline: on
   identical faces we are close on the aggregate and behind MediaPipe on
   every group where the two annotation conventions agree.
7. Done, with one result suspended. The frame to frame stability harness
   comparing both, its design recorded before any numbers existed in
   [docs/milestone7_design.md](docs/milestone7_design.md). Refinement damps
   jitter by roughly twelve times and the box, dropout and jitter results
   stand; the EAR false crossing result is suspended pending the responsiveness
   defect above.

## What actually runs

The values below are what `configs/layer1_base.yaml` ships, and every one was
chosen on a measurement rather than a default. They are here because the
config comments explain each decision at length and a reader should be able
to see the operating point without reading them.

| setting | value | chosen on |
|---------|-------|-----------|
| `face_detector.scale_factor` / `min_neighbors` | 1.05 / 5 | the pose sweep, then re-scored by the selection diagnostic |
| `face_detector.selection` | `confidence` | about 13 points over largest, and flat across detector settings |
| `face_detector.box_scale` | 1.60 | end to end NME with refinement on, not stage 1 alone |
| `face_detector.box_shift_y` | 0.00 | the same grid |
| `face_detector.box_shift_x` | 0.00 | measured and rejected: the horizontal bias appears only with pose, so no constant removes it |
| `detector.refine_stages` | 2 | worth 6.609 NME points; a third stage adds 0.126 |
| `detector.refine_expand` | 1.60 | the refinement sweep and the 24 point box sweep agree |
| `detector.cross_expand` | 1.60 | the same, for the component swap row |
| `model.width` / `input_size` | 32 / 112 | 0.59M parameters, 2.37 MB fp32 |
| `train.loss` | `wing` | 5.386% best val NME against L2's 7.285%. This line read `l2` in the config until milestone 7, so a retrain would have reproduced the losing run |
| `train.augment.framing` | 0.85 to 1.60 | the deploy framing envelope measured in milestone 6 |

The stage 1 box is deliberately one of the worst in the calibration grid on
its own: (1.60, 0.00) scores 16.488% at stage 1 and 9.879% end to end, while
the stage 1 winner (1.30, 0.11) scores 10.820% and loses after refinement.
Stage 1's job is to contain the face, and stage 2 rebuilds the framing from
the points found inside it.

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
no face. `OurLandmarkDetector` is the deployment path: the Haar detection
with the highest cascade confidence, a calibrated crop box, our model, and
the coordinates mapped back. Selection is by confidence rather than by size.
Milestone 6 measured "largest box" picking the subject on only 27.67% of
single-face WFLW frames at the shipped detector settings, because loose
settings emit oversized spurious boxes; confidence beats it by about 13
points and, unlike largest, is flat across detector settings. `MediaPipeLandmarkDetector` wraps the Tasks API `FaceLandmarker` and
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

# Milestone 6 findings: the full-frame path, and two proxy inversions

The first milestone-6 run produced numbers that could not be true together:
27.71% detection against milestone 3's 72.08% on the same cascade, settings
and split, and 29.2% NME through the Haar path against 6.8% on ground truth
boxes for the same faces. `scripts/diagnose_deploy_gap.py` separated the
causes with magnitudes.

## The sanity gate, which is what makes the rest interpretable

Ground truth box NME measured **through the live detector path** was 6.816%,
against 7.346% from the milestone-5 protocol on the full split. The model,
the weights loading, the crop extraction and the coordinate round trip are
all correct. Everything below is about which box is chosen and how it frames
the face, not about the landmark model.

The any-box detection rate reproduced at 72.48% against milestone 3's
72.08%, confirming the two runs measure the same cascade behaviour and that
the 27.71% was measuring something else.

## Finding 1: framing, the dominant error term

NME as a function of framing factor k, where k = 1.0 is the canonical box
(reference_expand 1.3) that training and the milestone-5 protocol use:

| k    | 1.00   | 1.10   | 1.20   | 1.30    | 1.40    | 1.56    | 1.70    | 1.90    |
|------|-------:|-------:|-------:|--------:|--------:|--------:|--------:|--------:|
| NME  | 6.714% | 7.949% | 9.862% | 12.933% | 16.616% | 24.814% | 33.428% | 47.248% |

Training covered k in [0.87, 1.18]. Deploy framing at the 1.75 calibration
sits at about 1.56. The model was being asked to work in a regime it had
never seen, and error grows roughly linearly with k once outside the trained
range. This single fact accounts for most of the 22 point gap.

## Finding 2: containment inverts against accuracy

The milestone-3 calibration was chosen to maximise landmark containment.
End NME on the same faces, with face choice excluded:

| box_scale | containment | NME     |
|-----------|------------:|--------:|
| 1.12      | 81.68%      | 13.288% |
| 1.45      | 98.47%      | 18.285% |
| 1.75      | 100.00%     | 28.997% |

Best containment, worst accuracy, monotonically. Containment asks whether
the answer is inside the picture. It cannot ask whether the model can read a
picture framed that way, and a wider box buys containment by making the face
smaller, which is precisely what degrades the model. The lesson is the same
one milestone 1 taught with the absolute chin tolerance, one level up: a
proxy that is easy to measure is not automatically aligned with the
objective, and the check is to measure the objective once, early, on the
cheapest possible sample.

## Finding 3: two-stage refinement is not a fix on its own

Predicting once at the current calibration, rebuilding a training-style box
from those points, and predicting again moves 29.0% to 25.2%, against a 5.5%
ceiling on the same faces. It cannot rescue a first stage that is far
outside the model's competent range, because the box it rebuilds is only as
good as the points it starts from. Worth revisiting after retraining, when
stage one lands close enough for stage two to sharpen it.

## Finding 4: selection, the second proxy inversion

The largest Haar box is the annotated subject on only 27.67% of
single-face images. WFLW crowd scenes explain part of the all-images gap but
cannot explain the single-face number, since on a single-face image the
largest box should usually be the face.

The suspect is the milestone-3 tuning itself. Those settings (scale_factor
1.05, min_neighbors 2) were chosen to maximise ANY-box matching, a metric
structurally blind to spurious detections, and they raised unmatched boxes
from 1.81 to 5.21 per frame. Deployment picks one box per frame, so every
spurious oversized box is a chance to pick wrong. `diagnose_face_selection.py`
measures what the winning boxes actually are and compares selection rules
(largest, cascade level weight, most central, and size-sanity variants)
against an oracle, crossed with detector settings. This matters for the
cabin as well as for WFLW: a headrest, a seat back or a window frame can
produce the same oversized box that a crowd scene does.

## The fix, and why it needs a cache rebuild first

Widening the framing augmentation is the right attack because it addresses
the cause rather than trading accuracy for containment. But the milestone-2
cache stores only the canonical box (crop_expand 1.3), and simulating a
wider framing from it means zooming out into content the cache does not
hold. Measured on the current cache: at k = 1.56 about 36% of the input
width would be zero padding, at k = 1.9 about 47%. The model would learn
"face surrounded by black", while deployment shows a face surrounded by real
background. Wide framing augmentation therefore requires a cache that stores
real context.

The geometry is now split into two config values so this cannot be confused
again:

* `preprocess.reference_expand` (1.3) is the canonical framing. k = 1.0, the
  milestone-5 protocol, and the calibration semantics all refer to it. It
  does not change.
* `preprocess.crop_expand` (2.2) is how much context the cache stores.
* `train.augment.framing` ([0.85, 1.60]) is the k range sampled in training,
  expressed against the reference.

The trainer computes `base_frac = reference_expand / crop_expand` and
refuses to run if the configured framing range would run past the cached
context, naming the crop_expand needed. A silently padded seven minute run
is exactly the failure this project keeps designing against.

Cache cost at crop_expand 2.2 and cache_size 224: about 500 MB for both
splits, against 167 MB before, and the canonical region stays at 132 px
against 128 px before, so k = 1.0 numbers remain comparable across the
rebuild.

## Results after the framing-aware retrain

Scale response, old model against the framing-aware one, on the same faces:

| k    | 1.00   | 1.10   | 1.20   | 1.30    | 1.40    | 1.56    | 1.70    | 1.90    |
|------|-------:|-------:|-------:|--------:|--------:|--------:|--------:|--------:|
| old  | 6.714% | 7.949% | 9.862% | 12.933% | 16.616% | 24.814% | 33.428% | 47.248% |
| new  | 8.598% | 8.480% | 8.851% |  9.370% | 10.112% | 11.825% | 14.131% | 19.380% |

The cliff is gone. The new model gives up 1.9 points at canonical framing
and gains 13 at deploy framing, crossing over just past k = 1.1. Its own
minimum sits at about k = 1.1 rather than 1.0, which is the training
distribution's mean (framing [0.85, 1.60], mean 1.22) pulling the sweet spot
wide. Re-centring the augmentation range on the realized deploy framing is
an available further experiment, not a recommendation yet.

Calibration grid, end NME, old against new:

| box_scale | old     | new     |
|-----------|--------:|--------:|
| 1.12      | 13.288% | 13.957% |
| 1.30      | 14.894% | 13.528% |
| 1.45      | 18.285% | 13.982% |
| 1.60      | 22.876% | 14.674% |
| 1.75      | 28.997% | 16.324% |

Peak accuracy is essentially tied (old 12.745% at 1.12/0.13, new 13.099% at
the same point). The difference is the shape: the new model is flat from
1.12 to 1.45 within 0.5 points, where the old one fell 15 points across the
same span.

**Decisions.** Adopt the framing-aware model, not because it wins on peak
accuracy (it does not) but because it removes the sensitivity to a
calibration this project has already got wrong once on a proxy. Set the
calibration to 1.30 / 0.13, the centre of the flat region rather than its
edge, so per-face variation in the Haar-to-face ratio moves along the flat
part instead of off a cliff. Adopt two-stage refinement, which now pays:
16.324% to 12.599% against a 7.768% ceiling, where on the old model it moved
28.997% to 25.216% and was not a fix.

Refinement and the stage-1 calibration are one joint decision, since the
rebuilt box is only as good as the points that build it, and the earlier
table measured refinement only from the 1.75 box. The diagnostic now crosses
the full calibration grid with refinement, sweeps the rebuilt box's own
framing (which need not be canonical, given the k = 1.1 optimum), and checks
a third stage for convergence.

## The remaining gap, and what it is not

Best end to end is about 12.6% against a 7.768% ceiling on the same faces.
That residual is not face selection: sections 3 and 4 run only on images
where the largest box already hits the target, so selection is excluded by
construction. It is box geometry, and the diagnostic now decomposes it into
the two ways a deploy box differs from the canonical one, by evaluating four
variants on the same faces: the ground truth box, the ground truth centre
with the deploy size, the deploy centre with the ground truth size, and the
deploy box. Size cost and centre cost are then separable, with the realized
framing and centre offset distributions printed beside them.

Face selection is a separate and now larger loss, unchanged at 27.67% on
single-face images and not yet diagnosed.

## Finding 4 resolved: selection, and where the classical front end caps

The oversized-box hypothesis was wrong. On single-face failures, 77.2% of
the winning boxes are DISJOINT from the face and only 3.3% contain it, with
a median of 6 detections per failing frame (up to 32). The cascade usually
does find the face: it was present in 281 of 368 failures, ranking a median
third by size. Its level weight on failing frames is 2.67 ABOVE the winner's,
so the cascade's own confidence ordering already knows which box is the face
and the largest-box rule was throwing that information away.

Selection rules on single-face images: largest 29.23%, confidence 42.31%,
central 26.35%, oracle 71.92%. Size sanity does nothing at any threshold,
which follows from the boxes being disjoint rather than oversized. Across
detector settings the confidence rule is flat near 40% while largest swings
from 27.67% to 38.00%, so confidence is robust to the tuning rather than
dependent on it. **Adopted: confidence selection**, worth about 13 points for
no retraining and no extra computation.

Settings, re-scored for the one-box-per-frame objective:

| sf, nb  | boxes/img | largest | confidence | oracle | ms  |
|---------|----------:|--------:|-----------:|-------:|----:|
| 1.05, 2 | 6.21      | 27.67%  | 39.67%     | 72.83% | 226 |
| 1.05, 5 | 3.77      | 35.50%  | 40.50%     | 69.17% | 229 |
| 1.10, 5 | 2.62      | 38.00%  | 38.50%     | 61.83% | 116 |

At n = 600 the standard error on these rates is about 2.0 points, so the
three confidence numbers are statistically indistinguishable (largest
pairwise difference 2.0 points, SE 2.8). The oracle differences are real
(11.0 points, SE 2.7). The choice therefore falls to the secondary criteria,
and **1.05 / 5 is adopted**: it ties for the best realized rate, costs the
same as 1.05 / 2, halves the spurious boxes, and keeps the oracle within 3.7
points of the maximum. Oracle headroom is worth protecting because temporal
tracking in a cabin re-acquires from the candidate set, and that is a rule
WFLW stills cannot test.

The milestone-3 hypothesis was half right. Loose settings do create the
spurious boxes that break largest-box selection, but tightening trades real
faces for fake ones: the oracle falls from 72.83% to 61.83%. There is no
setting that fixes this by itself.

## The ceiling, and how to state it honestly

On single-face WFLW images the decomposition is roughly:

* 28% the cascade never finds the face at all,
* 30% it finds the face but confidence selection still picks another box,
* 42% the pipeline acquires the target face.

The plain statement, which is the one to lead with: **with a Haar cascade
front end at any setting tested, target-face acquisition on the WFLW test
set caps near 40%, and even a perfect selection rule would cap near 72%.**
That is a property of the classical detector, not of the landmark model, and
it is the concrete reason production driver monitoring systems use learned
detectors. Say that first, without softening it.

Then bound its transferability, in both directions, because the honest
answer is that WFLW is harsher than a cabin AND part of this would follow
the pipeline into a cabin:

* Does not transfer. WFLW is web photography: most test images carry at
  least one of the six difficulty flags, and images contain faces that are
  not annotated, so a box that looks like a failure here may be a correct
  detection of a different real face. A driver-facing camera sees one
  cooperative subject at a roughly fixed distance under controlled framing.
* Does transfer. The cascade fires on non-face background, which a cabin
  supplies in the form of headrests, seat backs, window frames and door
  pillars. A front passenger is a genuine second face, so selection is a
  real cabin problem, not only a crowd-photo artifact. And the 28%
  no-detection share concentrates on exactly the turned heads that
  inattentiveness detection is about.
* Not yet known, and it decides how much weight each bullet carries: are
  the winning disjoint boxes real unannotated faces or background false
  positives? Section 3b of `diagnose_face_selection.py` now judges each
  winning crop with MediaPipe as an independent detector, with the true-face
  box run through the same judge as a control. Mostly-faces supports the
  dataset reading; mostly-not-faces supports the detector reading. Report
  the measured split rather than either intuition.

What survives regardless: the landmark model is evaluated on ground-truth
boxes under the standard WFLW protocol, so the acquisition ceiling does not
contaminate the landmark comparison, which is the project's actual
contribution. What does not survive: any claim that this classical full
pipeline is deployable as it stands. Both belong in the report.

## What this changes about the MediaPipe comparison

It changes it fundamentally, and the design has been adjusted rather than
the result explained away afterwards. MediaPipe carries its own learned
detector, which is not subject to any of the above, so a naive full-frame
comparison would be dominated by the detectors and would mostly restate a
known result (a learned detector beats a Haar cascade) while letting that
detector gap read as a landmark-model gap. The comparison now runs three
conditions:

| condition       | face from  | landmarks from | isolates                    |
|-----------------|------------|----------------|-----------------------------|
| ours            | Haar       | our model      | the deployable classical stack |
| ours_on_mp_box  | MediaPipe  | our model      | one component swapped       |
| mediapipe       | MediaPipe  | MediaPipe mesh | the reference stack         |

Row 1 against row 2 is the face detector's contribution. Row 2 against row 3
is the landmark model's contribution on identical inputs, which is the
project's question. `dms_layer1/detect/cross.py` implements row 2 behind the
same `LandmarkDetector` interface, so nothing downstream knows the
difference, and it is a legitimate deployable configuration for anyone
willing to ship MediaPipe's detector. The ground-truth-box protocol
(milestone 5) remains the fourth, detector-free comparison.

Timing needs the same care. Our landmark model runs in about 2.9 ms while
Haar detection costs 116 to 229 ms on the same machine, so at the pipeline
level our model's size advantage is invisible and MediaPipe's faster
detector will dominate any full-pipeline timing. Report model-only and
whole-pipeline timing as separate numbers, and state whether detection runs
every frame or is amortised by tracking, or the speed claim will be wrong in
one direction or the other.

## Guard added after a wasted run

A diagnostic run auto-loaded stale uploaded weights and reproduced the old
numbers exactly, which is indistinguishable from a real result in the
output. Checkpoints and exports now record the framing envelope, loss and
seed they were trained with, and every script that loads a model prints that
line before anything else. Weights trained before this record say
"framing UNRECORDED" rather than staying silent.

## Sequence from here

1. Rebuild the cache (crop_expand 2.2, cache_size 224). Done.
2. Retrain with framing [0.85, 1.60]. Done; results above.
3. Re-run the deploy gap diagnostic and choose the calibration on end NME.
   Done: 1.30 / 0.13 adopted, with refinement on. The re-run of the extended
   sections 4 and 5 confirms the joint calibration-plus-refinement choice
   and prices the residual.
4. Run the selection diagnostic and choose a selection rule, and if the
   settings grid says so, revisit the cascade settings under the
   one-box-per-frame objective. This is now the largest open loss.
5. Only then re-run `compare_detectors.py`, once, on a fixed path.

## The MediaPipe side had to be rebuilt before it could be compared

The first milestone-6 run produced one row instead of three because
MediaPipe never loaded. The legacy `solutions` Face Mesh the wrapper was
written against had been removed from the package, and the version pin that
was supposed to guarantee it (`>=0.10,<1`) did not: removal happened at
0.10.30, not at 1.0. Pinning exactly to 0.10.21, the last release that ships
solutions, was correct about the API and unusable in practice, because
0.10.21 needs protobuf < 5 and Kaggle's tensorflow needs protobuf >= 5.

The wrapper is now on the Tasks API, which needs no protobuf pin, loads its
model from a vendored `.task` bundle, and gets the iris points (the pupil
indices) from the bundle rather than from a `refine_landmarks` flag. The
24-index mapping was re-verified on the new mesh rather than assumed to
carry over: `scripts/verify_mediapipe_mapping.py`, contour points identical
to within 0.1 points of IOD, everything else within the difference between
two model generations. Full history and measurements in
`docs/dependency_notes.md`.

For the ablation this is a note about reproducibility, not about accuracy:
the comparison runs against whatever MediaPipe ships today, and what it
ships changed under the project. The report should name the exact version
and bundle used, which the comparison script now prints and records.

## The mapping question: nine flagged indices, and why the four eyelid ones stay

The mapping verification on the test split (n = 80 usable faces of 300) put
every point inside tolerance and the pupils at 1.79% of IOD, but its control
section flagged nine indices where a different mesh vertex sits closer to
WFLW's ground truth. Four are eyelid points, which is the group the EAR that
Layer 2 consumes is computed from:

| our point | configured | offset | closest | offset | gap |
|-----------|-----------|--------|---------|--------|-----|
| left_eye_upper_inner  | 158 | 5.16% | 157 | 2.90% | 2.26 |
| left_eye_lower_outer  | 144 | 4.94% | 163 | 1.91% | 3.03 |
| right_eye_upper_inner | 385 | 5.06% | 384 | 3.26% | 1.80 |
| right_eye_lower_outer | 373 | 4.70% | 390 | 2.33% | 2.37 |

The worry this raises is legitimate and worth stating in the report: if
MediaPipe runs on worse eyelid indices than it could, our model wins on the
group the project is about for a reason that has nothing to do with either
model.

### What the alternatives are

MediaPipe publishes its mesh topology, so this is answerable exactly rather
than by inspection. Both eyes are a 16-vertex contour ring. Walking the
image-left ring from the outer corner:

    33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7
     0    1    2    3    4    5    6    7    8    9   10   11   12   13   14  15

All four alternatives are on that same ring, exactly one vertex from the
configured index: 158 to 157 is one step toward the inner corner along the
upper lid, 144 to 163 one step toward the outer corner along the lower lid,
and the other eye mirrors both. They are the same anatomy sampled at a
slightly different point along the lid, not different features. WFLW's six
eye points sit slightly wider along the lid than MediaPipe's, and that
difference is what the control is measuring.

### Why moving to them would cost more than it gains

EAR is defined on chords: `(|p1 - p5| + |p2 - p4|) / (2 |p0 - p3|)`. It reads
the numerator as lid separation, which is only true when each chord is
perpendicular to the corner-to-corner axis. In ring positions, the configured
sextet is exactly symmetric about that axis:

| chord | configured | positions | alternative | positions |
|-------|-----------|-----------|-------------|-----------|
| p1 to p5 | 160, 144 | 3 and 13, mirrored about 8 | 160, 163 | 3 and 14, off by one |
| p2 to p4 | 158, 153 | 5 and 11, mirrored about 8 | 157, 153 | 6 and 11, off by one |

Both configured chords are vertical by construction. Each proposed swap tilts
one of them, so the numerator starts picking up eye width along with eye
opening. The configured set is the canonical six-point EAR selection for this
mesh for exactly that reason.

So the two candidate mappings optimise different things. The proximity
mapping minimises distance to WFLW's annotation convention; the configured
mapping preserves the geometry the aspect ratio is defined on. Fitting the
first on the evaluation split would also be fitting the mapping to the test
set.

### The decision rule adopted

1. The schema mapping stays chosen on semantics and stays fixed. Its offset
   from WFLW's convention is reported, not minimised.
2. The alternative is measured rather than argued about.
   `scripts/verify_mediapipe_mapping.py` now reports, for every flagged
   index, whether the closer vertex is on the same feature ring and how many
   vertices away, a paired per-face gain with its standard error, and what
   the swap does to that eye's EAR chord geometry. Section 4 reports both
   mappings side by side on the two things they are used for: NME over all
   24 points, and EAR agreement with ground truth (mean absolute difference,
   correlation, chord skew).
3. Any index change must be derived on the TRAIN split. The script says so
   when it is run on test.
4. The ablation itself carries the sensitivity check. Setting
   `mediapipe.indices_alt` in the schema adds a `mediapipe_alt_map` row to
   `scripts/compare_detectors.py`, scored off the same mesh, and the paired
   section prints our margin under both mappings. If the sign and the
   conclusion hold under both, the index choice is not what decided the
   comparison, and the report can say so with a number.

### Sample size, and what the 80 usable faces are

Of 300 test images, MediaPipe found no face in 178 and a different face in
42. That is a property of the Tasks bundle's short-range face detector rather
than a bug: it is built for faces that fill a reasonable part of the frame,
and WFLW is web photography full of small faces in group shots. The
verification script now prints the median target-face size for found versus
missed faces so the population the medians describe is visible.

Two consequences. For the mapping question, the pairing is per face and the
differences are systematic, so the standard error is small even at n = 80;
the script prints it rather than leaving it to be assumed. For the ablation,
MediaPipe's detection rate on full WFLW frames is measuring a short-range
detector against a long-range dataset, which is worth one sentence in the
report next to the rate itself.

### Contour

Contour keeps 234, 58, 454, 288. The closer alternatives are interior mesh
vertices on the cheek, not points on the face oval at all, so they are not
contour points in any semantic sense. The group exists to give Layer 2 a
left/right yaw asymmetry rather than a precise measurement, and the offset
from WFLW's contour parameterisation is a documented number rather than
something to chase.

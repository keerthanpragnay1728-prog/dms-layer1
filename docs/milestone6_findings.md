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

## The guard's own message was wrong

The stale-weights guard told a user their framing-aware model had been
trained on the narrow envelope. The export was made before `export_weights`
carried `train_meta` (added in d88db7a), so the file genuinely had no record,
and `describe_weights` filled the gap with "assume the narrow [0.87, 1.18]
envelope". That assumption was false for this file and stated as fact.

Fixed three ways. The message now distinguishes no record at all from a
record with no framing entry, and in neither case guesses an envelope: it
says the file cannot describe itself and points at the run's
config_used.yaml. `export_weights` warns loudly when the checkpoint it is
exporting has no record, so a provenance-less export cannot be produced
quietly again. And `scripts/export_weights.py --restamp` adds the record to
an existing file from a named config, marked `stamped_after_the_fact` so it
prints as "stamped from <config>, not by the trainer" and can never be
confused with the trainer's own stamp.

The general lesson for the report: a provenance guard that fills a gap with
an assumption is worse than one that says the gap exists. The first time this
guard fired on a file it did not understand, it produced a confident false
statement about the model under test.

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

## The mapping question: nine flagged indices, one falsified argument

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

### The chord-alignment argument, made and then falsified

This section originally argued that the configured sextet keeps both EAR
chords perpendicular to the corner axis and that every proposed swap tilts
one of them. The full-split run measured that claim and it is half wrong.
The wrong half is recorded here rather than deleted, because the corrected
justification only makes sense against it.

The argument was: EAR is `(|p1 - p5| + |p2 - p4|) / (2 |p0 - p3|)`, which
reads its numerator as lid separation only when each chord is perpendicular
to the corner-to-corner axis; in ring positions the configured sextet is
symmetric about that axis (chords at 3-13 and 5-11, both mirrored about the
inner corner at 8) while each swap moves one endpoint by a vertex; therefore
every swap tilts a chord.

Measured on 624 faces, one swap at a time:

| swap | lid | chord skew before | after | |
|------|-----|------------------|-------|--|
| 158 to 157 | upper | 0.0901 | 0.1057 | worse, as the argument predicted |
| 144 to 163 | lower | 0.0901 | 0.0713 | BETTER, contradicting it |
| 385 to 384 | upper | 0.0904 | 0.1017 | worse, as predicted |
| 373 to 390 | lower | 0.0904 | 0.0707 | BETTER, contradicting it |

The split is clean: upper-lid swaps tilt the chord, lower-lid swaps
straighten it. So the configured sextet is not the minimum-skew choice, and
the prediction that any swap must tilt a chord is false.

The mistake was treating a ring index as a geometric coordinate. It is a
topological one. Both lids carry seven vertices between the corners, but they
are distributed along arcs of different length, so mirrored ring positions do
not land at the same place along the eye axis. Measured directly on a fitted
mesh, the supposedly mirrored pair 160 and 144 sits at 0.214 and 0.249 of eye
width from the outer corner: the configured chord already leans by about
0.035 eye widths before anything is swapped. Nothing about verticality
follows from the index, in either direction.

What survives is the part that came from topology alone, which the
measurement does not touch: the four alternatives are neighbouring vertices
on the same 16-point eye ring, so they are the same anatomy sampled at a
different point along the lid, and the contour alternatives are interior mesh
vertices on no named feature. Both were checked against MediaPipe's published
connection sets and are confirmed by the full-split run.

### The corrected justification: correlation, not geometry

Section 4 on the full split:

| mapping | mean absolute EAR error vs gt | correlation with gt | chord skew |
|---------|------------------------------|--------------------|------------|
| ground truth | 0.0000 | 1.000 | 0.0325 |
| configured | 0.0563 | 0.760 | 0.0902 |
| proximity-optimal | 0.0442 | 0.723 | 0.0809 |

The proximity mapping is closer in absolute EAR and less skewed, and
correlates worse. Layer 2 calibrates per driver, so a constant offset is
absorbed and the relationship with the truth is not. Correlation is therefore
the criterion, and the configured mapping is kept on it.

That makes chord skew a diagnostic rather than a decision rule. It has now
failed twice as a predictor of the thing that matters: it mispredicted the
direction of two of the four swaps, and the mapping with the lower skew is
the one that correlates worse. It stays in the output because it explains
what an index change does to the geometry; it does not decide anything.

One caution attached to the number above. A gap of 0.760 against 0.723 is two
point estimates, and treating a point estimate as a fact is the same error
that produced the falsified argument. The script now reports the correlation
difference against the configured mapping as a paired bootstrap over faces
with a 95% interval. If that interval excludes zero, the justification is
"configured correlates better"; if it includes zero, the justification is
"the two are indistinguishable on the criterion that matters, so the mapping
chosen on semantics and documented in the schema stands". Either way the
mapping does not change, but the sentence in the report differs and only the
interval says which one is true.

### The decision rule adopted

1. The schema mapping stays chosen on semantics and stays fixed, and is kept
   on EAR correlation with ground truth rather than on proximity to WFLW's
   annotation convention. Its offset from that convention is reported, not
   minimised.
2. Candidates are measured rather than argued about.
   `scripts/verify_mediapipe_mapping.py` reports, for every flagged index,
   whether the closer vertex is on the same feature ring and how many
   vertices away, a paired per-face gain with its standard error, and what
   the swap does to that eye's chord geometry. Section 4 scores every
   candidate mapping (the schema's, any declared alternatives, and the
   proximity-optimal one) on NME over 24 points, on eyelid NME, and on EAR
   against ground truth: mean absolute error, correlation with a bootstrap
   interval on the difference, and each chord's skew separately, since the
   outer and inner chords do not respond alike to an index change and
   averaging them concealed exactly that.
3. Any index change must be derived on the TRAIN split. The script says so
   when it is run on test.
4. The ablation itself carries the sensitivity check. `mediapipe.indices_alt`
   in the schema takes one index list or several named ones, and each adds a
   `mediapipe_map_<name>` row to `scripts/compare_detectors.py`, scored off
   the same mesh. The paired section prints our margin under every mapping.
   If the sign and the conclusion hold across all of them, the index choice
   is not what decided the comparison, and the report can say so with a
   number.
5. The bound is reported whatever is chosen. On the full test split,
   switching every index to the proximity-optimal one moves MediaPipe's NME
   by 1.952 +- 0.041 points. That is the most the mapping choice can flatter
   or hurt either detector, and it belongs in the report next to the
   comparison itself.

### The mixed mapping, and how it would have to be decided

The measurement suggests a third candidate: configured on the upper lid,
swapped on the lower, which is where the skew actually improves.

    image-left   [33, 160, 158, 133, 153, 163]
    image-right  [362, 385, 387, 263, 390, 380]

There is no coherence objection to it. All six points stay on the same eye
ring, the two corners still define the denominator, and 163 and 390 are exact
mirrors of each other on their rings, so the two eyes stay comparable and the
flip permutation is unaffected. "Mixing conventions" is not the problem: the
commonly cited sextet is a community convention rather than anything
MediaPipe defines, and any six ring vertices are as legitimate as any other
six.

The objection is procedural. The mixed set exists because per-point results
on this data were inspected and the winners kept, which is selection on the
evaluation set. It also inherits its motivation from skew and proximity,
the two criteria that have now each failed to track EAR correlation. So there
is no reason to expect it to help, and the honest prediction is that it lands
indistinguishable from the configured mapping on correlation.

If it is to be adopted, the procedure is the one already written down:
declare it under `mediapipe.indices_alt` as a named alternative, derive the
decision on the TRAIN split, and adopt only if the bootstrap interval on the
correlation difference excludes zero there. Then report the test numbers
under every mapping. If the interval includes zero, the configured mapping
stands, and the tie goes to the documented semantic choice rather than to the
set that was fitted.

One soft argument on the same side: `[33, 160, 158, 133, 153, 144]` is the
sextet in common use for this mesh, so a reader can check it against other
work. A bespoke set is one more thing to defend, and it should only be
defended if it buys something measurable.

### What MediaPipe's 29% detection rate is measuring

On the full test split MediaPipe found the target face in 624 of 2118 frames:
no face at all in 1217, a different face in 277. The median target face is
0.267 of the shorter image side where it succeeded and 0.115 where it found
nothing, so the failures are concentrated in small faces rather than spread
across the split.

That is the bundle's face detector meeting web photography. The 3.76 MB
`.task` file carries four models, one of which is its own face detector at
0.23 MB, and that detector is built for faces that fill a reasonable part of
the frame. WFLW is full of group shots and crowd scenes where the annotated
face is a small part of a wide image.

So the aggregate detection rates in the ablation are not a detector-quality
comparison, and reporting them as one would repeat the milestone-6 mistake in
a new place. Both front ends fail on WFLW for reasons that are about fit
between their operating assumptions and the dataset, and the two reasons are
different in kind:

* Haar fails by firing on background structure. `diagnose_face_selection.py`
  section 3b judged the winning box on failing frames: 46.2% are a real face
  that is not the target, 53.8% are background, with the judge's control at
  85.6%.
* MediaPipe fails by not seeing small faces at all, which the size split
  above measures directly.

How this is reported:

1. The aggregate rate is stated, because omitting it would be worse, but
   never as a statement about detector quality on its own.
2. `scripts/compare_detectors.py` now prints detection rate per target-size
   band for every detector in the table. That turns one confounded number
   into a curve, and it makes the cabin-relevant regime visible: a driver's
   face fills a large part of the frame, so the top band is the one that
   transfers and the bottom band is the one that does not.
3. The landmark comparison stays on jointly matched faces, which is size
   matched by construction: both detectors had to find the same face for it
   to count. That is why the paired number is the headline and the per
   detector rates are context.
4. The report says plainly that a full-frame WFLW detection rate is a
   dataset-fit measurement for both front ends, and that neither number
   should be read as "this detector would find the driver n% of the time".

### Contour

Contour keeps 234, 58, 454, 288. The closer alternatives are interior mesh
vertices on the cheek, not points on the face oval at all, so they are not
contour points in any semantic sense. The group exists to give Layer 2 a
left/right yaw asymmetry rather than a precise measurement, and the offset
from WFLW's contour parameterisation is a documented number rather than
something to chase.

## Reading the full ablation: three things the first table could not say

The full-split run (n = 2118, jointly matched 408) produced three results
that each need care before they are written up.

### 1. ours_on_mp_box is worse than ours, and the table cannot yet say why

The numbers were 13.776% for the component-swap row against 9.569% for our
own Haar path, which invites the reading that the model is coupled to the
crop geometry our calibration produces rather than merely to face location.
That reading may well be right, but two confounds sit in front of it and both
are now measured rather than reasoned about.

**Different populations.** Each per-detector row is scored on the faces THAT
path found. A Haar cascade only fires on near-frontal faces, so its matched
set is filtered easy; MediaPipe's matched set includes profiles and harder
poses that it can still track and our model cannot. Comparing 9.569% on one
population against 13.776% on another measures the populations as much as the
pipelines. `scripts/compare_detectors.py` section 4 now fixes the faces
first: every row, plus a ground-truth-box ceiling row, scored on the faces
every path found.

**An uncalibrated path.** The Haar path's framing was chosen on end NME:
box_scale and box_shift_y came out of the calibration grid. The component-swap
path was never calibrated. It boxes MediaPipe's 24 points at the nominal
`reference_expand`, and a box around 24 points is not the box around 98 points
the cache was built from, so the same nominal number does not produce the same
framing. The ablation now prints the framing factor each path actually
delivers, measured against the canonical ground-truth box on the same face,
and `diagnose_deploy_gap.py` section 6 sweeps the expand a 24-point box needs,
using ground-truth points so detector error is out of the way.

There is also a specific hypothesis worth testing against that sweep. Training
samples the framing factor uniformly over [0.85, 1.60], so most of the mass
sits near 1.2 rather than at 1.0, and the Haar calibration may simply have
found that centre empirically while the component-swap path sits at the low
edge. If section 6 puts the best 24-point expand near the Haar path's measured
framing, the deficit is a missing calibration and the swap framing survives.

What is true either way, and belongs in the report: a crop-based regressor has
a preferred crop geometry, so every front end that feeds it needs its own
calibration to that geometry. Swapping the detector is not free, and the cost
is a calibration step rather than a line of code. That is a real limitation of
the design and is worth stating plainly. It is a weaker claim than "the model
cannot be treated as independent of its detector", and it is the one the
evidence currently supports.

`detector.cross_expand` is now a config value. Left at `auto` the path runs at
the nominal framing and prints that it is uncalibrated, so the row can never
again be read as a like-for-like swap without someone noticing.

### 2. The headline stays unweighted, and the ex-contour number is reported too

Per-point offsets run 5 to 7% of IOD on eyelids, pupils and mouth, and 15 to
25% on contour, so contour drags the aggregate. Down-weighting the group we
care least about, after seeing the results, is where special pleading starts,
so the headline stays the standard unweighted mean over 24 points. It is what
milestone 5 reports, it is what the WFLW literature reports, and changing the
definition to suit the result would cost more credibility than the points are
worth.

Two things sit next to it. Per-group NME is now in the ablation table for
every row, which is where a reader looks to see that the aggregate is carried
by one group. And the aggregate excluding contour is reported as a secondary
number, with the reason it is not a thumb on the scale: contour is where
WFLW's parameterisation and MediaPipe's face oval disagree most, so removing
it helps MediaPipe more than it helps us. Stating which way the alternative
metric moves the result is what separates a disclosed sensitivity from
special pleading, and here it moves against us.

If a task-weighted number is wanted at all, it has to be defined from what
Layer 2 consumes before the results are looked at, and labelled as a task
metric rather than as NME.

### 3. Detection rate: one sentence, and it does not lead with the aggregate

The aggregate ordering is ours 38.29% against MediaPipe 29.46%, and it
reverses in the regime this project is built for:

| target size, fraction of the shorter side | ours | mediapipe |
|-------------------------------------------|------|-----------|
| 0.00 to 0.15 | 30.0% | 4.5% |
| 0.15 to 0.25 | 38.2% | 42.6% |
| 0.25 to 0.40 | 55.8% | 71.6% |
| 0.40 to 1.01 | 65.6% | 83.8% |

The honest summary for the report:

> On full WFLW frames our Haar front end acquires the target face in 38.3% of
> images against MediaPipe's 29.5%, but that ordering is a property of the
> dataset rather than of the detectors: MediaPipe's near-range detector finds
> 4.5% of faces smaller than 0.15 of the frame and 83.8% of those larger than
> 0.40, so in the size band a driver's face occupies it leads 83.8% to 65.6%,
> and our aggregate lead comes entirely from small faces that a cabin camera
> never sees.

Leading with the aggregate would state a lead that reverses in deployment,
which is the worst kind of claim to have to defend. The size table goes in the
report next to the sentence.

## Section 5 moved the target: the residual is centre, not size

The residual decomposition on right-face images, so face selection is
excluded by construction:

| variant | NME | cost against the ceiling |
|---------|-----|--------------------------|
| A ground-truth box (ceiling) | 7.187% | |
| B ground-truth centre, deploy size | 7.532% | size alone, +0.346 |
| C deploy centre, ground-truth size | 11.106% | centre alone, +3.920 |
| D deploy box, both | 11.209% | both, +4.022 |

Binned, the same thing twice: 6.060% under 0.05 of a face side of centre
error against 47.759% past 0.20, and 8.177% at realized k between 1.10 and
1.30 against 44.083% past 1.50.

Size is almost free and centre carries the residual, which is awkward,
because every calibration decision since milestone 3 has been about scale.
box_scale was swept in milestone 3, re-chosen on end NME in milestone 6, and
swept again against refinement. box_shift_y has only ever been tried at two
values, 0.08 and 0.13, and there has never been a horizontal parameter at
all, even though `calibrate_haar_to_crop` has reported a `box_shift_x`
alongside the others since milestone 3 and nothing could apply it.

Three changes follow.

`haar_to_crop_box` takes a third parameter. `face_detector.box_shift_x` is a
config value, zero until measured, threaded through the detector, the
verification script and the diagnostic so every path uses one transform.

The calibration grid searches the centre properly: seven values of
box_shift_y from 0.00 to 0.20 crossed with the scales, then box_shift_x swept
at the best of those. The axes are searched separably rather than as a third
grid dimension, because section 5b reports the two biases independently and
they act on different components of the same error.

### 5b: whose centre error is it

The question section 5 cannot answer is whether the centre error is the Haar
box landing in the wrong place or our transform putting it there. Section 5b
splits it into the part a constant transform can remove and the part it
cannot:

* **Bias**, the mean signed offset in units of the deploy box side, which is
  exactly what `box_shift_x` and `box_shift_y` multiply. Our calibration
  putting the box in the same wrong place on every face. Removable.
* **Scatter**, the standard deviation of the same offsets. The detector
  landing differently face to face. No constant transform reaches it.

The section reports both, their ratio, the bias-cancelling config values, and
end NME under three conditions on the same faces: as configured, with the
bias removed, and with the ground-truth centre. The first gap is what
calibration is worth. The second is what is left for a better front end or
for refinement, which rebuilds the box from predicted points and is the only
mechanism in the pipeline that can respond per face.

It also splits the horizontal bias by the pose attribute. A frontal cascade
boxes the visible part of a turned head, so a horizontal offset that appears
only on pose-flagged faces is not a constant and a single `box_shift_x`
cannot remove it. That distinction decides whether the parameter is worth
having.

## The component-swap row was measuring a missing calibration

Section 6 settled the question left open above. The cross path's nominal 1.30
expand implied k = 0.921, below the trained envelope; the best 24-point expand
is 1.60 at k = 1.131, worth 8.949% against 10.584% at nominal. The 13.776%
ablation row was measuring the missing calibration, not the front end it was
meant to isolate.

So the stronger reading, that the model cannot be treated as independent of
the detector feeding it, is not supported. The claim that survives is the
weaker one: a crop-based regressor has a preferred crop geometry, and every
front end that feeds it needs calibrating to that geometry. Swapping the
detector costs a calibration step. `detector.cross_expand` is 1.60 and the
ablation re-runs with the cross row as a like-for-like swap.

`detector.refine_expand` moves to 1.60 on the same evidence: the section-4
sweep wants 1.45 to 1.60 and section 6 puts the best 24-point box at 1.60.
Both are the same construction, a box around 24 points, and 1.30 was never
the canonical framing for it.

Refinement itself is now characterised rather than assumed: the gain is
negative from tight stage-1 boxes and +2.434 from a loose 1.75 one, so it
rescues bad boxes rather than improving good ones, and a third stage is flat,
so it converges at two.

`face_detector.box_scale` and `box_shift_y` stay where they are until the
centre search runs. Adopting the (1.45, 0.08) winner of a scale-centric grid
would be settling the axis that carries 0.3 points and leaving the one that
carries 3.9.

## A gate that failed for the wrong reason

The diagnostic's any-box gate said "should reproduce 72.08%" and came back at
68.62% on unchanged settings, which reads as a regression. It is not one. The
72.08% was measured at min_neighbors 2, and min_neighbors was raised to 5 by
the selection diagnostic in c7ad69a: a deliberate trade of any-box recall for
fewer spurious boxes, since any-box recall falls when there are fewer boxes
to hit with. The gate compared a number against a baseline from different
settings.

The baseline is now stored with the settings that produced it, and the gate
prints the current settings, says whether they match, and explains what to
compare against when they do not. A bare number is not a gate.

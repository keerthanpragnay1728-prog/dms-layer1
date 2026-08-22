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

## Sequence from here

1. Rebuild the cache (crop_expand 2.2, cache_size 224).
2. Retrain with framing [0.85, 1.60]. The cost at k = 1.0 is measured by
   re-running the milestone-5 protocol and comparing against 7.346%.
3. Re-run the deploy gap diagnostic and choose the calibration on end NME.
4. Run the selection diagnostic and choose a selection rule, and if the
   settings grid says so, revisit the cascade settings under the
   one-box-per-frame objective.
5. Only then re-run `compare_detectors.py`, once, on a fixed path.

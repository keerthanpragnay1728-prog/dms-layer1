# Milestone 3 notes: the Haar front end on WFLW

> **Correction (milestone 6).** Two decisions recorded below were made on
> proxy objectives that turned out to invert against the real one. The
> calibration was chosen by landmark containment, and containment runs
> OPPOSITE to end accuracy: the 1.75 box that scores 100% containment is
> the worst box for NME. The detector settings were chosen by any-box
> matching, which cannot see spurious detections, while deployment picks
> one box per frame. Both corrections are recorded in
> [milestone6_findings.md](milestone6_findings.md) and marked in place
> below. Numbers in this file are correct as measured; it is the decisions
> drawn from them that needed revisiting.


Date: 2026-08-21. Dataset: WFLW test split. Detector:
`haarcascade_frontalface_default` (vendored in assets/), histogram
equalisation on. The detector parameters were re-decided by the pose sweep
below; the shipped operating point is scale_factor 1.05, min_neighbors 2,
min_size_frac 0.08. The first table was measured at the original settings
(1.10, 5, 0.08) and is kept as the sweep's baseline.

## Any-box detection rate at the original settings (1.10, 5, 0.08), IoU >= 0.3

This asks whether ANY returned box matches each annotated face. It is
the right question for measuring cascade coverage and the wrong one for
deployment, which picks a single box per frame (see the correction).

| population   | rate    |
|--------------|--------:|
| overall      | 60.04%  |
| no flags     | 82.21%  |
| pose         | 17.79%  |
| expression   | 53.82%  |
| illumination | 63.90%  |
| makeup       | 56.31%  |
| occlusion    | 46.60%  |
| blur         | 51.62%  |

3,843 unmatched Haar boxes (1.81 per image). Not a false positive rate,
because WFLW images contain unannotated faces, but useful as a relative
cost signal.

## First calibration decision: containment beats the median fit

Median fit of raw Haar boxes to the model crop box: box_scale 1.121 (IQR
1.056 to 1.205), box_shift_y 0.130 (IQR 0.108 to 0.155). Landmark
containment in the Haar derived crop: 98.47% with the config values
(1.45, 0.10) against 84.34% with the measured medians (1.12, 0.13).

We kept box_scale 1.45, reasoning that a landmark outside the crop is a
landmark the model cannot predict, so containment matters more than crop
tightness. **That reasoning was wrong, and milestone 6 measured it.**
Containment only says the answer is inside the picture; it says nothing
about whether the model can read a picture framed that way. Measured end to
end on the same faces:

| box_scale | containment | NME     |
|-----------|------------:|--------:|
| 1.12      | 81.68%      | 13.288% |
| 1.45      | 98.47%      | 18.285% |
| 1.75      | 100.00%     | 28.997% |

Best containment, worst accuracy, monotonically. The containment maximising
grid search that `verify_haar_pipeline.py` gained after this decision
optimises the wrong thing; the calibration is now chosen on end NME by
`scripts/diagnose_deploy_gap.py` section 3, after the model has been
retrained to tolerate the framing range.

**Second correction, and the original instinct comes out ahead.** The table
above was measured on a single-stage path with the pre-framing model. With
two stages and the framing-aware model, the calibration grid picks
box_scale 1.60 with no vertical shift: a wide containing box that scores
16.488% on its own and 9.879% after refinement, against a well-framed
(1.30, 0.11) box that scores 10.820% on its own and loses end to end.
Containment was the right criterion for the wrong stage. It fails as a
target for a single pass, because a contained face can still be framed
unreadably; it succeeds for stage 1 of a two-stage pipeline, because a point
outside the crop cannot be recovered later while a badly framed one can.
Full grid and the mechanism in `docs/milestone6_findings.md`.

## Final: any-box detection at the shipped settings (1.05, 2, 0.08)

Verifier re-run at commit `dcbe062`, clone provenance confirmed. Every
subset improved and pose nearly doubled:

| population   | original (1.10, 5) | shipped (1.05, 2) |
|--------------|-------------------:|------------------:|
| overall      | 60.04%             | 72.08%            |
| no flags     | 82.21%             | 89.33%            |
| pose         | 17.79%             | 33.74%            |
| expression   | 53.82%             | 71.97%            |
| illumination | 63.90%             | 75.64%            |
| makeup       | 56.31%             | 74.76%            |
| occlusion    | 46.60%             | 60.73%            |
| blur         | 51.62%             | 65.46%            |

Unmatched boxes went from 1.81 to 5.21 per image (11,029 total). This was
accepted on the grounds that deployment takes the largest box in a one face
cabin and most of the extra boxes in WFLW are unannotated faces. **The
first half of that is the second correction:** more boxes per frame is
exactly what makes the largest-box rule fail, and milestone 6 measured the
largest box being the annotated subject on only 27.67% of single-face
images. Selection rules and the settings that suit them are measured by
`scripts/diagnose_face_selection.py`. The coordinate round trip stayed
exact at 0.000000000 px. The integer readout quantisation bound moved from
0.79 to 0.95 frame px as the median crop side grew from 177 to 213 px with
the bigger calibrated box; the model regresses continuous coordinates, so
this bound does not apply to its output. Timing this session was 160 ms per
image, which matches the first session rather than the 83 ms one and
confirms the variance caveat below.

Milestone 3 closed on these numbers.

## Containment at the shipped settings, second calibration decision

Verifier re-run at (1.05, 2, 0.08):

| calibration candidate         | containment |
|-------------------------------|------------:|
| (1.45, 0.10) previous config  | 98.27%      |
| (1.45, 0.13) shift candidate  | 98.47%      |
| (1.12, 0.13) measured medians | 82.81%      |
| (1.75, 0.08) grid best        | 99.73%      |

Adopted at the time: (1.75, 0.08), by the same containment argument as
before, converting a 1.5% landmark loss rate into 0.27%. **Superseded.**
1.75 is the worst of the candidates on end NME (28.997% against 13.288% at
1.12), because it frames the face at about 1.56 times the framing the model
was trained on, and milestone 6 measured NME growing roughly linearly with
framing outside the trained range. The number to select on is NME.

For the record, the containment story was internally consistent to the end,
which is what made it convincing. On the larger matched population of the
final re-run, 1.75 held 99.33%, the containment grid optimum landed at
(1.75, 0.07) with the same 99.33%, and the tighter (1.45, 0.13) candidate
fell to 96.61%, which read at the time as confirmation that the faces a
looser detector newly finds are exactly the ones a tight box loses
landmarks on. Every one of those statements is still true. They were just
answering a question that does not determine accuracy.

Flagged cost, which milestone 6 confirmed and then some. This was recorded
as a cost to be priced later; it was in fact the dominant error term, worth
about 22 NME points, and pricing it earlier would have caught the
containment inversion at the time. The mechanism as flagged: The deploy crop is 1.75
times the Haar side. The ground truth crop that training uses measures
about 1.12 times the Haar side at the median, so the deploy crop is about
1.56 times wider than the training crop. The face spans about 77% of a
training crop (the 1.3 expand) but only about 49% of a deploy crop. That is
equivalent to a scale augmentation factor of about 0.64, outside the
configured [0.85, 1.15] range, and it shrinks the inter-ocular distance at
the 112 px input from roughly 35 px to roughly 22 px. The plan: train the
milestone 4 baseline as configured, since its NME protocol evaluates on
ground truth boxes and is unaffected. Milestone 6 measures NME through the
real Haar pipeline against ground truth boxes, which prices this mismatch
directly. If the gap is material, the options in order of preference are
widening the scale augmentation's lower bound to about 0.6 and retraining,
or falling back to the tighter (1.45, 0.13). No changes to the training
recipe before that number exists.

## Timing and environment facts

* Haar detection timing on the Kaggle CPU is not stable across sessions:
  identical settings measured 160 ms per image in one session and 83 ms in
  another. No single figure is citable as absolute. Report timings only as
  within-session relative comparisons, and later ours against MediaPipe on
  identical hardware in the same session. Deployment latency is a separate
  question anyway: single camera frame, optional downscale, detection every
  Nth frame with box persistence in between.
* The coordinate round trip from frame to crop space and back is exact.
* The `opencv-python` 5.0 wheel has no Haar `CascadeClassifier` and ships no
  cascade data files. We saw this directly in the wheel: zero files matching
  `haarcascade*`, and no `CascadeClassifier` in the type stubs. The project
  therefore pins `opencv-python>=4.8,<5` and vendors
  `haarcascade_frontalface_default.xml` and `haarcascade_profileface.xml`
  in `assets/`, extracted from the OpenCV 4.10 PyPI wheel with their
  original licence headers kept. Worth a sentence in the report: the
  classical detector this pipeline uses is being retired from the main
  OpenCV distribution.

  Scope correction (milestone 6): this is a statement about the
  `opencv-python` build, not about OpenCV 5 as such. `opencv-contrib-python`
  5.0.0.93 still has `CascadeClassifier`, `detectMultiScale3` and 17 cascade
  XMLs in `cv2.data.haarcascades`, and the repo's Haar tests pass under it.
  The API moved into the contrib build rather than disappearing. That matters
  in practice because mediapipe depends on `opencv-contrib-python`, so
  installing it can pull an OpenCV 5 contrib wheel that shadows the pinned
  4.x one. See `docs/dependency_notes.md`.

## The pose problem (17.79%), the sweep, and the decision

Detecting turned heads is exactly the inattentiveness case, so the 17.79%
pose rate was a problem to address, not a footnote. The sweep
(`scripts/sweep_haar_pose.py`) measured a parameter grid, scale_factor
{1.05, 1.10} by min_neighbors {2, 3, 5} by min_size_frac {0.05, 0.08}, and
the profile face cascade as a fallback, on every pose subset face plus a
250 image no-flag reference sample. Key rows (the full grid is in the sweep
run's `sweep_results.yaml`):

| combo (sf, mn, msf)      | pose  | no-flag | unmatched/img | ms/img |
|--------------------------|------:|--------:|--------------:|-------:|
| 1.10, 5, 0.08 (baseline) | 17.8% | 79.8%   | 1.80          | 84     |
| 1.05, 2, 0.05 (best pose)| 34.0% | 87.1%   | 7.32          | 272    |
| 1.05, 2, 0.08 (adopted)  | 33.7% | 86.8%   | 5.00          | 157    |
| 1.05, 2, 0.05 + profile  | 35.0% |         |               | 664    |

The no-flag rates here are on the sweep's sampled reference population and
differ slightly from the full split numbers above.

Tuning beats the profile fallback. The profile cascade helps most where the
frontal cascade is strict (17.8% to 27.9% on the baseline row), but it
never beats simply loosening the frontal cascade, and it roughly triples
latency everywhere. On the best row it adds 1.0 point for 272 to 664 ms.
So `profile_fallback` stays off. It was tried and rejected on evidence,
not skipped, and the code stays in the repo as the record of that.

The adopted operating point, (1.05, 2, 0.08), is within 0.3 points of the
best pose rate at 40% of its latency, and it also lifts the no-flag
population from 79.8% to 86.8%. The accepted cost is unmatched boxes going
from 1.80 to 5.00 per image. Doubling pose detection does not make Haar a
turned head detector; it moves the measured ceiling.

## Framing for the report

1. The ceiling was measured, not assumed. A 12 point parameter grid and the
   profile cascade fallback were both evaluated with their frontal side and
   latency costs, and the operating point was chosen from that table.
2. WFLW's pose subset is dominated by extreme yaw approaching profile,
   which is harsher than the in-cabin envelope. A driver facing camera sees
   a near frontal face most of the time, and mirror or shoulder glances are
   brief excursions. Report the detection envelope per subset rather than
   one number.
3. The deployed system is temporal while this layer is per frame. A Haar
   miss during a head turn does not erase the event: Layer 3 can hold the
   last box briefly, and a sustained loss of the frontal face is itself
   evidence for eyes off road. That use of loss of lock as a signal is to
   be validated in Layer 3, not assumed.
4. The scientific comparison stays clean by construction. Landmark NME
   (ours against MediaPipe) is evaluated on ground truth boxes, the
   standard WFLW protocol, so landmark quality is not confounded by the
   detector. The Haar envelope is reported separately as a property of the
   deployment front end. MediaPipe brings its own detector, so the full
   pipeline comparison is between complete stacks, and the detector
   difference is named explicitly as part of what differs.
5. If the report needs a closing argument: tuning within the Haar family
   moved pose detection from 17.8% to 33.7% and no further at acceptable
   cost, which is itself evidence for why modern in-cabin systems use
   learned detectors.

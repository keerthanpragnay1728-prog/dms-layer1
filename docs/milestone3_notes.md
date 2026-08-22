# Milestone 3 notes: the Haar front end on WFLW

Date: 2026-08-21. Dataset: WFLW test split. Detector:
`haarcascade_frontalface_default` (vendored in assets/), histogram
equalisation on. The detector parameters were re-decided by the pose sweep
below; the shipped operating point is scale_factor 1.05, min_neighbors 2,
min_size_frac 0.08. The first table was measured at the original settings
(1.10, 5, 0.08) and is kept as the sweep's baseline.

## Detection rate at the original settings (1.10, 5, 0.08), IoU >= 0.3

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

We kept box_scale 1.45. A landmark outside the crop is a landmark the model
cannot predict, so containment matters more than crop tightness, and a
median fit box loses the tails of the distribution. As a consequence,
`verify_haar_pipeline.py` recommends calibrations by a containment
maximising grid search rather than a median fit, and prints a candidate
table.

## Final: detection at the shipped settings (1.05, 2, 0.08)

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

Unmatched boxes went from 1.81 to 5.21 per image (11,029 total). We accept
that: deployment takes the largest box in a one face cabin, and most of the
extra boxes in WFLW are unannotated faces. The coordinate round trip stayed
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

Adopted: (1.75, 0.08), by the same containment argument as before. The grid
best converts a 1.5% landmark loss rate into 0.27%. On the larger matched
population of the final re-run it held 99.33%, and the grid optimum landed
at (1.75, 0.07) with the same 99.33%, so the shipped values sit at the
optimum. The (1.45, 0.13) candidate dropped to 96.61% on that population,
which supports the decision: the faces the looser detector newly finds are
exactly the ones a tight box loses landmarks on.

Flagged cost: less face per pixel at deploy time. The deploy crop is 1.75
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
* OpenCV 5.x removed the Haar `CascadeClassifier` API and stopped shipping
  the cascade data files. We saw this directly: the OpenCV 5.0 wheel has
  neither. The project therefore pins `opencv-python>=4.8,<5` and vendors
  `haarcascade_frontalface_default.xml` and `haarcascade_profileface.xml`
  in `assets/`, extracted from the OpenCV 4.10 PyPI wheel with their
  original licence headers kept. Worth a sentence in the report: the
  classical detector this pipeline uses is being retired from its home
  library.

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

# Milestone 7 design: what stability means here, and the order it is measured in

The spec calls for a frame-to-frame stability harness as a first-class
deliverable. This file records what it measures and why, before any numbers
exist, so the design cannot be reverse-fitted to the results.

## Why the order matters

Milestone 6 established two things that shape this milestone. The pipeline's
accuracy is dominated by acquisition rather than by the landmark model: at
the shipped calibration the ground-truth-box ceiling is 6.388% against 7.421%
deployed. And of the centre error that remains after calibration, 86% is
per-face scatter, the Haar box simply landing in a different place on each
face, which no constant transform can remove.

If that carries over from face to face into frame to frame, the box is moving
under the model every frame, and a landmark jitter number measured on the
deployed path would mostly be measuring the box. So the box is measured
first, in the same units as the calibration parameters, and the model's own
contribution is measured separately with acquisition removed.

## The five measurements

1. **Box jitter.** Stage-1 crop centre and side against a box that tracks the
   face perfectly, in box-side units. Bias and jitter are reported
   separately: a box consistently offset is a calibration matter and only a
   box that moves reaches the landmarks as noise. The units are chosen so
   this is directly comparable to the across-faces centre scatter measured in
   milestone 6 section 5b, which the harness reads from that run's YAML when
   pointed at it rather than quoting a number from memory.
2. **Model jitter.** Landmarks predicted in a per-frame ground-truth box, so
   acquisition is perfect and only the image changes. The model's own noise
   floor.
3. **End-to-end jitter.** The deployed path. The difference from (2) is the
   front end's contribution.
4. **Refinement's effect.** Stage 1 alone against two stages. Stage 2
   rebuilds its box from stage 1's points, which is a feedback path: it can
   damp box motion or amplify it, and nothing in milestone 6 predicts which.
   Since the shipped stage-1 box is deliberately wide and poorly framed, this
   is the measurement that says whether that choice is safe over time as well
   as on average.
5. **The Layer 2 signals.** EAR, MAR and the gaze proxy, which is the pupil's
   position inside the eye normalised by eye width. Jitter of each, plus a
   false-crossing rate against a blink threshold set to a fraction of the
   sequence's own mean EAR, which is the per-driver calibration Layer 2 would
   perform. A sequence with no blink in it that produces threshold crossings
   is producing false blinks, and the ground-truth signal is run through the
   same threshold as a control.

MediaPipe runs beside ours for (3) and (5), so the ablation extends to
stability rather than stopping at accuracy.

## Definitions that carry the design

**Jitter is the standard deviation of the residual, not of the trajectory.**
A landmark that moves because the face moved is not jitter. Ground truth is
known per frame on synthesised sequences, so the residual is available and
the face's real motion cancels. On a real clip there is no truth, so jitter
is estimated as deviation from the trajectory's own moving average, which
assumes real motion is slower than the smoothing window. The two estimators
are not comparable and never appear in one table.

**Bias and jitter are reported separately everywhere.** A prediction can be
steady and wrong or accurate and unsteady. A calibrated downstream stage
absorbs the first and cannot absorb the second, so collapsing them into one
number would hide the distinction that matters most to Layer 2.

**A dropped frame contributes nothing.** Holding the previous prediction
through a dropout turns the face's real motion into measured jitter, so a
detector that fails often would score as unsteady for the wrong reason.
Frames without a usable detection are excluded from the jitter computation
and reported as a dropout rate instead; a sequence with fewer than eight
usable frames produces no jitter number at all.

**Pupils get their own attention.** Milestone 6 put our pupil error at 6.42%
against MediaPipe's 3.42%, and that gap survives handing our model a perfect
box, so it lives in the model. Gaze depends on pupil position relative to the
eye corners rather than in the image, which is why the gaze proxy is
normalised that way: a constant pupil offset is absorbed by calibration and a
wandering one is not. Per-group jitter is reported for every row so the pupil
column can be read directly.

## What this cannot measure

Synthesised sequences are a 2D warp of a still. There is no out-of-plane
rotation, no blinking, no expression change, no real motion blur, and the
lighting changes only as a global brightness drift. Every jitter number from
them is a lower bound. The `--video` path exists so the harness can be run on
a real clip, where the numbers are realistic and the estimator is weaker, and
the two are reported apart.

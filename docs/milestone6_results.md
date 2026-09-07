# Milestone 6 results: our 24-point model against MediaPipe on identical faces

Final run at the shipped calibration (box_scale 1.60, box_shift_y 0.00,
box_shift_x 0.00, two-stage refinement at 1.60, cross_expand 1.60), WFLW
test split, 2118 images, one target face each.

Everything below is on the 410 faces every path found, so no row is scored on
a population another row missed. The per-detector rows on their own matched
sets are in `m6_results.yaml`; they are context, not the comparison.

## The table

| row | NME | fail@10% | eyelids | pupils | mouth | axis | contour |
|-----|-----|----------|---------|--------|-------|------|---------|
| ours (Haar box) | 7.421% | 15.85% | 6.05% | 6.42% | 7.29% | 9.37% | 11.18% |
| mediapipe | 7.100% | 8.05% | 4.54% | 3.42% | 3.83% | 6.58% | 20.14% |
| ours on MediaPipe's box | 7.991% | 19.27% | 7.28% | 7.65% | 7.44% | 9.36% | 10.17% |
| ours on ground-truth box | 6.388% | 12.68% | 5.54% | 5.91% | 6.18% | 7.84% | 8.66% |

Paired on the same faces: ours 7.421% against MediaPipe 7.100%, ours better
on 51.0% of faces. Excluding contour: ours 6.669% against 4.492%.

## What the headline is, and why it is not 7.42 against 7.10

The aggregate pair is close only because contour offsets everything else. We
lose on eyelids, pupils, mouth and axis, and win on contour by 9 points.

The contour win is a convention artifact and should be reported as one. Our
model is trained on WFLW's contour parameterisation, so it fits by
construction. MediaPipe's contour indices are the documented approximation of
WFLW 4/8/28/24, and `scripts/verify_mediapipe_mapping.py` measured that
approximation at 15 to 25% of IOD against ground truth on the faces MediaPipe
finds. MediaPipe's 20.14% contour NME is essentially that mapping offset. It
is not landmark failure, and a reader who takes it as such has been misled.

The same effect exists in every other group at smaller magnitude, and always
in our favour: MediaPipe's error against WFLW ground truth carries a
convention term everywhere, and ours carries none, because the annotations
scoring us are the annotations that trained us. So the ex-contour pair,
6.669% against 4.492%, is not a hostile framing of our result. It is the
closer of the two to a fair comparison, and it still understates MediaPipe.

The report leads with the per-group table. Where one number is needed, it is
the ex-contour pair with the convention caveat attached, and the standard
unweighted aggregate is given beside it for comparability with milestone 5
and with the literature.

## The landmark comparison with acquisition removed

The ground-truth-box row is the comparison this project set out to make: our
model given a perfect box, against MediaPipe running its own pipeline. It
favours us, since our acquisition error is removed and MediaPipe's is not.

| group | ours on gt box | mediapipe | gap at our ceiling | acquisition cost to us |
|-------|---------------|-----------|--------------------|------------------------|
| eyelids | 5.54% | 4.54% | +1.00 | +0.51 |
| pupils | 5.91% | 3.42% | +2.49 | +0.51 |
| mouth | 6.18% | 3.83% | +2.35 | +1.11 |
| axis | 7.84% | 6.58% | +1.26 | +1.53 |
| contour | 8.66% | 20.14% | -11.48 | +2.52 |

Two readings follow. On every group except the convention-driven one,
MediaPipe is ahead even when we are handed a perfect box, so the remaining
gap is the landmark model rather than the front end. And the largest gaps,
pupils and mouth, barely move when acquisition is removed (+0.51 and +1.11),
which locates them in the model rather than in the pipeline around it.

The likely cause for pupils is resolution. Our model regresses 24 points from
one 112x112 grayscale crop of the whole face, which leaves roughly 25 pixels
of eye width and perhaps 10 across the iris. MediaPipe runs a dedicated iris
head on a re-cropped eye region, so it works at several times that effective
resolution. That is a difference in what the two systems are doing, not only
in how well they do it, and it is the honest explanation to give: a
single-shot whole-face regressor at 112 pixels is not the same task as a
cascaded eye-crop refiner. Testing it means training at a larger input or
adding an eye-crop stage, which is future work rather than a fix.

## Failure rate, and why it leads alongside NME

Our failure rate at 10% is 15.85% against MediaPipe's 8.05% on the same
faces, roughly double, while the means differ by 0.32 points. Our centre is
comparable and our tail is heavier.

For a drowsiness system the tail is the operational quantity. A mean NME is a
per-face average nobody experiences; what the system experiences is frames,
and a frame at NME above 10% is a frame whose EAR or MAR is probably wrong.
So the report gives failure rate equal billing with NME rather than as a
footnote, and states the tail plainly: on this population our pipeline
produces roughly twice as many unusable frames as MediaPipe does.

This connects to the calibration. The shipped stage-1 box was chosen on
end-to-end mean NME, and a wide containing box that depends on refinement can
buy a better mean while carrying a worse tail than a tighter box would. The
calibration grid now reports failure rate beside the mean for exactly this
reason, and if the tighter (1.30, 0.11) configuration turns out to have a
lighter tail at a slightly worse mean, that is a legitimate choice for a
safety system and should be made explicitly rather than inherited from an
optimiser that was pointed at the mean.

## Acquisition, framing, and the price of the classical front end

The calibration price, our model on ground-truth boxes against the same model
on Haar boxes, is +1.033 NME points on these faces, down from +2.462 before
the centre work and +22.4 before the framing retrain.

The shipped stage-1 framing runs at a median k of 1.380 with a p90 of 1.750,
against a trained envelope of [0.85, 1.60]. Half the mass sits above 1.38 and
a tenth beyond the envelope entirely. That is the design working as intended
rather than drift: stage 1 is deliberately a wide containing box and stage 2
does the framing, and the grid selected it on end-to-end NME with refinement
on. It is worth saying out loud that it depends on refinement, since running
stage 1 outside the trained envelope is the exact condition that produced the
milestone-6 collapse, and here it is survivable only because a second stage
corrects it.

## Detection

| target size, fraction of the shorter side | ours | mediapipe |
|-------------------------------------------|------|-----------|
| 0.00 to 0.15 | 30.1% | 4.5% |
| 0.15 to 0.25 | 38.9% | 42.6% |
| 0.25 to 0.40 | 56.8% | 71.6% |
| 0.40 to 1.01 | 65.6% | 83.8% |

Reported as in `docs/milestone6_findings.md`: the aggregate ordering favours
our Haar front end and reverses in the band a driver's face occupies, so the
aggregate is never quoted alone.

## What this milestone establishes

Our model, given the same face, is close to MediaPipe on the aggregate and
behind it on every group where the two conventions agree. The gap is a
landmark-model gap rather than a pipeline gap: it survives handing our model
a perfect box, and it is largest on pupils, where the architectures differ
most.

Against that, the model is 2.4 MB of weights trained from random
initialisation on 6750 faces, with no pretrained backbone, running in 2.9 ms
on a CPU. MediaPipe is a 3.76 MB bundle of four models including a dedicated
iris head, trained on data and with methods that are not public. Close on the
aggregate and behind by 2.2 points ex-contour is the honest summary, and the
comparison is worth more than the ranking: it says where a small
from-scratch model loses, and the answer is fine detail at low effective
resolution rather than face-level geometry.

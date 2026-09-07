# Recording protocol for the milestone 7 real clip

Synthesised sequences cannot set their own noise level. Section 1 of the
harness measures box jitter under a perturbation model whose sensor noise is
a parameter someone chose, so the number is only as meaningful as that
choice. A real clip fixes it, and it is the only source that produces genuine
blinks, out-of-plane rotation and real sensor behaviour.

## What to record

Three segments, one file each, so they can be analysed separately. Total
about 90 seconds.

| segment | duration | what to do | what it measures |
|---------|----------|-----------|------------------|
| still | 20 to 30 s | sit as still as you can, eyes open, no talking, no nodding | the box jitter floor under real sensor noise alone. Analysed with `--static`, where the raw spread of the trajectory IS the jitter and no smoothing assumption is needed. This is the segment that calibrates the synthesised numbers. |
| natural | 30 s | sit normally, small posture adjustments, blink naturally, look around the "cabin" a little | the realistic operating case, and the only source of genuine blinks. Run without `--static`. |
| turn | 20 to 30 s | slow yaw left and right to about 30 to 40 degrees, then pitch up and down, then return to centre | the pose regime where the frontal cascade fails and where the horizontal box bias appeared. Expect dropouts; the gap structure is the result. |

## Camera and framing

* **Fixed camera.** A tripod, a stack of books, anything. If the camera moves,
  its motion adds to the box jitter and nothing can separate the two.
* **1280x720 at 30 fps** is the target. A driver monitoring camera is
  typically 640x480 to 1280x720, and the pipeline resizes every crop to 112
  pixels anyway, so more resolution buys nothing beyond this.
* **Face filling roughly a third to a half of the frame height**, which puts
  it in the top size band from milestone 6, the band that matches a cabin.
  About 50 to 70 cm from the lens for a typical webcam field of view.
* **One face in frame.** No one else in the background, no faces on posters
  or screens. Selection flips between candidate boxes are a separate failure
  and would confound the jitter measurement.
* **Ordinary indoor lighting**, reasonably even, no strong backlight. If a
  second still segment in dim light is easy, record it: sensor noise rises
  with gain, so it bounds the other end of the range.

## File format

* **mp4, H.264, high bitrate**, constant frame rate. Compression noise is
  part of what we are trying to measure, so heavy compression would put the
  codec's artifacts into the answer. If the phone or camera offers a quality
  setting, use the highest one.
* No stabilisation. Phone video stabilisation warps frames to remove motion,
  which is precisely the signal this measures. Turn it off if it can be
  turned off; if it cannot, note that in the run.
* No beauty filters, no auto-framing, no portrait mode.

## Running it

```
python scripts/stability_harness.py --config configs/layer1_base.yaml \
    --video still.mp4 --static --m6-yaml /kaggle/working/m6_diagnosis/m6_diagnosis.yaml

python scripts/stability_harness.py --config configs/layer1_base.yaml --video natural.mp4
python scripts/stability_harness.py --config configs/layer1_base.yaml --video turn.mp4
```

The still segment with `--static` is the one to run first: its box centre
jitter is directly comparable to the synthesised section 1 number, and the
ratio between them says how much of the synthetic result was the perturbation
model.

## What each result will mean

If the still clip's box jitter is close to the synthesised number, the
perturbation model is realistic and section 1 stands as written. If it is far
lower, the synthesised sensor noise is too aggressive and the synthetic
section 1 is measuring `stability.noise_sigma` rather than the detector, in
which case the config value should be set from the clip and the synthesised
sequences re-run.

On the natural clip the EAR crossings include real blinks, so the number is
not a false-positive rate on its own. Read it against the still segment,
where any crossing is false by construction.

On the turn clip, expect the Haar path to drop frames. The rate matters less
than the longest gap: Layer 2 aggregates over a window, and a two second
blackout during a head check is a different failure from the same number of
frames lost one at a time.

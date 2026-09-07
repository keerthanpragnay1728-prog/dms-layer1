# Running the model live on a webcam

`scripts/live_demo.py` runs the deployed pipeline on a camera and draws what
it is doing: the Haar rectangle the cascade selected, the stage-2 box
refinement rebuilt from the stage-1 points, the 24 landmarks in their schema
group colours, the eye aspect ratio Layer 2 would consume, the loop rate, and
a red border on frames where the detector loses the face. It also records the
milestone 7 clips, so the recording protocol can be satisfied from the same
session rather than filmed separately.

## Two resolutions, deliberately different

Inference and display run on a downscaled copy of the frame, `--infer-width`,
default 640. A Haar cascade at 720p costs a few hundred milliseconds and the
viewer would be unusable. This is a demo choice and the window labels the
resolution it is running at; the frame rate shown is not the pipeline's frame
rate at 720p.

Recording writes the camera's full-resolution frames with **no overlay**, on
the capture thread, at the camera's own rate. Two reasons. Drawing boxes over
the face and then measuring landmark stability on those frames would be
measuring the overlay. And writing frames at the inference rate would produce
a file that plays back several times too fast, which would silently corrupt
every temporal measurement made from it.

Each recording also writes a `.timestamps.txt` sidecar holding the capture
time of every frame, so the harness's conversion of gaps into seconds can be
checked rather than assumed.

## Keys

| key | action |
|-----|--------|
| `r` | start or stop recording |
| `1` `2` `3` | label the next recording still / natural / turn |
| `m` | mirror the view; the recording is never mirrored |
| `q` or Esc | quit |

## Codec

`--codec mp4v` is the default because that encoder is present in every OpenCV
wheel. The recording protocol asks for H.264: `--codec avc1` tries it first
and falls back to mp4v, and prints which one it used. When avc1 is absent the
probe makes the backend emit a page of red ffmpeg errors that look like a
crash and are not one, which is why it is opt in.

## What can go wrong

**The camera does not open.** Another application is usually holding it.
On Windows try `--backend msmf`, or `--camera 1` if there is more than one
device, and check that Settings, Privacy, Camera allows desktop apps.

**The window never appears.** `opencv-python-headless` has no GUI. Uninstall
it and install `opencv-python`.

**The requested resolution is ignored.** Cameras silently substitute a size
they support. The script prints what the camera actually gave; if it is not
1280x720, record that in the notes rather than assuming the protocol was met.

**The weights say "framing NOT RECORDED".** That is the provenance line
working as intended on an export made before the trainer stamped its
envelope. It is a statement about the file, not an error.

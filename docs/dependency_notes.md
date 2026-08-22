# Dependency notes: two libraries removed the APIs this project was built on

Both of Layer 1's non-PyTorch building blocks, the OpenCV Haar cascade and
MediaPipe Face Mesh, were deleted from their libraries while this project
was being written. Neither is a version-bump inconvenience: in both cases
the classical or first-generation API that a driver monitoring pipeline
would naturally reach for is gone from the current release, and the
replacement changes how the code is written and what has to ship with it.

This file records what was measured, what the project does about it, and
what is worth saying in the report.

## 1. The opencv-python 5.0 wheel has no Haar cascade API

Found in milestone 3, scope corrected in milestone 6.

Measured on the wheels rather than read from release notes:

| distribution              | version   | `CascadeClassifier` | cascade XMLs shipped |
|---------------------------|-----------|---------------------|----------------------|
| opencv-python             | 4.8 to 4.x| yes                 | yes                  |
| opencv-python             | 5.0.0.93  | no (absent from the stubs) | none (0 files matching `haarcascade*`) |
| opencv-contrib-python     | 5.0.0.93  | yes, with `detectMultiScale3` | 17 cascade XMLs in `cv2.data.haarcascades` |

So the Haar API moved into the contrib build in OpenCV 5; it did not
disappear from OpenCV. The milestone-3 note originally said "OpenCV 5.x
removed the Haar CascadeClassifier API", which was measured on
`opencv-python` and is too broad as written. It is corrected there.

What the project does:

* pins `opencv-python>=4.8,<5` in requirements.txt, which is also what
  Kaggle ships;
* pins `opencv-contrib-python>=4.8,<5` for the same reason. That
  distribution is not used directly, but mediapipe depends on it and would
  otherwise pull an OpenCV 5 contrib wheel. Both distributions install the
  same `cv2` package directory, so whichever is installed last decides what
  `import cv2` gives you, and an unpinned transitive dependency can silently
  replace a pinned direct one. The bound keeps one OpenCV generation in play,
  which the milestone-3 detection rates depend on for comparability;
* vendors `haarcascade_frontalface_default.xml` and
  `haarcascade_profileface.xml` in `assets/`, taken from the OpenCV 4.10
  PyPI wheel with their original licence headers, because even inside 4.x
  the data files are not guaranteed to be present in every build.

The face detector resolves its cascade from `assets/` first and falls back
to `cv2.data` only if this build has one, so the pipeline does not depend on
the wheel's contents. If it ever runs against a build with no Haar API at
all, it says so and names the pin instead of raising `AttributeError`.

## 2. MediaPipe removed the solutions API, and the last version that had it
   no longer installs

Found in milestone 6. This one took two rounds to get right, so the whole
history is here.

### The version history, as measured

The wrapper was first written against `mediapipe.solutions.face_mesh`
(the legacy Face Mesh with `refine_landmarks=True`). Requirements pinned
`mediapipe>=0.10,<1`, on the assumption that the solutions API survived
until the 1.0 release. That assumption was wrong, and a milestone 6 run
proved it: MediaPipe never loaded and the ablation produced one row instead
of three.

Inspecting the published wheels gives the real history:

| version | `mediapipe.solutions` | note                                        |
|---------|-----------------------|---------------------------------------------|
| 0.10.21 | present               | last published release that ships it        |
| 0.10.30 | removed               | PyPI has nothing between these two versions |
| 0.10.32 | removed               |                                             |
| 0.10.33 | removed               |                                             |
| 0.10.35 | removed               |                                             |
| 1.0.1   | removed               | Tasks API only                              |

So the removal happened at 0.10.30, well before the version numbering
suggests, and `>=0.10,<1` selected a version without the API it claimed to
guarantee.

The first fix was to pin exactly `mediapipe==0.10.21`. That pin is correct
about the API and useless in practice: 0.10.21 cannot be installed on
Kaggle's current image. It requires `protobuf < 5`; Kaggle's preinstalled
tensorflow requires `protobuf >= 5`; three attempts (protobuf 4.25.9,
4.25.3 and 7.36) each failed on one side or the other of that constraint.
There is no version satisfying both, so the pin was a dead end, not a
workaround.

### What the project does now

The wrapper is ported to the Tasks API
(`mediapipe.tasks.python.vision.FaceLandmarker`), which every release from
0.10.30 on ships. Requirements say `mediapipe>=0.10.30`, with no protobuf
pin, because the current package declares no protobuf dependency at all:
mediapipe 1.0.1's requirements are absl-py, certifi, numpy, sounddevice,
flatbuffers, opencv-contrib-python and matplotlib. The constraint that made
0.10.21 uninstallable is simply gone rather than worked around.

Three things change with the port:

* **The model is a file, not part of the wheel.** Tasks loads a `.task`
  bundle. `assets/face_landmarker.task` (3.76 MB) is vendored;
  `detector.mediapipe.model_path` accepts an explicit path or `auto`, which
  searches `assets/`, the repo root and `/kaggle/input` and accepts exactly
  one unambiguous hit, the same rule the crop cache and our weights use
  (`dms_layer1/assets.py`). With `allow_download: true` it fetches the
  bundle from Google's model store if nothing is attached.
* **`refine_landmarks` is a property of the bundle, not a flag.** The iris
  head is what produces mesh points 468 to 477, and the pupil indices
  (468, 473) do not exist without it. The wrapper checks the returned
  landmark count and raises a message naming the 468 versus 478 problem
  rather than silently indexing out of range.
* **The runtime links EGL/GLES even on CPU.** On a headless container it
  fails at landmarker construction with `libEGL.so.1` missing; installing
  `libegl1` and `libgles2` fixes it (note the package is `libgles2`, not
  `libglesv2-2`). Kaggle images already have them. The wrapper catches the
  `OSError` and says this.

### The mapping survived the port, measured rather than assumed

The 24-point mapping in `configs/landmarks_24.yaml` was chosen against the
solutions mesh. Tasks returns the same 478-point topology, so the indices
should carry over, but "should" is not a measurement.
`scripts/verify_mediapipe_mapping.py` re-runs the check: per-point offset
from ground truth in percent of inter-ocular distance, plus a control that
scores all 478 mesh points against each ground truth point and reports
whether some other index would have been closer.

Same schematic face, same check, before and after the port:

| point               | solutions | Tasks |
|---------------------|-----------|-------|
| left_pupil          | 0.8%      | 0.5%  |
| right_pupil         | 0.3%      | 1.1%  |
| eyelids, worst      | 4.6%      | 3.8%  |
| mouth, worst        | 2.4%      | 3.9%  |
| nose_tip            | 4.0%      | 1.9%  |
| chin                | 2.1%      | 2.1%  |
| contour_left_upper  | 10.0%     | 10.1% |
| contour_left_lower  | 8.1%      | 8.1%  |
| contour_right_upper | 9.3%      | 9.2%  |
| contour_right_lower | 8.2%      | 8.2%  |
| mean, all 24 points | 3.7%      | 3.2%  |

The four contour points agree to within 0.1 points of IOD, which is the
useful part: contour indices are far from the fine detail the two model
generations differ on, so an identical contour is evidence that the mesh
was not renumbered. The eye and mouth points move by up to 2.5 points of
IOD in both directions, which is the newer model predicting slightly
different positions for the same semantic points, not a mapping change.
The mean is marginally better after the port.

This is a schematic-face check, so it verifies the port rather than the
mapping. The mapping's own evidence comes from the same script run on WFLW
faces.

## What to say in the report

One sentence, backed by both findings: a perception layer built on
classical or first-generation vision APIs inherits their deprecation
schedule, and both of this layer's non-PyTorch components had their API
removed during the project. The practical consequences are the reason the
repository vendors its cascade XML and its `.task` bundle in `assets/`
instead of trusting a `pip install` to provide them, and the reason
`requirements.txt` carries a bound with a stated reason on every line that
has one.

It is also a small argument for the thing being compared: our own model is
a checkpoint we trained and a config we control, so it has no upstream that
can withdraw it.

#!/usr/bin/env python3
"""Live webcam viewer for the deployed pipeline, with a recorder for the
milestone 7 clips.

What the window shows: the raw Haar rectangle the cascade selected, the
stage-2 box refinement rebuilt from the stage-1 points, the 24 landmarks in
their schema group colours, the eye aspect ratio Layer 2 would consume, the
loop rate, and a red border on any frame where the detector loses the face.

Two resolutions are in play and they are deliberately different:

  * INFERENCE and display run on a downscaled copy, --infer-width, because a
    Haar cascade at 720p costs a few hundred milliseconds per frame and the
    viewer would be unusable. This is a demo choice, not the pipeline's
    behaviour, and the window labels the resolution it is running at.
  * RECORDING writes the camera's full-resolution frames, unannotated,
    because the milestone 7 harness has to see what the camera saw. Drawing
    boxes over the face and then measuring landmark stability on those frames
    would be measuring the overlay.

Recording runs on the capture thread at the camera's own frame rate, so the
file is a true 30 fps recording even though the viewer updates at whatever
rate inference manages. A sidecar .timestamps.txt records the capture time of
every written frame, so the harness's conversion of gaps into seconds can be
checked against what actually happened rather than assumed.

Keys:
    r        start or stop recording
    1 2 3    label the next recording still / natural / turn
    m        mirror the view (recording is never mirrored)
    q or Esc quit

Usage (Windows):
    python scripts\\webcam_demo.py --config configs\\layer1_base.yaml ^
        --weights landmarks24_framing.pt
"""

from __future__ import annotations

import argparse
import platform
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dms_layer1.assets import resolve_asset
from dms_layer1.config import load_config, require, resolve_path
from dms_layer1.detect.ours import OurLandmarkDetector
from dms_layer1.evaluation import signals
from dms_layer1.landmarks.schema import load_schema
from dms_layer1.viz.overlay import GROUP_COLORS

FONT = cv2.FONT_HERSHEY_SIMPLEX
WEIGHT_PATTERNS = ["landmarks24*.pt", "*.pt", "best.pth", "*.pth"]
SEGMENTS = {ord("1"): "still", ord("2"): "natural", ord("3"): "turn"}
BASELINE_SECONDS = 5.0        # window for the running EAR baseline


class Camera:
    """Capture on its own thread so recording keeps the camera's frame rate
    while inference runs slower on the main thread. Without this the file
    would be written at the inference rate and play back several times too
    fast, which would silently corrupt every temporal measurement made from
    it."""

    def __init__(self, source, width: int, height: int, fps: float,
                 backend: int, codec: str = "mp4v"):
        self.cap = (cv2.VideoCapture(source, backend) if isinstance(source, int)
                    else cv2.VideoCapture(str(source)))
        if isinstance(source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or fps
        self.codec = codec
        # A file has no clock of its own: without throttling the thread reads
        # it as fast as the disk allows, which is not a replay of anything.
        self._frame_interval = 1.0 / max(self.fps, 1.0) if not isinstance(source, int) else 0.0
        self._lock = threading.Lock()
        self._latest = None
        self._writer = None
        self._stamps = None
        self._frames = 0
        self._started = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                self._stop.set()
                break
            now = time.time()
            if self._frame_interval:
                time.sleep(self._frame_interval)
            with self._lock:
                self._latest = frame
                if self._writer is not None:
                    self._writer.write(frame)
                    self._stamps.write(f"{now:.6f}\n")
                    self._frames += 1

    def read(self):
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    @property
    def alive(self) -> bool:
        return not self._stop.is_set()

    @property
    def recording(self) -> bool:
        return self._writer is not None

    @property
    def recorded_frames(self) -> int:
        return self._frames

    @property
    def recorded_seconds(self) -> float:
        return time.time() - self._started if self._writer is not None else 0.0

    def start_recording(self, path: Path) -> tuple[Path, str]:
        """Returns the file and the four character code actually used. H.264
        is what the protocol asks for; OpenCV's wheels do not always ship an
        encoder for it, so this falls back and says which one it got rather
        than failing or pretending."""
        path.parent.mkdir(parents=True, exist_ok=True)
        # mp4v first by default because it is present in every OpenCV wheel.
        # Probing for avc1 when it is absent makes the backend print a wall of
        # red ffmpeg errors that look like a crash and are not one, so trying
        # it is opt in through --codec.
        codes = ("avc1", "mp4v") if self.codec == "avc1" else ("mp4v",)
        for code in codes:
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*code),
                                     self.fps, (self.width, self.height))
            if writer.isOpened():
                stamps = open(path.with_suffix(".timestamps.txt"), "w")
                with self._lock:
                    self._writer, self._stamps = writer, stamps
                    self._frames, self._started = 0, time.time()
                return path, code
            writer.release()
        raise RuntimeError(
            f"no usable mp4 encoder for {path}. Neither avc1 nor mp4v opened. "
            "This usually means opencv-python is not installed (a headless "
            "build has no encoders); reinstall with 'pip install opencv-python'.")

    def stop_recording(self) -> dict:
        with self._lock:
            writer, stamps = self._writer, self._stamps
            self._writer = self._stamps = None
            frames, started = self._frames, self._started
        if writer is None:
            return {}
        writer.release()
        stamps.close()
        elapsed = max(time.time() - started, 1e-6)
        return {"frames": frames, "seconds": elapsed, "measured_fps": frames / elapsed}

    def release(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.stop_recording()
        self.cap.release()


def draw_overlay(view: np.ndarray, det, schema, points, ear_now, baseline,
                 fps: float, infer_ms: float, state: dict) -> np.ndarray:
    """Everything the window shows. Drawn on the inference-resolution copy,
    never on the frames being recorded."""
    h, w = view.shape[:2]
    if points is None:
        cv2.rectangle(view, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
        # above the bottom strip, which is drawn later and would cover it
        cv2.putText(view, "NO FACE", (16, h - 44), FONT, 0.9, (0, 0, 255), 2,
                    cv2.LINE_AA)
    else:
        raw = det.last_haar_box
        if raw is not None:
            cv2.rectangle(view, (raw.x, raw.y), (raw.x + raw.w, raw.y + raw.h),
                          (150, 150, 150), 1, cv2.LINE_AA)
            cv2.putText(view, "haar", (raw.x, max(12, raw.y - 4)), FONT, 0.4,
                        (150, 150, 150), 1, cv2.LINE_AA)
        for i, box in enumerate(det.last_boxes or []):
            last = i == len(det.last_boxes) - 1
            colour = (90, 200, 255) if last else (90, 90, 90)
            cv2.rectangle(view, (box.x0, box.y0),
                          (box.x0 + box.side, box.y0 + box.side), colour,
                          2 if last else 1, cv2.LINE_AA)
            if last:
                cv2.putText(view, f"stage {i + 1} box", (box.x0,
                            max(12, box.y0 - 6)), FONT, 0.45, colour, 1,
                            cv2.LINE_AA)
        for p in schema.points:
            x, y = points[p.index]
            c = GROUP_COLORS[p.group]
            cv2.circle(view, (int(round(x)), int(round(y))), 5, (0, 0, 0), -1,
                       cv2.LINE_AA)
            cv2.circle(view, (int(round(x)), int(round(y))), 3, c, -1, cv2.LINE_AA)

    panel = view.copy()
    cv2.rectangle(panel, (0, 0), (w, 96), (25, 25, 25), -1)
    cv2.addWeighted(panel, 0.65, view, 0.35, 0, view)
    line1 = f"{fps:5.1f} fps   inference {infer_ms:5.0f} ms   {w}x{h}"
    cv2.putText(view, line1, (12, 24), FONT, 0.55, (235, 235, 235), 1, cv2.LINE_AA)

    if ear_now is None:
        cv2.putText(view, "EAR    --", (12, 48), FONT, 0.55, (150, 150, 150), 1,
                    cv2.LINE_AA)
    else:
        rel = ear_now / baseline if baseline else 1.0
        colour = (80, 220, 80) if rel > 0.8 else (60, 60, 255)
        cv2.putText(view, f"EAR {ear_now:5.3f}  L {state['ear_l']:.3f} "
                    f"R {state['ear_r']:.3f}  baseline {baseline:5.3f} "
                    f"({rel * 100:3.0f}%)", (12, 48), FONT, 0.55, colour, 1,
                    cv2.LINE_AA)
    cv2.putText(view, f"dropped {100 * state['dropout']:4.1f}% of the last "
                f"{state['window']} frames", (12, 70), FONT, 0.5,
                (200, 200, 200), 1, cv2.LINE_AA)

    # Recording state gets its own strip along the bottom: on a narrow
    # window a right-aligned indicator in the header collides with the
    # readouts, and the one thing that must never be ambiguous is whether
    # the camera is recording.
    strip = view.copy()
    cv2.rectangle(strip, (0, h - 30), (w, h), (25, 25, 25), -1)
    cv2.addWeighted(strip, 0.7, view, 0.3, 0, view)
    label = state["segment"]
    if state["recording"]:
        cv2.circle(view, (20, h - 15), 7, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(view, f"REC  {label}  {state['rec_seconds']:5.1f}s  "
                    f"{state['rec_frames']}f  (raw frames, no overlay)",
                    (36, h - 10), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(view, f"r record [{label}]   1 still  2 natural  3 turn   "
                    "m mirror   q quit", (12, h - 10), FONT, 0.45,
                    (190, 190, 190), 1, cv2.LINE_AA)
    return view


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/layer1_base.yaml")
    ap.add_argument("--weights", default=None,
                    help="landmarks24_framing.pt; searched for in assets/ and "
                         "the repo root when not given")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--source", default=None,
                    help="a video file instead of the camera, to replay a clip "
                         "through the same viewer")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--infer-width", type=int, default=640,
                    help="inference and display resolution; the recording is "
                         "always full camera resolution")
    ap.add_argument("--out-dir", default="clips")
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "dshow", "msmf", "any"],
                    help="capture backend; dshow is the reliable one on Windows")
    ap.add_argument("--codec", default="mp4v", choices=["mp4v", "avc1"],
                    help="mp4v is in every OpenCV wheel; avc1 is the H.264 "
                         "the protocol asks for and is often absent, in which "
                         "case the probe prints ffmpeg errors and falls back")
    ap.add_argument("--no-mirror", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    schema = load_schema(resolve_path(cfg, require(cfg, "landmark_schema")))
    weights = args.weights or str(resolve_asset(
        "auto", WEIGHT_PATTERNS, "trained landmark weights"))
    det = OurLandmarkDetector(cfg, weights=weights)

    backends = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}
    if args.backend == "auto":
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    else:
        backend = backends[args.backend]

    source = args.source if args.source else args.camera
    print(f"opening {'file ' + str(source) if args.source else f'camera {source}'} ...")
    cam = Camera(source, args.width, args.height, args.fps, backend,
                 codec=args.codec)
    if not cam.cap.isOpened():
        print("FAILED to open the camera.\n"
              "  Windows: another app may hold it (Teams, Zoom, the Camera "
              "app); close them and retry.\n"
              "  Try --backend msmf, or --camera 1 if you have more than one.\n"
              "  Check Settings > Privacy > Camera allows desktop apps.")
        return 1
    print(f"camera reports {cam.width}x{cam.height} at {cam.fps:.1f} fps")
    if (cam.width, cam.height) != (args.width, args.height):
        print(f"  NOTE: asked for {args.width}x{args.height}. Cameras ignore "
              "sizes they do not support. The recording protocol wants "
              "1280x720; anything else should be recorded in the notes.")
    cam.start()

    mirror = not args.no_mirror
    out_dir = Path(args.out_dir)
    segment = "still"
    times: deque = deque(maxlen=30)
    found: deque = deque(maxlen=90)
    ear_hist: deque = deque(maxlen=int(BASELINE_SECONDS * 10))
    state = {"segment": segment, "recording": False, "rec_seconds": 0.0,
             "rec_frames": 0, "dropout": 0.0, "window": 0, "ear_l": 0.0,
             "ear_r": 0.0}
    print("\nwindow keys: r record, 1/2/3 segment, m mirror, q quit")

    while cam.alive:
        frame = cam.read()
        if frame is None:
            time.sleep(0.01)
            continue
        scale = args.infer_width / max(frame.shape[1], 1)
        view = cv2.resize(frame, (args.infer_width,
                                  int(round(frame.shape[0] * scale))))
        if mirror:
            view = cv2.flip(view, 1)

        t0 = time.perf_counter()
        out = det.detect(view)
        infer_ms = (time.perf_counter() - t0) * 1000
        times.append(time.perf_counter())
        found.append(out is not None)

        points = None if out is None else out.points.astype(np.float64)
        ear_now = None
        if points is not None:
            left, right = signals.ear(points)
            ear_now = float(np.mean([left, right]))
            state["ear_l"], state["ear_r"] = left, right
            ear_hist.append(ear_now)
        baseline = float(np.median(ear_hist)) if ear_hist else 0.0

        fps = ((len(times) - 1) / (times[-1] - times[0])
               if len(times) > 1 and times[-1] > times[0] else 0.0)
        state.update(segment=segment, recording=cam.recording,
                     rec_seconds=cam.recorded_seconds,
                     rec_frames=cam.recorded_frames,
                     dropout=1.0 - (sum(found) / len(found)), window=len(found))
        cv2.imshow("dms layer 1 - live", draw_overlay(
            view, det, schema, points, ear_now, baseline, fps, infer_ms, state))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("m"):
            mirror = not mirror
        if key in SEGMENTS and not cam.recording:
            segment = SEGMENTS[key]
            print(f"next recording will be labelled '{segment}'")
        if key == ord("r"):
            if cam.recording:
                info = cam.stop_recording()
                print(f"stopped: {info['frames']} frames in "
                      f"{info['seconds']:.1f} s, measured "
                      f"{info['measured_fps']:.1f} fps")
                if info["frames"] == 0:
                    print("  the file has no frames: the camera stopped "
                          "delivering, or a file source ran out.")
                elif abs(info["measured_fps"] - cam.fps) > 0.1 * cam.fps:
                    print("  WARNING: the measured rate differs from the rate "
                          "written into the file, so playback timing is off. "
                          "The .timestamps.txt sidecar has the true capture "
                          "times; use it rather than assuming a constant rate.")
            else:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path, code = cam.start_recording(
                    out_dir / f"{segment}_{stamp}.mp4")
                print(f"recording {path} with '{code}' at {cam.width}x"
                      f"{cam.height} {cam.fps:.0f} fps (raw frames, no overlay)")
                if code != "avc1":
                    print("  NOTE: H.264 was not available so this is MPEG-4 "
                          "part 2. Fine for the harness; mention it in the "
                          "notes since the protocol asks for H.264.")

    cam.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())

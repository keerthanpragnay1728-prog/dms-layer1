"""MediaPipe Face Landmarker as a LandmarkDetector (milestone 6).

Ported from the legacy `mediapipe.solutions` Face Mesh to the Tasks API
after the solutions path became unrunnable on current environments (see
docs/dependency_notes.md). The Tasks API needs no protobuf pin, is the
supported path going forward, and loads its model from a `.task` bundle
rather than from files inside the wheel.

The wrapper's job is unchanged: MediaPipe finds the face and predicts its
478-point mesh; we select the 24 indices defined in
configs/landmarks_24.yaml so both detectors emit identical semantics. The
mesh topology is the same in both APIs, so the indices carry over, but that
was re-verified against ground truth rather than assumed
(scripts/verify_mediapipe_mapping.py).

Iris points: the 24-point mapping uses mesh indices 468 and 473 for the
pupils, which exist only when the model bundle includes the iris
("attention") head. The legacy API exposed this as refine_landmarks=True; in
the Tasks API it is a property of the bundle, so the wrapper checks the
returned landmark count and fails loudly if the bundle is the 468-point
kind.

Runtime note: the Tasks API loads a shared library that links EGL/GLES even
on CPU, so a headless container needs libegl1 and libgles2 installed. The
error is raised at landmarker construction with a pointer to that.
"""

from __future__ import annotations

import numpy as np

from dms_layer1.assets import resolve_asset
from dms_layer1.config import require
from dms_layer1.detect.interface import LandmarkDetector, Landmarks24, as_rgb
from dms_layer1.landmarks.schema import LandmarkSchema

MESH_POINTS_WITH_IRIS = 478
MODEL_PATTERNS = ["face_landmarker*.task", "*face_landmarker*.task"]


class MediaPipeUnavailable(Exception):
    """Raised when mediapipe, its runtime libraries, or its model bundle
    cannot be used."""


class MediaPipeLandmarkDetector(LandmarkDetector):
    name = "mediapipe"

    def __init__(self, cfg: dict, schema: LandmarkSchema):
        if schema.mediapipe_indices is None:
            raise ValueError(
                "The landmark schema has no mediapipe block; fill it in "
                "configs/landmarks_24.yaml before using this detector.")
        self.indices = list(schema.mediapipe_indices)
        # The schema's refine_landmarks flag is the legacy API's name for the
        # iris head. Under Tasks it is not a runtime flag but a property of
        # the bundle, so it becomes a requirement to check rather than a
        # setting to pass. Either the flag or an iris index in the mapping
        # means the 478-point bundle is required.
        self.needs_iris = bool(schema.mediapipe_refine) or max(self.indices) >= 468

        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError as e:
            raise MediaPipeUnavailable(
                "mediapipe is not installed. Install with 'pip install "
                "mediapipe' (any version with the Tasks API, 0.10.30 or "
                "newer; no protobuf pin needed).") from e
        if not hasattr(mp, "tasks"):
            raise MediaPipeUnavailable(
                f"mediapipe {mp.__version__} has no Tasks API. Install a "
                "newer version.")
        self._mp = mp

        model_path = resolve_asset(
            require(cfg, "detector.mediapipe.model_path"),
            MODEL_PATTERNS, "MediaPipe face_landmarker.task bundle",
            url=require(cfg, "detector.mediapipe.model_url"),
            allow_download=bool(require(cfg, "detector.mediapipe.allow_download")))
        self.model_path = model_path

        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=float(
                require(cfg, "detector.mediapipe.min_detection_confidence")),
        )
        try:
            self.landmarker = vision.FaceLandmarker.create_from_options(options)
        except OSError as e:
            raise MediaPipeUnavailable(
                f"MediaPipe's Tasks runtime failed to load a shared library: "
                f"{e}\nThe Tasks API links EGL/GLES even for CPU inference. "
                "On a headless image install them (apt-get install -y "
                "libegl1 libgles2). Kaggle images ship them already.") from e
        print(f"mediapipe: Tasks FaceLandmarker ready, bundle {model_path.name}")

    def close(self) -> None:
        """Release the landmarker. Worth calling explicitly: left to the
        garbage collector it is closed during interpreter teardown, by which
        point mediapipe's own worker pool is already gone, and the failure
        prints an alarming "Exception ignored in __del__" traceback after a
        run that in fact succeeded. Closing it while the interpreter is alive
        makes that path silent. An atexit hook is too late for the same
        reason: thread shutdown runs before atexit callbacks."""
        landmarker, self.landmarker = getattr(self, "landmarker", None), None
        if landmarker is not None:
            landmarker.close()

    def __enter__(self) -> "MediaPipeLandmarkDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def mesh(self, frame: np.ndarray) -> np.ndarray | None:
        """The full mesh in pixel coordinates, shape (478, 2), or None when no
        face is found. Exposed because verifying the index mapping needs every
        mesh point, not just the 24 we select from it."""
        rgb = np.ascontiguousarray(as_rgb(frame))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(image)
        if not result.face_landmarks:
            return None
        lms = result.face_landmarks[0]
        needed = (MESH_POINTS_WITH_IRIS if self.needs_iris
                  else max(self.indices) + 1)
        if len(lms) < needed:
            raise MediaPipeUnavailable(
                f"The bundle returned {len(lms)} landmarks, and this mapping "
                f"needs {needed} (highest mapped index {max(self.indices)}"
                + (", plus the iris head" if self.needs_iris else "") + "). "
                f"The pupil indices live in the iris head, which only the "
                f"{MESH_POINTS_WITH_IRIS}-point bundle has; this looks like a "
                "468-point bundle. Use the standard face_landmarker.task.")
        h, w = rgb.shape[:2]
        return np.array([(p.x * w, p.y * h) for p in lms], dtype=np.float32)

    def detect(self, frame: np.ndarray) -> Landmarks24 | None:
        pts = self.mesh(frame)
        if pts is None:
            return None
        return Landmarks24(points=pts[self.indices].copy(), source=self.name)

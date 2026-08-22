"""MediaPipe Face Mesh as a LandmarkDetector (milestone 6).

MediaPipe brings its own face detector and predicts 478 mesh points
(468 mesh + two 5-point irises when refine_landmarks is on). This wrapper
selects the 24 indices defined in configs/landmarks_24.yaml so both
detectors emit identical semantics, which is what makes the ablation a
comparison of landmark sources and not of output formats.

mediapipe is imported lazily so the rest of the package works without it
installed. The project pins mediapipe below 1.0: the 1.0 release removed
the legacy solutions API this wrapper uses (the same fate as OpenCV 5 and
the Haar API), and the 0.10 wheels bundle their models inside the package,
which keeps runs free of runtime downloads.
"""

from __future__ import annotations

import numpy as np

from dms_layer1.config import require
from dms_layer1.detect.interface import LandmarkDetector, Landmarks24, as_rgb
from dms_layer1.landmarks.schema import LandmarkSchema


class MediaPipeUnavailable(Exception):
    """Raised when mediapipe is not installed or too new."""


class MediaPipeLandmarkDetector(LandmarkDetector):
    name = "mediapipe"

    def __init__(self, cfg: dict, schema: LandmarkSchema):
        if schema.mediapipe_indices is None:
            raise ValueError(
                "The landmark schema has no mediapipe block; fill it in "
                "configs/landmarks_24.yaml before using this detector."
            )
        self.indices = list(schema.mediapipe_indices)
        try:
            import mediapipe as mp
        except ImportError as e:
            raise MediaPipeUnavailable(
                "mediapipe is not installed. Install with "
                "'pip install \"mediapipe>=0.10,<1\"'."
            ) from e
        if not hasattr(mp, "solutions"):
            raise MediaPipeUnavailable(
                f"mediapipe {mp.__version__} has no solutions API (removed in "
                "1.0). Install 'mediapipe>=0.10,<1'."
            )
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=schema.mediapipe_refine,
            max_num_faces=1,
            min_detection_confidence=float(
                require(cfg, "detector.mediapipe.min_detection_confidence")),
        )

    def detect(self, frame: np.ndarray) -> Landmarks24 | None:
        rgb = as_rgb(frame)
        result = self.mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None
        lms = result.multi_face_landmarks[0].landmark
        if len(lms) <= max(self.indices):
            raise MediaPipeUnavailable(
                f"FaceMesh returned {len(lms)} points but the mapping needs "
                f"index {max(self.indices)}; refine_landmarks must be on for "
                "the iris points."
            )
        h, w = rgb.shape[:2]
        pts = np.array([(lms[i].x * w, lms[i].y * h) for i in self.indices],
                       dtype=np.float32)
        return Landmarks24(points=pts, source=self.name)

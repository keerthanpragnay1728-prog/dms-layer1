"""The component-swap detector: MediaPipe finds the face, our model places
the points.

Milestone 6 measured our front end acquiring the target face on about 40% of
WFLW frames while the landmark model itself is competitive on ground-truth
boxes. A full-pipeline comparison against MediaPipe therefore measures mostly
the DETECTORS, and would let a detector difference masquerade as a landmark
difference. This detector swaps exactly one component so the ablation can
attribute the gap:

    ours            = Haar box        + our landmark model
    this            = MediaPipe box   + our landmark model
    mediapipe       = MediaPipe box   + MediaPipe mesh

Comparing row 1 to row 2 isolates the face detector. Comparing row 2 to row 3
isolates the landmark model on identical inputs, which is the comparison this
project is actually about. It is also a deployable configuration in its own
right for anyone willing to ship MediaPipe's detector.
"""

from __future__ import annotations

import numpy as np

from dms_layer1.config import require
from dms_layer1.data.crops import square_box_around
from dms_layer1.detect.interface import LandmarkDetector, Landmarks24, as_gray


class OurModelOnMediaPipeBox(LandmarkDetector):
    name = "ours_on_mp_box"

    def __init__(self, cfg: dict, ours, mediapipe):
        self.ours = ours
        self.mediapipe = mediapipe
        self.reference_expand = float(require(cfg, "preprocess.reference_expand"))
        self.refine_stages = int(require(cfg, "detector.refine_stages"))
        self.refine_expand = float(require(cfg, "detector.refine_expand"))

    def detect(self, frame: np.ndarray) -> Landmarks24 | None:
        found = self.mediapipe.detect(frame)
        if found is None:
            return None
        gray = as_gray(frame)
        # MediaPipe's points define the face; box them at the canonical
        # framing our model expects, then predict as usual.
        box = square_box_around(found.points.astype(np.float64),
                                self.reference_expand)
        pts = self.ours.predict_in_box(gray, box)
        for _ in range(self.refine_stages - 1):
            pts = self.ours.predict_in_box(
                gray, square_box_around(pts, self.refine_expand))
        return Landmarks24(points=pts.astype(np.float32), source=self.name)

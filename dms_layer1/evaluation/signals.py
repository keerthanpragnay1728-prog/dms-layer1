"""The Layer 2 facing signals, defined once.

Layer 2 does not consume landmarks, it consumes a few scalars derived from
them: eye aspect ratio for blink and drowsiness, mouth aspect ratio for
yawning, and pupil position within the eye for gaze. Those definitions live
here rather than in whichever script needed them first, because a landmark
comparison is only interesting insofar as it changes these, and two
implementations that drift apart would make that comparison meaningless.

The eyelid ordering contract in configs/landmarks_24.yaml is what makes the
EAR formula identical for both eyes: each eye is
[corner, upper, upper, corner, lower, lower].
"""

from __future__ import annotations

import numpy as np

from dms_layer1.landmarks.schema import LandmarkSchema

EYE_BLOCKS = ((0, 6), (6, 12))     # image-left eye, image-right eye
MOUTH_CORNERS = (14, 15)
MOUTH_MIDS = (16, 17)
PUPILS = (12, 13)


def eye_aspect_ratio(p6: np.ndarray) -> float:
    """EAR for one eye from its six points, in the schema's order.

        EAR = (|p1 - p5| + |p2 - p4|) / (2 |p0 - p3|)
    """
    p6 = np.asarray(p6, dtype=np.float64)
    return float((np.linalg.norm(p6[1] - p6[5]) + np.linalg.norm(p6[2] - p6[4]))
                 / (2 * max(np.linalg.norm(p6[0] - p6[3]), 1e-9)))


def ear(points24: np.ndarray) -> tuple[float, float]:
    """(image-left, image-right) eye aspect ratio."""
    p = np.asarray(points24, dtype=np.float64)
    return tuple(eye_aspect_ratio(p[lo:hi]) for lo, hi in EYE_BLOCKS)


def mar(points24: np.ndarray) -> float:
    """Mouth aspect ratio: vertical opening over corner-to-corner width."""
    p = np.asarray(points24, dtype=np.float64)
    width = np.linalg.norm(p[MOUTH_CORNERS[0]] - p[MOUTH_CORNERS[1]])
    return float(np.linalg.norm(p[MOUTH_MIDS[0]] - p[MOUTH_MIDS[1]])
                 / max(width, 1e-9))


def gaze_proxy(points24: np.ndarray) -> np.ndarray:
    """Pupil position inside each eye, as (4,): [left x, left y, right x,
    right y].

    Normalised by eye width and measured from the midpoint of that eye's two
    corners, so it is invariant to face scale and to where the face sits in
    the frame. This is the quantity a gaze-zone classifier consumes, and it
    is why pupil ACCURACY in pixels matters less than pupil STABILITY
    relative to the eye corners: a constant offset is absorbed by per-driver
    calibration, and a wandering one is not.
    """
    p = np.asarray(points24, dtype=np.float64)
    out = []
    for (lo, hi), pupil in zip(EYE_BLOCKS, PUPILS):
        eye = p[lo:hi]
        centre = (eye[0] + eye[3]) / 2.0
        width = max(float(np.linalg.norm(eye[0] - eye[3])), 1e-9)
        out.extend(((p[pupil] - centre) / width).tolist())
    return np.array(out, dtype=np.float64)


def signal_series(sequence_points: np.ndarray) -> dict[str, np.ndarray]:
    """Every Layer 2 signal for a sequence of (T, 24, 2) landmark sets."""
    pts = np.asarray(sequence_points, dtype=np.float64)
    return {
        "ear_left": np.array([ear(f)[0] for f in pts]),
        "ear_right": np.array([ear(f)[1] for f in pts]),
        "ear_mean": np.array([float(np.mean(ear(f))) for f in pts]),
        "mar": np.array([mar(f) for f in pts]),
        "gaze": np.stack([gaze_proxy(f) for f in pts]),
    }


def interocular_of(points24: np.ndarray, schema: LandmarkSchema) -> float:
    """Outer-corner inter-ocular distance, the normaliser used everywhere."""
    p = np.asarray(points24, dtype=np.float64)
    return float(np.linalg.norm(p[schema.nme_right_index] - p[schema.nme_left_index]))

"""Deciding whether a detector's output is the face we are scoring against.

Full-frame evaluation needs one rule for "the detector found the target
face", and every script that reports a detection rate must use the same one
or the rates are not comparable. It lives here rather than in a script so
scripts/compare_detectors.py and scripts/verify_mediapipe_mapping.py cannot
drift apart.
"""

from __future__ import annotations

import numpy as np

from dms_layer1.data.crops import square_box_around
from dms_layer1.detect.haar import box_iou

MATCH_IOU = 0.30
MATCH_EXPAND = 1.30   # same framing as preprocess.reference_expand


def rect_of(box) -> tuple[float, float, float, float]:
    return (box.x0, box.y0, box.x0 + box.side, box.y0 + box.side)


def match_target(pts24: np.ndarray, gt24: np.ndarray) -> bool:
    """Detected points count as the target face when the boxes around both
    24-point sets overlap at IoU >= MATCH_IOU. Matching on the POINTS (the
    only currency both detectors share) folds gross landmark failure into
    the detection rate: a detector that finds the face but scatters its
    points counts as a miss. With trained models that is the intended
    reading, "usable detection"."""
    det_box = square_box_around(pts24, MATCH_EXPAND)
    gt_box = square_box_around(gt24, MATCH_EXPAND)
    return box_iou(rect_of(det_box), rect_of(gt_box)) >= MATCH_IOU

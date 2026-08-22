"""Haar-cascade face detection front-end (milestone 3).

The landmark model refines points inside a face box; it does not find the
face. This module supplies the box: OpenCV Haar cascade detection, plus the
calibrated transform from a raw Haar box to the model's crop box - the same
kind of square box the training cache was built with (1.3x the 98-point
extent). Haar boxes frame a face differently (tighter, roughly brow-to-chin),
so the transform has two parameters, measured against ground truth by
scripts/verify_haar_pipeline.py and stored in the config:

    side   = max(w, h) * box_scale
    centre = haar box centre, shifted down by box_shift_y * side

Crop extraction and coordinate mapping reuse dms_layer1/data/crops.py, so
the detector and the training cache can never disagree on the transform.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from dms_layer1.config import require
from dms_layer1.data.crops import CropBox, square_box_around


class HaarError(Exception):
    """Raised when the cascade file cannot be found or loaded."""


def _load_cascade(name: str) -> cv2.CascadeClassifier:
    """Resolution order: the cascade vendored in this repo's assets/ (pinned
    -- OpenCV wheels do not reliably ship the data files, e.g. opencv 5.x
    wheels drop them), then OpenCV's bundled data dir."""
    candidates = [Path(__file__).resolve().parents[2] / "assets" / name,
                  Path(cv2.data.haarcascades) / name]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise HaarError(
            f"Cascade '{name}' not found. Looked in:\n"
            + "\n".join(f"  {p.parent}" for p in candidates)
        )
    cascade = cv2.CascadeClassifier(str(path))
    if cascade.empty():
        raise HaarError(f"Cascade file failed to load: {path}")
    return cascade


@dataclass(frozen=True)
class FaceBox:
    """Raw detector output, OpenCV convention (top-left x, y, width, height).
    `source` records which cascade produced it: 'frontal', 'profile'
    (left-facing pass) or 'profile_mirrored' (right-facing pass)."""
    x: int
    y: int
    w: int
    h: int
    source: str = "frontal"
    score: float = 0.0          # cascade level weight, higher is stronger

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2

    @property
    def area(self) -> int:
        return self.w * self.h


def box_iou(a: tuple[float, float, float, float],
            b: tuple[float, float, float, float]) -> float:
    """IoU of two (x0, y0, x1, y1) rectangles."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def haar_to_crop_box(box: FaceBox, box_scale: float, box_shift_y: float) -> CropBox:
    """Calibrated Haar box -> integer square model crop box."""
    side_f = max(box.w, box.h) * box_scale
    cx, cy = box.center
    cy += box_shift_y * side_f
    x0_f, y0_f = cx - side_f / 2, cy - side_f / 2
    x0, y0 = int(np.floor(x0_f)), int(np.floor(y0_f))
    side = int(np.ceil(max(x0_f + side_f - x0, y0_f + side_f - y0)))
    return CropBox(x0, y0, side)


def calibrate_haar_to_crop(gt_boxes: list[CropBox],
                           haar_boxes: list[FaceBox]) -> dict:
    """Given matched (ground-truth crop box, raw Haar box) pairs, measure the
    box_scale / box_shift_y that map one to the other. Returns medians and
    quartiles; the medians are the recommended config values."""
    if len(gt_boxes) != len(haar_boxes) or not gt_boxes:
        raise ValueError("need equal, non-empty lists of matched boxes")
    scales, shifts_x, shifts_y = [], [], []
    for gt, hb in zip(gt_boxes, haar_boxes):
        side_h = max(hb.w, hb.h)
        scales.append(gt.side / side_h)
        gcx, gcy = gt.x0 + gt.side / 2, gt.y0 + gt.side / 2
        hcx, hcy = hb.center
        # shift is expressed in units of the CALIBRATED side (= gt side for a
        # perfect match), matching how haar_to_crop_box applies it
        shifts_x.append((gcx - hcx) / gt.side)
        shifts_y.append((gcy - hcy) / gt.side)
    q = lambda v: {"p25": float(np.percentile(v, 25)),
                   "median": float(np.median(v)),
                   "p75": float(np.percentile(v, 75))}
    return {"n_pairs": len(gt_boxes), "box_scale": q(scales),
            "box_shift_x": q(shifts_x), "box_shift_y": q(shifts_y)}


SELECTION_RULES = ("largest", "confidence", "central",
                   "largest_sane", "confidence_sane")


def sane_boxes(boxes: list[FaceBox], image_shape, max_size_frac: float
               ) -> list[FaceBox]:
    """Drop boxes too large to be a face in this frame. Haar at loose
    settings emits oversized boxes on background texture; deployment would
    also see them from a headrest or a window frame."""
    limit = max_size_frac * min(image_shape[:2])
    return [b for b in boxes if max(b.w, b.h) <= limit]


def select_face(boxes: list[FaceBox], rule: str, image_shape,
                max_size_frac: float = 0.9) -> FaceBox | None:
    """Pick the one face to track from a frame's detections. Milestone 6
    showed this choice matters as much as the cascade settings: 'largest'
    is only right when the biggest box IS the subject."""
    if rule not in SELECTION_RULES:
        raise ValueError(f"Unknown selection rule '{rule}'; "
                         f"expected one of {SELECTION_RULES}")
    pool = boxes
    if rule.endswith("_sane"):
        pool = sane_boxes(boxes, image_shape, max_size_frac) or []
    if not pool:
        return None
    if rule in ("largest", "largest_sane"):
        return max(pool, key=lambda b: (b.area, b.score))
    if rule in ("confidence", "confidence_sane"):
        return max(pool, key=lambda b: (b.score, b.area))
    cy, cx = image_shape[0] / 2.0, image_shape[1] / 2.0
    return min(pool, key=lambda b: (b.center[0] - cx) ** 2 + (b.center[1] - cy) ** 2)


class HaarFaceDetector:
    """OpenCV Haar cascade wrapper configured entirely from the YAML config."""

    def __init__(self, cfg: dict):
        self.cascade = _load_cascade(require(cfg, "face_detector.haar_cascade"))
        self.scale_factor = float(require(cfg, "face_detector.scale_factor"))
        self.min_neighbors = int(require(cfg, "face_detector.min_neighbors"))
        self.min_size_frac = float(require(cfg, "face_detector.min_size_frac"))
        self.equalize_hist = bool(require(cfg, "face_detector.equalize_hist"))
        self.box_scale = float(require(cfg, "face_detector.box_scale"))
        self.box_shift_y = float(require(cfg, "face_detector.box_shift_y"))
        # Optional profile-face fallback for turned heads: runs only when the
        # frontal cascade finds nothing. The stock profile cascade detects
        # LEFT-facing profiles, so a mirrored second pass covers right-facing.
        self.selection = str(require(cfg, "face_detector.selection"))
        self.max_size_frac = float(require(cfg, "face_detector.max_size_frac"))
        if self.selection not in SELECTION_RULES:
            raise ValueError(f"face_detector.selection '{self.selection}' "
                             f"must be one of {SELECTION_RULES}")
        self.profile_fallback = bool(require(cfg, "face_detector.profile_fallback"))
        self.profile_cascade = (
            _load_cascade(require(cfg, "face_detector.profile_cascade"))
            if self.profile_fallback else None)

    def _run(self, cascade, img: np.ndarray, min_side: int,
             source: str) -> list[FaceBox]:
        """detectMultiScale3 also returns per-box level weights, which are
        the closest thing a cascade has to a confidence. Falls back to the
        plain call (all scores 0) if the build lacks it."""
        try:
            found, _levels, weights = cascade.detectMultiScale3(
                img, scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors, minSize=(min_side, min_side),
                outputRejectLevels=True)
            scores = [float(w) for w in np.asarray(weights).reshape(-1)]
        except (cv2.error, AttributeError):
            found = cascade.detectMultiScale(
                img, scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors, minSize=(min_side, min_side))
            scores = [0.0] * len(found)
        if len(scores) != len(found):
            scores = [0.0] * len(found)
        return [FaceBox(int(x), int(y), int(w), int(h), source, sc)
                for (x, y, w, h), sc in zip(found, scores)]

    def detect(self, gray: np.ndarray) -> list[FaceBox]:
        """All detected faces, largest first. Input must be grayscale. With
        profile_fallback on, the profile passes run only when the frontal
        cascade finds nothing in the whole image."""
        if gray.ndim != 2:
            raise ValueError(f"expected a grayscale image, got shape {gray.shape}")
        img = cv2.equalizeHist(gray) if self.equalize_hist else gray
        min_side = int(min(gray.shape) * self.min_size_frac)
        boxes = self._run(self.cascade, img, min_side, "frontal")
        if not boxes and self.profile_fallback:
            boxes = self._run(self.profile_cascade, img, min_side, "profile")
            w_img = img.shape[1]
            for b in self._run(self.profile_cascade, cv2.flip(img, 1),
                               min_side, "profile_mirrored"):
                boxes.append(FaceBox(w_img - b.x - b.w, b.y, b.w, b.h, b.source))
        return sorted(boxes, key=lambda b: -b.area)

    def primary_crop_box(self, gray: np.ndarray) -> CropBox | None:
        """The model crop box for the largest detected face, or None. This is
        the front half of the deployment pipeline: frame -> Haar -> crop box;
        crops.extract_square / to_frame_space complete it."""
        box = self.select(gray)
        if box is None:
            return None
        return haar_to_crop_box(box, self.box_scale, self.box_shift_y)

    def select(self, gray: np.ndarray) -> FaceBox | None:
        """The single face this frame is about, per face_detector.selection."""
        return select_face(self.detect(gray), self.selection, gray.shape,
                           self.max_size_frac)


def gt_crop_box(landmarks98: np.ndarray, expand: float) -> CropBox:
    """The ground-truth model crop box for an annotated face - by definition
    identical to what the training cache used."""
    return square_box_around(landmarks98, expand)

"""Tests for the Haar face-detection front-end: cascade loading, the
Haar-box -> crop-box transform, calibration recovery, and detection on
rendered synthetic faces."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dms_layer1.config import load_config
from dms_layer1.data.crops import CropBox
from dms_layer1.data.synthetic import generate_face
from dms_layer1.detect.haar import (FaceBox, HaarFaceDetector, box_iou,
                                    calibrate_haar_to_crop, gt_crop_box,
                                    haar_to_crop_box)

CFG = load_config(REPO / "configs" / "layer1_base.yaml")


def test_cascade_loads_from_vendored_assets():
    det = HaarFaceDetector(CFG)
    assert not det.cascade.empty()


def test_box_iou():
    a = (0, 0, 10, 10)
    assert box_iou(a, a) == 1.0
    assert box_iou(a, (10, 10, 20, 20)) == 0.0
    assert abs(box_iou(a, (5, 0, 15, 10)) - 1 / 3) < 1e-9   # half overlap


def test_haar_to_crop_box_geometry():
    hb = FaceBox(x=100, y=80, w=60, h=64)
    crop = haar_to_crop_box(hb, box_scale=1.5, box_shift_y=0.1)
    side_f = 64 * 1.5
    assert side_f <= crop.side <= side_f + 2          # int box covers float box
    cx, cy = hb.center
    assert abs((crop.x0 + crop.side / 2) - cx) <= 1.0
    assert abs((crop.y0 + crop.side / 2) - (cy + 0.1 * side_f)) <= 1.0


def test_calibration_recovers_known_transform():
    """Fabricate Haar boxes from ground-truth crop boxes with a known scale
    and shift; calibration must recover both, and applying the recovered
    values must reproduce the ground-truth boxes."""
    rng = np.random.default_rng(3)
    true_scale, true_shift_y = 1.42, 0.11
    gts, haars = [], []
    for _ in range(60):
        side = int(rng.integers(120, 400))
        gt = CropBox(int(rng.integers(0, 500)), int(rng.integers(0, 500)), side)
        side_h = side / true_scale
        gcx, gcy = gt.x0 + side / 2, gt.y0 + side / 2
        hcx, hcy = gcx, gcy - true_shift_y * side
        haars.append(FaceBox(int(round(hcx - side_h / 2)), int(round(hcy - side_h / 2)),
                             int(round(side_h)), int(round(side_h))))
        gts.append(gt)

    cal = calibrate_haar_to_crop(gts, haars)
    assert abs(cal["box_scale"]["median"] - true_scale) < 0.03
    assert abs(cal["box_shift_y"]["median"] - true_shift_y) < 0.02
    assert abs(cal["box_shift_x"]["median"]) < 0.02

    for gt, hb in zip(gts, haars):
        rebuilt = haar_to_crop_box(hb, cal["box_scale"]["median"],
                                   cal["box_shift_y"]["median"])
        assert abs(rebuilt.side - gt.side) <= 0.05 * gt.side
        assert abs(rebuilt.x0 - gt.x0) <= 0.05 * gt.side
        assert abs(rebuilt.y0 - gt.y0) <= 0.05 * gt.side


def test_detects_rendered_synthetic_faces():
    """The cascade should find most schematic faces (they are frontal,
    high-contrast, face-like). Lenient threshold: cascade behaviour on
    drawings is not the real benchmark - that runs on WFLW on Kaggle."""
    det = HaarFaceDetector(CFG)
    hits = 0
    for i in range(12):
        rng = np.random.default_rng(i)
        img, pts = generate_face(
            angle_deg=float(rng.uniform(-12, 12)),
            scale=float(rng.uniform(0.8, 1.1)),
            shift=(float(rng.uniform(-25, 25)), float(rng.uniform(-25, 25))))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        boxes = det.detect(gray)
        if boxes:
            gt = gt_crop_box(pts, 1.3)
            iou = box_iou((boxes[0].x, boxes[0].y, boxes[0].x + boxes[0].w,
                           boxes[0].y + boxes[0].h),
                          (gt.x0, gt.y0, gt.x0 + gt.side, gt.y0 + gt.side))
            hits += iou > 0.2
    assert hits >= 6, f"cascade found only {hits}/12 schematic faces"


def _fb(x, y, side, score=0.0):
    from dms_layer1.detect.haar import FaceBox
    return FaceBox(x, y, side, side, "frontal", score)


def test_selection_rules_pick_the_intended_box():
    from dms_layer1.detect.haar import sane_boxes, select_face
    shape = (400, 400)
    face = _fb(170, 170, 60, score=9.0)          # small, central, confident
    spurious = _fb(0, 0, 320, score=1.0)         # big, off-centre, weak
    boxes = [face, spurious]
    assert select_face(boxes, "largest", shape) is spurious
    assert select_face(boxes, "confidence", shape) is face
    assert select_face(boxes, "central", shape) is face
    # size sanity drops the oversized box, so 'largest' then finds the face
    assert select_face(boxes, "largest_sane", shape, max_size_frac=0.5) is face
    assert select_face(boxes, "confidence_sane", shape, max_size_frac=0.5) is face
    assert len(sane_boxes(boxes, shape, 0.5)) == 1
    assert select_face([], "largest", shape) is None
    # a rule that filters everything out returns None rather than guessing
    assert select_face([spurious], "largest_sane", shape, max_size_frac=0.1) is None
    try:
        select_face(boxes, "biggest", shape)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an unknown rule")


def test_detector_scores_are_populated():
    """detectMultiScale3 level weights make the 'confidence' rule possible;
    without them every score is 0 and confidence degenerates to largest."""
    det = HaarFaceDetector(CFG)
    img, _ = generate_face()
    boxes = det.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    assert boxes, "cascade found nothing on a schematic face"
    assert any(b.score != 0.0 for b in boxes)

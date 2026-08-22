"""Milestone 6 tests: the interface contract, our detector end to end on
synthetic data, and the MediaPipe wrapper (skipped with a note when the
package is not installed)."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dms_layer1.config import load_config
from dms_layer1.data.synthetic import generate_face
from dms_layer1.detect.interface import Landmarks24
from dms_layer1.detect.ours import OurLandmarkDetector
from dms_layer1.landmarks.schema import load_schema
from dms_layer1.train.loop import Trainer
from tests.test_resume import _build_test_cache, _cfg

SCHEMA = load_schema(REPO / "configs" / "landmarks_24.yaml")
BASE_CFG = load_config(REPO / "configs" / "layer1_base.yaml")


def test_landmarks24_contract():
    ok = Landmarks24(points=np.random.rand(24, 2).astype(np.float32), source="x")
    assert ok.points.shape == (24, 2) and ok.points.dtype == np.float32
    for bad in (np.zeros((23, 2)), np.full((24, 2), np.nan)):
        try:
            Landmarks24(points=bad, source="x")
        except ValueError:
            continue
        raise AssertionError("invalid Landmarks24 accepted")


def _tiny_detector(tmp: Path) -> OurLandmarkDetector:
    cache_dir = _build_test_cache(tmp)
    cfg = _cfg(cache_dir, tmp / "run", epochs=3)
    Trainer(cfg).train()
    # detector reads haar + pixel settings from the base config, weights and
    # architecture from the checkpoint's embedded arch record
    merged = {**BASE_CFG, "model": cfg["model"], "train": BASE_CFG["train"]}
    return OurLandmarkDetector(merged, weights=str(tmp / "run" / "ckpt" / "best.pth"))


def test_our_detector_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        det = _tiny_detector(Path(tmp))
        img, pts98 = generate_face()
        out = det.detect(img)                      # BGR frame in
        assert out is not None and out.source == "ours"
        h, w = img.shape[:2]
        assert (out.points[:, 0] > -w).all() and (out.points[:, 0] < 2 * w).all()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        assert det.detect(gray) is not None        # gray frame also accepted
        blank = np.zeros((240, 240, 3), dtype=np.uint8)
        assert det.detect(blank) is None


def test_mediapipe_detector_on_schematic():
    if importlib.util.find_spec("mediapipe") is None:
        print("SKIP: mediapipe not installed")
        return
    from dms_layer1.detect.mediapipe_detector import MediaPipeLandmarkDetector

    det = MediaPipeLandmarkDetector(BASE_CFG, SCHEMA)
    img, pts98 = generate_face()
    out = det.detect(img)
    assert out is not None and out.points.shape == (24, 2)
    gt24 = pts98[SCHEMA.wflw_indices]
    iod = np.linalg.norm(gt24[SCHEMA.nme_right_index] - gt24[SCHEMA.nme_left_index])
    off = np.linalg.norm(out.points - gt24, axis=1) / iod
    # pupils are the strictest check of the mapping: iris indices 468/473
    assert off[12] < 0.05 and off[13] < 0.05, f"pupil offsets {off[12:14]}"
    # eyes, mouth and axis must land on their features (loose: cartoon face)
    assert off[:12].max() < 0.15, f"eyelid offsets up to {off[:12].max():.3f}"
    assert off[14:20].max() < 0.15
    # contour is the known approximation; just keep it in the neighbourhood
    assert off[20:24].max() < 0.30
    assert det.detect(np.zeros((240, 240, 3), dtype=np.uint8)) is None

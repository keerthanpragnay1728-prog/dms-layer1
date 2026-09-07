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
    """The Tasks-API wrapper: it loads, it returns the iris head, and its 24
    mapped points land on the anatomy they claim. Real verification is
    scripts/verify_mediapipe_mapping.py on WFLW; this pins the code path and
    catches an index typo."""
    if importlib.util.find_spec("mediapipe") is None:
        print("SKIP: mediapipe not installed")
        return
    from dms_layer1.detect.mediapipe_detector import (MediaPipeLandmarkDetector,
                                                      MediaPipeUnavailable)
    try:
        det = MediaPipeLandmarkDetector(BASE_CFG, SCHEMA)
    except MediaPipeUnavailable as e:
        # no .task bundle attached, or no EGL/GLES on this machine
        print(f"SKIP: mediapipe unavailable ({e})")
        return
    with det:   # closed here rather than at teardown; see close()'s docstring
        _check_mediapipe_mapping(det)


def _check_mediapipe_mapping(det) -> None:
    img, pts98 = generate_face()
    mesh = det.mesh(img)
    assert mesh is not None and mesh.shape[1] == 2
    # 478 not 468: the iris head is the Tasks equivalent of refine_landmarks,
    # and the pupil indices live in it
    assert mesh.shape[0] >= 478, f"bundle returned {mesh.shape[0]} points"

    out = det.detect(img)
    assert out is not None and out.points.shape == (24, 2)
    # a tuple would be read as multi-dimensional indexing, hence list()
    assert np.allclose(out.points, mesh[list(SCHEMA.mediapipe_indices)])
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


def test_two_stage_refinement_runs_and_changes_the_answer():
    """Refinement must be a real second pass: same interface, same output
    contract, different (rebuilt-box) prediction."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cache_dir = _build_test_cache(Path(tmp))
        cfg = _cfg(cache_dir, tmp / "run", epochs=3)
        Trainer(cfg).train()
        merged = {**BASE_CFG, "model": cfg["model"], "train": BASE_CFG["train"]}
        ckpt = str(tmp / "run" / "ckpt" / "best.pth")

        one = {**merged, "detector": {**BASE_CFG["detector"], "refine_stages": 1}}
        two = {**merged, "detector": {**BASE_CFG["detector"], "refine_stages": 2}}
        img, _ = generate_face()
        a = OurLandmarkDetector(one, weights=ckpt).detect(img)
        b = OurLandmarkDetector(two, weights=ckpt).detect(img)
        assert a is not None and b is not None
        assert b.points.shape == (24, 2) and b.source == "ours"
        assert not np.allclose(a.points, b.points), "second stage changed nothing"

        bad = {**merged, "detector": {**BASE_CFG["detector"], "refine_stages": 0}}
        try:
            OurLandmarkDetector(bad, weights=ckpt)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for refine_stages < 1")


def test_checkpoints_record_their_framing_envelope():
    """The stale-weights guard: a checkpoint says what framing it was trained
    for, so an old upload cannot masquerade as a fresh run."""
    from dms_layer1.model.io import describe_weights, load_model

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cache_dir = _build_test_cache(Path(tmp))
        cfg = _cfg(cache_dir, tmp / "run", epochs=1)
        cfg["train"]["augment"]["framing"] = [0.9, 1.0]
        Trainer(cfg).train()
        ckpt = tmp / "run" / "ckpt" / "best.pth"
        _, meta = load_model(ckpt, cfg)
        assert meta["train_meta"]["framing"] == [0.9, 1.0]
        assert meta["train_meta"]["loss"] == "wing"
        line = describe_weights(ckpt, meta)
        assert "framing [0.90, 1.00]" in line and "val NME" in line
        # A file without the record says so rather than staying silent, and
        # must not guess an envelope: saying "assume the narrow one" once told
        # a user the opposite of the truth about their own model.
        none_line = describe_weights(ckpt, {"epoch": 1, "val_nme": None})
        assert "NOT RECORDED" in none_line
        assert "0.87" not in none_line and "assume" not in none_line.lower()
        # a record that exists but has no framing entry is a third case
        partial = describe_weights(ckpt, {"epoch": 1, "val_nme": None,
                                          "train_meta": {"loss": "wing"}})
        assert "NOT RECORDED" in partial and "no framing entry" in partial
        # a record stamped after the fact says so
        stamped = describe_weights(ckpt, {"epoch": 1, "val_nme": None,
                                          "train_meta": {"framing": [0.85, 1.6],
                                                         "stamped_after_the_fact": "run.yaml"}})
        assert "framing [0.85, 1.60]" in stamped and "not by the trainer" in stamped


def test_alternative_mapping_reads_the_same_mesh():
    """The second mapping must be a different SELECTION off one mesh, not a
    second model: that is what makes 'would other indices change the verdict'
    answerable at all."""
    if importlib.util.find_spec("mediapipe") is None:
        print("SKIP: mediapipe not installed")
        return
    from dms_layer1.detect.mediapipe_detector import (MediaPipeLandmarkDetector,
                                                      MediaPipeUnavailable)
    try:
        det = MediaPipeLandmarkDetector(BASE_CFG, SCHEMA)
    except MediaPipeUnavailable as e:
        print(f"SKIP: mediapipe unavailable ({e})")
        return
    with det:
        img, _ = generate_face()
        mesh = det.mesh(img)
        alt = list(SCHEMA.mediapipe_indices)
        alt[2] = 157        # a neighbouring vertex on the same eyelid ring
        remapped = det.remapped(alt, "alt")
        out = remapped.detect(img)
        assert out is not None and out.source == "alt"
        assert np.allclose(out.points, mesh[alt])
        assert not np.allclose(out.points, det.detect(img).points)
        try:
            det.remapped(alt[:5], "short")
        except ValueError:
            pass
        else:
            raise AssertionError("accepted a wrong-length index list")

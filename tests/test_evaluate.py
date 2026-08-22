"""Metric correctness tests: NME, per-group NME, failure rate and subset
splits computed on hand-crafted arrays with known answers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dms_layer1.evaluation import metrics
from dms_layer1.landmarks.schema import load_schema

SCHEMA = load_schema(REPO / "configs" / "landmarks_24.yaml")


def _gt_face() -> np.ndarray:
    """A face whose inter-ocular distance is exactly 0.5."""
    gt = np.random.default_rng(0).uniform(0.2, 0.8, size=(24, 2))
    gt[SCHEMA.nme_left_index] = (0.25, 0.4)
    gt[SCHEMA.nme_right_index] = (0.75, 0.4)
    return gt


def test_nme_known_answer():
    gt = _gt_face()[None]
    pred = gt.copy()
    assert metrics.nme_per_face(pred, gt, SCHEMA)[0] == 0.0
    # shift every point by 0.05 in x: error 0.05 / IOD 0.5 = 10% exactly
    pred2 = gt.copy()
    pred2[0, :, 0] += 0.05
    nme = metrics.nme_per_face(pred2, gt, SCHEMA)
    assert abs(nme[0] - 0.10) < 1e-9


def test_group_nme_isolates_the_broken_group():
    gt = _gt_face()[None]
    pred = gt.copy()
    for i in SCHEMA.indices_of_group("pupils"):
        pred[0, i, 1] += 0.10                    # 20% of IOD, pupils only
    g = metrics.group_nme(pred, gt, SCHEMA)
    assert abs(g["pupils"] - 0.20) < 1e-9
    for other in ("eyelids", "mouth", "axis", "contour"):
        assert g[other] == 0.0


def test_failure_rate_and_subsets():
    nme = np.array([0.05, 0.15, 0.09, 0.30])
    assert metrics.failure_rate(nme, 0.10) == 0.5
    attrs = np.array([[1, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0]], dtype=np.uint8)
    names = ["pose", "expression", "illumination", "makeup", "occlusion", "blur"]
    out = metrics.subset_nme(nme, attrs, names, 0.10)
    assert out["largepose"]["n"] == 2
    assert abs(out["largepose"]["nme_pct"] - 10.0) < 1e-9      # (5% + 15%) / 2
    assert out["largepose"]["failure_pct"] == 50.0
    assert out["expression"]["n"] == 1 and out["expression"]["failure_pct"] == 100.0
    assert out["no_flags"]["n"] == 1 and out["no_flags"]["failure_pct"] == 0.0
    assert out["illumination"]["n"] == 0 and out["illumination"]["nme_pct"] is None


def test_scale_invariance():
    """NME must be identical in [0,1] crop space and in pixel space."""
    rng = np.random.default_rng(3)
    gt = rng.uniform(0.1, 0.9, size=(5, 24, 2))
    pred = gt + rng.normal(0, 0.01, size=gt.shape)
    a = metrics.nme_per_face(pred, gt, SCHEMA)
    b = metrics.nme_per_face(pred * 448, gt * 448, SCHEMA)
    assert np.abs(a - b).max() < 1e-12

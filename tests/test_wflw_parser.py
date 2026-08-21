"""Tests for WFLW annotation parsing and dataset path resolution, using a
synthetic dataset tree that mirrors the real layout."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dms_layer1.data import wflw
from dms_layer1.data.synthetic import write_synthetic_dataset

ATTRS = ["pose", "expression", "illumination", "makeup", "occlusion", "blur"]


def _cfg(root: str | Path) -> dict:
    return {
        "dataset": {
            "root": str(root),
            "images_dir": "WFLW_images",
            "annotations_dir": "WFLW_annotations",
            "train_list": "list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt",
            "test_list": "list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt",
            "attribute_names": ATTRS,
        }
    }


def test_roundtrip_parse():
    with tempfile.TemporaryDirectory() as tmp:
        root = write_synthetic_dataset(tmp, ATTRS, num_per_split=6, seed=3)
        records, paths = wflw.load_split(_cfg(root), "test")
        assert len(records) == 6
        for rec in records:
            assert rec.landmarks98.shape == (98, 2)
            assert np.isfinite(rec.landmarks98).all()
            x0, y0, x1, y1 = rec.bbox
            assert x0 < x1 and y0 < y1
            # bbox was written as landmark extent +12px margin
            assert (rec.landmarks98.min(axis=0) >= np.array([x0, y0]) - 0.51).all()
            assert (rec.landmarks98.max(axis=0) <= np.array([x1, y1]) + 0.51).all()
            assert set(rec.attributes) == set(ATTRS)
            assert wflw.image_path(paths, rec).is_file()


def test_subset_selection():
    with tempfile.TemporaryDirectory() as tmp:
        root = write_synthetic_dataset(tmp, ATTRS, num_per_split=9, seed=3)
        records, _ = wflw.load_split(_cfg(root), "test")
        largepose = wflw.subset_records(records, "largepose")
        occluded = wflw.subset_records(records, "occlusion")
        assert all(r.attributes["pose"] == 1 for r in largepose)
        assert all(r.attributes["occlusion"] == 1 for r in occluded)
        assert len(largepose) == 3 and len(occluded) == 3  # every 3rd face flagged


def _expect_wflw_error(fn, label):
    try:
        fn()
    except wflw.WFLWError:
        return
    raise AssertionError(f"expected WFLWError for {label}")


def test_bad_paths_fail_clearly():
    _expect_wflw_error(lambda: wflw.resolve_paths(_cfg("/nonexistent/wflw/root")),
                       "missing dataset root")
    with tempfile.TemporaryDirectory() as tmp:
        root = write_synthetic_dataset(tmp, ATTRS, num_per_split=2, seed=0)
        cfg = _cfg(root)
        cfg["dataset"]["images_dir"] = "WFLW_imagez"
        try:
            wflw.resolve_paths(cfg)
        except wflw.WFLWError as e:
            # error must show what actually exists so the config can be fixed
            assert "WFLW_images" in str(e)
        else:
            raise AssertionError("expected WFLWError for wrong images_dir")


def test_malformed_lines_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        good = "1.0 " * 196 + "0 0 10 10 " + "0 0 0 0 0 0" + " img.jpg"
        cases = {
            "truncated": "1.0 " * 195 + "0 0 10 10 0 0 0 0 0 0 img.jpg",
            "bad attr": "1.0 " * 196 + "0 0 10 10 " + "0 0 7 0 0 0" + " img.jpg",
            "degenerate rect": "1.0 " * 196 + "10 10 0 0 " + "0 0 0 0 0 0" + " img.jpg",
        }
        p = Path(tmp) / "ok.txt"
        p.write_text(good + "\n")
        assert len(wflw.parse_annotation_file(p, ATTRS)) == 1
        for label, line in cases.items():
            p = Path(tmp) / "bad.txt"
            p.write_text(line + "\n")
            _expect_wflw_error(lambda: wflw.parse_annotation_file(p, ATTRS), label)

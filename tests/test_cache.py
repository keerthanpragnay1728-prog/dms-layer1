"""End-to-end test of the crop cache: build from a synthetic WFLW tree,
reload, and confirm the cached labels reproduce the annotations."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dms_layer1.data import cache as cachemod
from dms_layer1.data import wflw
from dms_layer1.data.synthetic import write_synthetic_dataset
from dms_layer1.landmarks.schema import load_schema

ATTRS = ["pose", "expression", "illumination", "makeup", "occlusion", "blur"]


def _cfg(root, out_dir) -> dict:
    return {
        "_config_path": str(REPO / "configs" / "layer1_base.yaml"),
        "landmark_schema": "landmarks_24.yaml",
        "seed": 0,
        "dataset": {
            "root": str(root),
            "images_dir": "WFLW_images",
            "annotations_dir": "WFLW_annotations",
            "train_list": "list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt",
            "test_list": "list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt",
            "attribute_names": ATTRS,
        },
        "preprocess": {"cache_size": 96, "crop_expand": 1.3,
                       "num_preview": 4, "out_dir": str(out_dir)},
    }


def test_build_load_and_label_fidelity():
    with tempfile.TemporaryDirectory() as tmp:
        root = write_synthetic_dataset(Path(tmp) / "ds", ATTRS, num_per_split=8, seed=5)
        cfg = _cfg(root, Path(tmp) / "cache")
        manifest = cachemod.build_cache(cfg, "test")
        assert manifest["n_faces"] == 8
        assert manifest["landmark_min"] >= 0.0 and manifest["landmark_max"] <= 1.0

        data = cachemod.load_cache(cfg["preprocess"]["out_dir"], "test")
        assert data.crops.shape == (8, 96, 96) and data.crops.dtype == np.uint8
        assert data.landmarks.shape == (8, 24, 2) and data.landmarks.dtype == np.float32
        assert data.attrs.shape == (8, 6) and len(data.rel_paths) == 8
        assert data.crops.std() > 5, "crops look empty"

        # cached [0,1] labels + boxes must reproduce the annotation coords
        schema = load_schema(REPO / "configs" / "landmarks_24.yaml")
        records, _ = wflw.load_split(cfg, "test")
        truth = np.stack([r.landmarks98[schema.wflw_indices] for r in records])
        boxes = data.boxes.astype(np.float64)
        rebuilt = (data.landmarks.astype(np.float64) * boxes[:, None, 2:3]
                   + boxes[:, None, 0:2])
        assert np.abs(rebuilt - truth).max() < 0.01

        # attribute flags survive the round trip, index-aligned
        for i, rec in enumerate(records):
            assert list(data.attrs[i]) == [rec.attributes[a] for a in ATTRS]


def test_load_missing_cache_fails_clearly():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            cachemod.load_cache(tmp, "train")
        except cachemod.CacheError as e:
            assert "build_crop_cache" in str(e)
        else:
            raise AssertionError("expected CacheError for empty cache dir")

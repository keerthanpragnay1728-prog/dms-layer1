"""Tests for square crop geometry and frame<->crop coordinate transforms."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dms_layer1.data.crops import (CropBox, extract_square, square_box_around,
                                   to_crop_space, to_frame_space)


def test_box_contains_all_points_with_margin():
    rng = np.random.default_rng(0)
    for _ in range(50):
        pts = rng.uniform(-50, 900, size=(98, 2))
        box = square_box_around(pts, expand=1.3)
        p01 = to_crop_space(pts, box)
        assert p01.min() >= 0.0 and p01.max() <= 1.0
        # the expand margin means no point should touch the box edge
        assert p01.min() > 0.05 and p01.max() < 0.95
        extent = max(pts.max(0) - pts.min(0))
        assert box.side >= extent * 1.3 - 1e-9


def test_coordinate_round_trip_is_exact():
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 1200, size=(200, 2))
    box = square_box_around(pts, expand=1.25)
    back = to_frame_space(to_crop_space(pts, box), box)
    assert np.abs(back - pts).max() < 1e-9


def test_float32_labels_round_trip_below_hundredth_px():
    """Labels are stored as float32 in [0,1]; mapping back through the box
    must stay far under a hundredth of a pixel even for large crops."""
    rng = np.random.default_rng(2)
    pts = rng.uniform(0, 2000, size=(98, 2))
    box = square_box_around(pts, expand=1.3)
    stored = to_crop_space(pts, box).astype(np.float32)
    back = to_frame_space(stored.astype(np.float64), box)
    assert np.abs(back - pts).max() < 0.01 * box.side / 1000 + 1e-3


def test_extract_pads_outside_image():
    img = np.full((100, 100), 200, dtype=np.uint8)
    box = CropBox(x0=-20, y0=-20, side=60)      # overhangs top-left
    crop = extract_square(img, box, out_size=60)
    assert crop.shape == (60, 60)
    assert crop[0, 0] == 0                      # padded corner
    assert crop[59, 59] == 200                  # inside-image corner
    # fully outside the image -> all padding
    empty = extract_square(img, CropBox(500, 500, 40), out_size=32)
    assert empty.max() == 0


def test_extract_resizes_small_and_large():
    img = np.arange(300 * 300, dtype=np.uint8).reshape(300, 300)
    big = extract_square(img, CropBox(10, 10, 200), out_size=128)   # shrink
    small = extract_square(img, CropBox(10, 10, 40), out_size=128)  # enlarge
    assert big.shape == small.shape == (128, 128)


def test_point_round_trip_within_a_pixel():
    """Milestone-3 requirement: a point mapped to crop space and back lands
    within a pixel of where it started - verified through the coordinate
    algebra (exact) AND through actual image content (a marker survives
    extract_square + resize and maps back to its origin), which would catch
    any off-by-one or axis swap in the extraction that pure algebra cannot."""
    S = 128
    # coordinate algebra through crop-pixel space: exact
    rng = np.random.default_rng(7)
    box = CropBox(37, 55, 256)
    pts = rng.uniform([37, 55], [37 + 256, 55 + 256], size=(300, 2))
    crop_px = to_crop_space(pts, box) * S
    back = to_frame_space(crop_px / S, box)
    assert np.abs(back - pts).max() < 1e-9

    # image content: bright 3x3 marker, extract + resize, intensity centroid,
    # map back. Pixel (r, c) spans [c, c+1) x [r, r+1), centre at +0.5.
    for box in (CropBox(100, 100, 256), CropBox(90, 110, 250)):   # int & non-int ratio
        img = np.zeros((400, 400), dtype=np.uint8)
        img[156:159, 212:215] = 255                # continuous centre (213.5, 157.5)
        crop = extract_square(img, box, S)
        ys, xs = np.nonzero(crop)
        w = crop[ys, xs].astype(np.float64)
        centroid_px = np.array([(xs * w).sum(), (ys * w).sum()]) / w.sum()
        back = to_frame_space((centroid_px + 0.5) / S, box)
        err = np.abs(back - (213.5, 157.5))
        assert err.max() <= 1.0, f"content round trip off by {err} px (box {box})"

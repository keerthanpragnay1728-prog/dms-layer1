"""Model tests: shape contract, size budget, centre initialisation."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dms_layer1.model.net import LandmarkNet, model_size_mb


def test_output_shape_and_size_budget():
    m = LandmarkNet(num_points=24, width=32, input_size=112)
    out = m(torch.randn(3, 1, 112, 112))
    assert out.shape == (3, 24, 2)
    assert model_size_mb(m) < 5.0, f"model is {model_size_mb(m):.2f} MB, budget is 5"


def test_untrained_model_predicts_crop_centre():
    torch.manual_seed(0)
    m = LandmarkNet(width=16, input_size=112).eval()
    with torch.no_grad():
        out = m(torch.rand(8, 1, 112, 112))
    assert (out.mean() - 0.5).abs() < 0.05


def test_input_size_must_fit_downsampling():
    try:
        LandmarkNet(input_size=100)
    except ValueError:
        return
    raise AssertionError("expected ValueError for input_size not divisible by 16")

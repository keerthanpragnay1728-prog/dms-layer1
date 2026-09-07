"""Loss tests: zero at target, Wing continuity at the regime switch, and
config selection."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dms_layer1.train.loss import l2_loss, make_loss, wing_loss

W, EPS = 10.0, 2.0


def _cfg(kind: str) -> dict:
    return {"train": {"loss": kind, "wing": {"w": W, "epsilon": EPS}},
            "model": {"input_size": 112}}


def test_zero_at_target():
    x = torch.rand(4, 24, 2)
    assert float(l2_loss(x, x)) == 0.0
    assert float(wing_loss(x, x, W, EPS)) == 0.0


def test_wing_matches_closed_form():
    t = torch.zeros(1, 1, 2)
    small = torch.full((1, 1, 2), 1.0)          # |x| = 1 < w
    expected = W * math.log(1 + 1.0 / EPS)
    assert abs(float(wing_loss(small, t, W, EPS)) - expected) < 1e-5
    big = torch.full((1, 1, 2), 25.0)           # |x| = 25 > w
    c = W - W * math.log(1 + W / EPS)
    assert abs(float(wing_loss(big, t, W, EPS)) - (25.0 - c)) < 1e-5


def test_wing_is_continuous_at_w():
    t = torch.zeros(1, 1, 2)
    lo = float(wing_loss(torch.full((1, 1, 2), W - 1e-4), t, W, EPS))
    hi = float(wing_loss(torch.full((1, 1, 2), W + 1e-4), t, W, EPS))
    assert abs(lo - hi) < 1e-3


def test_selection_by_config():
    p, t = torch.rand(2, 24, 2), torch.rand(2, 24, 2)
    assert float(make_loss(_cfg("l2"))(p, t)) > 0
    assert float(make_loss(_cfg("wing"))(p, t)) > 0
    try:
        make_loss(_cfg("huber"))
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown loss kind")

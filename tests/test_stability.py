"""Milestone 7 tests: the sequence generator's image/annotation coupling, and
the jitter metrics' behaviour on cases whose answer is known by construction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dms_layer1.data.crops import CropBox
from dms_layer1.data.synthetic import canonical_landmarks98
from dms_layer1.evaluation import signals
from dms_layer1.landmarks.schema import load_schema
from dms_layer1.stability.metrics import (box_jitter, crossing_rate, gap_runs,
                                          highfreq_jitter, residual_jitter,
                                          rule_of_three, split_jumps,
                                          sustained_crossing_rate,
                                          threshold_margin)
from dms_layer1.stability.sequences import SequenceSpec, make_sequence

SCHEMA = load_schema(REPO / "configs" / "landmarks_24.yaml")


def test_annotation_follows_the_image_exactly():
    """The whole harness rests on this: the points must be warped by the same
    affine as the pixels, or ground truth silently stops being ground truth
    and every jitter number is wrong in a way nothing else would catch."""
    img = np.zeros((120, 160), dtype=np.uint8)
    img[40, 90] = 255                       # one bright pixel
    pts = np.array([[90.0, 40.0]])          # a "landmark" on that pixel
    spec = SequenceSpec(frames=6, translate_frac=0.05, rotate_deg=3.0,
                        scale_frac=0.02, noise_sigma=0.0, brightness_amp=0.0)
    frames, seq = make_sequence(img, pts, spec, np.random.default_rng(0),
                                face_side=40.0)
    assert len(frames) == 6 and seq.shape == (6, 1, 2)
    for frame, p in zip(frames, seq):
        yx = np.unravel_index(int(np.argmax(frame)), frame.shape)
        moved = np.array([yx[1], yx[0]], dtype=np.float64)
        assert np.linalg.norm(moved - p[0]) <= 1.5, (
            f"pixel at {moved} but annotation says {p[0]}")


def test_zero_amplitude_sequence_is_static():
    img = (np.random.RandomState(0).rand(80, 80) * 255).astype(np.uint8)
    pts = np.array([[10.0, 20.0], [30.0, 40.0]])
    spec = SequenceSpec(frames=5, translate_frac=0.0, rotate_deg=0.0,
                        scale_frac=0.0, noise_sigma=0.0, brightness_amp=0.0)
    frames, seq = make_sequence(img, pts, spec, np.random.default_rng(1), 50.0)
    for f in frames:
        assert np.array_equal(f, frames[0])
    assert np.allclose(seq, seq[0])


def test_sequences_are_seeded():
    img = (np.random.RandomState(2).rand(60, 60) * 255).astype(np.uint8)
    pts = np.array([[20.0, 20.0]])
    spec = SequenceSpec(frames=4)
    a = make_sequence(img, pts, spec, np.random.default_rng(7), 30.0)[1]
    b = make_sequence(img, pts, spec, np.random.default_rng(7), 30.0)[1]
    c = make_sequence(img, pts, spec, np.random.default_rng(8), 30.0)[1]
    assert np.allclose(a, b) and not np.allclose(a, c)


def test_jitter_separates_wobble_from_offset():
    """A prediction can be steady and wrong, or accurate and unsteady. The
    metric must not confuse them: a calibrated downstream stage absorbs the
    first and cannot absorb the second."""
    gt = np.tile(np.random.RandomState(3).rand(24, 2) * 100, (30, 1, 1))
    iod = np.full(30, 50.0)
    assert residual_jitter(gt, gt, iod)["mean_pct"] == 0.0

    offset = residual_jitter(gt + 5.0, gt, iod)
    assert offset["mean_pct"] < 1e-9, "a constant offset is not jitter"
    assert abs(offset["bias_mean_pct"] - 100 * np.hypot(5, 5) / 50) < 1e-6

    noisy = gt + np.random.RandomState(4).normal(0, 0.5, gt.shape)
    got = residual_jitter(noisy, gt, iod)["mean_pct"]
    assert abs(got - 100 * 0.5 * np.sqrt(2) / 50) < 0.25


def test_jitter_is_measured_against_motion_not_position():
    """A perfect tracker on a moving face has zero jitter. This is why the
    residual, not the trajectory, is what gets a standard deviation."""
    base = np.random.RandomState(5).rand(24, 2) * 100
    drift = np.linspace(0, 40, 25)[:, None, None]
    gt = base[None] + drift
    assert residual_jitter(gt, gt, np.full(25, 50.0))["mean_pct"] == 0.0


def test_box_jitter_units_and_split():
    gt = [CropBox(100, 100, 200) for _ in range(20)]
    steady = [CropBox(110, 100, 200) for _ in range(20)]   # offset, not moving
    st = box_jitter(steady, gt)
    assert abs(st["centre_bias_x"] - 0.05) < 1e-9
    assert st["centre_jitter_x"] < 1e-9

    rng = np.random.RandomState(6)
    moving = [CropBox(100 + int(rng.normal(0, 10)), 100, 200) for _ in range(200)]
    mv = box_jitter(moving, gt)
    assert 0.02 < mv["centre_jitter_x"] < 0.08     # 10 px sd on a 200 px box
    assert box_jitter([None] * 3, gt[:3])["n"] == 0


def test_crossing_rate_and_highfreq_estimator():
    assert crossing_rate(np.full(20, 0.30), 0.2) == 0.0
    assert crossing_rate(np.array([0.1, 0.3] * 10), 0.2) == 1.0
    smooth = np.sin(np.linspace(0, 2, 60))
    assert highfreq_jitter(smooth) < 0.02
    assert highfreq_jitter(smooth + np.random.RandomState(9).normal(0, 0.1, 60)) > 0.05


def test_layer2_signals_have_the_expected_geometry():
    p = canonical_landmarks98()[SCHEMA.wflw_indices]
    left, right = signals.ear(p)
    assert 0.1 < left < 0.6 and abs(left - right) < 1e-6, "eyes are symmetric"
    assert 0.05 < signals.mar(p) < 1.0

    # a centred pupil reads zero; moving it to the outer corner reads -0.5
    assert np.allclose(signals.gaze_proxy(p), 0.0, atol=1e-6)
    moved = p.copy()
    moved[12] = p[0]                       # left pupil onto the outer corner
    assert abs(signals.gaze_proxy(moved)[0] + 0.5) < 1e-6

    series = signals.signal_series(np.stack([p, p, p]))
    assert series["gaze"].shape == (3, 4)
    assert np.allclose(series["ear_mean"], series["ear_mean"][0])


def test_jumps_are_separated_from_wobble():
    """A cascade steps as well as wobbles, and one standard deviation cannot
    tell a few large steps from constant small noise."""
    rng = np.random.RandomState(0)
    v = rng.normal(0, 0.01, 200)
    v[80:] += 0.5                      # one discrete step
    r = split_jumps(v, jump_threshold=0.1)
    assert abs(r["jump_rate"] - 1 / 199) < 1e-9
    assert abs(r["continuous_sd"] - 0.01) < 0.003, "wobble should survive the step"
    assert r["largest_jump"] > 0.4

    calm = split_jumps(rng.normal(0, 0.01, 200), 0.1)
    assert calm["jump_rate"] == 0.0
    assert abs(calm["continuous_sd"] - 0.01) < 0.003


def test_gap_structure_not_just_rate():
    """The same dropout rate can be one blackout or scattered singles, and
    Layer 2 aggregates over a window, so the difference matters."""
    scattered = gap_runs([True, False] * 10)
    burst = gap_runs([True] * 10 + [False] * 10)
    assert abs(scattered["dropout"] - burst["dropout"]) < 1e-9
    assert scattered["longest_gap"] == 1 and scattered["gaps"] == 10
    assert burst["longest_gap"] == 10 and burst["gaps"] == 1
    assert gap_runs([])["dropout"] == 1.0


def test_margin_and_null_bound():
    steady = np.random.RandomState(1).normal(0.30, 0.01, 300)
    assert 4.0 < threshold_margin(steady, 0.25) < 6.0
    assert threshold_margin(np.full(10, 0.3), 0.25) == float("inf")
    # a null result is a bound, not a zero
    assert abs(rule_of_three(0, 1800) - 3 / 1800) < 1e-12
    assert abs(rule_of_three(4, 1800) - 4 / 1800) < 1e-12


def test_sustained_crossings_ignore_single_frame_dips():
    sig = np.full(60, 0.30)
    sig[10] = 0.10                     # one-frame blip, not a blink
    assert sustained_crossing_rate(sig, 0.2, 3) == 0.0
    sig[30:35] = 0.10                  # a real closure
    assert sustained_crossing_rate(sig, 0.2, 3) > 0
    assert crossing_rate(sig, 0.2) > sustained_crossing_rate(sig, 0.2, 3)

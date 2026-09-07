"""Tests for the shared face-frame magnitudes: on the symmetric canonical
face every asymmetry magnitude must be ~0, and all magnitudes must be
invariant to in-plane rotation (the property the checks rely on)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dms_layer1.data import frame as ff
from dms_layer1.data.synthetic import canonical_landmarks98


def _rotated(lm: np.ndarray, deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    center = lm.mean(axis=0)
    return (lm - center) @ rot.T + center


def test_symmetric_face_has_zero_asymmetry():
    fr = ff.FaceFrame(canonical_landmarks98()[None])
    lowest, margin = ff.chin_lowest(fr)
    assert lowest[0] == 16 and margin[0] < 1e-6
    assert ff.pair_height_mismatch(fr, 4, 28)[0] < 1e-6
    assert ff.pair_height_mismatch(fr, 8, 24)[0] < 1e-6
    assert abs(ff.nose_offset(fr)[0]) < 1e-6
    assert abs(ff.yaw_proxy_contour(fr)[0]) < 1e-6
    assert abs(ff.yaw_proxy_eyewidth(fr)[0]) < 1e-6
    assert ff.check_chin(fr)[0] and ff.check_nose_midline(fr)[0]
    assert ff.check_pair(fr, 4, 28)[0] and ff.check_pair(fr, 8, 24)[0]


def test_magnitudes_are_roll_invariant():
    lm = canonical_landmarks98()
    lm_rot = _rotated(lm, 25.0)
    a, b = ff.FaceFrame(lm[None]), ff.FaceFrame(lm_rot[None])
    for fn in (lambda f: ff.chin_lowest(f)[1],
               lambda f: ff.pair_height_mismatch(f, 4, 28),
               lambda f: ff.nose_offset(f),
               lambda f: ff.yaw_proxy_contour(f),
               lambda f: f.face_height,
               lambda f: f.iod_corner):
        assert abs(fn(a)[0] - fn(b)[0]) < 1e-6, "magnitude changed under roll"


def _yawed(lm: np.ndarray, theta_deg: float) -> np.ndarray:
    """Cheap 3D yaw: treat the face as a vertical cylinder (radius R around
    the facial midline), lift each point to its cylinder depth and rotate
    about the vertical axis. Good enough to produce realistic asymmetry."""
    out = lm.copy()
    cx, radius = 224.0, 170.0
    dx = lm[:, 0] - cx
    z = np.sqrt(np.clip(radius**2 - dx**2, 0.0, None))
    z[51:60] += [10, 18, 26, 35, 12, 14, 16, 14, 12]   # the nose protrudes
    t = np.deg2rad(theta_deg)
    out[:, 0] = cx + dx * np.cos(t) - z * np.sin(t)
    return out


def test_yaw_machinery_detects_yaw():
    """The diagnostic's yaw proxy must grow with actual yaw, keep a
    consistent sign per direction, and relate consistently to the nose
    offset. Deliberately sign-agnostic: the polarity depends on annotation
    convention, so only consistency and monotonicity are asserted."""
    lm = canonical_landmarks98()
    thetas = [5, 10, 20, 30]
    proxies, offsets = [], []
    for t in thetas:
        fr = ff.FaceFrame(_yawed(lm, t)[None])
        proxies.append(ff.yaw_proxy_contour(fr)[0])
        offsets.append(ff.nose_offset(fr)[0] / fr.iod_corner[0])

    mags = [abs(p) for p in proxies]
    assert all(b > a for a, b in zip(mags, mags[1:])), \
        f"|yaw proxy| not increasing with yaw: {proxies}"
    assert len({np.sign(p) for p in proxies}) == 1, "proxy sign flips within one direction"
    assert len({np.sign(o) for o in offsets}) == 1, "nose offset sign flips within one direction"
    assert abs(offsets[-1]) > 0.1, "nose offset should be large at 30 deg yaw"

    # opposite yaw direction flips both signs
    fr_neg = ff.FaceFrame(_yawed(lm, -20)[None])
    assert np.sign(ff.yaw_proxy_contour(fr_neg)[0]) == -np.sign(proxies[0])
    assert np.sign(ff.nose_offset(fr_neg)[0]) == -np.sign(offsets[0])


def test_frame_axes_are_orthonormal():
    fr = ff.FaceFrame(canonical_landmarks98()[None])
    assert abs(np.linalg.norm(fr.u[0]) - 1) < 1e-9
    assert abs(np.linalg.norm(fr.v[0]) - 1) < 1e-9
    assert abs(np.dot(fr.u[0], fr.v[0])) < 1e-9
    # on an upright face u points image-right and v points image-down
    assert fr.u[0, 0] > 0.99 and fr.v[0, 1] > 0.99

"""Face-aligned frame and layout-check magnitudes on WFLW 98-point arrays.

Shared by scripts/verify_layout.py (applies pass/fail thresholds) and
scripts/diagnose_layout_failures.py (reports the underlying magnitudes).
One implementation so the two scripts can never disagree about what a
"failing face" is.

Frame definition: u = unit vector from the centroid of eye contour 60-67 to
the centroid of eye contour 68-75 ("across" the face, image-left to
image-right on an upright frontal face); v = u rotated 90 degrees ("down" the
face). Projections are taken relative to eye_mid, the midpoint of the two
centroids. This presumes 60-67/68-75 are the eyes, which the frame-free
checks in verify_layout (pupil containment, image-left/right ordering)
validate first.
"""

from __future__ import annotations

import numpy as np

# Thresholds used by verify_layout's pass/fail versions of these checks.
CHIN_TOL_PX = 2.0     # absolute pixels (note: scale-dependent by design flaw;
                      # diagnose_layout_failures reports IOD-relative margins)
PAIR_TOL_FRAC = 0.20  # fraction of the eye-line -> chin distance
NOSE_TOL_FRAC = 0.35  # fraction of the centroid inter-ocular distance


class FaceFrame:
    """Vectorised face-aligned frame over an (N, 98, 2) landmark array."""

    def __init__(self, lm: np.ndarray):
        lm = np.asarray(lm, dtype=np.float64)
        if lm.ndim != 3 or lm.shape[1:] != (98, 2):
            raise ValueError(f"expected (N, 98, 2) landmarks, got {lm.shape}")
        self.lm = lm
        c_left = lm[:, 60:68].mean(axis=1)
        c_right = lm[:, 68:76].mean(axis=1)
        diff = c_right - c_left
        self.iod_centroid = np.linalg.norm(diff, axis=1)      # centroid-to-centroid
        self.u = diff / (self.iod_centroid[:, None] + 1e-9)
        self.v = np.stack([-self.u[:, 1], self.u[:, 0]], axis=1)
        self.eye_mid = (c_left + c_right) / 2
        # The project's NME normaliser: outer eye corner distance (60 <-> 72).
        self.iod_corner = np.linalg.norm(lm[:, 72] - lm[:, 60], axis=1)

    def _proj(self, idx, axis_vec: np.ndarray) -> np.ndarray:
        pts = self.lm[:, idx]                  # (N, 2) or (N, k, 2) for slices
        if pts.ndim == 3:
            return np.sum((pts - self.eye_mid[:, None]) * axis_vec[:, None], axis=-1)
        return np.sum((pts - self.eye_mid) * axis_vec, axis=-1)

    def across(self, idx) -> np.ndarray:
        return self._proj(idx, self.u)

    def down(self, idx) -> np.ndarray:
        return self._proj(idx, self.v)

    @property
    def face_height(self) -> np.ndarray:
        """Eye line to chin distance along the face axis."""
        return self.down(16)


# ---- magnitudes behind the four pose-sensitive checks ----------------------

def chin_lowest(frame: FaceFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per face: which contour index (0-32) is farthest below the eye line,
    and by how many pixels more than point 16 (0 when 16 itself is lowest)."""
    d = frame.down(slice(0, 33))               # (N, 33)
    return d.argmax(axis=1), d.max(axis=1) - frame.down(16)


def pair_height_mismatch(frame: FaceFrame, i: int, j: int) -> np.ndarray:
    """|height difference| of a symmetric contour pair along the face axis."""
    return np.abs(frame.down(i) - frame.down(j))


def nose_offset(frame: FaceFrame) -> np.ndarray:
    """Signed across-axis offset of nose tip 54 from eye_mid (the midline
    reference point). Negative = toward image-left on an upright face."""
    return frame.across(54)


# ---- pass/fail forms, exactly as verify_layout applies them ----------------

def check_chin(frame: FaceFrame) -> np.ndarray:
    _, margin = chin_lowest(frame)
    return margin <= CHIN_TOL_PX


def check_pair(frame: FaceFrame, i: int, j: int) -> np.ndarray:
    return pair_height_mismatch(frame, i, j) < PAIR_TOL_FRAC * frame.face_height


def check_nose_midline(frame: FaceFrame) -> np.ndarray:
    return np.abs(nose_offset(frame)) < NOSE_TOL_FRAC * frame.iod_centroid


# ---- head-yaw proxies (for diagnostics only, not part of any check) --------

def yaw_proxy_contour(frame: FaceFrame) -> np.ndarray:
    """Signed yaw estimate from contour half-width asymmetry:
    (right extent - left extent) / (right + left), measured along the across
    axis from eye_mid. 0 = symmetric; sign follows which half is wider.
    This mirrors how Layer 2 will estimate yaw from contour asymmetry."""
    left_ext = -frame.across(slice(0, 16)).min(axis=1)
    right_ext = frame.across(slice(17, 33)).max(axis=1)
    return (right_ext - left_ext) / (right_ext + left_ext + 1e-9)


def yaw_proxy_eyewidth(frame: FaceFrame) -> np.ndarray:
    """Independent yaw estimate from apparent eye-width asymmetry (the far
    eye forshortens under yaw): (left width - right width) / (sum).
    Uses only corner points 60/64 and 68/72, which the containment and
    ordering checks validate. Used to cross-check the contour proxy."""
    lm = frame.lm
    w_left = np.linalg.norm(lm[:, 64] - lm[:, 60], axis=1)
    w_right = np.linalg.norm(lm[:, 72] - lm[:, 68], axis=1)
    return (w_left - w_right) / (w_left + w_right + 1e-9)

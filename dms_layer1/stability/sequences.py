"""Synthesised video sequences from WFLW stills.

WFLW is stills, and stability is a temporal property, so the harness needs
sequences. Two sources exist and they measure different things:

  * synthesised, here: a still is moved under a smooth random walk with
    sensor noise on top, and the annotation is carried through the same
    warp. Ground truth is known for every frame, so jitter can be measured
    as variation of the RESIDUAL rather than of the raw trajectory, which is
    the only way to separate wobble from the face genuinely moving.
  * a real clip, which has no ground truth and is scored on self-consistency
    instead. It is the honest source for anything involving how a detector
    responds to real sensor noise and real motion.

What synthesis cannot do: a 2D affine moves the face without changing what
it looks like, so out-of-plane rotation, blinking, expression and lighting
change are absent. The numbers it produces are a lower bound on jitter, and
the harness labels them that way.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SequenceSpec:
    """Amplitudes are per frame, in units that scale with the face, so a
    sequence built from a small face and a large one perturb comparably."""
    frames: int = 60
    translate_frac: float = 0.010   # of the face side, per-frame step
    rotate_deg: float = 0.40        # per-frame step
    scale_frac: float = 0.004       # per-frame step
    smoothness: float = 0.85        # AR(1) coefficient: 0 = white, 1 = drift
    noise_sigma: float = 2.0        # gray levels, iid per pixel per frame
    brightness_amp: float = 3.0     # gray levels, smooth drift


def _walk(n: int, step: float, smoothness: float, rng: np.random.Generator
          ) -> np.ndarray:
    """Smooth zero-mean random walk. Real head motion is correlated frame to
    frame, so white noise would understate the low-frequency component and
    overstate the high-frequency one, which is exactly the split this
    harness is trying to measure."""
    out = np.zeros(n)
    v = 0.0
    for i in range(n):
        v = smoothness * v + (1.0 - smoothness) * rng.normal(0.0, 1.0)
        out[i] = v
    # rescale so the per-frame step matches the requested amplitude
    diffs = np.abs(np.diff(out))
    if diffs.size and diffs.mean() > 0:
        out = out * (step / diffs.mean())
    return out


def make_sequence(image: np.ndarray, pts: np.ndarray, spec: SequenceSpec,
                  rng: np.random.Generator, face_side: float
                  ) -> tuple[list[np.ndarray], np.ndarray]:
    """Return (frames, points per frame). `pts` is any (N, 2) annotation; it
    is warped by the same affine as the image, so it stays ground truth."""
    n = spec.frames
    tx = _walk(n, spec.translate_frac * face_side, spec.smoothness, rng)
    ty = _walk(n, spec.translate_frac * face_side, spec.smoothness, rng)
    rot = _walk(n, spec.rotate_deg, spec.smoothness, rng)
    scale = 1.0 + _walk(n, spec.scale_frac, spec.smoothness, rng)
    bright = _walk(n, spec.brightness_amp, spec.smoothness, rng)

    h, w = image.shape[:2]
    centre = (w / 2.0, h / 2.0)
    frames, all_pts = [], []
    for i in range(n):
        m = cv2.getRotationMatrix2D(centre, float(rot[i]), float(scale[i]))
        m[0, 2] += tx[i]
        m[1, 2] += ty[i]
        frame = cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
        if spec.noise_sigma > 0 or spec.brightness_amp > 0:
            noisy = frame.astype(np.float32)
            noisy += rng.normal(0.0, spec.noise_sigma, noisy.shape)
            noisy += bright[i]
            frame = np.clip(noisy, 0, 255).astype(np.uint8)
        frames.append(frame)
        ones = np.ones((len(pts), 1))
        all_pts.append((np.hstack([np.asarray(pts, dtype=np.float64), ones])
                        @ m.T))
    return frames, np.stack(all_pts)

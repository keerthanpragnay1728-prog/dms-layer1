"""The swappable landmark-source interface (milestone 6).

This is the seam the whole ablation turns on. A LandmarkDetector takes a
full camera frame and returns 24 (x, y) points in original frame
coordinates, in the exact order and semantics of configs/landmarks_24.yaml,
or None when it finds no face. Everything downstream (Layer 2 features,
Layer 3 decisions, the stability harness) consumes this interface and must
never know which implementation produced the coordinates.

Frame contract: uint8 numpy array, either grayscale (H, W) or BGR (H, W, 3)
as OpenCV delivers it. Implementations convert internally to whatever they
need.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Landmarks24:
    """24 (x, y) points in original frame pixel coordinates, schema order."""
    points: np.ndarray          # (24, 2) float32
    source: str                 # which detector produced them, for logging

    def __post_init__(self):
        pts = np.asarray(self.points, dtype=np.float32)
        if pts.shape != (24, 2):
            raise ValueError(f"Landmarks24 needs shape (24, 2), got {pts.shape}")
        if not np.isfinite(pts).all():
            raise ValueError("Landmarks24 contains non-finite coordinates")
        object.__setattr__(self, "points", pts)


class LandmarkDetector(ABC):
    """One face per frame: the largest, matching the one-driver cabin."""

    name: str = "abstract"

    @abstractmethod
    def detect(self, frame: np.ndarray) -> Landmarks24 | None:
        """Returns 24 (x, y) points in original frame coordinates, or None
        if no face was found."""


def as_gray(frame: np.ndarray) -> np.ndarray:
    import cv2
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"expected (H, W) gray or (H, W, 3) BGR frame, got {frame.shape}")


def as_rgb(frame: np.ndarray) -> np.ndarray:
    import cv2
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    raise ValueError(f"expected (H, W) gray or (H, W, 3) BGR frame, got {frame.shape}")

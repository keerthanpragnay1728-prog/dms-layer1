"""Square face-crop geometry and crop-space coordinate transforms.

One module owns the crop box definition and the frame<->crop coordinate
mapping, because the training cache (milestone 2), the Haar detector wrapper
(milestone 3) and both LandmarkDetector implementations (milestone 6) must
all agree on it exactly.

Conventions:
  * A crop box is integer (x0, y0, side), axis-aligned and square, possibly
    extending outside the image (extraction pads with zeros).
  * Crop-space coordinates are normalised to [0, 1] over the box:
    p01 = (p - (x0, y0)) / side. They are resolution-independent, so the
    same labels serve any cache_size / model input_size.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Pixels outside the image are filled with this value when a crop box
# overhangs the border (same constant the rotation augmentation will use, so
# the model only ever sees one kind of border).
PAD_VALUE = 0


@dataclass(frozen=True)
class CropBox:
    x0: int
    y0: int
    side: int

    def as_array(self) -> np.ndarray:
        return np.array([self.x0, self.y0, self.side], dtype=np.int32)


def square_box_around(points: np.ndarray, expand: float) -> CropBox:
    """Integer square box centred on the points' bounding box, with side
    max(width, height) * expand. Guaranteed to contain every input point for
    expand >= 1 (the float box does; the integer box is grown to cover it)."""
    points = np.asarray(points, dtype=np.float64)
    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)
    side_f = max(xmax - xmin, ymax - ymin, 1.0) * expand
    x0_f = (xmin + xmax) / 2 - side_f / 2
    y0_f = (ymin + ymax) / 2 - side_f / 2
    x0, y0 = int(np.floor(x0_f)), int(np.floor(y0_f))
    side = int(np.ceil(max(x0_f + side_f - x0, y0_f + side_f - y0)))
    return CropBox(x0, y0, side)


def extract_square(image: np.ndarray, box: CropBox, out_size: int) -> np.ndarray:
    """Cut the box out of a grayscale image (zero-padding any part outside
    the image) and resize it to out_size x out_size. INTER_AREA when
    shrinking, INTER_LINEAR when enlarging (small faces)."""
    if image.ndim != 2:
        raise ValueError(f"expected a grayscale image, got shape {image.shape}")
    h, w = image.shape
    canvas = np.full((box.side, box.side), PAD_VALUE, dtype=image.dtype)
    xs0, ys0 = max(0, box.x0), max(0, box.y0)
    xs1, ys1 = min(w, box.x0 + box.side), min(h, box.y0 + box.side)
    if xs1 > xs0 and ys1 > ys0:
        canvas[ys0 - box.y0:ys1 - box.y0, xs0 - box.x0:xs1 - box.x0] = \
            image[ys0:ys1, xs0:xs1]
    interp = cv2.INTER_AREA if box.side >= out_size else cv2.INTER_LINEAR
    return cv2.resize(canvas, (out_size, out_size), interpolation=interp)


def to_crop_space(points: np.ndarray, box: CropBox) -> np.ndarray:
    """Frame pixels -> [0, 1] crop coordinates."""
    return (np.asarray(points, dtype=np.float64) - (box.x0, box.y0)) / box.side


def to_frame_space(points01: np.ndarray, box: CropBox) -> np.ndarray:
    """[0, 1] crop coordinates -> original frame pixels."""
    return np.asarray(points01, dtype=np.float64) * box.side + (box.x0, box.y0)

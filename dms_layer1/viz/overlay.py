"""Overlay rendering for mapping verification (milestone 1).

For each face this produces one canvas:
  * main panel: the face (cropped around the landmarks, upscaled), all 98
    WFLW points as faint gray dots for context, our 24 points in group
    colours; mouth/axis/contour points carry their output-index label.
  * three zoom insets (left eye, right eye, mouth) rendered from the CLEAN
    image and drawn at zoom scale, so eyelid/pupil placement can be judged
    at pixel precision; every point in an inset is labelled with its output
    index. Eye insets get a crosshair through the pupil point to make
    "pupil dead centre" easy to check.
  * a title row (image, annotation line, attribute flags) and a legend row.

Colour code (BGR), one colour per landmark group:
    eyelids green, pupils red, mouth orange, axis (nose tip + chin) yellow,
    contour magenta.
"""

from __future__ import annotations

import numpy as np
import cv2

from dms_layer1.landmarks.schema import LandmarkSchema

GROUP_COLORS = {
    "eyelids": (80, 220, 80),
    "pupils": (60, 60, 255),
    "mouth": (0, 165, 255),
    "axis": (60, 230, 230),
    "contour": (255, 100, 255),
}
GRAY_98 = (150, 150, 150)
BG = (35, 35, 35)
FONT = cv2.FONT_HERSHEY_SIMPLEX

MAIN_HEIGHT = 640          # face crop is scaled to this height in the main panel
CROP_MARGIN = 0.25         # margin around the landmark extent when cropping
LABELED_ON_MAIN = ("mouth", "axis", "contour")   # eyes/pupils are labelled in insets


def _put_label(img: np.ndarray, xy: tuple[float, float], text: str,
               color: tuple[int, int, int], scale: float = 0.45) -> None:
    x, y = int(round(xy[0])) + 5, int(round(xy[1])) - 5
    cv2.putText(img, text, (x, y), FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale, color, 1, cv2.LINE_AA)


def _face_crop_region(lm98: np.ndarray, img_shape) -> tuple[int, int, int, int]:
    """Crop box around the landmark extent plus margin, clipped to the image."""
    h, w = img_shape[:2]
    xmin, ymin = lm98.min(axis=0)
    xmax, ymax = lm98.max(axis=0)
    mx = (xmax - xmin) * CROP_MARGIN
    my = (ymax - ymin) * CROP_MARGIN
    x0 = int(max(0, np.floor(xmin - mx)))
    y0 = int(max(0, np.floor(ymin - my)))
    x1 = int(min(w, np.ceil(xmax + mx)))
    y1 = int(min(h, np.ceil(ymax + my)))
    return x0, y0, x1, y1


def _render_main_panel(image: np.ndarray, lm98: np.ndarray,
                       schema: LandmarkSchema) -> np.ndarray:
    x0, y0, x1, y1 = _face_crop_region(lm98, image.shape)
    crop = image[y0:y1, x0:x1]
    s = MAIN_HEIGHT / max(1, crop.shape[0])
    panel = cv2.resize(crop, (max(1, int(round(crop.shape[1] * s))), MAIN_HEIGHT),
                       interpolation=cv2.INTER_CUBIC)
    to_panel = lambda p: ((p[0] - x0) * s, (p[1] - y0) * s)

    for i in range(98):                       # context: the full 98-point set
        px, py = to_panel(lm98[i])
        cv2.circle(panel, (int(round(px)), int(round(py))), 2, GRAY_98, -1, cv2.LINE_AA)

    for p in schema.points:                   # our 24, colour-coded
        px, py = to_panel(lm98[p.wflw])
        color = GROUP_COLORS[p.group]
        cv2.circle(panel, (int(round(px)), int(round(py))), 4, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(panel, (int(round(px)), int(round(py))), 3, color, -1, cv2.LINE_AA)
        if p.group in LABELED_ON_MAIN:
            _put_label(panel, (px, py), str(p.index), color)
    return panel


def _render_inset(image: np.ndarray, lm98: np.ndarray, schema: LandmarkSchema,
                  point_indices: list[int], inset_w: int, inset_h: int,
                  caption: str, crosshair_index: int | None = None) -> np.ndarray:
    """Zoomed crop around a point subset, drawn AFTER upscaling so markers
    stay small and precise relative to the zoomed image."""
    wflw_ids = [schema.points[i].wflw for i in point_indices]
    sub = lm98[wflw_ids]
    xmin, ymin = sub.min(axis=0)
    xmax, ymax = sub.max(axis=0)
    span = max(xmax - xmin, ymax - ymin, 8.0)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half_w, half_h = span * 0.85, span * 0.85 * inset_h / inset_w
    h, w = image.shape[:2]
    x0, y0 = max(0, int(cx - half_w)), max(0, int(cy - half_h))
    x1, y1 = min(w, int(np.ceil(cx + half_w))), min(h, int(np.ceil(cy + half_h)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return np.full((inset_h, inset_w, 3), BG, dtype=np.uint8)

    crop = image[y0:y1, x0:x1]
    sx, sy = inset_w / crop.shape[1], inset_h / crop.shape[0]
    inset = cv2.resize(crop, (inset_w, inset_h), interpolation=cv2.INTER_CUBIC)

    if crosshair_index is not None:
        px = (lm98[schema.points[crosshair_index].wflw][0] - x0) * sx
        py = (lm98[schema.points[crosshair_index].wflw][1] - y0) * sy
        ch_color = (120, 120, 240)
        cv2.line(inset, (int(round(px)), 0), (int(round(px)), inset_h), ch_color, 1)
        cv2.line(inset, (0, int(round(py))), (inset_w, int(round(py))), ch_color, 1)

    for i in point_indices:
        p = schema.points[i]
        px, py = (lm98[p.wflw][0] - x0) * sx, (lm98[p.wflw][1] - y0) * sy
        if not (0 <= px < inset_w and 0 <= py < inset_h):
            continue
        color = GROUP_COLORS[p.group]
        cv2.circle(inset, (int(round(px)), int(round(py))), 4, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(inset, (int(round(px)), int(round(py))), 3, color, -1, cv2.LINE_AA)
        _put_label(inset, (px, py), str(p.index), color, scale=0.5)

    # caption on its own band so it never collides with point labels
    cv2.rectangle(inset, (0, inset_h - 22), (inset_w, inset_h), (25, 25, 25), -1)
    cv2.putText(inset, caption, (6, inset_h - 7), FONT, 0.45,
                (235, 235, 235), 1, cv2.LINE_AA)
    cv2.rectangle(inset, (0, 0), (inset_w - 1, inset_h - 1), (90, 90, 90), 1)
    return inset


def _legend_row(width: int, height: int = 44) -> np.ndarray:
    row = np.full((height, width, 3), BG, dtype=np.uint8)
    x = 10
    for group, color in GROUP_COLORS.items():
        cv2.circle(row, (x + 6, height // 2), 5, color, -1, cv2.LINE_AA)
        cv2.putText(row, group, (x + 16, height // 2 + 5), FONT, 0.5,
                    (230, 230, 230), 1, cv2.LINE_AA)
        x += 16 + 11 * len(group) + 22
    note = "gray dots = all 98 WFLW points"
    cv2.putText(row, note, (x + 10, height // 2 + 5), FONT, 0.5,
                (160, 160, 160), 1, cv2.LINE_AA)
    return row


def render_face_overlay(image: np.ndarray, lm98: np.ndarray,
                        schema: LandmarkSchema, title: str,
                        inset_size: int = 320) -> np.ndarray:
    """Full verification canvas for one face. `image` is BGR (or grayscale,
    converted here); `lm98` is (98, 2) in original image coordinates."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    lm98 = np.asarray(lm98, dtype=np.float64)

    main = _render_main_panel(image, lm98, schema)
    inset_h = MAIN_HEIGHT // 3
    insets = [
        _render_inset(image, lm98, schema, [0, 1, 2, 3, 4, 5, 12], inset_size,
                      inset_h, "left eye (ours 0-5) + pupil 12", crosshair_index=12),
        _render_inset(image, lm98, schema, [6, 7, 8, 9, 10, 11, 13], inset_size,
                      inset_h, "right eye (ours 6-11) + pupil 13", crosshair_index=13),
        _render_inset(image, lm98, schema, [14, 15, 16, 17], inset_size,
                      inset_h, "mouth (ours 14-17)"),
    ]
    right_col = np.concatenate(insets, axis=0)
    if right_col.shape[0] != main.shape[0]:
        pad = np.full((main.shape[0] - right_col.shape[0], inset_size, 3), BG, np.uint8)
        right_col = np.concatenate([right_col, pad], axis=0)

    body = np.concatenate([main, right_col], axis=1)
    title_row = np.full((36, body.shape[1], 3), BG, dtype=np.uint8)
    cv2.putText(title_row, title, (10, 24), FONT, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    return np.concatenate([title_row, body, _legend_row(body.shape[1])], axis=0)


def make_contact_sheet(canvases: list[np.ndarray], cols: int = 3,
                       thumb_width: int = 640) -> np.ndarray:
    """Grid of shrunken overlay canvases for a single-glance check."""
    thumbs = []
    for c in canvases:
        s = thumb_width / c.shape[1]
        thumbs.append(cv2.resize(c, (thumb_width, int(round(c.shape[0] * s)))))
    th = max(t.shape[0] for t in thumbs)
    thumbs = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0,
                                 cv2.BORDER_CONSTANT, value=BG) for t in thumbs]
    rows = []
    for i in range(0, len(thumbs), cols):
        row = thumbs[i:i + cols]
        while len(row) < cols:
            row.append(np.full((th, thumb_width, 3), BG, dtype=np.uint8))
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)

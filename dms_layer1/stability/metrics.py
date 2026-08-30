"""Jitter measures.

The central distinction: a landmark that moves because the face moved is not
jitter. On synthesised sequences ground truth is known per frame, so jitter
is the variation of the RESIDUAL (prediction minus truth) over time, and the
face's real motion cancels. On a real clip there is no truth, so jitter is
estimated as the high-frequency part of the trajectory, which assumes real
motion is smooth and noise is not. The two estimators are not
interchangeable and the harness never mixes them in one table.

Everything is normalised: landmark jitter by inter-ocular distance, box
jitter by the box side. Box jitter is therefore in the same units as
face_detector.box_shift_x and box_shift_y, and directly comparable to the
per-face scatter measured in the milestone 6 residual decomposition.
"""

from __future__ import annotations

import numpy as np


def residual_jitter(pred: np.ndarray, gt: np.ndarray,
                    iod: np.ndarray) -> dict:
    """Per-point jitter of a (T, P, 2) prediction sequence against its truth.

    Returns the per-point standard deviation of the residual over time, in
    percent of inter-ocular distance, plus the mean over points. The MEAN
    residual is reported separately as bias: a prediction can be badly
    offset and perfectly steady, and for a calibrated downstream stage those
    are very different failures.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    iod = np.asarray(iod, dtype=np.float64).reshape(-1, 1, 1)
    resid = (pred - gt) / np.maximum(iod, 1e-9)          # (T, P, 2)
    per_point = np.linalg.norm(resid.std(axis=0), axis=1)  # (P,)
    bias = np.linalg.norm(resid.mean(axis=0), axis=1)
    return {"per_point_pct": 100 * per_point,
            "mean_pct": float(100 * per_point.mean()),
            "worst_point": int(np.argmax(per_point)),
            "worst_pct": float(100 * per_point.max()),
            "bias_mean_pct": float(100 * bias.mean())}


def highfreq_jitter(series: np.ndarray, window: int = 5) -> float:
    """Jitter estimate with no ground truth: the root mean square deviation
    of a series from its own moving average. Assumes real motion is slower
    than the window and noise is faster, which is why it is only used on
    real clips and always labelled as a different estimator."""
    x = np.asarray(series, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if len(x) < window + 1:
        return float("nan")
    kernel = np.ones(window) / window
    smooth = np.stack([np.convolve(col, kernel, mode="same") for col in x.T], axis=1)
    edge = window // 2
    resid = (x - smooth)[edge:len(x) - edge]
    return float(np.sqrt((resid ** 2).sum(axis=1).mean()))


def box_jitter(boxes: list, gt_boxes: list) -> dict:
    """Stage-1 box stability, in box-side units.

    Reports centre bias and centre jitter separately for the same reason the
    milestone 6 decomposition did: a box consistently offset is a
    calibration matter, a box that moves is not, and only the second one
    reaches the landmarks as noise.
    """
    dx, dy, sides = [], [], []
    for b, g in zip(boxes, gt_boxes):
        if b is None:
            continue
        bc = (b.x0 + b.side / 2.0, b.y0 + b.side / 2.0)
        gc = (g.x0 + g.side / 2.0, g.y0 + g.side / 2.0)
        dx.append((bc[0] - gc[0]) / b.side)
        dy.append((bc[1] - gc[1]) / b.side)
        sides.append(b.side / g.side)
    if not dx:
        return {"n": 0}
    dx, dy, sides = np.array(dx), np.array(dy), np.array(sides)
    return {"n": len(dx),
            "centre_bias_x": float(dx.mean()), "centre_bias_y": float(dy.mean()),
            "centre_jitter_x": float(dx.std()), "centre_jitter_y": float(dy.std()),
            "centre_jitter": float(np.hypot(dx.std(), dy.std())),
            "side_ratio_median": float(np.median(sides)),
            "side_jitter": float(sides.std())}


def crossing_rate(series: np.ndarray, threshold: float) -> float:
    """Fraction of consecutive frame pairs that cross a threshold.

    The false-blink proxy: on a sequence where the eye state never changes,
    every crossing of a blink threshold is an error the decision layer would
    act on. A signal can have small jitter and still cross often if it sits
    near the threshold, which is why this is reported next to jitter rather
    than derived from it.
    """
    x = np.asarray(series, dtype=np.float64)
    if len(x) < 2:
        return 0.0
    above = x > threshold
    return float(np.mean(above[1:] != above[:-1]))

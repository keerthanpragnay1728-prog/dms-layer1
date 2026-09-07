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


def split_jumps(values: np.ndarray, jump_threshold: float) -> dict:
    """Separate a per-frame series into continuous wobble and discrete jumps.

    A cascade detector does not drift smoothly. It searches a scale pyramid on
    a grid and merges neighbours, so a small change in the image can flip
    which candidate wins or which pyramid level it wins at, and the box moves
    in a step rather than a slide. Averaging the two into one standard
    deviation describes neither: a few large steps and constant small wobble
    produce the same number and call for different fixes.

    Returns the jump rate, and the spread of the frame-to-frame differences
    with jumps excluded.
    """
    v = np.asarray(values, dtype=np.float64)
    if len(v) < 2:
        return {"jump_rate": 0.0, "continuous_sd": 0.0, "largest_jump": 0.0}
    d = np.abs(np.diff(v, axis=0))
    if d.ndim > 1:
        d = np.linalg.norm(d, axis=1)
    jumps = d > jump_threshold
    # Measured on the DIFFERENCES, not on the values: a step leaves a
    # permanent level change, so a standard deviation of positions counts
    # every jump forever. For white wobble of size sigma the differences have
    # spread sigma * sqrt(2), so dividing recovers the per-frame wobble.
    calm = d[~jumps]
    sd = float(np.sqrt((calm ** 2).mean()) / np.sqrt(2)) if calm.size else 0.0
    return {"jump_rate": float(jumps.mean()), "continuous_sd": sd,
            "largest_jump": float(d.max())}


def gap_runs(valid: list[bool]) -> dict:
    """Dropout structure, not just its rate.

    A cabin loses a drowsiness estimate on every dropped frame, and losing 3%
    of frames one at a time is a different failure from losing the same 3% in
    one blackout. Layer 2 aggregates over a window, so what matters is the
    longest gap relative to that window.
    """
    v = list(valid)
    if not v:
        return {"dropout": 1.0, "longest_gap": 0, "gaps": 0}
    longest = run = gaps = 0
    for ok in v:
        if ok:
            run = 0
        else:
            run += 1
            if run == 1:
                gaps += 1
            longest = max(longest, run)
    return {"dropout": float(1.0 - sum(v) / len(v)),
            "longest_gap": int(longest), "gaps": int(gaps)}


def threshold_margin(series: np.ndarray, threshold: float) -> float:
    """How far a signal sits from a decision threshold, in units of its own
    jitter.

    Zero false crossings is a result about the sequences tested, not a
    property of the system: a signal one standard deviation from the
    threshold crosses often and a signal five away essentially never. This
    turns the binary into the quantity that generalises.
    """
    x = np.asarray(series, dtype=np.float64)
    sd = float(x.std())
    if sd <= 1e-12:
        return float("inf")
    return float(abs(x.mean() - threshold) / sd)


def rule_of_three(events: int, trials: int) -> float:
    """Upper 95% bound on a rate when the count is zero.

    Zero crossings in n frames does not mean the rate is zero, and a
    drowsiness system running at 30 fps accumulates frames quickly. With no
    events the bound is 3/n, which is the honest way to report a null.
    """
    if trials <= 0:
        return float("nan")
    if events > 0:
        return float(events / trials)
    return float(3.0 / trials)


def sustained_crossing_rate(series: np.ndarray, threshold: float,
                            min_frames: int = 3) -> float:
    """Crossings a blink detector would act on: runs of at least `min_frames`
    consecutive frames on the far side of the threshold, per frame.

    A single frame below an EAR threshold is not a blink and no sensible
    Layer 2 would treat it as one, so the raw crossing rate is a pessimistic
    proxy. This is the one that corresponds to a decision.
    """
    x = np.asarray(series, dtype=np.float64)
    below = x < threshold
    runs, count, n = 0, 0, len(x)
    for b in below:
        count = count + 1 if b else 0
        if count == min_frames:
            runs += 1
    return float(runs / max(n, 1))

"""Evaluation metrics (milestone 5), shared later by the milestone-6
detector comparison and the milestone-7 stability harness.

All functions take numpy arrays of predicted and ground-truth landmarks in
the SAME coordinate space - [0, 1] crop space or pixels, it does not matter,
because every reported number is normalised by the inter-ocular distance
(outer eye corners) measured from the ground truth in that same space.

Definitions:
  * per-face NME: mean over the 24 points of the Euclidean point error,
    divided by the face's inter-ocular distance (in percent when printed).
  * per-group NME: the same, averaged over one group's points only -
    reported separately because pupil error matters far more than contour
    error for this pipeline, and an overall average hides that.
  * failure rate: fraction of faces with per-face NME above a threshold
    (0.10 = the standard "10%" failure criterion).
"""

from __future__ import annotations

import numpy as np

from dms_layer1.landmarks.schema import LandmarkSchema


def interocular(gt: np.ndarray, schema: LandmarkSchema) -> np.ndarray:
    """(N,) inter-ocular distance per face, from ground truth."""
    d = np.linalg.norm(gt[:, schema.nme_left_index] - gt[:, schema.nme_right_index],
                       axis=1)
    return np.maximum(d, 1e-9)


def point_errors_norm(pred: np.ndarray, gt: np.ndarray,
                      schema: LandmarkSchema) -> np.ndarray:
    """(N, 24) per-point error / inter-ocular distance."""
    err = np.linalg.norm(np.asarray(pred, dtype=np.float64)
                         - np.asarray(gt, dtype=np.float64), axis=2)
    return err / interocular(gt, schema)[:, None]


def nme_per_face(pred: np.ndarray, gt: np.ndarray,
                 schema: LandmarkSchema) -> np.ndarray:
    """(N,) per-face NME (fractional; multiply by 100 for percent)."""
    return point_errors_norm(pred, gt, schema).mean(axis=1)


def failure_rate(nme: np.ndarray, threshold: float) -> float:
    return float(np.mean(nme > threshold))


def group_nme(pred: np.ndarray, gt: np.ndarray,
              schema: LandmarkSchema) -> dict[str, float]:
    """Mean NME restricted to each landmark group."""
    pe = point_errors_norm(pred, gt, schema)
    return {g: float(pe[:, schema.indices_of_group(g)].mean())
            for g in schema.groups}


def subset_nme(nme: np.ndarray, attrs: np.ndarray, attr_names: list[str],
               threshold: float) -> dict[str, dict]:
    """Per WFLW test subset (attribute flag == 1), plus the no-flag subset:
    mean NME, failure rate, and n."""
    out: dict[str, dict] = {}

    def entry(mask: np.ndarray) -> dict:
        return {"n": int(mask.sum()),
                "nme_pct": float(100 * nme[mask].mean()) if mask.any() else None,
                "failure_pct": float(100 * failure_rate(nme[mask], threshold))
                if mask.any() else None}

    for k, name in enumerate(attr_names):
        label = "largepose" if name == "pose" else name
        out[label] = entry(attrs[:, k] == 1)
    out["no_flags"] = entry(attrs.sum(axis=1) == 0)
    return out

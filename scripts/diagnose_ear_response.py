#!/usr/bin/env python3
"""Does the landmark model's eye aspect ratio respond to eye closure?

The live viewer shows EAR moving under 4% between open and closed eyes,
where it should roughly halve, and the two eyes converging on one value when
closed. That is the signature of a prediction collapsing to a fixed eyelid
shape rather than of a shape that moves too little.

Two candidate causes, and they need different responses:

  * the DATA has no closed eyes, so the model never had a reason to learn
    that the lids move. WFLW is web photography and people photograph
    themselves with their eyes open.
  * the MODEL cannot resolve the lid gap, or tracks it on WFLW faces and
    fails on a webcam, which would be a resolution or domain problem rather
    than a dataset one.

Sections:
  1. The ground-truth EAR distribution on the split. If closed eyes are
     absent from the labels, no amount of training reaches them.
  2. Our model's predicted EAR against ground-truth EAR on the SAME faces,
     using ground-truth boxes so acquisition is out of the way. The
     regression slope is the number that matters: a model that has learned a
     fixed shape predicts the same EAR whatever the truth is, giving a slope
     near zero, while a model that tracks the lids gives a slope near one.
  3. MediaPipe on the same faces, as a reference for what a responsive model
     looks like on this data and this EAR definition. If MediaPipe's slope is
     high and ours is not, the EAR computation is not the problem.

Read section 1 first: if the ground-truth EAR range is narrow, section 2's
slope is measured only over that narrow range and says nothing about full
closure. The script says so rather than leaving it implied.

Usage (Kaggle):
    python scripts/diagnose_ear_response.py --config configs/layer1_base.yaml \
        --split train --limit 3000
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dms_layer1.config import load_config, require, resolve_path, save_config_snapshot
from dms_layer1.data import wflw
from dms_layer1.data.crops import square_box_around
from dms_layer1.detect.interface import as_gray
from dms_layer1.detect.ours import OurLandmarkDetector
from dms_layer1.evaluation import signals
from dms_layer1.evaluation.matching import match_target
from dms_layer1.landmarks.schema import load_schema

# Thresholds to count faces below. 0.15 is roughly the EAR of a half-closed
# eye and 0.10 a shut one for the six-point formula; both are conventions
# from the blink-detection literature rather than anything measured here, so
# the whole distribution is printed and not only the counts.
COUNT_BELOW = [0.30, 0.25, 0.20, 0.15, 0.10]
_lines: list[str] = []


def say(text: str = "") -> None:
    print(text)
    _lines.append(text)


def describe(name: str, values: np.ndarray) -> dict:
    q = {p: float(np.percentile(values, p)) for p in (1, 5, 10, 25, 50, 75, 90, 99)}
    say(f"  {name}: n={len(values)}  mean {values.mean():.4f}  sd {values.std():.4f}")
    say("    percentiles " + "  ".join(f"p{p}={v:.3f}" for p, v in q.items()))
    return q


def histogram(values: np.ndarray, lo: float = 0.0, hi: float = 0.55,
              bins: int = 22) -> None:
    counts, edges = np.histogram(values, bins=bins, range=(lo, hi))
    top = max(counts.max(), 1)
    for c, a, b in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(round(48 * c / top))
        say(f"    {a:.3f} to {b:.3f} {c:6d} {bar}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=3000,
                    help="faces for the distribution; 0 = all")
    ap.add_argument("--model-limit", type=int, default=600,
                    help="faces run through the models in sections 2 and 3")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--out-dir", default="/kaggle/working/ear_response")
    args = ap.parse_args()

    cfg = load_config(args.config)
    schema = load_schema(resolve_path(cfg, require(cfg, "landmark_schema")))
    seed = int(require(cfg, "seed"))
    expand = float(require(cfg, "preprocess.reference_expand"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records, paths = wflw.load_split(cfg, args.split)
    rng = random.Random(seed)
    if args.limit and args.limit < len(records):
        records = rng.sample(records, args.limit)
    say(f"=== 1. Ground-truth EAR on {args.split}, {len(records)} annotated faces ===")
    say("  Computed from the annotation only, so this is a property of the "
        "DATASET and nothing to do with the model.")

    gt24 = np.stack([r.landmarks98[schema.wflw_indices] for r in records]).astype(np.float64)
    ear_pairs = np.array([signals.ear(p) for p in gt24])
    ear_mean = ear_pairs.mean(axis=1)
    left, right = ear_pairs[:, 0], ear_pairs[:, 1]

    stats = {"mean_eye": describe("mean of both eyes", ear_mean),
             "left_eye": describe("image-left eye", left),
             "right_eye": describe("image-right eye", right)}
    say("\n  distribution of the mean:")
    histogram(ear_mean)

    say("\n  faces below a threshold (mean of both eyes, then both eyes):")
    below = {}
    for t in COUNT_BELOW:
        n_mean = int((ear_mean < t).sum())
        n_both = int(((left < t) & (right < t)).sum())
        below[t] = {"mean_eye": n_mean, "both_eyes": n_both}
        say(f"    < {t:.2f}   {n_mean:6d} ({100 * n_mean / len(ear_mean):5.2f}%)"
            f"   both eyes {n_both:6d} ({100 * n_both / len(ear_mean):5.2f}%)")
    say("  WFLW has no eye-state label, so a low EAR here is squinting, "
        "blinking mid-shutter, blur or an unusual eye shape rather than a "
        "verified closed eye. The count is an upper bound on how much closed "
        "eye evidence the training signal could contain.")

    # ---- 2 and 3: do the models track it? ---------------------------------
    say(f"\n=== 2. Does OUR model track it? ===")
    say("  Predicted EAR against ground-truth EAR on the same faces, from "
        "ground-truth boxes, so this is the landmark model with acquisition "
        "removed. The slope is the answer: near 1 means the lids move with "
        "the truth, near 0 means one fixed shape whatever the eye is doing.")
    model_records = records if args.model_limit >= len(records) else rng.sample(
        records, args.model_limit)
    det = OurLandmarkDetector(cfg, weights=args.weights)
    mp_det = None
    try:
        from dms_layer1.detect.mediapipe_detector import (MediaPipeLandmarkDetector,
                                                          MediaPipeUnavailable)
        try:
            mp_det = MediaPipeLandmarkDetector(cfg, schema)
        except MediaPipeUnavailable as e:
            say(f"  mediapipe unavailable, section 3 skipped: {e}")
    except ImportError as e:
        say(f"  mediapipe import failed, section 3 skipped: {e}")

    gt_e, our_e, mp_e = [], [], []
    for n, rec in enumerate(model_records, 1):
        frame = cv2.imread(str(wflw.image_path(paths, rec)), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        g24 = rec.landmarks98[schema.wflw_indices].astype(np.float64)
        box = square_box_around(rec.landmarks98, expand)
        pred = det.predict_in_box(as_gray(frame), box)
        gt_e.append(float(np.mean(signals.ear(g24))))
        our_e.append(float(np.mean(signals.ear(pred))))
        if mp_det is not None:
            out = mp_det.detect(frame)
            mp_e.append(float(np.mean(signals.ear(out.points.astype(np.float64))))
                        if out is not None and match_target(
                            out.points.astype(np.float64), g24) else np.nan)
        if n % 200 == 0:
            say(f"  {n}/{len(model_records)} faces")

    gt_e, our_e = np.array(gt_e), np.array(our_e)
    mp_e = np.array(mp_e) if mp_e else None

    def response(name: str, pred: np.ndarray, truth: np.ndarray) -> dict:
        ok = np.isfinite(pred) & np.isfinite(truth)
        p, t = pred[ok], truth[ok]
        if len(p) < 10:
            say(f"  {name}: too few usable faces ({len(p)})")
            return {}
        slope, intercept = np.polyfit(t, p, 1)
        r = float(np.corrcoef(p, t)[0, 1])
        say(f"  {name}: n={len(p)}  slope {slope:+.3f}  intercept {intercept:+.3f}  "
            f"correlation {r:+.3f}")
        say(f"    predicted spread p10 to p90: {np.percentile(p, 10):.3f} to "
            f"{np.percentile(p, 90):.3f}   (truth "
            f"{np.percentile(t, 10):.3f} to {np.percentile(t, 90):.3f})")
        edges = np.percentile(t, [0, 20, 40, 60, 80, 100])
        say(f"    {'truth bin':<22} {'n':>5} {'mean truth':>11} {'mean pred':>10}")
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (t >= lo) & (t <= hi)
            if m.sum():
                say(f"    {f'{lo:.3f} to {hi:.3f}':<22} {int(m.sum()):5d} "
                    f"{t[m].mean():11.3f} {p[m].mean():10.3f}")
        return {"n": len(p), "slope": float(slope), "intercept": float(intercept),
                "correlation": r,
                "pred_p10": float(np.percentile(p, 10)),
                "pred_p90": float(np.percentile(p, 90)),
                "truth_p10": float(np.percentile(t, 10)),
                "truth_p90": float(np.percentile(t, 90))}

    ours_resp = response("ours on gt boxes", our_e, gt_e)
    mp_resp = {}
    if mp_e is not None:
        say(f"\n=== 3. Does MEDIAPIPE track it, on the same faces? ===")
        say("  Same EAR definition and the same ground truth. If this slope is "
            "high and ours is not, the formula and the mapping are fine and "
            "the difference is the landmark model.")
        mp_resp = response("mediapipe", mp_e, gt_e)

    # ---- verdict ----------------------------------------------------------
    say("\n=== Verdict ===")
    span = stats["mean_eye"][90] - stats["mean_eye"][10]
    say(f"  ground-truth EAR spans {span:.3f} between p10 and p90, and "
        f"{100 * below[0.15]['mean_eye'] / len(ear_mean):.2f}% of faces sit "
        "below 0.15.")
    if below[0.15]["mean_eye"] / len(ear_mean) < 0.01:
        say("  Closed eyes are effectively ABSENT from the labels. A landmark "
            "model trained on this cannot have learned that the lids move, "
            "and no change to the loss, the architecture or the augmentation "
            "fixes a signal that is not in the data.")
    if ours_resp and mp_resp:
        say(f"  slope: ours {ours_resp['slope']:+.3f} against mediapipe "
            f"{mp_resp['slope']:+.3f} on the same faces.")
        say("  If ours is near zero while MediaPipe's is not, our model has "
            "learned a fixed eyelid shape. If both are low, the ground-truth "
            "range on this split is too narrow to measure a response at all, "
            "and the test itself is inconclusive: check the span above.")
    say("  Either way this measures WFLW faces. The live failure could still "
        "be domain shift on top; the webcam log from scripts/webcam_demo.py "
        "--log-csv is what settles that, since it has real closures in it.")

    results = {"split": args.split, "faces": len(records),
               "gt_ear": {k: v for k, v in stats.items()},
               "below_threshold": {str(k): v for k, v in below.items()},
               "ours_response": ours_resp, "mediapipe_response": mp_resp}
    with open(out_dir / "ear_response.yaml", "w") as f:
        yaml.safe_dump(results, f, sort_keys=False)
    (out_dir / "report.txt").write_text("\n".join(_lines) + "\n")
    np.save(out_dir / "gt_ear_all.npy", ear_mean)
    save_config_snapshot(cfg, out_dir)
    say(f"\nresults + report + config snapshot in {out_dir}")
    if mp_det is not None:
        mp_det.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

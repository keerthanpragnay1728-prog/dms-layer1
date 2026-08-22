#!/usr/bin/env python3
"""Diagnose face SELECTION: what are the boxes that beat the real face?

Milestone 6 measured that the largest Haar box is the annotated subject on
only 27.67% of SINGLE-face WFLW images, while any-box matching reproduces
milestone 3's 72.48%. On a single-face image the largest box should usually
be the face, so something is producing bigger boxes that are not faces. This
script identifies them and tests selection rules that could avoid them. The
question matters for the cabin too: a headrest, a window frame or a seat
back can produce the same kind of oversized spurious box.

Sections:
  1. Anatomy of the failures on single-face images: how much bigger the
     winning box is than the face, whether it contains the face, sits
     disjoint from it, or overlaps partially, how many boxes the frame has,
     where the true face ranks by size, and how the cascade level weights
     (its nearest thing to a confidence) compare between the two.
  2. Selection rules compared on the same frames: largest, highest level
     weight, most central, and the two size-sanity variants, against an
     oracle that always picks the best available box. The oracle is the
     ceiling any selection rule could reach at these detector settings.
  3. Detector settings crossed with selection rules. The milestone-3 tuning
     was chosen to maximise ANY-box matching, which is blind to spurious
     boxes; deployment picks ONE box, so the same settings may be the wrong
     ones here. Scored on target-hit rate, with boxes per frame and runtime.
  4. Renders of failure cases: every Haar box drawn, the selected one and
     the ground truth marked.

Usage (Kaggle):
    python scripts/diagnose_face_selection.py --config configs/layer1_base.yaml --split test
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dms_layer1.config import (load_config, require, resolve_path,
                               save_config_snapshot)
from dms_layer1.data import wflw
from dms_layer1.data.crops import square_box_around
from dms_layer1.detect.haar import (SELECTION_RULES, HaarFaceDetector, box_iou,
                                    sane_boxes, select_face)

SETTINGS_GRID = [(1.05, 2), (1.05, 3), (1.05, 5), (1.10, 2), (1.10, 3), (1.10, 5)]
SANE_FRACS = [0.5, 0.7, 0.9]
FONT = cv2.FONT_HERSHEY_SIMPLEX
_lines: list[str] = []


def say(text: str = "") -> None:
    print(text)
    _lines.append(text)


def rect_of(box) -> tuple[float, float, float, float]:
    if hasattr(box, "side"):
        return (box.x0, box.y0, box.x0 + box.side, box.y0 + box.side)
    return (box.x, box.y, box.x + box.w, box.y + box.h)


def contains(outer, inner, slack: float = 0.9) -> bool:
    """Is `inner` essentially inside `outer`?"""
    ox0, oy0, ox1, oy1 = rect_of(outer)
    ix0, iy0, ix1, iy1 = rect_of(inner)
    inter_w = max(0.0, min(ox1, ix1) - max(ox0, ix0))
    inter_h = max(0.0, min(oy1, iy1) - max(oy0, iy0))
    inner_area = max(1e-9, (ix1 - ix0) * (iy1 - iy0))
    return (inter_w * inter_h) / inner_area >= slack


def pct(x) -> str:
    return f"{100 * float(np.mean(x)):6.2f}%" if len(x) else "   n/a"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=600,
                    help="images to analyse (seeded sample); 0 = all")
    ap.add_argument("--out-dir", default="/kaggle/working/m6_selection")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.synthetic:
        from dms_layer1.data.synthetic import write_synthetic_dataset
        say("SYNTHETIC MODE: schematic faces, code smoke test only.")
        root = write_synthetic_dataset(
            Path(__import__("tempfile").mkdtemp(prefix="sel_synth_")),
            require(cfg, "dataset.attribute_names"), seed=require(cfg, "seed"))
        cfg["dataset"]["root"] = str(root)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    expand = float(require(cfg, "preprocess.reference_expand"))
    match_iou = float(require(cfg, "face_detector.match_iou"))
    max_frac = float(require(cfg, "face_detector.max_size_frac"))
    rng = random.Random(require(cfg, "seed"))

    records, paths = wflw.load_split(cfg, args.split)
    by_image: dict[str, list[wflw.FaceRecord]] = {}
    for rec in records:
        by_image.setdefault(rec.image_rel_path, []).append(rec)
    images = sorted(by_image)
    if args.limit and args.limit < len(images):
        rng.shuffle(images)
        images = sorted(images[:args.limit])

    grays = {}
    for rel in images:
        gray = cv2.imread(str(wflw.image_path(paths, by_image[rel][0])),
                          cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise wflw.WFLWError(f"cv2 could not decode {rel}")
        grays[rel] = gray

    targets = {rel: max(by_image[rel],
                        key=lambda r: square_box_around(r.landmarks98, 1.0).side)
               for rel in images}
    gt_box = {rel: square_box_around(targets[rel].landmarks98, expand)
              for rel in images}
    single = [r for r in images if len(by_image[r]) == 1]
    say(f"analysing {len(images)} images ({len(single)} single-face) at the "
        "configured settings")

    det = HaarFaceDetector(cfg)
    boxes = {rel: det.detect(grays[rel]) for rel in images}

    def best_box(rel):
        """The detection that best matches the target face, and its IoU."""
        best, best_iou = None, 0.0
        for b in boxes[rel]:
            iou = box_iou(rect_of(b), rect_of(gt_box[rel]))
            if iou > best_iou:
                best, best_iou = b, iou
        return best, best_iou

    # ---- 1. anatomy --------------------------------------------------------
    say("\n=== 1. Anatomy of the failures (single-face images) ===")
    fails, ratios, ranks, n_boxes, score_gaps = [], [], [], [], []
    rel_cls = {"contains the face": 0, "inside the face box": 0,
               "partial overlap": 0, "disjoint from the face": 0,
               "no detections at all": 0}
    for rel in single:
        bs = boxes[rel]
        if not bs:
            rel_cls["no detections at all"] += 1
            fails.append(rel)
            continue
        picked = select_face(bs, "largest", grays[rel].shape, max_frac)
        if box_iou(rect_of(picked), rect_of(gt_box[rel])) >= match_iou:
            continue
        fails.append(rel)
        n_boxes.append(len(bs))
        ratios.append(max(picked.w, picked.h) / gt_box[rel].side)
        tgt, _ = best_box(rel)
        order = sorted(bs, key=lambda b: -b.area)
        ranks.append(order.index(tgt) + 1 if tgt is not None else 0)
        if tgt is not None:
            score_gaps.append(picked.score - tgt.score)
        iou = box_iou(rect_of(picked), rect_of(gt_box[rel]))
        if contains(picked, gt_box[rel]):
            rel_cls["contains the face"] += 1
        elif contains(gt_box[rel], picked):
            rel_cls["inside the face box"] += 1
        elif iou > 0:
            rel_cls["partial overlap"] += 1
        else:
            rel_cls["disjoint from the face"] += 1

    say(f"  single-face images where 'largest' misses: {len(fails)}/{len(single)} "
        f"({100 * len(fails) / max(1, len(single)):.2f}%)")
    say("  what the winning box is, relative to the face:")
    for k, v in rel_cls.items():
        say(f"    {k:<26} {v:>5}  ({100 * v / max(1, len(fails)):5.1f}% of failures)")
    if ratios:
        q = np.percentile(ratios, [50, 75, 90, 100])
        say(f"  winning box side / face box side: median {q[0]:.2f}  p75 {q[1]:.2f}  "
            f"p90 {q[2]:.2f}  max {q[3]:.2f}")
        say(f"  boxes per failing frame: median {np.median(n_boxes):.0f}  "
            f"max {max(n_boxes)}")
        found = [r for r in ranks if r > 0]
        say(f"  the true face WAS detected in {len(found)}/{len(fails)} failures; "
            f"its rank by size: median {np.median(found) if found else float('nan'):.0f}")
        if score_gaps:
            say(f"  level weight of the winner minus the true face: "
                f"median {np.median(score_gaps):+.2f} "
                f"(negative means confidence would have picked the face)")

    # ---- 2. selection rules -----------------------------------------------
    say("\n=== 2. Selection rules at the configured settings ===")
    say(f"  {'rule':<18} {'all images':>12} {'single-face':>12}")
    rule_rows = {}
    for rule in SELECTION_RULES:
        hits_all, hits_single = [], []
        for rel in images:
            picked = select_face(boxes[rel], rule, grays[rel].shape, max_frac)
            hit = picked is not None and box_iou(
                rect_of(picked), rect_of(gt_box[rel])) >= match_iou
            hits_all.append(hit)
            if len(by_image[rel]) == 1:
                hits_single.append(hit)
        rule_rows[rule] = (float(np.mean(hits_all)), float(np.mean(hits_single)))
        say(f"  {rule:<18} {pct(hits_all):>12} {pct(hits_single):>12}")
    oracle_all = [best_box(r)[1] >= match_iou for r in images]
    oracle_single = [best_box(r)[1] >= match_iou for r in single]
    say(f"  {'ORACLE (any box)':<18} {pct(oracle_all):>12} {pct(oracle_single):>12}"
        "   <- ceiling for any selection rule at these settings")

    say("\n  size-sanity threshold sweep (largest_sane, all images):")
    for frac in SANE_FRACS:
        hits = []
        for rel in images:
            picked = select_face(boxes[rel], "largest_sane", grays[rel].shape, frac)
            hits.append(picked is not None and box_iou(
                rect_of(picked), rect_of(gt_box[rel])) >= match_iou)
        dropped = np.mean([1 - len(sane_boxes(boxes[r], grays[r].shape, frac))
                           / max(1, len(boxes[r])) for r in images if boxes[r]])
        say(f"    max_size_frac {frac:.2f}: hit {pct(hits)}  "
            f"(drops {100 * dropped:.1f}% of detections)")

    # ---- 3. settings x rules ----------------------------------------------
    say("\n=== 3. Detector settings crossed with selection rules ===")
    say("  (milestone 3 tuned these for ANY-box matching, which cannot see "
        "spurious boxes; this scores the deployment question instead)")
    header = (f"  {'sf':>5} {'nb':>3} {'boxes/img':>10} {'largest':>9} "
              f"{'confidence':>11} {'conf_sane':>10} {'oracle':>8} {'ms':>6}")
    say(header)
    grid_rows = []
    for sf, nb in SETTINGS_GRID:
        c = copy.deepcopy(cfg)
        c["face_detector"].update({"scale_factor": sf, "min_neighbors": nb})
        d = HaarFaceDetector(c)
        per_image, ms = {}, []
        for rel in images:
            t0 = time.perf_counter()
            per_image[rel] = d.detect(grays[rel])
            ms.append((time.perf_counter() - t0) * 1000)

        def hit_rate(rule):
            hits = []
            for rel in images:
                p = select_face(per_image[rel], rule, grays[rel].shape, max_frac)
                hits.append(p is not None and box_iou(
                    rect_of(p), rect_of(gt_box[rel])) >= match_iou)
            return float(np.mean(hits))

        def oracle_rate():
            hits = []
            for rel in images:
                hits.append(any(box_iou(rect_of(b), rect_of(gt_box[rel])) >= match_iou
                                for b in per_image[rel]))
            return float(np.mean(hits))

        row = {"scale_factor": sf, "min_neighbors": nb,
               "boxes_per_image": round(float(np.mean([len(per_image[r]) for r in images])), 2),
               "largest_pct": round(100 * hit_rate("largest"), 2),
               "confidence_pct": round(100 * hit_rate("confidence"), 2),
               "confidence_sane_pct": round(100 * hit_rate("confidence_sane"), 2),
               "oracle_pct": round(100 * oracle_rate(), 2),
               "ms_median": round(float(np.median(ms)), 1)}
        grid_rows.append(row)
        say(f"  {sf:>5.2f} {nb:>3} {row['boxes_per_image']:>10.2f} "
            f"{row['largest_pct']:>8.2f}% {row['confidence_pct']:>10.2f}% "
            f"{row['confidence_sane_pct']:>9.2f}% {row['oracle_pct']:>7.2f}% "
            f"{row['ms_median']:>6.0f}")

    # ---- 3b. are the winning boxes faces at all? ---------------------------
    say("\n=== 3b. What the winning boxes really are, judged independently ===")
    say("  A box that beats the true face is either a REAL unannotated face")
    say("  (WFLW is web photography, so this is a dataset property that a")
    say("  one-driver cabin does not share) or a false positive on background")
    say("  (a detector property that a cabin DOES share: headrest, seat back,")
    say("  window frame). The two readings support opposite report claims, so")
    say("  MediaPipe judges each crop independently. The true-face box is run")
    say("  through the same judge as a control: if the control is not high,")
    say("  the judge is unreliable on these crops and the split means nothing.")
    judged = {"winner is a face": 0, "winner is not a face": 0}
    control_hits, control_total = 0, 0
    try:
        from dms_layer1.detect.mediapipe_detector import (MediaPipeLandmarkDetector,
                                                          MediaPipeUnavailable)
        from dms_layer1.landmarks.schema import load_schema as _load_schema
        judge = MediaPipeLandmarkDetector(
            cfg, _load_schema(resolve_path(cfg, require(cfg, "landmark_schema"))))

        def has_face(gray, box, margin: float = 0.35) -> bool:
            x0, y0, x1, y1 = rect_of(box)
            m = margin * max(x1 - x0, y1 - y0)
            h, w = gray.shape[:2]
            xa, ya = max(0, int(x0 - m)), max(0, int(y0 - m))
            xb, yb = min(w, int(x1 + m)), min(h, int(y1 + m))
            if xb - xa < 24 or yb - ya < 24:
                return False
            return judge.detect(gray[ya:yb, xa:xb]) is not None

        for rel in rng.sample(fails, min(150, len(fails))):
            picked = select_face(boxes[rel], "largest", grays[rel].shape, max_frac)
            if picked is None:
                continue
            judged["winner is a face" if has_face(grays[rel], picked)
                   else "winner is not a face"] += 1
            control_total += 1
            control_hits += has_face(grays[rel], gt_box[rel])
        n_j = sum(judged.values())
        if n_j:
            for k, v in judged.items():
                say(f"    {k:<22} {v:>4}  ({100 * v / n_j:5.1f}%)")
            say(f"    control, judge finds a face in the TRUE face box: "
                f"{100 * control_hits / max(1, control_total):.1f}% "
                f"(n={control_total}); a low control invalidates the split above")
        else:
            say("    no failures to judge on this population")
    except (ImportError, MediaPipeUnavailable) as e:
        say(f"    skipped: {e}")
        n_j = 0

    # ---- 4. renders --------------------------------------------------------
    tiles = []
    for rel in rng.sample(fails, min(6, len(fails))):
        view = cv2.cvtColor(grays[rel], cv2.COLOR_GRAY2BGR)
        for b in boxes[rel]:
            cv2.rectangle(view, (b.x, b.y), (b.x + b.w, b.y + b.h), (160, 160, 160), 2)
        picked = select_face(boxes[rel], "largest", grays[rel].shape, max_frac)
        if picked is not None:
            cv2.rectangle(view, (picked.x, picked.y),
                          (picked.x + picked.w, picked.y + picked.h), (60, 60, 255), 3)
        g = gt_box[rel]
        cv2.rectangle(view, (g.x0, g.y0), (g.x0 + g.side, g.y0 + g.side),
                      (80, 220, 80), 3)
        s = 420 / max(1, view.shape[0])
        view = cv2.resize(view, (max(1, int(view.shape[1] * s)), 420))
        strip = np.full((26, view.shape[1], 3), (25, 25, 25), dtype=np.uint8)
        cv2.putText(strip, f"{rel}  gray=all boxes, red=largest, green=face",
                    (6, 18), FONT, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
        tiles.append(np.concatenate([view, strip], axis=0))
    if tiles:
        w = max(t.shape[1] for t in tiles)
        h = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, w - t.shape[1],
                                    cv2.BORDER_CONSTANT, value=(35, 35, 35))
                 for t in tiles]
        rows = [np.concatenate(tiles[i:i + 2], axis=1)
                for i in range(0, len(tiles) - len(tiles) % 2, 2)]
        if rows:
            cv2.imwrite(str(out_dir / "selection_failures.png"),
                        np.concatenate(rows, axis=0))
            say(f"\nfailure renders: {out_dir / 'selection_failures.png'}")

    results = {"n_images": len(images), "n_single_face": len(single),
               "failure_anatomy": {k: int(v) for k, v in rel_cls.items()},
               "rule_hit_rates_pct": {k: [round(100 * v[0], 2), round(100 * v[1], 2)]
                                      for k, v in rule_rows.items()},
               "oracle_pct": [round(100 * float(np.mean(oracle_all)), 2),
                              round(100 * float(np.mean(oracle_single)), 2)],
               "settings_grid": grid_rows,
               "winner_identity": judged if n_j else None}
    with open(out_dir / "m6_selection.yaml", "w") as f:
        yaml.safe_dump(results, f, sort_keys=False)
    (out_dir / "report.txt").write_text("\n".join(_lines) + "\n")
    save_config_snapshot(cfg, out_dir)
    say(f"\nresults + report + config snapshot in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

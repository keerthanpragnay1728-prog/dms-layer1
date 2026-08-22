#!/usr/bin/env python3
"""Diagnose the milestone-6 full-frame collapse with magnitudes.

The observed facts: detection 27.71% here vs 72.08% in milestone 3, and
Haar-box NME 29.2% vs GT-box NME 6.8% on the same faces, while GT-box NME
through the live path matches milestone 5. So the model and the coordinate
machinery are right, and the damage is confined to WHICH box is chosen and
WHAT FRAMING it imposes. This script separates and measures the candidate
causes:

  1. Face choice. Milestone 3 asked "does ANY Haar box match each annotated
     face" (that is the 72.08%). Deployment takes the LARGEST box, and WFLW
     images are crowd scenes: the largest Haar box is often some other,
     bigger face. Measured here: the any-box rate (must reproduce milestone
     3), the largest-box-hits-target rate, and both split by single-face
     versus multi-face images. The single-face split is the cabin analogue.
  2. Framing scale. Training crops frame the face at the ground-truth box
     (1.3x the 98-point extent); the deploy crop is box_scale x the Haar
     side, which the milestone-3 medians put at about 1.56x the training
     framing, outside anything augmentation showed the model. Measured
     here: the model's NME as a function of framing factor k on
     ground-truth-centred boxes (its scale-response curve), with the
     training envelope marked.
  3. The calibration grid, re-decided on NME. Milestone 3 chose
     (box_scale, shift) by landmark containment; if the containment proxy
     lied, this table shows it: end NME per calibration on the images where
     the largest Haar box IS the target face (so face choice is excluded).
  4. Two-stage refinement, the no-retrain candidate fix: predict once with
     the current calibration, rebuild a training-style box from the
     predicted points, predict again. Measured against the GT-box ceiling.

Usage (Kaggle):
    python scripts/diagnose_deploy_gap.py --config configs/layer1_base.yaml --split test
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dms_layer1.config import load_config, require, resolve_path, save_config_snapshot
from dms_layer1.data import wflw
from dms_layer1.data.crops import CropBox, extract_square, square_box_around, to_frame_space
from dms_layer1.detect.haar import box_iou, haar_to_crop_box
from dms_layer1.detect.ours import OurLandmarkDetector
from dms_layer1.evaluation import metrics
from dms_layer1.landmarks.schema import load_schema

SCALE_CURVE = [1.0, 1.1, 1.2, 1.3, 1.4, 1.56, 1.7, 1.9]
GRID_SCALES = [1.12, 1.30, 1.45, 1.60, 1.75]
GRID_SHIFTS = [0.08, 0.13]
_lines: list[str] = []


def say(text: str = "") -> None:
    print(text)
    _lines.append(text)


def rect_of(box) -> tuple[float, float, float, float]:
    if hasattr(box, "side"):
        return (box.x0, box.y0, box.x0 + box.side, box.y0 + box.side)
    return (box.x, box.y, box.x + box.w, box.y + box.h)


def scaled_box(base: CropBox, k: float, shift_y: float = 0.0) -> CropBox:
    side = base.side * k
    cx = base.x0 + base.side / 2
    cy = base.y0 + base.side / 2 + shift_y * side
    return CropBox(int(np.floor(cx - side / 2)), int(np.floor(cy - side / 2)),
                   int(np.ceil(side)))


def nme_stats(vals: list[float]) -> str:
    if not vals:
        return "n/a (no faces)"
    a = np.array(vals)
    return f"mean {100 * a.mean():6.3f}%  median {100 * np.median(a):6.3f}%  n={len(a)}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=500,
                    help="images to analyse (seeded sample); 0 = all")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--out-dir", default="/kaggle/working/m6_diagnosis")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.synthetic:
        from dms_layer1.data.synthetic import synthetic_trained_setup
        cfg, ckpt = synthetic_trained_setup(cfg)
        cfg["detector"]["weights"] = str(ckpt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = load_schema(resolve_path(cfg, require(cfg, "landmark_schema")))
    expand = float(require(cfg, "preprocess.reference_expand"))
    match_iou = float(require(cfg, "face_detector.match_iou"))
    rng = random.Random(require(cfg, "seed"))

    det = OurLandmarkDetector(cfg, weights=args.weights)

    def run_model(gray: np.ndarray, box: CropBox) -> np.ndarray:
        crop = extract_square(gray, box, det.input_size)
        x = (torch.from_numpy(np.ascontiguousarray(crop)).float() / 255.0
             - det.pixel_mean) / det.pixel_std
        with torch.no_grad():
            pred01 = det.model(x.unsqueeze(0).unsqueeze(0))[0].numpy().astype(np.float64)
        return to_frame_space(pred01, box)

    def nme(pred_frame: np.ndarray, gt24: np.ndarray) -> float:
        return float(metrics.nme_per_face(pred_frame[None], gt24[None], schema)[0])

    records, paths = wflw.load_split(cfg, args.split)
    by_image: dict[str, list[wflw.FaceRecord]] = {}
    for rec in records:
        by_image.setdefault(rec.image_rel_path, []).append(rec)
    images = sorted(by_image)
    if args.limit and args.limit < len(images):
        rng.shuffle(images)
        images = sorted(images[:args.limit])
    say(f"analysing {len(images)} images "
        f"({sum(len(by_image[r]) for r in images)} annotated faces)")

    # decode once; run haar once per image
    grays: dict[str, np.ndarray] = {}
    haar_boxes: dict[str, list] = {}
    for rel in images:
        gray = cv2.imread(str(wflw.image_path(paths, by_image[rel][0])),
                          cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise wflw.WFLWError(f"cv2 could not decode {rel}")
        grays[rel] = gray
        haar_boxes[rel] = det.haar.detect(gray)

    targets = {rel: max(recs, key=lambda r: square_box_around(r.landmarks98, 1.0).side)
               for rel, recs in by_image.items() if rel in grays}
    gt_box = {rel: square_box_around(targets[rel].landmarks98, expand)
              for rel in images}
    gt24 = {rel: targets[rel].landmarks98[schema.wflw_indices].astype(np.float64)
            for rel in images}

    # ---- 1. face choice ----------------------------------------------------
    say("\n=== 1. Face choice: any-box matching vs largest-box deployment ===")
    per_face_hits, per_face_total = 0, 0
    for rel in images:
        boxes = haar_boxes[rel]
        taken: set[int] = set()
        for rec in by_image[rel]:
            g = square_box_around(rec.landmarks98, expand)
            best_i, best = -1, 0.0
            for i, hb in enumerate(boxes):
                if i in taken:
                    continue
                iou = box_iou(rect_of(hb), rect_of(g))
                if iou > best:
                    best_i, best = i, iou
            per_face_total += 1
            if best >= match_iou:
                taken.add(best_i)
                per_face_hits += 1
    say(f"  any-box rate, ALL annotated faces (the milestone-3 definition): "
        f"{100 * per_face_hits / max(1, per_face_total):.2f}% "
        f"({per_face_hits}/{per_face_total})  <- should reproduce 72.08%")

    def largest_hits_target(rel) -> bool:
        boxes = haar_boxes[rel]
        return bool(boxes) and box_iou(rect_of(boxes[0]),
                                       rect_of(gt_box[rel])) >= match_iou

    def any_hits_target(rel) -> bool:
        return any(box_iou(rect_of(hb), rect_of(gt_box[rel])) >= match_iou
                   for hb in haar_boxes[rel])

    single = [r for r in images if len(by_image[r]) == 1]
    multi = [r for r in images if len(by_image[r]) > 1]
    for label, pool in (("all images", images), ("single-face images", single),
                        ("multi-face images", multi)):
        if not pool:
            continue
        a = 100 * np.mean([any_hits_target(r) for r in pool])
        l = 100 * np.mean([largest_hits_target(r) for r in pool])
        say(f"  {label:<20} n={len(pool):<5} any-box hits target {a:6.2f}%   "
            f"LARGEST box hits target {l:6.2f}%")
    say("  (the largest-box column is what deployment and the milestone-6 "
        "comparison use; the single-face split is the cabin analogue)")

    right_face = [r for r in images if largest_hits_target(r)]

    # ---- sanity + 2. scale response ---------------------------------------
    say("\n=== 2. Framing scale response of the trained model ===")
    sample = sorted(rng.sample(images, min(300, len(images))))
    gtb_nme = [nme(run_model(grays[r], gt_box[r]), gt24[r]) for r in sample]
    say(f"  GT-box framing through the LIVE path: {nme_stats(gtb_nme)}")
    say("  (must sit near the milestone-5 7.346%; it did in your run: 6.816%)")
    a = require(cfg, "train.augment")
    if "framing" in a:
        k_lo, k_hi = float(a["framing"][0]), float(a["framing"][1])
    else:
        k_lo, k_hi = 1.0 / float(a["scale"][1]), 1.0 / float(a["scale"][0])
    say(f"  the CONFIGURED training envelope is k in [{k_lo:.2f}, {k_hi:.2f}]; "
        "the checkpoint under test was trained with whatever its own "
        "config_used.yaml says, so read the curve, not the config")
    say(f"  {'k':>6} {'NME':>26}")
    for k in SCALE_CURVE:
        vals = [nme(run_model(grays[r], scaled_box(gt_box[r], k)), gt24[r])
                for r in sample]
        say(f"  {k:>6.2f} {nme_stats(vals):>40}")
    vals = [nme(run_model(grays[r], scaled_box(gt_box[r], 1.56, 0.08)), gt24[r])
            for r in sample]
    say(f"  deploy-like (k=1.56, shift 0.08): {nme_stats(vals)}")

    # ---- 3. calibration grid, decided on NME -------------------------------
    say("\n=== 3. Calibration grid on right-face images (face choice "
        "excluded), decided on END NME, not containment ===")
    say(f"  right-face images: {len(right_face)}")
    say(f"  {'scale':>6} {'shift':>6} {'containment':>12} {'NME':>26}")
    grid_rows = []
    for s in GRID_SCALES:
        for d in GRID_SHIFTS:
            vals, contained = [], 0
            for rel in right_face:
                box = haar_to_crop_box(haar_boxes[rel][0], s, d)
                pts01_gt = (gt24[rel] - (box.x0, box.y0)) / box.side
                contained += bool(pts01_gt.min() >= 0 and pts01_gt.max() <= 1)
                vals.append(nme(run_model(grays[rel], box), gt24[rel]))
            row = {"box_scale": s, "box_shift_y": d,
                   "containment_pct": round(100 * contained / max(1, len(right_face)), 2),
                   "nme_pct_mean": round(100 * float(np.mean(vals)), 3) if vals else None,
                   "nme_pct_median": round(100 * float(np.median(vals)), 3) if vals else None}
            grid_rows.append(row)
            say(f"  {s:>6.2f} {d:>6.2f} {row['containment_pct']:>11.2f}% "
                f"{nme_stats(vals):>40}")

    # ---- 4. two-stage refinement -------------------------------------------
    say("\n=== 4. Two-stage refinement (no retraining): stage-1 box from the "
        "current calibration, stage-2 box rebuilt from stage-1 points ===")
    s1_vals, s2_vals = [], []
    for rel in right_face:
        box1 = haar_to_crop_box(haar_boxes[rel][0], det.haar.box_scale,
                                det.haar.box_shift_y)
        pred1 = run_model(grays[rel], box1)
        s1_vals.append(nme(pred1, gt24[rel]))
        box2 = square_box_around(pred1, expand)
        s2_vals.append(nme(run_model(grays[rel], box2), gt24[rel]))
    say(f"  stage 1 (current {det.haar.box_scale}, {det.haar.box_shift_y}): "
        f"{nme_stats(s1_vals)}")
    say(f"  stage 2 (refined box):                {nme_stats(s2_vals)}")
    say(f"  GT-box ceiling on the same population: "
        f"{nme_stats([nme(run_model(grays[r], gt_box[r]), gt24[r]) for r in right_face])}")

    results = {
        "n_images": len(images),
        "any_box_all_faces_pct": round(100 * per_face_hits / max(1, per_face_total), 2),
        "largest_hits_target_pct": round(
            100 * float(np.mean([largest_hits_target(r) for r in images])), 2),
        "single_face_largest_hits_target_pct": round(
            100 * float(np.mean([largest_hits_target(r) for r in single])), 2)
        if single else None,
        "gt_box_live_path_nme_pct_mean": round(100 * float(np.mean(gtb_nme)), 3),
        "calibration_grid": grid_rows,
        "two_stage": {"stage1_nme_pct_mean": round(100 * float(np.mean(s1_vals)), 3)
                      if s1_vals else None,
                      "stage2_nme_pct_mean": round(100 * float(np.mean(s2_vals)), 3)
                      if s2_vals else None},
    }
    with open(out_dir / "m6_diagnosis.yaml", "w") as f:
        yaml.safe_dump(results, f, sort_keys=False)
    (out_dir / "report.txt").write_text("\n".join(_lines) + "\n")
    save_config_snapshot(cfg, out_dir)
    say(f"\nresults + report + config snapshot in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

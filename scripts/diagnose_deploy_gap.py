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
from dms_layer1.model.io import describe_weights

BOX24_EXPANDS = [1.30, 1.45, 1.60, 1.75, 1.90, 2.05]
SCALE_CURVE = [1.0, 1.1, 1.2, 1.3, 1.4, 1.56, 1.7, 1.9]
GRID_SCALES = [1.12, 1.30, 1.45, 1.60, 1.75]
# Centre, not scale, is where the residual lives (section 5), so the shift
# axes are searched properly rather than sampled at the two values milestone 3
# happened to try. Section 5b measures the bias first and this grid confirms
# it on end NME.
GRID_SHIFTS = [0.00, 0.04, 0.08, 0.11, 0.13, 0.16, 0.20]
GRID_SHIFTS_X = [-0.06, -0.03, 0.00, 0.03, 0.06]
REFINE_EXPANDS = [1.15, 1.30, 1.45, 1.60]
# The milestone-3 any-box baseline, WITH the settings that produced it. A
# bare number is not a gate: this one read as a regression when the only
# change was min_neighbors 2 -> 5, adopted deliberately in c7ad69a.
ANY_BOX_BASELINE = {"rate_pct": 72.08, "settings": (1.05, 2, 0.08)}
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


def centred_box(cx: float, cy: float, side: float) -> CropBox:
    """Square box of a given side centred on a point."""
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
    say("")
    say(describe_weights(args.weights or require(cfg, "detector.weights"), det.meta))
    say("  ^ check this line first: a stale weights file reproduces old "
        "numbers exactly and is otherwise invisible in the output.")

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
    sf = require(cfg, "face_detector.scale_factor")
    nb = require(cfg, "face_detector.min_neighbors")
    mf = require(cfg, "face_detector.min_size_frac")
    say(f"  any-box rate, ALL annotated faces (the milestone-3 definition): "
        f"{100 * per_face_hits / max(1, per_face_total):.2f}% "
        f"({per_face_hits}/{per_face_total})")
    say(f"  this run: scale_factor {sf}, min_neighbors {nb}, "
        f"min_size_frac {mf}")
    if (sf, nb, mf) == ANY_BOX_BASELINE["settings"]:
        say(f"  baseline at these settings: {ANY_BOX_BASELINE['rate_pct']}% "
            "(milestone 3). A gap here is a real regression.")
    else:
        say(f"  NOTE: the {ANY_BOX_BASELINE['rate_pct']}% milestone-3 baseline "
            f"was measured at {ANY_BOX_BASELINE['settings']}, not at these "
            "settings, so this number is NOT expected to reproduce it. "
            "min_neighbors was raised from 2 to 5 by the selection diagnostic, "
            "which was a deliberate trade: fewer spurious boxes, and any-box "
            "recall falls because there are fewer boxes to hit with. Compare "
            "against the largest-box and confidence-selected rates below, "
            "which are what deployment uses.")

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
    say(f"  {'scale':>6} {'shift':>6} {'containment':>12} {'NME':>26} "
        f"{'fail@thr':>9}")
    grid_rows = []

    def grid_point(s: float, d: float, dx: float) -> dict:
        vals, contained = [], 0
        for rel in right_face:
            box = haar_to_crop_box(haar_boxes[rel][0], s, d, dx)
            pts01_gt = (gt24[rel] - (box.x0, box.y0)) / box.side
            contained += bool(pts01_gt.min() >= 0 and pts01_gt.max() <= 1)
            vals.append(nme(run_model(grays[rel], box), gt24[rel]))
        # failure rate as well as the mean: a driver monitoring system cares
        # about the tail more than the average, and a configuration that wins
        # on the mean can carry a heavier one
        thr = float(require(cfg, "eval.failure_threshold"))
        return {"box_scale": s, "box_shift_y": d, "box_shift_x": dx,
                "containment_pct": round(100 * contained / max(1, len(right_face)), 2),
                "nme_pct_mean": round(100 * float(np.mean(vals)), 3) if vals else None,
                "nme_pct_median": round(100 * float(np.median(vals)), 3) if vals else None,
                "failure_pct": round(100 * float(np.mean(np.array(vals) > thr)), 2)
                if vals else None,
                "_vals": vals}

    for s in GRID_SCALES:
        for d in GRID_SHIFTS:
            row = grid_point(s, d, 0.0)
            grid_rows.append(row)
            say(f"  {s:>6.2f} {d:>6.2f} {row['containment_pct']:>11.2f}% "
                f"{nme_stats(row['_vals']):>40} {row['failure_pct']:>8.2f}%")

    # horizontal offset, swept at the best (scale, shift_y) above. Separable
    # to keep this a 1D pass rather than a third grid dimension: the axes act
    # on different components of the centre error and section 5b reports both
    # biases independently.
    best_sy = min(grid_rows, key=lambda r: r["nme_pct_mean"])
    say(f"\n  horizontal offset at the best (scale {best_sy['box_scale']:.2f}, "
        f"shift_y {best_sy['box_shift_y']:.2f}); box_shift_x has existed in "
        "the calibration report since milestone 3 and never in the transform:")
    say(f"  {'shift_x':>8} {'NME':>26}")
    for dx in GRID_SHIFTS_X:
        row = grid_point(best_sy["box_scale"], best_sy["box_shift_y"], dx)
        grid_rows.append(row)
        say(f"  {dx:>8.2f} {nme_stats(row['_vals']):>40} "
            f"{row['failure_pct']:>8.2f}%")
    for r in grid_rows:
        r.pop("_vals", None)

    # ---- 4. refinement, crossed with the stage-1 calibration ---------------
    say("\n=== 4. Two-stage refinement crossed with the stage-1 calibration ===")
    say("  Refinement quality depends on the points it starts from, so the "
        "stage-1 box and refinement are ONE joint decision, not two.")
    ceiling = [nme(run_model(grays[r], gt_box[r]), gt24[r]) for r in right_face]
    say(f"  GT-box ceiling on this population: {nme_stats(ceiling)}")
    say(f"  {'scale':>6} {'shift':>6} {'stage 1':>10} {'stage 2':>10} {'gain':>8}")
    refine_rows, best = [], None
    for sc in GRID_SCALES:
        for sh in GRID_SHIFTS:
            s1, s2 = [], []
            for rel in right_face:
                box1 = haar_to_crop_box(haar_boxes[rel][0], sc, sh,
                                        det.haar.box_shift_x)
                pred1 = run_model(grays[rel], box1)
                s1.append(nme(pred1, gt24[rel]))
                box2 = square_box_around(pred1, det.refine_expand)
                s2.append(nme(run_model(grays[rel], box2), gt24[rel]))
            m1, m2 = float(np.mean(s1)), float(np.mean(s2))
            refine_rows.append({"box_scale": sc, "box_shift_y": sh,
                                "stage1_nme_pct": round(100 * m1, 3),
                                "stage2_nme_pct": round(100 * m2, 3)})
            say(f"  {sc:>6.2f} {sh:>6.2f} {100 * m1:>9.3f}% {100 * m2:>9.3f}% "
                f"{100 * (m1 - m2):>+7.3f}")
            if best is None or m2 < best[0]:
                best = (m2, sc, sh)
    say(f"  best end-to-end: ({best[1]:.2f}, {best[2]:.2f}) at "
        f"{100 * best[0]:.3f}%, against the {100 * np.mean(ceiling):.3f}% ceiling")

    say("\n  refine_expand sweep from that stage-1 box (the rebuilt box does "
        "not have to be the canonical framing; the model's own optimum may "
        "sit slightly wide of k = 1.0):")
    exp_rows = []
    for re_exp in REFINE_EXPANDS:
        vals = []
        for rel in right_face:
            box1 = haar_to_crop_box(haar_boxes[rel][0], best[1], best[2],
                                    det.haar.box_shift_x)
            pred1 = run_model(grays[rel], box1)
            vals.append(nme(run_model(grays[rel], square_box_around(pred1, re_exp)),
                            gt24[rel]))
        exp_rows.append({"refine_expand": re_exp,
                         "nme_pct": round(100 * float(np.mean(vals)), 3)})
        mark = "  <- config" if abs(re_exp - det.refine_expand) < 1e-9 else ""
        say(f"    refine_expand {re_exp:.2f} (k = {re_exp / expand:.2f}): "
            f"{nme_stats(vals)}{mark}")

    say("\n  a third stage, to check refinement has converged:")
    best_exp = min(exp_rows, key=lambda r: r["nme_pct"])["refine_expand"]
    s3 = []
    for rel in right_face:
        pred = run_model(grays[rel], haar_to_crop_box(haar_boxes[rel][0],
                                                      best[1], best[2],
                                                      det.haar.box_shift_x))
        for _ in range(2):
            pred = run_model(grays[rel], square_box_around(pred, best_exp))
        s3.append(nme(pred, gt24[rel]))
    say(f"    stage 3 at refine_expand {best_exp:.2f}: {nme_stats(s3)}")

    # ---- 5. what is left, decomposed ---------------------------------------
    say("\n=== 5. The residual, decomposed: scale error vs centre error ===")
    say("  Face SELECTION is excluded by construction here (right-face images "
        "only), so the gap to the ceiling is box GEOMETRY: the deploy box has "
        "both the wrong size and the wrong centre. These four variants "
        "separate the two.")
    cal_sc, cal_sh = det.haar.box_scale, det.haar.box_shift_y
    cal_sx = det.haar.box_shift_x
    variants = {"A gt box (ceiling)": [], "B gt centre, deploy size": [],
                "C deploy centre, gt size": [], "D deploy box (both)": []}
    realized_k, offsets = [], []
    for rel in right_face:
        g = gt_box[rel]
        d_box = haar_to_crop_box(haar_boxes[rel][0], cal_sc, cal_sh, cal_sx)
        gcx, gcy = g.x0 + g.side / 2, g.y0 + g.side / 2
        dcx, dcy = d_box.x0 + d_box.side / 2, d_box.y0 + d_box.side / 2
        realized_k.append(d_box.side / g.side)
        offsets.append(np.hypot(dcx - gcx, dcy - gcy) / g.side)
        variants["A gt box (ceiling)"].append(nme(run_model(grays[rel], g), gt24[rel]))
        variants["B gt centre, deploy size"].append(
            nme(run_model(grays[rel], centred_box(gcx, gcy, d_box.side)), gt24[rel]))
        variants["C deploy centre, gt size"].append(
            nme(run_model(grays[rel], centred_box(dcx, dcy, g.side)), gt24[rel]))
        variants["D deploy box (both)"].append(
            nme(run_model(grays[rel], d_box), gt24[rel]))
    for label, vals in variants.items():
        say(f"  {label:<26} {nme_stats(vals)}")
    a, b, c, d = (float(np.mean(variants[k])) for k in variants)
    say(f"  cost of size alone   (B - A): {100 * (b - a):+.3f} NME points")
    say(f"  cost of centre alone (C - A): {100 * (c - a):+.3f} NME points")
    say(f"  cost of both         (D - A): {100 * (d - a):+.3f} NME points")
    say(f"  interaction (D - A) - (B - A) - (C - A): "
        f"{100 * (d - a - (b - a) - (c - a)):+.3f}")
    say(f"\n  realized framing k at ({cal_sc}, {cal_sh}): "
        f"median {np.median(realized_k):.2f}  "
        f"p10 {np.percentile(realized_k, 10):.2f}  "
        f"p90 {np.percentile(realized_k, 90):.2f}")
    say(f"  centre offset / face side: median {np.median(offsets):.3f}  "
        f"p90 {np.percentile(offsets, 90):.3f}")
    dv = np.array(variants["D deploy box (both)"])
    for name, arr, edges in (("realized k", np.array(realized_k), [0, 1.1, 1.3, 1.5, 9]),
                             ("centre offset", np.array(offsets), [0, 0.05, 0.1, 0.2, 9])):
        say(f"  deploy NME binned by {name}:")
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (arr >= lo) & (arr < hi)
            if m.sum():
                say(f"    {lo:.2f} to {hi:.2f}: {100 * dv[m].mean():6.3f}%  (n={m.sum()})")

    # ---- 5b. centre error: detector scatter vs calibration bias ------------
    say("\n=== 5b. Whose centre error is it? ===")
    say("  Section 5 says the residual is centre, not size. This splits the "
        "centre error into the part a constant transform can remove (a BIAS: "
        "our calibration puts the box in the wrong place on every face the "
        "same way) and the part it cannot (SCATTER: the Haar box lands "
        "differently face to face). Offsets are in units of the DEPLOY box "
        "side, which is what box_shift_x and box_shift_y are multiplied by.")
    dxs, dys = [], []
    for rel in right_face:
        g = gt_box[rel]
        d_box = haar_to_crop_box(haar_boxes[rel][0], cal_sc, cal_sh, cal_sx)
        gcx, gcy = g.x0 + g.side / 2, g.y0 + g.side / 2
        dcx, dcy = d_box.x0 + d_box.side / 2, d_box.y0 + d_box.side / 2
        dxs.append((dcx - gcx) / d_box.side)
        dys.append((dcy - gcy) / d_box.side)
    dxs, dys = np.array(dxs), np.array(dys)
    say(f"  horizontal: bias {dxs.mean():+.4f}  scatter (sd) {dxs.std():.4f}  "
        f"p10 {np.percentile(dxs, 10):+.4f}  p90 {np.percentile(dxs, 90):+.4f}")
    say(f"  vertical:   bias {dys.mean():+.4f}  scatter (sd) {dys.std():.4f}  "
        f"p10 {np.percentile(dys, 10):+.4f}  p90 {np.percentile(dys, 90):+.4f}")
    say(f"  bias / scatter ratio: horizontal {abs(dxs.mean()) / max(dxs.std(), 1e-9):.2f}, "
        f"vertical {abs(dys.mean()) / max(dys.std(), 1e-9):.2f}. Above 1 means "
        "the systematic part dominates and calibration is worth more than a "
        "better detector; below 1 means the reverse.")
    say(f"  the bias-cancelling config values would be "
        f"box_shift_x {cal_sx - dxs.mean():+.4f}, "
        f"box_shift_y {cal_sh - dys.mean():+.4f}")

    # A horizontal bias that only appears on turned heads is not a constant:
    # a frontal cascade boxes the visible part of a yawed face, so the offset
    # follows the pose. Split it, because a constant shift_x can only fix the
    # part that is there on frontal faces too.
    posed = np.array([bool(targets[rel].attributes.get("pose", 0))
                      for rel in right_face])
    if posed.any() and (~posed).any():
        say(f"  horizontal bias on frontal faces {dxs[~posed].mean():+.4f} "
            f"(n={int((~posed).sum())}) against {dxs[posed].mean():+.4f} on "
            f"pose-flagged faces (n={int(posed.sum())}). A bias that only "
            "appears with pose is not a constant and a single shift_x cannot "
            "remove it.")

    corrected = [nme(run_model(grays[rel],
                               haar_to_crop_box(haar_boxes[rel][0], cal_sc,
                                                cal_sh - float(dys.mean()),
                                                cal_sx - float(dxs.mean()))),
                     gt24[rel]) for rel in right_face]
    say("\n  end NME on the same faces:")
    say(f"    as configured                {nme_stats(variants['D deploy box (both)'])}")
    say(f"    bias removed (best constant) {nme_stats(corrected)}")
    say(f"    ground-truth centre (oracle) {nme_stats(variants['B gt centre, deploy size'])}")
    dd = float(np.mean(variants["D deploy box (both)"]))
    cc = float(np.mean(corrected))
    bb = float(np.mean(variants["B gt centre, deploy size"]))
    say(f"    removable by calibration: {100 * (dd - cc):+.3f} NME points")
    say(f"    left as detector scatter: {100 * (cc - bb):+.3f} NME points")
    say("  The first number is ours to fix in the config. The second is the "
        "Haar box landing in a different place on each face, and no constant "
        "transform reaches it: it needs a better front end, or a second stage "
        "that rebuilds the box from predicted points, which is what "
        "refinement does.")

    # ---- 6. boxing from 24 landmarks, which is what a landmark front end
    # ---- hands the model ---------------------------------------------------
    say("\n=== 6. What expand a 24-POINT box needs ===")
    say("  The Haar path frames the face with a calibrated transform of a "
        "detector rectangle. A landmark-based front end (ours_on_mp_box, and "
        "our own second refinement stage) instead boxes 24 points, which is a "
        "different construction: the 24-point extent is not the 98-point "
        "extent the cache was built from, so the same nominal expand does not "
        "produce the same framing. Ground-truth points are used here so this "
        "measures the BOX GEOMETRY with detector error removed.")
    say(f"  {'expand':>7} {'implied k':>10} {'NME':>26}")
    box24_rows = []
    for e in BOX24_EXPANDS:
        vals, ks = [], []
        for rel in sample:
            box = square_box_around(gt24[rel], e)
            vals.append(nme(run_model(grays[rel], box), gt24[rel]))
            ks.append(box.side / max(gt_box[rel].side, 1e-9))
        box24_rows.append({"expand": e, "k_median": round(float(np.median(ks)), 3),
                           "nme_pct_mean": round(100 * float(np.mean(vals)), 3)})
        say(f"  {e:>7.2f} {float(np.median(ks)):>10.3f} {nme_stats(vals):>40}")
    best24 = min(box24_rows, key=lambda r: r["nme_pct_mean"])
    say(f"  best: expand {best24['expand']:.2f} (k = {best24['k_median']:.3f}) "
        f"at {best24['nme_pct_mean']:.3f}%")
    say("  Set detector.cross_expand and detector.refine_expand from this, the "
        "same way box_scale was set from section 3. A landmark front end left "
        "at the nominal reference_expand is an UNCALIBRATED path, and "
        "comparing it against the calibrated Haar path measures the missing "
        "calibration rather than the front end.")

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
        "refinement_grid": refine_rows,
        "refine_expand_sweep": exp_rows,
        "box_from_24_points_sweep": box24_rows,
        "best_end_to_end": {"box_scale": best[1], "box_shift_y": best[2],
                            "nme_pct": round(100 * best[0], 3)},
        "residual_decomposition": {k: round(100 * float(np.mean(v)), 3)
                                   for k, v in variants.items()},
        "centre_error": {
            "shift_x_bias": round(float(dxs.mean()), 4),
            "shift_x_scatter": round(float(dxs.std()), 4),
            "shift_y_bias": round(float(dys.mean()), 4),
            "shift_y_scatter": round(float(dys.std()), 4),
            "suggested_box_shift_x": round(cal_sx - float(dxs.mean()), 4),
            "suggested_box_shift_y": round(cal_sh - float(dys.mean()), 4),
            "nme_as_configured_pct": round(100 * dd, 3),
            "nme_bias_removed_pct": round(100 * cc, 3),
            "nme_oracle_centre_pct": round(100 * bb, 3)},
    }
    with open(out_dir / "m6_diagnosis.yaml", "w") as f:
        yaml.safe_dump(results, f, sort_keys=False)
    (out_dir / "report.txt").write_text("\n".join(_lines) + "\n")
    save_config_snapshot(cfg, out_dir)
    say(f"\nresults + report + config snapshot in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

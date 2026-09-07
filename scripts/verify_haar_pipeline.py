#!/usr/bin/env python3
"""Milestone 3: verify the Haar face-detection front-end on WFLW.

Runs the cascade over every image of a split and reports:
  1. Detection rate - the fraction of annotated faces matched by a Haar box
    (IoU vs the ground-truth crop box >= face_detector.match_iou), overall
    and per attribute subset. Large-pose/occlusion rates matter for in-cabin
    use. Unmatched Haar boxes are counted per image but NOT called false
    positives: WFLW images contain unannotated faces.
  2. Calibration - measured box_scale / box_shift from matched pairs, i.e.
    how a raw Haar box maps to the crop box the model trains on. Prints
    recommended config values to replace the provisional ones.
  3. Containment - fraction of faces whose 24 landmarks all fall inside the
    calibrated crop box (with the current config and with the measured
    medians). If the crop misses landmarks, the model cannot predict them.
  4. Round trip - landmarks mapped frame -> calibrated crop space -> back
    (max error, exact math), plus the quantisation bound at the model input
    resolution.
  5. Timing - detection ms/image on this CPU (relative comparison only; not
    a deployment figure).
  6. Previews - matched faces (ground-truth box green, raw Haar red,
    calibrated crop cyan; plus the extracted model crop with ground-truth
    points), and a page of missed faces.

Usage (Kaggle):
    python scripts/verify_haar_pipeline.py --config configs/layer1_base.yaml --split test
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dms_layer1.config import load_config, require, resolve_path, save_config_snapshot
from dms_layer1.data import wflw
from dms_layer1.data.crops import extract_square, to_crop_space, to_frame_space
from dms_layer1.detect.haar import (HaarFaceDetector, box_iou,
                                    calibrate_haar_to_crop, gt_crop_box,
                                    haar_to_crop_box)
from dms_layer1.landmarks.schema import load_schema
from dms_layer1.viz.overlay import GROUP_COLORS

FONT = cv2.FONT_HERSHEY_SIMPLEX
_lines: list[str] = []


def say(text: str = "") -> None:
    print(text)
    _lines.append(text)


def rect(b) -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1) from a FaceBox or CropBox."""
    if hasattr(b, "side"):
        return (b.x0, b.y0, b.x0 + b.side, b.y0 + b.side)
    return (b.x, b.y, b.x + b.w, b.y + b.h)


def containment_arrays(haar_matches, schema):
    """Vectorisation prep: per matched face, the Haar side/centre and the 24
    ground-truth landmark coordinates."""
    side = np.array([max(hb.w, hb.h) for _, hb in haar_matches], dtype=np.float64)
    cx = np.array([hb.center[0] for _, hb in haar_matches])
    cy = np.array([hb.center[1] for _, hb in haar_matches])
    lm = np.stack([rec.landmarks98[schema.wflw_indices] for rec, _ in haar_matches])
    return side, cx, cy, lm


def containment(side, cx, cy, lm, scale: float, shift_y: float) -> float:
    """Fraction of matched faces whose 24 landmarks all sit inside the crop
    box derived from their Haar box with the given calibration."""
    s = side * scale
    x0 = cx - s / 2
    y0 = cy + shift_y * s - s / 2
    inside_x = (lm[:, :, 0] >= x0[:, None]) & (lm[:, :, 0] <= (x0 + s)[:, None])
    inside_y = (lm[:, :, 1] >= y0[:, None]) & (lm[:, :, 1] <= (y0 + s)[:, None])
    return float(np.mean(np.all(inside_x & inside_y, axis=1)))


def best_containment_calibration(side, cx, cy, lm) -> tuple[float, float, float]:
    """Grid-search (box_scale, box_shift_y) for maximum landmark containment;
    ties break toward the smallest scale (tighter crop = more face pixels)
    then the smallest |shift|. Containment is what matters operationally: a
    median-fit box loses the tails, and a landmark outside the crop is a
    landmark the model cannot predict."""
    best = (0.0, None, None)
    for scale in np.arange(1.10, 1.751, 0.05):
        for shift in np.arange(0.00, 0.251, 0.01):
            c = containment(side, cx, cy, lm, float(scale), float(shift))
            key = (c, -scale, -abs(shift))
            if best[1] is None or key > (best[0], -best[1], -abs(best[2])):
                best = (c, float(scale), float(shift))
    return best


def render_matched(image, rec, hb, det, schema, input_size, expand) -> np.ndarray:
    gt = gt_crop_box(rec.landmarks98, expand)
    crop_box = haar_to_crop_box(hb, det.box_scale, det.box_shift_y,
                                det.box_shift_x)
    view = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for b, color in ((gt, (80, 220, 80)), (hb, (60, 60, 255)), (crop_box, (255, 220, 60))):
        x0, y0, x1, y1 = (int(round(v)) for v in rect(b))
        cv2.rectangle(view, (x0, y0), (x1, y1), color, 2)
    x0, y0, x1, y1 = rect(crop_box)
    m = crop_box.side * 0.25
    vx0, vy0 = max(0, int(x0 - m)), max(0, int(y0 - m))
    vx1, vy1 = min(image.shape[1], int(x1 + m)), min(image.shape[0], int(y1 + m))
    view = view[vy0:vy1, vx0:vx1]
    s = 360 / max(1, view.shape[0])
    view = cv2.resize(view, (max(1, int(view.shape[1] * s)), 360))

    crop = extract_square(image, crop_box, input_size)
    big = 360
    tile = cv2.cvtColor(cv2.resize(crop, (big, big), interpolation=cv2.INTER_NEAREST),
                        cv2.COLOR_GRAY2BGR)
    lm01 = to_crop_space(rec.landmarks98[schema.wflw_indices], crop_box)
    for p in schema.points:
        x, y = lm01[p.index] * big
        if 0 <= x < big and 0 <= y < big:
            cv2.circle(tile, (int(round(x)), int(round(y))), 4, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(tile, (int(round(x)), int(round(y))), 3,
                       GROUP_COLORS[p.group], -1, cv2.LINE_AA)
    panel = np.concatenate([view, tile], axis=1)
    strip = np.full((26, panel.shape[1], 3), (25, 25, 25), dtype=np.uint8)
    flags = [k for k, v in rec.attributes.items() if v] or ["frontal"]
    cv2.putText(strip, f"{rec.image_rel_path} line {rec.line_number} [{','.join(flags)}]  "
                       "green=gt-box red=haar cyan=calibrated-crop",
                (6, 18), FONT, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
    return np.concatenate([panel, strip], axis=0)


def render_missed(image, rec, expand) -> np.ndarray:
    gt = gt_crop_box(rec.landmarks98, expand)
    view = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    x0, y0, x1, y1 = (int(round(v)) for v in rect(gt))
    cv2.rectangle(view, (x0, y0), (x1, y1), (80, 220, 80), 2)
    m = gt.side
    vx0, vy0 = max(0, x0 - m), max(0, y0 - m)
    vx1, vy1 = min(image.shape[1], x1 + m), min(image.shape[0], y1 + m)
    view = view[vy0:vy1, vx0:vx1]
    s = 360 / max(1, view.shape[0])
    view = cv2.resize(view, (max(1, int(view.shape[1] * s)), 360))
    strip = np.full((26, view.shape[1], 3), (25, 25, 25), dtype=np.uint8)
    flags = [k for k, v in rec.attributes.items() if v] or ["frontal"]
    cv2.putText(strip, f"MISSED  {rec.image_rel_path} line {rec.line_number} "
                       f"[{','.join(flags)}]", (6, 18), FONT, 0.45,
                (120, 120, 245), 1, cv2.LINE_AA)
    return np.concatenate([view, strip], axis=0)


def grid(tiles: list[np.ndarray], cols: int = 2) -> np.ndarray:
    w = max(t.shape[1] for t in tiles)
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, w - t.shape[1],
                                cv2.BORDER_CONSTANT, value=(35, 35, 35)) for t in tiles]
    rows = []
    for i in range(0, len(tiles), cols):
        row = tiles[i:i + cols]
        while len(row) < cols:
            row.append(np.full((h, w, 3), (35, 35, 35), dtype=np.uint8))
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0,
                    help="process only this many images (seeded sample); 0 = all")
    ap.add_argument("--out-dir", default="/kaggle/working/m3_haar")
    ap.add_argument("--synthetic", action="store_true",
                    help="run on generated schematic faces (code smoke test only)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.synthetic:
        import tempfile
        from dms_layer1.data.synthetic import write_synthetic_dataset
        say("SYNTHETIC MODE: schematic faces - smoke test of the code path only.")
        root = write_synthetic_dataset(tempfile.mkdtemp(prefix="wflw_synth_"),
                                       require(cfg, "dataset.attribute_names"),
                                       seed=require(cfg, "seed"))
        cfg["dataset"]["root"] = str(root)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = load_schema(resolve_path(cfg, require(cfg, "landmark_schema")))
    det = HaarFaceDetector(cfg)
    expand = float(require(cfg, "preprocess.reference_expand"))
    input_size = int(require(cfg, "model.input_size"))
    match_iou = float(require(cfg, "face_detector.match_iou"))
    rng = random.Random(require(cfg, "seed"))

    records, paths = wflw.load_split(cfg, args.split)
    by_image: dict[str, list[wflw.FaceRecord]] = {}
    for rec in records:
        by_image.setdefault(rec.image_rel_path, []).append(rec)
    images = list(by_image)
    if args.limit and args.limit < len(images):
        rng.shuffle(images)
        images = sorted(images[:args.limit])
        say(f"limiting to a seeded sample of {len(images)} images")

    matched: list[tuple[wflw.FaceRecord, object]] = []   # (record, FaceBox)
    missed: list[wflw.FaceRecord] = []
    unmatched_boxes = 0
    detect_ms: list[float] = []
    # keep decoded grays only for a seeded subset, as preview candidates
    preview_rels = set(rng.sample(images, min(300, len(images))))
    gray_cache: dict[str, np.ndarray] = {}

    for n_img, rel in enumerate(images, 1):
        img_path = wflw.image_path(paths, by_image[rel][0])
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise wflw.WFLWError(f"cv2 could not decode {img_path}")
        t0 = time.perf_counter()
        boxes = det.detect(gray)
        detect_ms.append((time.perf_counter() - t0) * 1000)

        taken = set()
        for rec in by_image[rel]:
            gt = gt_crop_box(rec.landmarks98, expand)
            best_i, best_iou = -1, 0.0
            for i, hb in enumerate(boxes):
                if i in taken:
                    continue
                iou = box_iou(rect(hb), rect(gt))
                if iou > best_iou:
                    best_i, best_iou = i, iou
            if best_iou >= match_iou:
                taken.add(best_i)
                matched.append((rec, boxes[best_i]))
            else:
                missed.append(rec)
        unmatched_boxes += len(boxes) - len(taken)
        if rel in preview_rels:
            gray_cache[rel] = gray
        if n_img % 200 == 0:
            say(f"  {n_img}/{len(images)} images")

    n_faces = len(matched) + len(missed)
    say(f"\nSplit '{args.split}': {len(images)} images, {n_faces} annotated faces")
    say(f"detector: scale_factor={det.scale_factor} min_neighbors={det.min_neighbors} "
        f"min_size_frac={det.min_size_frac} equalize={det.equalize_hist}")

    say("\n=== 1. Detection rate (Haar box matches GT crop box at "
        f"IoU >= {match_iou}) ===")
    def rate(pred):
        m = sum(1 for r, _ in matched if pred(r))
        t = m + sum(1 for r in missed if pred(r))
        return f"{100 * m / t:6.2f}%  (n={t})" if t else "   n/a  (n=0)"
    say(f"  overall        {rate(lambda r: True)}")
    say(f"  no flags       {rate(lambda r: sum(r.attributes.values()) == 0)}")
    for a in require(cfg, "dataset.attribute_names"):
        say(f"  {a:<14} {rate(lambda r, a=a: r.attributes[a] == 1)}")
    say(f"  unmatched Haar boxes: {unmatched_boxes} total "
        f"({unmatched_boxes / max(1, len(images)):.2f}/image) - NOT necessarily "
        "false positives; WFLW images contain unannotated faces.")

    say("\n=== 2. Calibration: raw Haar box -> model crop box ===")
    cal = None
    best = None
    if matched:
        cal = calibrate_haar_to_crop(
            [gt_crop_box(r.landmarks98, expand) for r, _ in matched],
            [hb for _, hb in matched])
        say("  descriptive median fit (how boxes relate on the typical face -")
        say("  NOT the recommendation; a median-fit box loses the tails):")
        for k in ("box_scale", "box_shift_x", "box_shift_y"):
            q = cal[k]
            say(f"    {k:<12} median {q['median']:+.3f}   IQR [{q['p25']:+.3f}, {q['p75']:+.3f}]")
        side, cx, cy, lm = containment_arrays(matched, schema)
        best = best_containment_calibration(side, cx, cy, lm)
        say("  recommendation (containment-maximising grid search over")
        say("  scale 1.10-1.75 x shift 0.00-0.25; ties -> tighter crop):")
        say(f"    box_scale: {best[1]:.2f}")
        say(f"    box_shift_y: {best[2]:.2f}")
        say(f"    (containment {100 * best[0]:.2f}%)")

    say("\n=== 3. Landmark containment by calibration candidate ===")
    if matched:
        candidates = [
            (f"current config ({det.box_scale:.2f}, {det.box_shift_y:.2f})",
             det.box_scale, det.box_shift_y),
            ("candidate (1.45, 0.13)", 1.45, 0.13),
            (f"measured medians ({cal['box_scale']['median']:.2f}, "
             f"{cal['box_shift_y']['median']:.2f})",
             cal["box_scale"]["median"], cal["box_shift_y"]["median"]),
            (f"grid best ({best[1]:.2f}, {best[2]:.2f})", best[1], best[2]),
        ]
        for label, s_, d_ in candidates:
            c = containment(side, cx, cy, lm, s_, d_)
            say(f"  {label:<38} {100 * c:6.2f}% of matched faces")

    say("\n=== 4. Coordinate round trip through the deployed crop ===")
    if matched:
        errs = []
        for rec, hb in matched[:500]:
            crop = haar_to_crop_box(hb, det.box_scale, det.box_shift_y,
                                    det.box_shift_x)
            pts = rec.landmarks98[schema.wflw_indices]
            back = to_frame_space(to_crop_space(pts, crop), crop)
            errs.append(np.abs(back - pts).max())
        say(f"  frame -> crop space -> frame, max error: {max(errs):.9f} px (exact)")
        med_side = int(np.median([haar_to_crop_box(hb, det.box_scale,
                                                  det.box_shift_y,
                                                  det.box_shift_x).side
                                  for _, hb in matched]))
        say(f"  quantisation bound at model input {input_size}px: half a crop pixel "
            f"= {0.5 * med_side / input_size:.2f} frame px at the median crop side "
            f"({med_side}px) - the model regresses continuous coords, so this "
            "bound applies only to integer-pixel readouts.")

    say("\n=== 5. Detection timing (this CPU - relative use only) ===")
    say(f"  per image: median {np.median(detect_ms):.0f} ms, "
        f"mean {np.mean(detect_ms):.0f} ms, p90 {np.percentile(detect_ms, 90):.0f} ms")

    n_prev = 8
    prev_pool = [(r, hb) for r, hb in matched if r.image_rel_path in gray_cache]
    sample = rng.sample(prev_pool, min(n_prev, len(prev_pool)))
    if sample:
        tiles = [render_matched(gray_cache[r.image_rel_path], r, hb, det, schema,
                                input_size, expand) for r, hb in sample]
        cv2.imwrite(str(out_dir / "matched_preview.png"), grid(tiles))
        say(f"\npreview (matched): {out_dir / 'matched_preview.png'}")
    miss_pool = [r for r in missed if r.image_rel_path in gray_cache]
    sample_m = rng.sample(miss_pool, min(4, len(miss_pool)))
    if sample_m:
        tiles = [render_missed(gray_cache[r.image_rel_path], r, expand) for r in sample_m]
        cv2.imwrite(str(out_dir / "missed_preview.png"), grid(tiles))
        say(f"preview (missed) : {out_dir / 'missed_preview.png'}")

    results = {
        "split": args.split, "n_images": len(images), "n_faces": n_faces,
        "n_matched": len(matched), "n_missed": len(missed),
        "unmatched_haar_boxes": unmatched_boxes,
        "calibration_median_fit": cal,
        "recommended_containment_max": (
            {"box_scale": best[1], "box_shift_y": best[2],
             "containment": best[0]} if best else None),
        "containment_by_candidate": (
            {label: containment(side, cx, cy, lm, s_, d_)
             for label, s_, d_ in candidates} if matched else None),
        "detect_ms_median": float(np.median(detect_ms)) if detect_ms else None,
    }
    with open(out_dir / "m3_results.yaml", "w") as f:
        yaml.safe_dump(results, f, sort_keys=False)
    (out_dir / "report.txt").write_text("\n".join(_lines) + "\n")
    save_config_snapshot(cfg, out_dir)
    say(f"\nresults + report + config snapshot written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

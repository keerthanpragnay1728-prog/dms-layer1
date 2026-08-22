#!/usr/bin/env python3
"""Milestone 6: run both LandmarkDetector implementations on full WFLW
frames and compare them on identical faces.

Sections:
  1. MediaPipe mapping verification overlays: MediaPipe's mapped 24 points
     rendered with the milestone-1 canvas (labelled points, eye and mouth
     insets, pupil crosshairs) on seeded frontal, pose and occluded faces.
     The mapping is not trusted until these are checked by eye.
  2. Side-by-side comparison renders: ground truth, ours, and MediaPipe on
     the same faces.
  3. Full-frame ablation table: per detector, detection rate on the target
     face (the largest annotated face per image, matching the one-driver
     cabin), NME overall and per group on its matched faces, failure rate,
     and per-frame timing on this machine. Then the paired comparison on
     jointly matched faces, which is the number the ablation turns on, and
     a per-point cross-detector offset table that would expose any mapping
     error as a systematic offset on one specific point.
  4. The calibration price: our model evaluated on ground-truth boxes
     against Haar-derived boxes on the same faces (the deploy resolution
     cost flagged in milestone 3, measured at last).
  5. Footprints: our exported weights and cascade files against the model
     files bundled inside the mediapipe package.

Timing caveats: static image mode for MediaPipe (its video mode with
tracking is faster in deployment); all times are this machine, one session,
relative comparison only.

Usage (Kaggle):
    python scripts/compare_detectors.py --config configs/layer1_base.yaml --split test
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dms_layer1.config import load_config, require, resolve_path, save_config_snapshot
from dms_layer1.data import wflw
from dms_layer1.data.crops import extract_square, square_box_around, to_crop_space
from dms_layer1.detect.haar import box_iou
from dms_layer1.detect.interface import as_gray
from dms_layer1.detect.ours import OurLandmarkDetector
from dms_layer1.evaluation import metrics
from dms_layer1.landmarks.schema import load_schema
from dms_layer1.viz.overlay import GROUP_COLORS, render_points_overlay

FONT = cv2.FONT_HERSHEY_SIMPLEX
MATCH_IOU = 0.30
_lines: list[str] = []


def say(text: str = "") -> None:
    print(text)
    _lines.append(text)


def rect_of(box) -> tuple[float, float, float, float]:
    return (box.x0, box.y0, box.x0 + box.side, box.y0 + box.side)


def match_target(pts24: np.ndarray, gt24: np.ndarray) -> bool:
    """Detected points count as the target face when the boxes around both
    24-point sets overlap at IoU >= MATCH_IOU. Matching on the POINTS (the
    only currency both detectors share) folds gross landmark failure into
    the detection rate: a detector that finds the face but scatters its
    points counts as a miss. With trained models that is the intended
    reading, "usable detection"."""
    det_box = square_box_around(pts24, 1.3)
    gt_box = square_box_around(gt24, 1.3)
    return box_iou(rect_of(det_box), rect_of(gt_box)) >= MATCH_IOU


def render_compare(frame, gt24, results: dict, title: str) -> np.ndarray:
    """One face: ground truth as small green dots, each detector's points in
    group colours (ours solid, mediapipe rings)."""
    gt_box = square_box_around(gt24, 1.6)
    h, w = frame.shape[:2]
    x0, y0 = max(0, gt_box.x0), max(0, gt_box.y0)
    x1, y1 = min(w, gt_box.x0 + gt_box.side), min(h, gt_box.y0 + gt_box.side)
    view = frame[y0:y1, x0:x1]
    s = 480 / max(1, view.shape[0])
    view = cv2.resize(view, (max(1, int(view.shape[1] * s)), 480))
    if view.ndim == 2:
        view = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)

    def draw(pts, solid: bool):
        for i, (px, py) in enumerate(pts):
            px, py = (px - x0) * s, (py - y0) * s
            if not (0 <= px < view.shape[1] and 0 <= py < view.shape[0]):
                continue
            color = GROUP_COLORS[SCHEMA.points[i].group]
            if solid:
                cv2.circle(view, (int(round(px)), int(round(py))), 3, color, -1,
                           cv2.LINE_AA)
            else:
                cv2.circle(view, (int(round(px)), int(round(py))), 5, color, 1,
                           cv2.LINE_AA)

    for gx, gy in gt24:
        gx, gy = (gx - x0) * s, (gy - y0) * s
        if 0 <= gx < view.shape[1] and 0 <= gy < view.shape[0]:
            cv2.circle(view, (int(round(gx)), int(round(gy))), 1, (80, 220, 80), -1)
    if results.get("ours") is not None:
        draw(results["ours"], solid=True)
    if results.get("mediapipe") is not None:
        draw(results["mediapipe"], solid=False)

    strip = np.full((26, view.shape[1], 3), (25, 25, 25), dtype=np.uint8)
    cv2.putText(strip, title + "  (green dot=gt, solid=ours, ring=mediapipe)",
                (6, 18), FONT, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
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


def detector_stats(name, matched_nme, n_targets, per_subset, ms, threshold):
    say(f"\n  {name}")
    say(f"    detection rate (target faces): "
        f"{100 * len(matched_nme) / max(1, n_targets):.2f}%  "
        f"({len(matched_nme)}/{n_targets})")
    for label, (hit, tot) in per_subset.items():
        if tot:
            say(f"      {label:<14} {100 * hit / tot:6.2f}%  (n={tot})")
    if matched_nme:
        arr = np.array([v for v, _ in matched_nme])
        say(f"    NME on matched faces: mean {100 * arr.mean():.3f}%  "
            f"median {100 * np.median(arr):.3f}%  "
            f"failure@{threshold:.0%} {100 * np.mean(arr > threshold):.2f}%")
    say(f"    per-frame time: median {np.median(ms):.0f} ms "
        f"(this CPU, static image mode, relative only)")


def main() -> int:
    global SCHEMA
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0,
                    help="images to process (seeded sample); 0 = all")
    ap.add_argument("--weights", default=None, help="override detector.weights")
    ap.add_argument("--out-dir", default="/kaggle/working/m6_compare")
    ap.add_argument("--skip-mediapipe", action="store_true")
    ap.add_argument("--synthetic", action="store_true",
                    help="run on schematic faces with a briefly trained model "
                         "(code smoke test only)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.synthetic:
        cfg = _synthetic_pipeline(cfg)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    SCHEMA = load_schema(resolve_path(cfg, require(cfg, "landmark_schema")))
    threshold = float(require(cfg, "eval.failure_threshold"))
    rng = random.Random(require(cfg, "seed"))

    ours = OurLandmarkDetector(cfg, weights=args.weights)
    say(f"ours: weights loaded (epoch {ours.meta['epoch']}, "
        f"val NME {ours.meta['val_nme']:.3f}%)" if ours.meta["val_nme"]
        else "ours: weights loaded")
    mp_det = None
    if not args.skip_mediapipe:
        try:
            from dms_layer1.detect.mediapipe_detector import (
                MediaPipeLandmarkDetector, MediaPipeUnavailable)
            try:
                mp_det = MediaPipeLandmarkDetector(cfg, SCHEMA)
                say("mediapipe: FaceMesh ready (refine_landmarks on)")
            except MediaPipeUnavailable as e:
                say(f"mediapipe UNAVAILABLE: {e}\ncontinuing ours-only")
        except ImportError as e:
            say(f"mediapipe import failed: {e}\ncontinuing ours-only")

    records, paths = wflw.load_split(cfg, args.split)
    by_image: dict[str, list[wflw.FaceRecord]] = {}
    for rec in records:
        by_image.setdefault(rec.image_rel_path, []).append(rec)
    images = sorted(by_image)
    if args.limit and args.limit < len(images):
        rng.shuffle(images)
        images = sorted(images[:args.limit])
        say(f"limiting to a seeded sample of {len(images)} images")

    # target face per image: the largest annotated face (one-driver framing)
    targets: dict[str, wflw.FaceRecord] = {}
    for rel, recs in by_image.items():
        targets[rel] = max(recs, key=lambda r: square_box_around(r.landmarks98, 1.0).side)

    results = {"ours": {}, "mediapipe": {}}      # rel -> (nme, pts) for matches
    misses = {"ours": [], "mediapipe": []}
    times = {"ours": [], "mediapipe": []}
    unmatched = {"ours": 0, "mediapipe": 0}
    frames_cache: dict[str, np.ndarray] = {}
    preview_rels = set(rng.sample(images, min(250, len(images))))

    detectors = {"ours": ours}
    if mp_det is not None:
        detectors["mediapipe"] = mp_det

    for n_img, rel in enumerate(images, 1):
        frame = cv2.imread(str(wflw.image_path(paths, by_image[rel][0])),
                           cv2.IMREAD_COLOR)
        if frame is None:
            raise wflw.WFLWError(f"cv2 could not decode {rel}")
        gt24 = targets[rel].landmarks98[SCHEMA.wflw_indices].astype(np.float64)
        for name, det in detectors.items():
            t0 = time.perf_counter()
            out = det.detect(frame)
            times[name].append((time.perf_counter() - t0) * 1000)
            if out is None:
                misses[name].append(rel)
            elif match_target(out.points.astype(np.float64), gt24):
                nme = float(metrics.nme_per_face(
                    out.points[None].astype(np.float64), gt24[None], SCHEMA)[0])
                results[name][rel] = (nme, out.points.astype(np.float64))
            else:
                unmatched[name] += 1
                misses[name].append(rel)
        if rel in preview_rels:
            frames_cache[rel] = frame
        if n_img % 200 == 0:
            say(f"  {n_img}/{len(images)} images")

    n_targets = len(images)
    say(f"\nSplit '{args.split}': {n_targets} images, one target face each "
        "(the largest annotated face; detectors return one face, matching "
        "the one-driver cabin)")

    # ---- 1 + 2: renders ---------------------------------------------------
    def pick(flag: str | None, k: int) -> list[str]:
        pool = [r for r in images if r in frames_cache
                and (flag is None and sum(targets[r].attributes.values()) == 0
                     or flag is not None and targets[r].attributes.get(flag, 0) == 1)]
        return rng.sample(pool, min(k, len(pool)))

    mapping_tiles = []
    if mp_det is not None:
        for rel in pick(None, 2) + pick("pose", 1) + pick("occlusion", 1):
            out = mp_det.detect(frames_cache[rel])
            if out is not None:
                mapping_tiles.append(render_points_overlay(
                    frames_cache[rel], out.points, SCHEMA,
                    f"MediaPipe mapping check: {rel}",
                    inset_size=require(cfg, "visualize.inset_size")))
        if mapping_tiles:
            cv2.imwrite(str(out_dir / "mediapipe_mapping_overlays.png"),
                        grid(mapping_tiles, cols=1))
            say(f"\nmapping overlays (CHECK BY EYE, milestone-1 checklist): "
                f"{out_dir / 'mediapipe_mapping_overlays.png'}")

    compare_tiles = []
    for rel in pick(None, 4) + pick("pose", 2) + pick("occlusion", 2):
        gt24 = targets[rel].landmarks98[SCHEMA.wflw_indices].astype(np.float64)
        per = {n: results[n].get(rel, (None, None))[1] for n in detectors}
        flags = [k for k, v in targets[rel].attributes.items() if v] or ["frontal"]
        nmes = "  ".join(f"{n}:{100 * results[n][rel][0]:.1f}%" if rel in results[n]
                         else f"{n}:miss" for n in detectors)
        compare_tiles.append(render_compare(frames_cache[rel], gt24, per,
                                            f"[{','.join(flags)}] {nmes}"))
    if compare_tiles:
        cv2.imwrite(str(out_dir / "side_by_side.png"), grid(compare_tiles))
        say(f"side-by-side renders: {out_dir / 'side_by_side.png'}")

    # ---- 3: the ablation table -------------------------------------------
    say("\n=== Full-frame ablation (per detector, then paired) ===")
    attr_names = require(cfg, "dataset.attribute_names")
    for name in detectors:
        per_subset = {}
        for a in attr_names + ["no_flags"]:
            sel = [r for r in images
                   if (a == "no_flags" and sum(targets[r].attributes.values()) == 0)
                   or (a != "no_flags" and targets[r].attributes.get(a, 0) == 1)]
            per_subset["largepose" if a == "pose" else a] = (
                sum(1 for r in sel if r in results[name]), len(sel))
        detector_stats(name, list(results[name].values()), n_targets,
                       per_subset, times[name], threshold)
        say(f"    detected-but-not-target frames: {unmatched[name]}")

    paired_stats = None
    if mp_det is not None:
        joint = sorted(set(results["ours"]) & set(results["mediapipe"]))
        say(f"\n  paired comparison on jointly matched faces (n={len(joint)}):")
        if joint:
            a = np.array([results["ours"][r][0] for r in joint])
            b = np.array([results["mediapipe"][r][0] for r in joint])
            say(f"    ours      mean {100 * a.mean():.3f}%  median {100 * np.median(a):.3f}%")
            say(f"    mediapipe mean {100 * b.mean():.3f}%  median {100 * np.median(b):.3f}%")
            say(f"    ours better on {100 * np.mean(a < b):.1f}% of faces")
            # per-point cross-detector offsets: mapping errors show up here
            # as a large systematic offset on one specific point
            offs = []
            for r in joint:
                gt24 = targets[r].landmarks98[SCHEMA.wflw_indices]
                iod = np.linalg.norm(gt24[SCHEMA.nme_right_index]
                                     - gt24[SCHEMA.nme_left_index])
                offs.append(np.linalg.norm(results["ours"][r][1]
                                           - results["mediapipe"][r][1], axis=1)
                            / max(iod, 1e-9))
            med = np.median(np.stack(offs), axis=0)
            say(f"    per-point ours-vs-mediapipe median offset (% of IOD):")
            for p in SCHEMA.points:
                say(f"      {p.index:>2} {p.name:<22} {100 * med[p.index]:6.1f}%")
            paired_stats = {"n": len(joint),
                            "ours_nme_pct_mean": float(100 * a.mean()),
                            "mediapipe_nme_pct_mean": float(100 * b.mean()),
                            "ours_win_rate_pct": float(100 * np.mean(a < b))}
            np.save(out_dir / "paired_ours_nme.npy", a)
            np.save(out_dir / "paired_mediapipe_nme.npy", b)

    # ---- 4: the calibration price (GT box vs Haar box, our model) ---------
    say("\n=== Our model: ground-truth boxes vs the Haar pipeline ===")
    expand = float(require(cfg, "preprocess.crop_expand"))
    gt_nmes = []
    for rel in results["ours"]:
        rec = targets[rel]
        frame = frames_cache.get(rel)
        if frame is None:
            continue
        gray = as_gray(frame)
        box = square_box_around(rec.landmarks98, expand)
        crop = extract_square(gray, box, ours.input_size)
        x = (torch.from_numpy(np.ascontiguousarray(crop)).float() / 255.0
             - ours.pixel_mean) / ours.pixel_std
        with torch.no_grad():
            pred01 = ours.model(x.unsqueeze(0).unsqueeze(0))[0].numpy().astype(np.float64)
        gt24 = rec.landmarks98[SCHEMA.wflw_indices].astype(np.float64)
        gt01 = to_crop_space(gt24, box)
        gt_nmes.append((float(metrics.nme_per_face(pred01[None], gt01[None], SCHEMA)[0]),
                        results["ours"][rel][0]))
    if gt_nmes:
        g = np.array([v for v, _ in gt_nmes])
        hb = np.array([v for _, v in gt_nmes])
        say(f"  same {len(gt_nmes)} faces (the preview-cached subset of matches):")
        say(f"    GT-box NME   mean {100 * g.mean():.3f}%  median {100 * np.median(g):.3f}%")
        say(f"    Haar-box NME mean {100 * hb.mean():.3f}%  median {100 * np.median(hb):.3f}%")
        say(f"    calibration price: {100 * (hb.mean() - g.mean()):+.3f} NME points "
            "(the deploy-resolution cost flagged in milestone 3)")

    # ---- 5: footprints -----------------------------------------------------
    say("\n=== Footprints ===")
    from dms_layer1.model.net import model_size_mb
    say(f"  ours: model {model_size_mb(ours.model):.2f} MB (fp32 params) "
        "+ cascade XML 0.93 MB")
    try:
        import mediapipe as mp_pkg
        mp_dir = Path(mp_pkg.__file__).parent / "modules"
        files = sorted(mp_dir.glob("face_*/*.tflite")) + sorted(
            mp_dir.glob("face_*/*.binarypb"))
        total = sum(f.stat().st_size for f in files) / 1e6
        say(f"  mediapipe face modules: {total:.2f} MB across {len(files)} "
            "bundled model files (approximate; the package carries more)")
    except Exception:
        say("  mediapipe footprint: package not inspectable here")

    yaml_out = {
        "n_images": n_targets,
        "detection": {n: {"matched": len(results[n]), "rate_pct":
                          float(100 * len(results[n]) / max(1, n_targets)),
                          "ms_median": float(np.median(times[n]))}
                      for n in detectors},
        "paired": paired_stats,
    }
    with open(out_dir / "m6_results.yaml", "w") as f:
        yaml.safe_dump(yaml_out, f, sort_keys=False)
    (out_dir / "report.txt").write_text("\n".join(_lines) + "\n")
    save_config_snapshot(cfg, out_dir)
    say(f"\nresults + report + config snapshot in {out_dir}")
    return 0


def _synthetic_pipeline(cfg: dict) -> dict:
    """Smoke path: schematic dataset, a briefly trained model, and the
    detector pointed at its checkpoint. Loudly not a real comparison."""
    import tempfile
    from dms_layer1.data.cache import build_cache
    from dms_layer1.data.synthetic import write_synthetic_dataset
    from dms_layer1.train.loop import Trainer

    print("=" * 70)
    print("SYNTHETIC MODE: schematic faces + a briefly trained model.")
    print("Code smoke test only; numbers are meaningless.")
    print("=" * 70)
    tmp = Path(tempfile.mkdtemp(prefix="m6_synth_"))
    root = write_synthetic_dataset(tmp / "ds", require(cfg, "dataset.attribute_names"),
                                   seed=require(cfg, "seed"))
    cfg["dataset"]["root"] = str(root)
    cfg["preprocess"]["out_dir"] = str(tmp / "cache")
    for split in ("train", "test"):
        build_cache(cfg, split)
    cfg["cache"] = {"dir": str(tmp / "cache")}
    cfg["model"]["width"] = 8
    cfg["train"].update({"epochs": 6, "batch_size": 8, "num_workers": 0,
                         "stop_after_epochs": None, "resume": False,
                         "checkpoint_dir": str(tmp / "ckpt"),
                         "metrics_csv": str(tmp / "metrics.csv"),
                         "curves_png": str(tmp / "curves.png")})
    Trainer(cfg).train()
    cfg["detector"]["weights"] = str(tmp / "ckpt" / "best.pth")
    return cfg


if __name__ == "__main__":
    sys.exit(main())

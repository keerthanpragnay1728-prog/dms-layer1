#!/usr/bin/env python3
"""Milestone 3 follow-up: can Haar detection on turned heads be lifted?

The full-pipeline run measured 17.79% detection on the WFLW pose subset -
a problem, not a curiosity, because inattentiveness (gaze off road) is
exactly the turned-head case. This sweep answers, with numbers:

  (a) does tuning scale_factor / min_neighbors / min_size_frac meaningfully
      lift pose detection, and what does each setting cost on frontal faces
      (spurious boxes, runtime)?
  (b) does the OpenCV profile-face cascade as a fallback (image + mirrored
      pass, only when the frontal cascade finds nothing) buy detection, and
      what does it cost in unmatched boxes and milliseconds?

Both are evaluated in one pass per parameter combo: the detector runs with
the profile fallback enabled, so 'frontal-only' numbers come from boxes with
source == 'frontal' and 'with-profile' numbers from the union. Populations:
every pose-flagged face, and a seeded sample of no-flag faces as the frontal
reference. Unmatched boxes are NOT a false-positive rate (WFLW images
contain unannotated faces) but their per-image growth is a valid relative
cost signal.

Usage (Kaggle):
    python scripts/sweep_haar_pose.py --config configs/layer1_base.yaml --split test
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

from dms_layer1.config import load_config, require, save_config_snapshot
from dms_layer1.data import wflw
from dms_layer1.detect.haar import (HaarFaceDetector, box_iou,
                                    calibrate_haar_to_crop, gt_crop_box)

# The sweep grid. The baseline (1.10, 5, 0.08) is the current config.
SCALE_FACTORS = [1.05, 1.10]
MIN_NEIGHBORS = [2, 3, 5]
MIN_SIZE_FRACS = [0.05, 0.08]

_lines: list[str] = []


def say(text: str = "") -> None:
    print(text)
    _lines.append(text)


def rect(b):
    if hasattr(b, "side"):
        return (b.x0, b.y0, b.x0 + b.side, b.y0 + b.side)
    return (b.x, b.y, b.x + b.w, b.y + b.h)


def evaluate_combo(cfg, grays, faces_by_image, expand, match_iou):
    """One sweep point. Returns per-population match flags (frontal-only and
    with the profile fallback), unmatched-box counts, timing, and the
    profile-matched pairs (for calibration of profile boxes)."""
    det = HaarFaceDetector(cfg)
    out = {"frontal": {}, "union": {}, "unmatched_frontal": 0, "unmatched_union": 0,
           "ms_no_profile": [], "ms_profile_triggered": [], "profile_pairs": []}
    for rel, gray in grays.items():
        t0 = time.perf_counter()
        boxes = det.detect(gray)
        ms = (time.perf_counter() - t0) * 1000
        profile_ran = any(b.source != "frontal" for b in boxes) or (
            not boxes and det.profile_fallback)
        (out["ms_profile_triggered"] if profile_ran else out["ms_no_profile"]).append(ms)

        frontal_boxes = [b for b in boxes if b.source == "frontal"]
        taken_f, taken_u = set(), set()
        for rec in faces_by_image[rel]:
            gt = rect(gt_crop_box(rec.landmarks98, expand))
            def best(cands, taken):
                bi, bio = -1, 0.0
                for i, hb in enumerate(cands):
                    if i in taken:
                        continue
                    iou = box_iou(rect(hb), gt)
                    if iou > bio:
                        bi, bio = i, iou
                return bi, bio
            fi, fio = best(frontal_boxes, taken_f)
            ok_f = fio >= match_iou
            if ok_f:
                taken_f.add(fi)
            ui, uio = best(boxes, taken_u)
            ok_u = uio >= match_iou
            if ok_u:
                taken_u.add(ui)
                if boxes[ui].source != "frontal":
                    out["profile_pairs"].append((rec, boxes[ui]))
            key = (rel, rec.line_number)
            out["frontal"][key] = ok_f
            out["union"][key] = ok_u
        out["unmatched_frontal"] += len(frontal_boxes) - len(taken_f)
        out["unmatched_union"] += len(boxes) - len(taken_u)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--frontal-sample", type=int, default=250,
                    help="no-flag images used as the frontal reference set")
    ap.add_argument("--out-dir", default="/kaggle/working/m3_pose_sweep")
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
    expand = float(require(cfg, "preprocess.reference_expand"))
    match_iou = float(require(cfg, "face_detector.match_iou"))
    rng = random.Random(require(cfg, "seed"))

    records, paths = wflw.load_split(cfg, args.split)
    pose_faces = [r for r in records if r.attributes["pose"] == 1]
    noflag_faces = [r for r in records if sum(r.attributes.values()) == 0]
    pose_images = sorted({r.image_rel_path for r in pose_faces})
    noflag_images = sorted({r.image_rel_path for r in noflag_faces})
    if args.frontal_sample and args.frontal_sample < len(noflag_images):
        rng.shuffle(noflag_images)
        noflag_images = sorted(noflag_images[:args.frontal_sample])
    say(f"pose faces: {len(pose_faces)} in {len(pose_images)} images | "
        f"no-flag reference: {len(noflag_images)} images")

    # Decode every needed image once, up front.
    grays: dict[str, np.ndarray] = {}
    faces_by_image: dict[str, list[wflw.FaceRecord]] = {}
    for rel in pose_images + [r for r in noflag_images if r not in set(pose_images)]:
        recs = [r for r in records if r.image_rel_path == rel]
        first = recs[0]
        gray = cv2.imread(str(wflw.image_path(paths, first)), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise wflw.WFLWError(f"cv2 could not decode {rel}")
        grays[rel] = gray
        # evaluate only the population faces in each image
        faces_by_image[rel] = [r for r in recs
                               if r.attributes["pose"] == 1
                               or sum(r.attributes.values()) == 0]
    pose_keys = {(r.image_rel_path, r.line_number) for r in pose_faces
                 if r.image_rel_path in grays}
    noflag_keys = {(r.image_rel_path, r.line_number) for r in noflag_faces
                   if r.image_rel_path in noflag_images}

    def rate(match_map, keys):
        vals = [match_map[k] for k in keys if k in match_map]
        return float(100 * np.mean(vals)) if vals else float("nan")

    combos = [(sf, mn, ms) for sf in SCALE_FACTORS for mn in MIN_NEIGHBORS
              for ms in MIN_SIZE_FRACS]
    say(f"\nsweeping {len(combos)} combos x {len(grays)} images "
        "(profile fallback evaluated in the same pass)\n")
    header = (f"{'scale':>6} {'nbrs':>5} {'minsz':>6} | "
              f"{'pose-F':>7} {'pose+P':>7} | {'flat-F':>7} {'flat+P':>7} | "
              f"{'unm/img F':>9} {'unm/img +P':>10} | {'ms':>5} {'ms+P':>6}")
    say(header)
    say("-" * len(header))

    all_results = []
    profile_pairs_baseline = None
    for sf, mn, ms in combos:
        c = copy.deepcopy(cfg)
        c["face_detector"].update({"scale_factor": sf, "min_neighbors": mn,
                                   "min_size_frac": ms, "profile_fallback": True})
        r = evaluate_combo(c, grays, faces_by_image, expand, match_iou)
        n_img = len(grays)
        row = {
            "scale_factor": sf, "min_neighbors": mn, "min_size_frac": ms,
            "pose_frontal": rate(r["frontal"], pose_keys),
            "pose_union": rate(r["union"], pose_keys),
            "noflag_frontal": rate(r["frontal"], noflag_keys),
            "noflag_union": rate(r["union"], noflag_keys),
            "unmatched_per_img_frontal": r["unmatched_frontal"] / n_img,
            "unmatched_per_img_union": r["unmatched_union"] / n_img,
            "ms_median_frontal_only": float(np.median(r["ms_no_profile"]))
                if r["ms_no_profile"] else float("nan"),
            "ms_median_profile_triggered": float(np.median(r["ms_profile_triggered"]))
                if r["ms_profile_triggered"] else float("nan"),
            "n_profile_matches": len(r["profile_pairs"]),
        }
        all_results.append(row)
        if (sf, mn, ms) == (1.10, 5, 0.08):
            profile_pairs_baseline = r["profile_pairs"]
        say(f"{sf:>6.2f} {mn:>5} {ms:>6.2f} | "
            f"{row['pose_frontal']:>6.1f}% {row['pose_union']:>6.1f}% | "
            f"{row['noflag_frontal']:>6.1f}% {row['noflag_union']:>6.1f}% | "
            f"{row['unmatched_per_img_frontal']:>9.2f} "
            f"{row['unmatched_per_img_union']:>10.2f} | "
            f"{row['ms_median_frontal_only']:>5.0f} "
            f"{row['ms_median_profile_triggered']:>6.0f}")

    say("\ncolumns: pose-F = pose detection, frontal cascade only; +P = with the")
    say("profile fallback; flat = the no-flag reference population; unm/img =")
    say("unmatched boxes per image (relative cost signal, not a false-positive")
    say("rate); ms = median per image without/with the profile passes triggering.")
    say("Baseline combo is (1.10, 5, 0.08) - the current config.")

    if profile_pairs_baseline and len(profile_pairs_baseline) >= 30:
        cal = calibrate_haar_to_crop(
            [gt_crop_box(rec.landmarks98, expand) for rec, _ in profile_pairs_baseline],
            [hb for _, hb in profile_pairs_baseline])
        say(f"\nprofile-box calibration (baseline combo, {cal['n_pairs']} matches) -")
        say("profile boxes frame faces differently from frontal ones:")
        for k in ("box_scale", "box_shift_x", "box_shift_y"):
            q = cal[k]
            say(f"  {k:<12} median {q['median']:+.3f}   IQR [{q['p25']:+.3f}, {q['p75']:+.3f}]")
    elif profile_pairs_baseline is not None:
        say(f"\nprofile-box calibration: only {len(profile_pairs_baseline)} matches "
            "at the baseline combo - too few to calibrate.")

    with open(out_dir / "sweep_results.yaml", "w") as f:
        yaml.safe_dump({"combos": all_results}, f, sort_keys=False)
    (out_dir / "report.txt").write_text("\n".join(_lines) + "\n")
    save_config_snapshot(cfg, out_dir)
    say(f"\nresults + report + config snapshot written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

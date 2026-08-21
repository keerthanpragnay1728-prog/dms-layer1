#!/usr/bin/env python3
"""Verify the assumed WFLW 98-point index layout against the actual
annotation file, BEFORE trusting the 98->24 mapping.

Runs geometric sanity checks that must hold if (and only if) the documented
index layout is correct — e.g. "point 96 lies inside the polygon of points
60-67" can only pass if 96 really is the pupil of that eye. Checks sensitive
to head pose are evaluated on the frontal subset (pose flag == 0); each check
reports the fraction of faces satisfying it.

Usage:
    python scripts/verify_layout.py --config configs/layer1_base.yaml --split test
    python scripts/verify_layout.py --config configs/layer1_base.yaml --synthetic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dms_layer1.config import load_config, require
from dms_layer1.data import frame, wflw
from dms_layer1.landmarks.schema import format_mapping_table, load_schema
from dms_layer1.config import resolve_path

PASS_AT = 0.99   # fraction of applicable faces a check must satisfy
WARN_AT = 0.95


def check(name, mask, results, note=""):
    frac = float(np.mean(mask)) if len(mask) else 0.0
    verdict = "PASS" if frac >= PASS_AT else ("WARN" if frac >= WARN_AT else "FAIL")
    results.append((name, frac, len(mask), verdict, note))


def run_checks(records: list[wflw.FaceRecord]) -> list[tuple]:
    """Vertical/horizontal relations are evaluated in a FACE-ALIGNED frame so
    in-plane roll cannot fail them: u = unit vector from the 60-67 eye
    centroid to the 68-75 eye centroid, v = u rotated 90 deg (down the face).
    That frame presumes 60-67/68-75 are the eyes — which the frame-free
    checks (pupil containment, image-left/right ordering) validate first."""
    lm = np.stack([r.landmarks98 for r in records])          # (N, 98, 2)
    frontal = np.array([r.attributes["pose"] == 0 for r in records])
    lf = lm[frontal]
    results: list[tuple] = []

    def eye_box(pts, sl):
        lo = pts[:, sl, :].min(axis=1)
        hi = pts[:, sl, :].max(axis=1)
        pad = (hi - lo).max(axis=1, keepdims=True) * 0.25 + 1.0
        return lo - pad, hi + pad

    # Frame-free: pupils belong to the eyes we think they belong to (all faces).
    for pupil, sl, label in ((96, slice(60, 68), "96 inside eye 60-67"),
                             (97, slice(68, 76), "97 inside eye 68-75")):
        lo, hi = eye_box(lm, sl)
        inside = np.all((lm[:, pupil] >= lo) & (lm[:, pupil] <= hi), axis=1)
        check(f"pupil {label}", inside, results)

    # Frame-free left/right identities in image space (frontal faces only;
    # extreme yaw can flip the x order, roll up to ~45 deg cannot).
    check("eye 60-67 is image-left of eye 68-75",
          lf[:, 60:68, 0].mean(axis=1) < lf[:, 68:76, 0].mean(axis=1), results, "frontal")
    check("pupil 96 image-left of pupil 97",
          lf[:, 96, 0] < lf[:, 97, 0], results, "frontal")
    check("contour 0-15 image-left of 17-32",
          lf[:, 0:16, 0].mean(axis=1) < lf[:, 17:33, 0].mean(axis=1), results, "frontal")

    # Face-aligned frame for the frontal subset (shared with the diagnostic
    # script so both agree exactly on what a failing face is).
    fr = frame.FaceFrame(lf)
    across = fr.across
    down = fr.down

    # Upper lids above lower lids. 1px tolerance for closed eyes where the
    # arcs nearly coincide.
    check("eye 61-63 above 65-67 (upper/lower lid)",
          down(slice(61, 64)).mean(axis=1) < down(slice(65, 68)).mean(axis=1) + 1.0,
          results, "frontal")
    check("eye 69-71 above 73-75 (upper/lower lid)",
          down(slice(69, 72)).mean(axis=1) < down(slice(73, 76)).mean(axis=1) + 1.0,
          results, "frontal")

    # Chin: the contour point farthest below the eye line (tolerance is
    # IOD-relative so face scale cannot bias it; see frame.CHIN_TOL_IOD).
    check("16 is the lowest contour point (chin, tol 3% IOD)",
          frame.check_chin(fr), results, "frontal")

    # Chosen yaw pairs sit on opposite sides at similar face-height.
    check("contour pair 4/28 on opposite sides",
          (across(4) < 0) & (across(28) > 0), results, "frontal")
    check("contour pair 8/24 on opposite sides",
          (across(8) < 0) & (across(24) > 0), results, "frontal")
    check("pair (4,28) height match < 20% eye-chin dist",
          frame.check_pair(fr, 4, 28), results, "frontal")
    check("pair (8,24) height match < 20% eye-chin dist",
          frame.check_pair(fr, 8, 24), results, "frontal")

    # Nose tip on the facial axis, between the eyes and the mouth.
    check("nose tip 54 below eyes, above upper lip 79",
          (down(54) > 0) & (down(54) < down(79)), results, "frontal")
    check("nose tip 54 near the facial midline",
          frame.check_nose_midline(fr), results, "frontal")

    # Mouth: corner order and mid ordering on the outer lip.
    check("mouth corner 76 left of corner 82",
          across(76) < across(82), results, "frontal")
    check("upper-lip mid 79 above lower-lip mid 85",
          down(79) < down(85), results, "frontal")
    check("79/85 horizontally between the corners",
          (across(76) < across(79)) & (across(79) < across(82))
          & (across(76) < across(85)) & (across(85) < across(82)),
          results, "frontal")

    # Eyebrows (33-50) sit above the eyes — confirms 33-50 aren't lips etc.
    check("33-50 above eye region (eyebrows)",
          down(slice(33, 51)).mean(axis=1) < 0, results, "frontal")

    # Face rect roughly contains the landmarks (sanity of the rect columns).
    bbox = np.array([r.bbox for r in records])
    pad = 0.30 * np.maximum(bbox[:, 2] - bbox[:, 0], bbox[:, 3] - bbox[:, 1])[:, None, None]
    lo = bbox[:, None, (0, 1)] - pad
    hi = bbox[:, None, (2, 3)] + pad
    inside_frac = np.mean(np.all((lm >= lo) & (lm <= hi), axis=2), axis=1)
    check("landmarks inside face rect (+30% pad)", inside_frac >= 0.90, results)
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default=None, help="train or test (default: config visualize.split)")
    ap.add_argument("--synthetic", action="store_true",
                    help="run against a generated schematic dataset (code smoke test only)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    schema = load_schema(resolve_path(cfg, require(cfg, "landmark_schema")))
    split = args.split or require(cfg, "visualize.split")

    if args.synthetic:
        import tempfile
        from dms_layer1.data.synthetic import write_synthetic_dataset
        print("=" * 70)
        print("SYNTHETIC MODE: schematic faces, NOT the real WFLW dataset.")
        print("This validates the code path only. Layout verification is only")
        print("meaningful against the real annotation file (run on Kaggle).")
        print("=" * 70)
        root = write_synthetic_dataset(tempfile.mkdtemp(prefix="wflw_synth_"),
                                       require(cfg, "dataset.attribute_names"),
                                       seed=require(cfg, "seed"))
        cfg["dataset"]["root"] = str(root)

    records, paths = wflw.load_split(cfg, split)
    n_images = len({r.image_rel_path for r in records})
    print(f"\nParsed {len(records)} faces / {n_images} unique images "
          f"from {paths.root} [{split} split]")
    counts = {k: sum(r.attributes[k] for r in records) for k in records[0].attributes}
    print("attribute counts:", counts)

    print("\n" + format_mapping_table(schema))

    results = run_checks(records)
    print("\nLayout checks (PASS >= {:.0%} of applicable faces, WARN >= {:.0%}):"
          .format(PASS_AT, WARN_AT))
    print(f"{'check':<52} {'ok':>8} {'n':>6}  verdict")
    print("-" * 78)
    worst = "PASS"
    for name, frac, n, verdict, note in results:
        suffix = f"  [{note}]" if note else ""
        print(f"{name:<52} {frac:>7.2%} {n:>6}  {verdict}{suffix}")
        if verdict == "FAIL" or (verdict == "WARN" and worst != "FAIL"):
            worst = verdict
    print("-" * 78)
    if worst == "PASS":
        print("All layout assumptions hold. The 98->24 mapping indices are safe "
              "to verify visually next (scripts/visualize_mapping.py).")
        return 0
    print(f"Result: {worst} — inspect the failing checks before trusting the mapping.")
    return 1 if worst == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())

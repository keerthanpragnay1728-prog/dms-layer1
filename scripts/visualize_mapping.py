#!/usr/bin/env python3
"""Milestone 1 verification overlays: render the 98->24 mapping on sample
faces so the point subset can be confirmed by eye before anything trains.

Samples frontal, large-pose and occluded faces (counts from the config),
writes one labelled overlay PNG per face plus a contact sheet, prints the
mapping table, and snapshots the config + table into the output directory.

What to confirm on the overlays (from the milestone definition):
  * pupil points (12, 13) dead centre in each eye - crosshair in the insets
  * eyelid points 0-5 / 6-11 trace the lids, consistent order
  * mouth points 14-17 on corners and outer-lip midpoints
  * nose tip 18 and chin 19 on the vertical facial axis
  * contour points 20-23 symmetric left/right at similar heights

Usage:
    python scripts/visualize_mapping.py --config configs/layer1_base.yaml
    python scripts/visualize_mapping.py --config configs/layer1_base.yaml --synthetic
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dms_layer1.config import (load_config, require, resolve_path,
                               save_config_snapshot)
from dms_layer1.data import wflw
from dms_layer1.landmarks.schema import format_mapping_table, load_schema
from dms_layer1.viz.overlay import make_contact_sheet, render_face_overlay


def select_samples(records: list[wflw.FaceRecord], cfg: dict,
                   rng: random.Random) -> list[tuple[str, wflw.FaceRecord]]:
    """Seeded sample of frontal / large-pose / occluded faces."""
    frontal = [r for r in records
               if r.attributes["pose"] == 0 and r.attributes["occlusion"] == 0]
    largepose = [r for r in records if r.attributes["pose"] == 1]
    occluded = [r for r in records if r.attributes["occlusion"] == 1]

    picks: list[tuple[str, wflw.FaceRecord]] = []
    for label, pool, want_key in (("frontal", frontal, "num_frontal"),
                                  ("largepose", largepose, "num_largepose"),
                                  ("occluded", occluded, "num_occluded")):
        want = require(cfg, f"visualize.{want_key}")
        if len(pool) < want:
            print(f"note: only {len(pool)} '{label}' faces available, wanted {want}")
        chosen = rng.sample(pool, min(want, len(pool)))
        picks.extend((label, r) for r in chosen)
    return picks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default=None, help="train or test (default: config visualize.split)")
    ap.add_argument("--out-dir", default=None, help="default: config visualize.out_dir")
    ap.add_argument("--synthetic", action="store_true",
                    help="render on generated schematic faces (code smoke test only)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    schema = load_schema(resolve_path(cfg, require(cfg, "landmark_schema")))
    split = args.split or require(cfg, "visualize.split")
    out_dir = Path(args.out_dir or require(cfg, "visualize.out_dir"))
    seed = require(cfg, "seed")
    rng = random.Random(seed)

    if args.synthetic:
        import tempfile
        from dms_layer1.data.synthetic import write_synthetic_dataset
        print("=" * 70)
        print("SYNTHETIC MODE: schematic faces, NOT the real WFLW dataset.")
        print("Use this only to check the tooling runs; the milestone sign-off")
        print("needs overlays on real WFLW images (run on Kaggle).")
        print("=" * 70)
        root = write_synthetic_dataset(tempfile.mkdtemp(prefix="wflw_synth_"),
                                       require(cfg, "dataset.attribute_names"),
                                       seed=seed)
        cfg["dataset"]["root"] = str(root)

    table = format_mapping_table(schema)
    print("\n" + table + "\n")

    records, paths = wflw.load_split(cfg, split)
    print(f"Parsed {len(records)} faces from the {split} split (seed {seed}).")
    picks = select_samples(records, cfg, rng)

    out_dir.mkdir(parents=True, exist_ok=True)
    canvases = []
    for n, (label, rec) in enumerate(picks):
        img = cv2.imread(str(wflw.image_path(paths, rec)), cv2.IMREAD_COLOR)
        if img is None:
            raise wflw.WFLWError(f"cv2 could not decode {rec.image_rel_path}")
        flags = [k for k, v in rec.attributes.items() if v] or ["frontal"]
        title = (f"[{label}] {rec.image_rel_path}  (line {rec.line_number}, "
                 f"flags: {','.join(flags)})")
        canvas = render_face_overlay(img, rec.landmarks98, schema, title,
                                     inset_size=require(cfg, "visualize.inset_size"))
        name = f"{n:02d}_{label}_line{rec.line_number}.png"
        cv2.imwrite(str(out_dir / name), canvas)
        print(f"  wrote {out_dir / name}")
        canvases.append(canvas)

    sheet = make_contact_sheet(canvases)
    cv2.imwrite(str(out_dir / "contact_sheet.png"), sheet)
    print(f"  wrote {out_dir / 'contact_sheet.png'}")

    # Reproducibility: the exact config and mapping that produced these images.
    save_config_snapshot(cfg, out_dir)
    (out_dir / "mapping_table.txt").write_text(table + "\n")
    print(f"\nDone: {len(canvases)} overlays in {out_dir}")
    print("Check each face against the milestone checklist in this script's "
          "docstring before approving the mapping.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Milestone 2: build the WFLW crop cache, verify it, and render previews.

For each requested split this script:
  1. builds the cache (decode every image once, square-crop each face around
     its 98-point extent, resize to preprocess.cache_size grayscale, store
     crop-space [0,1] labels for our 24 points),
  2. reloads the written files (read-back check),
  3. reconstructs every face's landmarks from the cached [0,1] labels + crop
     box and compares them against a fresh parse of the annotation file -
     proving the cache encodes the labels faithfully (max error is float32
     rounding, well under 0.01 px),
  4. renders a preview grid of decoded crops with the cached points drawn on
     them, for the eyeball check that crops and labels line up.

Usage (Kaggle):
    python scripts/build_crop_cache.py --config configs/layer1_base.yaml --split both

Then publish the output directory (/kaggle/working/cache) as a Kaggle
dataset so training runs attach the cache instead of raw WFLW.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dms_layer1.config import load_config, require, resolve_path
from dms_layer1.data import cache as cachemod
from dms_layer1.data import wflw
from dms_layer1.landmarks.schema import load_schema
from dms_layer1.viz.overlay import GROUP_COLORS

PREVIEW_SCALE = 3          # cached crops are upscaled this much for previews
FONT = cv2.FONT_HERSHEY_SIMPLEX


def reconstruct_error_px(cfg: dict, data: cachemod.CacheData, split: str) -> float:
    """Max |cached labels mapped back to frame - annotation| in pixels."""
    schema = load_schema(resolve_path(cfg, require(cfg, "landmark_schema")))
    records, _ = wflw.load_split(cfg, split)
    truth = np.stack([r.landmarks98[schema.wflw_indices] for r in records])
    boxes = data.boxes.astype(np.float64)
    rebuilt = (data.landmarks.astype(np.float64) * boxes[:, None, 2:3]
               + boxes[:, None, 0:2])
    return float(np.abs(rebuilt - truth).max())


def render_preview(data: cachemod.CacheData, schema, indices: list[int],
                   out_path: Path) -> None:
    tiles = []
    size = data.crops.shape[1]
    big = size * PREVIEW_SCALE
    for i in indices:
        tile = cv2.cvtColor(cv2.resize(data.crops[i], (big, big),
                                       interpolation=cv2.INTER_NEAREST),
                            cv2.COLOR_GRAY2BGR)
        for p in schema.points:
            x, y = data.landmarks[i, p.index] * big
            cv2.circle(tile, (int(round(x)), int(round(y))), 4, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(tile, (int(round(x)), int(round(y))), 3,
                       GROUP_COLORS[p.group], -1, cv2.LINE_AA)
        strip = np.full((26, big, 3), (25, 25, 25), dtype=np.uint8)
        flags = [a for k, a in enumerate(data.manifest["attribute_names"])
                 if data.attrs[i, k]]
        caption = f"#{i}  {'+'.join(flags) if flags else 'frontal'}  side={data.boxes[i, 2]}px"
        cv2.putText(strip, caption, (6, 18), FONT, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        tiles.append(np.concatenate([tile, strip], axis=0))

    cols = 4
    rows = []
    blank = np.zeros_like(tiles[0])
    for r in range(0, len(tiles), cols):
        row = tiles[r:r + cols] + [blank] * (cols - len(tiles[r:r + cols]))
        rows.append(np.concatenate(row, axis=1))
    cv2.imwrite(str(out_path), np.concatenate(rows, axis=0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="both", choices=["train", "test", "both"])
    ap.add_argument("--out-dir", default=None, help="default: config preprocess.out_dir")
    ap.add_argument("--synthetic", action="store_true",
                    help="build from generated schematic faces (code smoke test only)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.synthetic:
        import tempfile
        from dms_layer1.data.synthetic import write_synthetic_dataset
        print("SYNTHETIC MODE: schematic faces - smoke test of the code path only.")
        root = write_synthetic_dataset(tempfile.mkdtemp(prefix="wflw_synth_"),
                                       require(cfg, "dataset.attribute_names"),
                                       seed=require(cfg, "seed"))
        cfg["dataset"]["root"] = str(root)

    out_dir = Path(args.out_dir or require(cfg, "preprocess.out_dir"))
    schema = load_schema(resolve_path(cfg, require(cfg, "landmark_schema")))
    rng = random.Random(require(cfg, "seed"))
    splits = ["train", "test"] if args.split == "both" else [args.split]

    for split in splits:
        print(f"\n=== building '{split}' cache -> {out_dir} ===")
        manifest = cachemod.build_cache(cfg, split, out_dir)
        for key in ("n_faces", "n_images", "cache_size", "crop_expand",
                    "pixel_mean", "pixel_std", "landmark_min", "landmark_max",
                    "crop_side_px", "crops_padded_at_image_border",
                    "build_seconds", "faces_per_second"):
            print(f"  {key:<30} {manifest[key]}")
        total_mb = sum(manifest["bytes"].values()) / 1e6
        for name, b in manifest["bytes"].items():
            print(f"  {name + ' bytes':<30} {b / 1e6:.1f} MB")
        print(f"  {'total':<30} {total_mb:.1f} MB")

        data = cachemod.load_cache(out_dir, split)          # read-back check
        err = reconstruct_error_px(cfg, data, split)
        print(f"  read-back OK; label round-trip max error {err:.5f} px "
              f"(float32 rounding only)")
        if err > 0.01:
            print("  ERROR: round-trip error exceeds 0.01 px - cache labels "
                  "do not faithfully encode the annotations. Do not train on this.")
            return 1

        n_prev = min(int(require(cfg, "preprocess.num_preview")), data.crops.shape[0])
        indices = sorted(rng.sample(range(data.crops.shape[0]), n_prev))
        preview = out_dir / f"{split}_preview.png"
        render_preview(data, schema, indices, preview)
        print(f"  preview ({n_prev} crops): {preview}")

    print("\nDone. Check the previews: crops must look like centred faces and "
          "every point must sit on its feature at crop resolution.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

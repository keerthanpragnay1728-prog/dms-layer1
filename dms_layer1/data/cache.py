"""Milestone 2: one-time preprocessing of WFLW into an in-RAM training cache.

Each split becomes a set of parallel arrays, index-aligned per face:

    {split}_crops.npy      uint8   (N, S, S)   grayscale face crops
    {split}_landmarks.npy  float32 (N, 24, 2)  our 24 points, [0,1] crop space
    {split}_boxes.npy      int32   (N, 3)      crop box (x0, y0, side) in the
                                               original frame, for mapping back
    {split}_attrs.npy      uint8   (N, 6)      WFLW attribute flags (subset eval)
    {split}_paths.txt      text    image rel-path + annotation line per face
    {split}_manifest.yaml  build stats + settings (plus config_used.yaml)

The crop box is a square around the extent of ALL 98 annotated points (not
just our 24 - that keeps forehead/brow context, closer to what a face
detector box contains), expanded by preprocess.crop_expand. Landmarks are
stored resolution-independent in [0,1] over the box, so the same cache
serves any model input size. The training loop (milestone 4) loads these
arrays fully into RAM and augments on the fly; images are decoded exactly
once, here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from dms_layer1.config import require, resolve_path, save_config_snapshot
from dms_layer1.data import wflw
from dms_layer1.data.crops import CropBox, extract_square, square_box_around, to_crop_space
from dms_layer1.landmarks.schema import load_schema


class CacheError(Exception):
    """Raised when building or loading a cache fails validation."""


def _file_names(split: str) -> dict[str, str]:
    return {
        "crops": f"{split}_crops.npy",
        "landmarks": f"{split}_landmarks.npy",
        "boxes": f"{split}_boxes.npy",
        "attrs": f"{split}_attrs.npy",
        "paths": f"{split}_paths.txt",
        "manifest": f"{split}_manifest.yaml",
    }


@dataclass
class CacheData:
    crops: np.ndarray
    landmarks: np.ndarray
    boxes: np.ndarray
    attrs: np.ndarray
    rel_paths: list[str]       # "image_rel_path\tannotation_line" per face
    manifest: dict


def build_cache(cfg: dict, split: str, out_dir: str | Path | None = None) -> dict:
    """Build the cache for one split. Returns the manifest dict."""
    out_dir = Path(out_dir or require(cfg, "preprocess.out_dir"))
    size = int(require(cfg, "preprocess.cache_size"))
    expand = float(require(cfg, "preprocess.crop_expand"))
    if expand < 1.0:
        raise CacheError(f"preprocess.crop_expand must be >= 1 (got {expand}); "
                         "smaller values would push landmarks outside the crop")
    schema = load_schema(resolve_path(cfg, require(cfg, "landmark_schema")))
    attr_names = require(cfg, "dataset.attribute_names")

    records, paths = wflw.load_split(cfg, split)
    n = len(records)
    crops = np.empty((n, size, size), dtype=np.uint8)
    landmarks = np.empty((n, 24, 2), dtype=np.float32)
    boxes = np.empty((n, 3), dtype=np.int32)
    attrs = np.empty((n, len(attr_names)), dtype=np.uint8)

    # Decode each image once even when it carries several annotated faces.
    by_image: dict[str, list[int]] = {}
    for i, rec in enumerate(records):
        by_image.setdefault(rec.image_rel_path, []).append(i)

    t0 = time.time()
    n_padded = 0
    done = 0
    for rel, indices in by_image.items():
        img_path = wflw.image_path(paths, records[indices[0]])
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise CacheError(f"cv2 could not decode {img_path}")
        h, w = gray.shape
        for i in indices:
            rec = records[i]
            box = square_box_around(rec.landmarks98, expand)
            crops[i] = extract_square(gray, box, size)
            lm01 = to_crop_space(rec.landmarks98[schema.wflw_indices], box)
            if lm01.min() < 0.0 or lm01.max() > 1.0:
                raise CacheError(
                    f"landmarks outside the crop for {rec.image_rel_path} "
                    f"(line {rec.line_number}): range [{lm01.min():.3f}, "
                    f"{lm01.max():.3f}] - this cannot happen with a box built "
                    "from all 98 points; the annotation or box code is broken"
                )
            landmarks[i] = lm01.astype(np.float32)
            boxes[i] = box.as_array()
            attrs[i] = [rec.attributes[a] for a in attr_names]
            if (box.x0 < 0 or box.y0 < 0
                    or box.x0 + box.side > w or box.y0 + box.side > h):
                n_padded += 1
            done += 1
            if done % 1000 == 0:
                rate = done / (time.time() - t0)
                print(f"  {done}/{n} faces  ({rate:.0f} faces/s)")

    build_s = time.time() - t0
    out_dir.mkdir(parents=True, exist_ok=True)
    names = _file_names(split)
    np.save(out_dir / names["crops"], crops)
    np.save(out_dir / names["landmarks"], landmarks)
    np.save(out_dir / names["boxes"], boxes)
    np.save(out_dir / names["attrs"], attrs)
    with open(out_dir / names["paths"], "w") as f:
        for rec in records:
            f.write(f"{rec.image_rel_path}\t{rec.line_number}\n")

    manifest = {
        "split": split,
        "n_faces": n,
        "n_images": len(by_image),
        "cache_size": size,
        "crop_expand": expand,
        "attribute_names": list(attr_names),
        "attribute_counts": {a: int(attrs[:, k].sum()) for k, a in enumerate(attr_names)},
        "pixel_mean": round(float(crops.mean()), 3),
        "pixel_std": round(float(crops.std()), 3),
        "landmark_min": round(float(landmarks.min()), 5),
        "landmark_max": round(float(landmarks.max()), 5),
        "crop_side_px": {"min": int(boxes[:, 2].min()),
                         "median": int(np.median(boxes[:, 2])),
                         "max": int(boxes[:, 2].max())},
        "crops_padded_at_image_border": n_padded,
        "bytes": {k: (out_dir / v).stat().st_size for k, v in names.items()
                  if k != "manifest"},
        "build_seconds": round(build_s, 1),
        "faces_per_second": round(n / build_s, 1),
        "source_annotations": str(paths.train_list if split == "train" else paths.test_list),
    }
    with open(out_dir / names["manifest"], "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    save_config_snapshot(cfg, out_dir)
    return manifest


# Where Kaggle mounts attached inputs. Mount layouts vary by input kind
# (dataset: /kaggle/input/<slug>/..., notebook output:
# /kaggle/input/notebooks/<user>/<slug>/...), which is why discovery exists.
KAGGLE_INPUT = Path("/kaggle/input")


def _has_split_files(cache_dir: Path, split: str) -> bool:
    return cache_dir.is_dir() and all(
        (cache_dir / v).is_file() for v in _file_names(split).values())


def discover_cache_dirs(split: str, search_root: Path | None = None) -> list[Path]:
    """Directories under the Kaggle input mount that contain the split's
    cache files. Depth-limited glob (mounts are shallow; covers both the
    dataset layout and notebooks/<user>/<slug>/cache)."""
    search_root = KAGGLE_INPUT if search_root is None else search_root
    if not search_root.is_dir():
        return []
    marker = _file_names(split)["crops"]
    hits: set[Path] = set()
    for depth in range(1, 5):
        pattern = "/".join(["*"] * depth) + f"/{marker}"
        hits.update(p.parent for p in search_root.glob(pattern))
    return sorted(h for h in hits if _has_split_files(h, split))


def load_cache_from_cfg(cfg: dict, split: str) -> CacheData:
    """Load the cache from cache.dir. Because Kaggle mount paths vary by how
    the cache is attached, a configured path that does not hold the cache
    files (or the value 'auto') falls back to discovery under /kaggle/input:
    exactly one hit is used WITH A LOUD NOTE; zero or several hits raise a
    clear error listing what was found. Build-time code uses
    preprocess.out_dir instead."""
    configured = str(require(cfg, "cache.dir"))
    if configured != "auto" and _has_split_files(Path(configured), split):
        return load_cache(configured, split)

    hits = discover_cache_dirs(split)
    if len(hits) == 1:
        print(f"NOTE: cache.dir='{configured}' does not hold the {split} cache; "
              f"discovered it at {hits[0]} and using that. Set cache.dir to "
              "this path (or leave 'auto') to silence this note.")
        return load_cache(hits[0], split)

    detail = (f"none under {KAGGLE_INPUT}" if not hits
              else "several candidates:\n" + "\n".join(f"    {h}" for h in hits))
    raise CacheError(
        f"Cannot locate the {split} crop cache.\n"
        f"  cache.dir: {configured} (missing or lacks the cache files)\n"
        f"  discovery: {detail}\n"
        "Attach the cache (dataset or the milestone-2 notebook's output) and "
        "set cache.dir to its mount path, or leave 'auto' when exactly one "
        "cache is attached."
    )


def load_cache(cache_dir: str | Path, split: str) -> CacheData:
    """Load one split's cache into RAM, validating file consistency."""
    cache_dir = Path(cache_dir)
    names = _file_names(split)
    missing = [v for v in names.values() if not (cache_dir / v).is_file()]
    if missing:
        listing = sorted(p.name for p in cache_dir.iterdir()) if cache_dir.is_dir() else []
        raise CacheError(
            f"Cache files missing in {cache_dir}: {missing}\n"
            f"Found instead: {listing}\n"
            "Build with scripts/build_crop_cache.py, or point at the Kaggle "
            "dataset that holds the cache."
        )
    crops = np.load(cache_dir / names["crops"])
    landmarks = np.load(cache_dir / names["landmarks"])
    boxes = np.load(cache_dir / names["boxes"])
    attrs = np.load(cache_dir / names["attrs"])
    rel_paths = (cache_dir / names["paths"]).read_text().splitlines()
    with open(cache_dir / names["manifest"]) as f:
        manifest = yaml.safe_load(f)

    n = crops.shape[0]
    if not (landmarks.shape == (n, 24, 2) and boxes.shape == (n, 3)
            and attrs.shape[0] == n and len(rel_paths) == n):
        raise CacheError(
            f"Cache arrays disagree on face count: crops {crops.shape}, "
            f"landmarks {landmarks.shape}, boxes {boxes.shape}, "
            f"attrs {attrs.shape}, paths {len(rel_paths)}"
        )
    if crops.dtype != np.uint8 or landmarks.dtype != np.float32:
        raise CacheError(f"Unexpected dtypes: crops {crops.dtype}, landmarks {landmarks.dtype}")
    if landmarks.min() < 0.0 or landmarks.max() > 1.0:
        raise CacheError("Cached landmarks outside [0, 1] - cache is corrupt")
    return CacheData(crops, landmarks, boxes, attrs, rel_paths, manifest)

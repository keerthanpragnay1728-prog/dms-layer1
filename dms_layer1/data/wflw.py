"""WFLW annotation parsing and dataset path resolution.

Annotation format (one face per line; an image file can appear on several
lines, one per face):

    x0 y0 x1 y1 ... x97 y97  xmin ymin xmax ymax  pose expr illum makeup occl blur  path/to/image.jpg

= 196 coordinate floats + 4 face-rect values + 6 binary attribute flags
+ 1 relative image path = 207 whitespace-separated tokens.

This module deliberately has no torch dependency: milestone 1 (mapping
verification) must run anywhere. The torch Dataset for training arrives with
milestone 4 and will consume these same records.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dms_layer1.config import require

NUM_COORD_TOKENS = 196  # 98 landmarks * (x, y)
NUM_RECT_TOKENS = 4
NUM_ATTR_TOKENS = 6
TOKENS_PER_LINE = NUM_COORD_TOKENS + NUM_RECT_TOKENS + NUM_ATTR_TOKENS + 1


class WFLWError(Exception):
    """Raised for missing dataset paths or malformed annotation lines."""


@dataclass
class FaceRecord:
    """One annotated face."""
    landmarks98: np.ndarray      # (98, 2) float32, original image coordinates
    bbox: tuple[float, float, float, float]  # xmin, ymin, xmax, ymax
    attributes: dict[str, int]   # e.g. {"pose": 0, ..., "blur": 1}
    image_rel_path: str          # relative to the WFLW_images directory
    line_number: int             # 1-based line in the annotation file


def _listing(path: Path, limit: int = 20) -> str:
    """A short directory listing for error messages, so a wrong path tells
    you what IS there instead of just failing."""
    if not path.exists():
        return f"  (directory does not exist: {path})"
    entries = sorted(p.name + ("/" if p.is_dir() else "") for p in path.iterdir())
    shown = "\n".join(f"  - {e}" for e in entries[:limit])
    if len(entries) > limit:
        shown += f"\n  ... and {len(entries) - limit} more"
    return shown or "  (empty directory)"


def _find_candidates(root: Path, target_name: str) -> list[Path]:
    """Look up to two levels below root for a directory with the expected
    name, purely to make the error message actionable. We never silently use
    a candidate - the config must be fixed explicitly."""
    hits = []
    if root.is_dir():
        for level1 in root.iterdir():
            if level1.name == target_name and level1.is_dir():
                hits.append(level1)
            elif level1.is_dir():
                sub = level1 / target_name
                if sub.is_dir():
                    hits.append(sub)
    return hits


@dataclass
class WFLWPaths:
    root: Path
    images_dir: Path
    annotations_dir: Path
    train_list: Path
    test_list: Path


def resolve_paths(cfg: dict) -> WFLWPaths:
    """Resolve and validate every dataset path from the config, failing with
    a message that says exactly what was expected, what exists instead, and
    where a likely candidate lives."""
    root = Path(require(cfg, "dataset.root"))
    if not root.is_dir():
        parent = root.parent
        raise WFLWError(
            f"WFLW dataset root not found:\n  {root}\n"
            f"Contents of {parent}:\n{_listing(parent)}\n"
            "Fix dataset.root in the config (configs/layer1_base.yaml). On Kaggle, "
            "check the dataset is attached and the mount path under /kaggle/input matches."
        )

    paths = WFLWPaths(
        root=root,
        images_dir=root / require(cfg, "dataset.images_dir"),
        annotations_dir=root / require(cfg, "dataset.annotations_dir"),
        train_list=root / require(cfg, "dataset.annotations_dir") / require(cfg, "dataset.train_list"),
        test_list=root / require(cfg, "dataset.annotations_dir") / require(cfg, "dataset.test_list"),
    )

    for label, p, kind in [
        ("dataset.images_dir", paths.images_dir, "dir"),
        ("dataset.annotations_dir", paths.annotations_dir, "dir"),
        ("dataset.train_list", paths.train_list, "file"),
        ("dataset.test_list", paths.test_list, "file"),
    ]:
        ok = p.is_dir() if kind == "dir" else p.is_file()
        if not ok:
            msg = (
                f"WFLW path for '{label}' not found:\n  {p}\n"
                f"Contents of {p.parent}:\n{_listing(p.parent)}"
            )
            candidates = _find_candidates(root, p.name)
            if candidates:
                found = "\n".join(f"  {c}" for c in candidates)
                msg += (
                    f"\nA directory/file with that name exists at:\n{found}\n"
                    "If that is the right one, update the config to point at it "
                    "(this loader never guesses silently)."
                )
            raise WFLWError(msg)
    return paths


def parse_annotation_file(path: str | Path, attribute_names: list[str]) -> list[FaceRecord]:
    """Parse a WFLW rect_attr annotation file into FaceRecords, validating
    every line's token count and value ranges."""
    path = Path(path)
    if not path.is_file():
        raise WFLWError(f"Annotation file not found: {path}")
    if len(attribute_names) != NUM_ATTR_TOKENS:
        raise WFLWError(f"Expected {NUM_ATTR_TOKENS} attribute names, got {attribute_names}")

    records: list[FaceRecord] = []
    with open(path, "r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            if len(tokens) != TOKENS_PER_LINE:
                raise WFLWError(
                    f"{path.name} line {line_no}: expected {TOKENS_PER_LINE} tokens "
                    f"(196 coords + 4 rect + 6 attrs + image path), got {len(tokens)}. "
                    "This does not look like a list_98pt_rect_attr_* file."
                )
            coords = np.array([float(t) for t in tokens[:NUM_COORD_TOKENS]],
                              dtype=np.float32).reshape(98, 2)
            if not np.isfinite(coords).all():
                raise WFLWError(f"{path.name} line {line_no}: non-finite landmark coordinate")
            rect = tuple(float(t) for t in tokens[NUM_COORD_TOKENS:NUM_COORD_TOKENS + 4])
            if not (rect[0] < rect[2] and rect[1] < rect[3]):
                raise WFLWError(f"{path.name} line {line_no}: degenerate face rect {rect}")
            attr_tokens = tokens[NUM_COORD_TOKENS + 4:NUM_COORD_TOKENS + 10]
            attrs = {}
            for name, t in zip(attribute_names, attr_tokens):
                v = int(t)
                if v not in (0, 1):
                    raise WFLWError(
                        f"{path.name} line {line_no}: attribute '{name}'={t}, expected 0/1"
                    )
                attrs[name] = v
            records.append(FaceRecord(
                landmarks98=coords,
                bbox=rect,
                attributes=attrs,
                image_rel_path=tokens[-1],
                line_number=line_no,
            ))
    if not records:
        raise WFLWError(f"Annotation file is empty: {path}")
    return records


def load_split(cfg: dict, split: str) -> tuple[list[FaceRecord], WFLWPaths]:
    """Load 'train' or 'test' records plus resolved paths."""
    paths = resolve_paths(cfg)
    if split == "train":
        list_path = paths.train_list
    elif split == "test":
        list_path = paths.test_list
    else:
        raise WFLWError(f"Unknown split '{split}', expected 'train' or 'test'")
    records = parse_annotation_file(list_path, require(cfg, "dataset.attribute_names"))
    return records, paths


def image_path(paths: WFLWPaths, record: FaceRecord) -> Path:
    p = paths.images_dir / record.image_rel_path
    if not p.is_file():
        raise WFLWError(
            f"Image referenced by annotations not found:\n  {p}\n"
            f"Contents of {p.parent.parent if not p.parent.exists() else p.parent}:\n"
            f"{_listing(p.parent.parent if not p.parent.exists() else p.parent)}"
        )
    return p


def subset_records(records: list[FaceRecord], subset: str) -> list[FaceRecord]:
    """The official WFLW test subsets, derived from the attribute flags
    (pose flag -> 'largepose' to match the official file naming)."""
    attr = {"largepose": "pose"}.get(subset, subset)
    hits = [r for r in records if r.attributes.get(attr, 0) == 1]
    if not hits:
        known = sorted(records[0].attributes.keys()) if records else []
        raise WFLWError(f"No faces with attribute '{attr}'. Known attributes: {known}")
    return hits

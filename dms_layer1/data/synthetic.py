"""Synthetic schematic face with WFLW-layout 98-point annotations.

Purpose: local smoke-testing of the parsing -> mapping -> overlay pipeline on
a machine that does not have the real WFLW dataset. The 98 points are placed
according to the documented WFLW index layout and the face drawing is derived
FROM those points (eyelid polygons, lip polygons, contour polyline), so an
overlay on a synthetic face makes mapping mistakes visible the same way a
real face would.

THIS IS NOT DATASET VERIFICATION. The real annotation file must still be
checked with scripts/verify_layout.py and by eye with
scripts/visualize_mapping.py on the actual WFLW data.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

CANVAS = 448


def _ellipse_arc(center, rx, ry, deg_start, deg_end, n) -> np.ndarray:
    """n points along an ellipse arc; angles in degrees, y-down image coords."""
    cx, cy = center
    angs = np.deg2rad(np.linspace(deg_start, deg_end, n))
    return np.stack([cx + rx * np.cos(angs), cy + ry * np.sin(angs)], axis=1)


def _eye_points(center) -> np.ndarray:
    """8-point eye contour in WFLW order: image-left corner, upper arc
    left->right (3), image-right corner, lower arc right->left (3)."""
    cx, cy = center
    offsets = [(-32, 0), (-16, -11), (0, -14), (16, -11),
               (32, 0), (16, 9), (0, 12), (-16, 9)]
    return np.array([(cx + dx, cy + dy) for dx, dy in offsets], dtype=np.float64)


def canonical_landmarks98() -> np.ndarray:
    """98 points laid out per the documented WFLW indexing, on a 448x448
    canvas, frontal upright face."""
    pts = np.zeros((98, 2), dtype=np.float64)

    # 0-32 face contour: image-left temple -> chin (index 16) -> image-right temple
    pts[0:33] = _ellipse_arc((224, 210), 150, 190, 195, -15, 33)

    # 33-50 eyebrows (unused by our 24; plausible arcs are enough).
    # 195 -> 345 degrees sweeps over the TOP of the ellipse in y-down coords.
    pts[33:42] = _ellipse_arc((154, 158), 46, 12, 195, 345, 9)   # image-left brow
    pts[42:51] = _ellipse_arc((294, 158), 46, 12, 195, 345, 9)   # image-right brow

    # 51-59 nose: 51-54 bridge (51 between brows, 54 = tip), 55-59 bottom edge
    pts[51] = (224, 192)
    pts[52] = (224, 212)
    pts[53] = (224, 232)
    pts[54] = (224, 252)
    pts[55:60] = [(194, 262), (209, 267), (224, 269), (239, 267), (254, 262)]

    # 60-67 image-left eye, 68-75 image-right eye, WFLW point order
    pts[60:68] = _eye_points((154, 190))
    pts[68:76] = _eye_points((294, 190))

    # 76-87 outer lip: 76 left corner, 77-81 upper edge (79 = top mid),
    # 82 right corner, 83-87 lower edge (85 = bottom mid)
    mx, my = 224, 320
    outer = [(-55, 0), (-35, -14), (-16, -18), (0, -20), (16, -18), (35, -14),
             (55, 0), (35, 12), (16, 17), (0, 19), (-16, 17), (-35, 12)]
    pts[76:88] = [(mx + dx, my + dy) for dx, dy in outer]

    # 88-95 inner lip
    inner = [(-38, 0), (-19, -6), (0, -8), (19, -6), (38, 0), (19, 5), (0, 7), (-19, 5)]
    pts[88:96] = [(mx + dx, my + dy) for dx, dy in inner]

    # 96/97 pupils: centre of the image-left eye, then the image-right eye
    pts[96] = (154, 190)
    pts[97] = (294, 190)
    return pts


def render_face(pts: np.ndarray, size: int = CANVAS) -> np.ndarray:
    """Draw a schematic face whose visible structures are built from the
    landmark points themselves."""
    img = np.full((size, size, 3), (60, 60, 60), dtype=np.uint8)  # background
    skin, skin_edge = (150, 180, 220), (110, 140, 180)
    ipts = lambda sl: pts[sl].round().astype(np.int32)

    # head: fill the region bounded by the contour, closed over the forehead
    # (-15 -> -165 degrees passes the ellipse top, mirroring the contour arc)
    forehead = _ellipse_arc((224, 210), 150, 190, -15, -165, 40)[1:-1]
    head_poly = np.concatenate([pts[0:33], forehead]).round().astype(np.int32)
    cv2.fillPoly(img, [head_poly], skin)
    cv2.polylines(img, [ipts(slice(0, 33))], False, skin_edge, 2, cv2.LINE_AA)

    for sl in (slice(33, 42), slice(42, 51)):                      # eyebrows
        cv2.polylines(img, [ipts(sl)], False, (60, 80, 110), 5, cv2.LINE_AA)

    cv2.polylines(img, [ipts(slice(51, 55))], False, skin_edge, 2, cv2.LINE_AA)  # nose
    cv2.polylines(img, [ipts(slice(55, 60))], False, skin_edge, 2, cv2.LINE_AA)

    for eye_sl, pupil_idx in ((slice(60, 68), 96), (slice(68, 76), 97)):
        cv2.fillPoly(img, [ipts(eye_sl)], (250, 250, 250))         # sclera
        c = tuple(pts[pupil_idx].round().astype(int))
        cv2.circle(img, c, 9, (120, 90, 60), -1, cv2.LINE_AA)      # iris
        cv2.circle(img, c, 4, (25, 25, 25), -1, cv2.LINE_AA)       # pupil
        cv2.polylines(img, [ipts(eye_sl)], True, (90, 110, 140), 1, cv2.LINE_AA)

    cv2.fillPoly(img, [ipts(slice(76, 88))], (100, 100, 190))      # lips
    cv2.fillPoly(img, [ipts(slice(88, 96))], (60, 60, 120))
    return img


def generate_face(rng: np.ndarray | None = None, angle_deg: float = 0.0,
                  scale: float = 1.0, shift: tuple[float, float] = (0.0, 0.0)
                  ) -> tuple[np.ndarray, np.ndarray]:
    """A canonical face warped by rotation/scale/translation (image and
    points through the same affine), so synthetic samples differ."""
    pts = canonical_landmarks98()
    img = render_face(pts)
    m = cv2.getRotationMatrix2D((CANVAS / 2, CANVAS / 2), angle_deg, scale)
    m[:, 2] += shift
    img = cv2.warpAffine(img, m, (CANVAS, CANVAS), flags=cv2.INTER_LINEAR,
                         borderValue=(60, 60, 60))
    ones = np.ones((98, 1))
    pts = (np.hstack([pts, ones]) @ m.T).astype(np.float32)
    return img, pts


def annotation_line(pts: np.ndarray, attributes: dict[str, int],
                    attribute_names: list[str], rel_path: str) -> str:
    """Format one face as a WFLW rect_attr annotation line."""
    xy = " ".join(f"{v:.3f}" for v in pts.reshape(-1))
    xmin, ymin = pts.min(axis=0) - 12
    xmax, ymax = pts.max(axis=0) + 12
    rect = f"{xmin:.0f} {ymin:.0f} {xmax:.0f} {ymax:.0f}"
    attrs = " ".join(str(attributes.get(n, 0)) for n in attribute_names)
    return f"{xy} {rect} {attrs} {rel_path}"


def synthetic_trained_setup(cfg: dict, epochs: int = 6) -> tuple[dict, Path]:
    """Shared smoke-test scaffolding for scripts that need a model: build a
    schematic dataset and cache, train a tiny model briefly, and point the
    config at the artefacts. Returns (cfg, checkpoint_path). Loudly not a
    real experiment; numbers from it are meaningless."""
    import tempfile
    from dms_layer1.config import require
    from dms_layer1.data.cache import build_cache
    from dms_layer1.train.loop import Trainer

    print("=" * 70)
    print("SYNTHETIC MODE: schematic faces + a briefly trained model.")
    print("Code smoke test only; numbers are meaningless.")
    print("=" * 70)
    tmp = Path(tempfile.mkdtemp(prefix="synth_run_"))
    root = write_synthetic_dataset(tmp / "ds", require(cfg, "dataset.attribute_names"),
                                   seed=require(cfg, "seed"))
    cfg["dataset"]["root"] = str(root)
    cfg["preprocess"]["out_dir"] = str(tmp / "cache")
    for split in ("train", "test"):
        build_cache(cfg, split)
    cfg["cache"] = {"dir": str(tmp / "cache")}
    cfg["model"]["width"] = 8
    cfg["train"].update({"epochs": epochs, "batch_size": 8, "num_workers": 0,
                         "stop_after_epochs": None, "resume": False,
                         "checkpoint_dir": str(tmp / "ckpt"),
                         "metrics_csv": str(tmp / "metrics.csv"),
                         "curves_png": str(tmp / "curves.png")})
    Trainer(cfg).train()
    return cfg, tmp / "ckpt" / "best.pth"


def write_synthetic_dataset(root: str | Path, attribute_names: list[str],
                            num_per_split: int = 12, seed: int = 0) -> Path:
    """Create a directory tree that mirrors the real WFLW layout (images dir,
    annotations dir, rect_attr train/test lists) filled with schematic faces.
    Returns the dataset root to point dataset.root at."""
    root = Path(root)
    rng = np.random.default_rng(seed)
    img_dir = root / "WFLW_images" / "synthetic"
    ann_dir = root / "WFLW_annotations" / "list_98pt_rect_attr_train_test"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "test"):
        lines = []
        for i in range(num_per_split):
            angle = float(rng.uniform(-12, 12))
            scale = float(rng.uniform(0.8, 1.1))
            shift = (float(rng.uniform(-25, 25)), float(rng.uniform(-25, 25)))
            img, pts = generate_face(angle_deg=angle, scale=scale, shift=shift)
            # cycle flags so frontal / pose / occlusion sampling has material
            attrs = {n: 0 for n in attribute_names}
            if i % 3 == 1:
                attrs["pose"] = 1
            elif i % 3 == 2:
                attrs["occlusion"] = 1
            rel = f"synthetic/{split}_{i:03d}.jpg"
            cv2.imwrite(str(root / "WFLW_images" / rel), img)
            lines.append(annotation_line(pts, attrs, attribute_names, rel))
        with open(ann_dir / f"list_98pt_rect_attr_{split}.txt", "w") as f:
            f.write("\n".join(lines) + "\n")
    return root

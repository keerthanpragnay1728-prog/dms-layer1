#!/usr/bin/env python3
"""Milestone 5: evaluate a trained checkpoint on the WFLW test set.

Reports, per the project brief:
  1. overall NME (inter-ocular normalised) on the 2,500-face test split
  2. NME per landmark group (eyelids, pupils, mouth, axis, contour) -
     separately, because pupil error matters far more than contour error
  3. NME + failure rate per WFLW test subset (largepose, expression,
     illumination, makeup, occlusion, blur) and the no-flag subset
  4. failure rate at NME > 10%
  5. model size (MB, params) and CPU inference time per frame - labelled as
     measured on THIS machine's CPU; use only as a relative comparison

Protocol note: evaluation runs on the cached ground-truth-box crops (the
standard WFLW protocol), so landmark quality is not confounded by the face
detector. The full-pipeline (Haar) comparison belongs to milestone 6.

Outputs: printed report, m5_results.yaml, per-face NME array (.npy - kept
for the milestone-6 ablation statistics), worst-K face renders (prediction
vs ground truth), and the config snapshot.

Usage (Kaggle; attach the cache and the training notebook's output):
    python scripts/evaluate.py --config configs/layer1_base.yaml \
        --checkpoint /kaggle/input/notebooks/<user>/<train-nb>/checkpoints/best.pth
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from dms_layer1.config import load_config, require, resolve_path, save_config_snapshot
from dms_layer1.data.cache import KAGGLE_INPUT, load_cache_from_cfg
from dms_layer1.evaluation import metrics
from dms_layer1.landmarks.schema import load_schema
from dms_layer1.model.io import load_model
from dms_layer1.model.net import model_size_mb
from dms_layer1.train.data import eval_transform
from dms_layer1.viz.overlay import GROUP_COLORS

_lines: list[str] = []


def say(text: str = "") -> None:
    print(text)
    _lines.append(text)


def resolve_checkpoint(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_file():
        return path
    candidates: list[Path] = []
    if KAGGLE_INPUT.is_dir():
        for depth in range(1, 6):
            pattern = "/".join(["*"] * depth)
            candidates += list(KAGGLE_INPUT.glob(f"{pattern}/best.pth"))
            candidates += list(KAGGLE_INPUT.glob(f"{pattern}/last.pth"))
    found = "\n".join(f"    {c}" for c in sorted(set(candidates))) or "    (none)"
    raise FileNotFoundError(
        f"Checkpoint not found: {path}\n"
        f"Checkpoints visible under {KAGGLE_INPUT}:\n{found}\n"
        "Pass --checkpoint with the exact path (auto-picking is deliberately "
        "not done here: several runs may be attached at once)."
    )


def predict_all(model, crops: np.ndarray, landmarks: np.ndarray,
                cfg: dict, device: torch.device) -> np.ndarray:
    """Deterministic eval transform + batched forward. Returns (N, 24, 2)
    predictions in [0, 1] crop space. The input size comes from the model's
    own arch record, so checkpoints of any size evaluate correctly."""
    input_size = model.arch["input_size"]
    mean = float(require(cfg, "train.pixel_mean"))
    std = float(require(cfg, "train.pixel_std"))
    batch = int(require(cfg, "eval.batch_size"))
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(crops), batch):
            imgs = [eval_transform(c, l, input_size)[0]
                    for c, l in zip(crops[i:i + batch], landmarks[i:i + batch])]
            x = torch.from_numpy(np.stack(imgs)).float().unsqueeze(1) / 255.0
            x = ((x - mean) / std).to(device)
            preds.append(model(x).cpu().numpy())
    return np.concatenate(preds).astype(np.float64)


def cpu_timing(model, cfg: dict) -> dict:
    """Median per-frame time on CPU, batch size 1: model forward alone, and
    preprocess (resize + normalise) + forward."""
    input_size = model.arch["input_size"]
    iters = int(require(cfg, "eval.cpu_timing_iters"))
    model_cpu = model.to("cpu").eval()
    crop = (np.random.default_rng(0).uniform(0, 255, (128, 128))).astype(np.uint8)

    def frame_once(with_pre: bool) -> float:
        t0 = time.perf_counter()
        if with_pre:
            img = cv2.resize(crop, (input_size, input_size), interpolation=cv2.INTER_AREA)
            x = (torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0 - 0.5) / 0.5
        else:
            x = frame_once.x
        with torch.no_grad():
            model_cpu(x)
        return (time.perf_counter() - t0) * 1000

    frame_once.x = torch.zeros(1, 1, input_size, input_size)
    for _ in range(20):                      # warmup
        frame_once(True)
    fwd = [frame_once(False) for _ in range(iters)]
    full = [frame_once(True) for _ in range(iters)]
    return {"forward_ms_median": float(np.median(fwd)),
            "preprocess_and_forward_ms_median": float(np.median(full)),
            "iters": iters}


def render_worst(crops: np.ndarray, gt: np.ndarray, pred: np.ndarray,
                 nme: np.ndarray, schema, k: int, out_path: Path) -> None:
    """Worst-k faces: ground truth green, prediction in group colours."""
    order = np.argsort(-nme)[:k]
    tiles = []
    big = crops.shape[1] * 3
    for i in order:
        tile = cv2.cvtColor(cv2.resize(crops[i], (big, big),
                                       interpolation=cv2.INTER_NEAREST),
                            cv2.COLOR_GRAY2BGR)
        for p in schema.points:
            gx, gy = gt[i, p.index] * big
            cv2.circle(tile, (int(round(gx)), int(round(gy))), 3, (80, 220, 80), -1,
                       cv2.LINE_AA)
            px, py = pred[i, p.index] * big
            if 0 <= px < big and 0 <= py < big:
                cv2.circle(tile, (int(round(px)), int(round(py))), 3,
                           GROUP_COLORS[p.group], 1, cv2.LINE_AA)
        strip = np.full((24, big, 3), (25, 25, 25), dtype=np.uint8)
        cv2.putText(strip, f"idx {i}  NME {100 * nme[i]:.1f}%  (green=gt)",
                    (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1,
                    cv2.LINE_AA)
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
    ap.add_argument("--checkpoint", default=None,
                    help="default: config eval.checkpoint")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--out-dir", default=None, help="default: config eval.out_dir")
    ap.add_argument("--synthetic", action="store_true",
                    help="build synthetic data, train a few epochs, evaluate "
                         "(code smoke test only)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.synthetic:
        cfg = _synthetic_pipeline(cfg)
    ckpt_path = resolve_checkpoint(args.checkpoint or require(cfg, "eval.checkpoint"))
    out_dir = Path(args.out_dir or require(cfg, "eval.out_dir"))
    out_dir.mkdir(parents=True, exist_ok=True)
    threshold = float(require(cfg, "eval.failure_threshold"))
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)

    schema = load_schema(resolve_path(cfg, require(cfg, "landmark_schema")))
    model, meta = load_model(ckpt_path, cfg, device)
    val_nme = meta["val_nme"]
    say(f"checkpoint: {ckpt_path} (trained epoch {meta['epoch']}, "
        + (f"val NME {val_nme:.3f}%)" if val_nme is not None else "no val record)"))
    say(f"device: {device}  |  model: {model_size_mb(model):.2f} MB, "
        f"{sum(p.numel() for p in model.parameters()):,} params")

    data = load_cache_from_cfg(cfg, "test")
    say(f"test split: {data.crops.shape[0]} faces "
        "(ground-truth-box crops - the standard WFLW protocol; the full "
        "Haar-pipeline comparison is milestone 6)")
    pred = predict_all(model, data.crops, data.landmarks, cfg, device)
    gt = data.landmarks.astype(np.float64)

    nme = metrics.nme_per_face(pred, gt, schema)
    say("\n=== 1. Overall ===")
    say(f"  NME (inter-ocular): mean {100 * nme.mean():.3f}%   "
        f"median {100 * np.median(nme):.3f}%")
    say(f"  failure rate @ NME > {threshold:.0%}: "
        f"{100 * metrics.failure_rate(nme, threshold):.2f}%")

    say("\n=== 2. Per landmark group ===")
    groups = metrics.group_nme(pred, gt, schema)
    for g, v in groups.items():
        say(f"  {g:<8} {100 * v:6.3f}%")

    say("\n=== 3. Per WFLW test subset ===")
    subsets = metrics.subset_nme(nme, data.attrs,
                                 data.manifest["attribute_names"], threshold)
    say(f"  {'subset':<14} {'n':>5} {'NME':>8} {'failure':>9}")
    for label, e in subsets.items():
        if e["n"]:
            say(f"  {label:<14} {e['n']:>5} {e['nme_pct']:>7.3f}% {e['failure_pct']:>8.2f}%")
        else:
            say(f"  {label:<14} {e['n']:>5} {'n/a':>8} {'n/a':>9}")

    say("\n=== 4. CPU inference time (this machine's CPU - relative use only) ===")
    timing = cpu_timing(model, cfg)
    say(f"  forward only          : {timing['forward_ms_median']:.2f} ms/frame")
    say(f"  preprocess + forward  : {timing['preprocess_and_forward_ms_median']:.2f} ms/frame")
    say(f"  (median of {timing['iters']} single-frame iterations, batch 1)")

    k = int(require(cfg, "eval.save_worst"))
    if k > 0:
        worst_png = out_dir / "worst_faces.png"
        render_worst(data.crops, gt, pred, nme, schema, min(k, len(nme)), worst_png)
        say(f"\nworst-{k} render: {worst_png}")

    np.save(out_dir / "nme_per_face.npy", nme)
    results = {
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": meta["epoch"],
        "n_test_faces": int(len(nme)),
        "overall": {"nme_pct_mean": float(100 * nme.mean()),
                    "nme_pct_median": float(100 * np.median(nme)),
                    "failure_pct": float(100 * metrics.failure_rate(nme, threshold)),
                    "failure_threshold": threshold},
        "per_group_nme_pct": {g: float(100 * v) for g, v in groups.items()},
        "per_subset": subsets,
        "model": {"size_mb": round(model_size_mb(model), 3),
                  "params": int(sum(p.numel() for p in model.parameters()))},
        "cpu_timing_ms": timing,
    }
    with open(out_dir / "m5_results.yaml", "w") as f:
        yaml.safe_dump(results, f, sort_keys=False)
    (out_dir / "report.txt").write_text("\n".join(_lines) + "\n")
    save_config_snapshot(cfg, out_dir)
    say(f"\nresults + report + per-face NME + config snapshot in {out_dir}")
    return 0


def _synthetic_pipeline(cfg: dict) -> dict:
    """Smoke path: synthetic dataset -> cache -> short training -> cfg
    pointed at the artefacts. Loudly not a real evaluation."""
    import tempfile
    from dms_layer1.data.cache import build_cache
    from dms_layer1.data.synthetic import write_synthetic_dataset
    from dms_layer1.train.loop import Trainer

    print("=" * 70)
    print("SYNTHETIC MODE: schematic faces + a briefly trained model.")
    print("Code smoke test only - numbers are meaningless.")
    print("=" * 70)
    tmp = Path(tempfile.mkdtemp(prefix="m5_synth_"))
    root = write_synthetic_dataset(tmp / "ds", require(cfg, "dataset.attribute_names"),
                                   seed=require(cfg, "seed"))
    cfg["dataset"]["root"] = str(root)
    cfg["preprocess"]["out_dir"] = str(tmp / "cache")
    for split in ("train", "test"):
        build_cache(cfg, split)
    cfg["cache"]["dir"] = str(tmp / "cache")
    cfg["model"]["width"] = 8
    cfg["train"].update({"epochs": 6, "batch_size": 8, "num_workers": 0,
                         "stop_after_epochs": None, "resume": False,
                         "checkpoint_dir": str(tmp / "ckpt"),
                         "metrics_csv": str(tmp / "metrics.csv"),
                         "curves_png": str(tmp / "curves.png")})
    Trainer(cfg).train()
    cfg["eval"]["checkpoint"] = str(tmp / "ckpt" / "best.pth")
    cfg["eval"]["out_dir"] = str(tmp / "eval")
    return cfg


if __name__ == "__main__":
    sys.exit(main())

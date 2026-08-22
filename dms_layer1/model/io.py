"""Loading and exporting model weights.

Two file kinds exist:
  * training checkpoints (checkpoints/best.pth): model weights plus
    optimiser, scheduler and RNG state, so a run can resume. Roughly 3x the
    model size.
  * exported weights (from scripts/export_weights.py): the state dict plus
    a small "arch" record (num_points, width, input_size) and training
    metadata. This is the shippable file and the size quoted in the report.

load_weights accepts both. When a file carries an "arch" record it is
checked against the model being loaded, so a config/weights mismatch fails
with a clear message instead of a shape error deep inside load_state_dict.
"""

from __future__ import annotations

from pathlib import Path

import torch

from dms_layer1.config import require
from dms_layer1.model.net import LandmarkNet


def build_model_from_cfg(cfg: dict) -> LandmarkNet:
    return LandmarkNet(num_points=int(require(cfg, "model.num_points")),
                       width=int(require(cfg, "model.width")),
                       input_size=int(require(cfg, "model.input_size")))


def _arch_of(cfg: dict) -> dict:
    return {"num_points": int(require(cfg, "model.num_points")),
            "width": int(require(cfg, "model.width")),
            "input_size": int(require(cfg, "model.input_size"))}


def load_weights(model: LandmarkNet, path: str | Path, cfg: dict,
                 device: torch.device | str = "cpu") -> dict:
    """Load a training checkpoint or an exported weights file into `model`.
    Returns metadata: {'epoch': int or None, 'val_nme': float or None}."""
    ck = torch.load(path, map_location=device, weights_only=False)
    if "model" not in ck:
        raise ValueError(f"{path} has no 'model' entry; not a checkpoint "
                         "or exported weights file from this project")
    if "arch" in ck and ck["arch"] != _arch_of(cfg):
        raise ValueError(
            f"Architecture mismatch loading {path}:\n"
            f"  file  : {ck['arch']}\n  config: {_arch_of(cfg)}\n"
            "Point --config at the config the weights were trained with."
        )
    model.load_state_dict(ck["model"])
    return {"epoch": ck.get("trained_epoch", ck.get("epoch")),
            "val_nme": ck.get("val_nme", ck.get("best_nme"))}


def load_model(path: str | Path, cfg: dict,
               device: torch.device | str = "cpu") -> tuple[LandmarkNet, dict]:
    """Build the right model for a weights file and load it. Files that
    carry an 'arch' record (new checkpoints and all exports) are
    self-describing: the model is built from the record, so one config can
    evaluate runs of different widths side by side. Files without the
    record fall back to the config's model section, with a clear error if
    the shapes do not fit."""
    ck = torch.load(path, map_location=device, weights_only=False)
    if "model" not in ck:
        raise ValueError(f"{path} has no 'model' entry; not a checkpoint "
                         "or exported weights file from this project")
    if "arch" in ck:
        model = LandmarkNet(**ck["arch"])
        if ck["arch"] != _arch_of(cfg):
            print(f"note: using the architecture stored in {Path(path).name} "
                  f"({ck['arch']}); the config's model section differs and is "
                  "ignored for loading")
    else:
        model = build_model_from_cfg(cfg)
        try:
            model.load_state_dict(ck["model"])
        except RuntimeError as e:
            first = str(e).strip().splitlines()[0]
            raise ValueError(
                f"Weights in {path} do not fit the model built from the "
                f"config ({_arch_of(cfg)}). This file predates embedded arch "
                "records; pass --config pointing at the config it was trained "
                "with (its config_used.yaml sits next to the checkpoint). "
                f"Original error: {first}"
            ) from e
        meta = {"epoch": ck.get("trained_epoch", ck.get("epoch")),
                "val_nme": ck.get("val_nme", ck.get("best_nme"))}
        return model.to(device), meta
    model.load_state_dict(ck["model"])
    meta = {"epoch": ck.get("trained_epoch", ck.get("epoch")),
            "val_nme": ck.get("val_nme", ck.get("best_nme"))}
    return model.to(device), meta


def export_weights(checkpoint_path: str | Path, cfg: dict,
                   out_path: str | Path) -> dict:
    """Strip a training checkpoint down to deployable weights. Verifies the
    exported file by reloading it and comparing a forward pass against the
    original, then returns a size summary."""
    checkpoint_path, out_path = Path(checkpoint_path), Path(out_path)
    model, meta = load_model(checkpoint_path, cfg)
    model.eval()

    payload = {
        "model": model.state_dict(),
        "arch": model.arch,
        "trained_epoch": meta["epoch"],
        "val_nme": meta["val_nme"],
        "source_checkpoint": checkpoint_path.name,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)

    # verification: the exported file must reproduce the checkpoint exactly
    model2, _ = load_model(out_path, cfg)
    model2.eval()
    torch.manual_seed(0)
    size = model.arch["input_size"]
    x = torch.randn(2, 1, size, size)
    with torch.no_grad():
        diff = float((model(x) - model2(x)).abs().max())
    if diff != 0.0:
        raise RuntimeError(f"Export verification failed: outputs differ by {diff}")

    return {"checkpoint_mb": round(checkpoint_path.stat().st_size / 1e6, 2),
            "exported_mb": round(out_path.stat().st_size / 1e6, 2),
            "params": sum(p.numel() for p in model.parameters()),
            "trained_epoch": meta["epoch"], "val_nme": meta["val_nme"],
            "out_path": str(out_path)}

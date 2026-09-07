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


def _meta_of(ck: dict) -> dict:
    return {"epoch": ck.get("trained_epoch", ck.get("epoch")),
            "val_nme": ck.get("val_nme", ck.get("best_nme")),
            "train_meta": ck.get("train_meta")}


def describe_weights(path, meta: dict) -> str:
    """One line naming exactly which weights are in play. Printed by every
    script that loads a model: milestone 6 wasted a run on stale uploaded
    weights that were indistinguishable from the new ones in the output."""
    tm = meta.get("train_meta") or {}
    if tm.get("framing"):
        framing = f"framing [{tm['framing'][0]:.2f}, {tm['framing'][1]:.2f}]"
        if tm.get("stamped_after_the_fact"):
            framing += f" (stamped from {tm['stamped_after_the_fact']}, not by "
            framing += "the trainer)"
    elif meta.get("train_meta") is not None:
        framing = ("framing NOT RECORDED: this file carries a train_meta "
                   "record with no framing entry")
    else:
        # Never guess an envelope here. An export made before the record
        # existed says nothing about what it was trained on, and asserting
        # the narrow envelope told one user the opposite of the truth about
        # their own model.
        framing = ("framing NOT RECORDED: this file has no train_meta record, "
                   "which means it was exported before provenance was "
                   "stamped, NOT that it was trained narrow. The training "
                   "run's config_used.yaml records train.augment.framing; "
                   "re-export from the checkpoint, or restamp with "
                   "scripts/export_weights.py --restamp, to make the file "
                   "self-describing")
    val = f"{meta['val_nme']:.3f}%" if meta.get("val_nme") is not None else "n/a"
    return (f"weights: {path}\n"
            f"         trained epoch {meta.get('epoch')}, val NME {val}, "
            f"loss {tm.get('loss', 'n/a')}, {framing}")


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
    return _meta_of(ck)


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
        return model.to(device), _meta_of(ck)
    model.load_state_dict(ck["model"])
    return model.to(device), _meta_of(ck)


def resolve_weights(path_or_auto: str | Path) -> Path:
    """Resolve a weights file for the deployed detector. An explicit path is
    used as-is; 'auto' (or a missing path) searches /kaggle/input for
    exported weights (landmarks24*.pt) and falls back to best.pth files. One
    unambiguous hit is used with a printed note; zero or several raise with
    the candidate list, because auto-picking between runs would be silent
    substitution."""
    path = Path(path_or_auto)
    if str(path_or_auto) != "auto" and path.is_file():
        return path
    kaggle_input = Path("/kaggle/input")
    exported: list[Path] = []
    checkpoints: list[Path] = []
    if kaggle_input.is_dir():
        for depth in range(1, 6):
            pattern = "/".join(["*"] * depth)
            exported += list(kaggle_input.glob(f"{pattern}/landmarks24*.pt"))
            checkpoints += list(kaggle_input.glob(f"{pattern}/best.pth"))
    hits = sorted(set(exported)) or sorted(set(checkpoints))
    if len(hits) == 1:
        print(f"NOTE: detector weights '{path_or_auto}' not found; using the "
              f"single candidate {hits[0]}. Set detector.weights explicitly "
              "to silence this note.")
        return hits[0]
    listing = "\n".join(f"    {h}" for h in sorted(set(exported + checkpoints))) or "    (none)"
    raise FileNotFoundError(
        f"Detector weights not found: {path_or_auto}\n"
        f"Weights visible under {kaggle_input}:\n{listing}\n"
        "Set detector.weights (or pass --weights) to the exact file."
    )


def export_weights(checkpoint_path: str | Path, cfg: dict,
                   out_path: str | Path) -> dict:
    """Strip a training checkpoint down to deployable weights. Verifies the
    exported file by reloading it and comparing a forward pass against the
    original, then returns a size summary."""
    checkpoint_path, out_path = Path(checkpoint_path), Path(out_path)
    model, meta = load_model(checkpoint_path, cfg)
    model.eval()

    if meta.get("train_meta") is None:
        print(f"WARNING: {checkpoint_path.name} carries no train_meta record, "
              "so the exported file cannot say which framing envelope, loss "
              "or seed produced it. Every script that loads it will say so "
              "rather than guess. Export from a checkpoint written by the "
              "current trainer, or add the record with --restamp.")
    payload = {
        "model": model.state_dict(),
        "arch": model.arch,
        "trained_epoch": meta["epoch"],
        "val_nme": meta["val_nme"],
        "train_meta": meta.get("train_meta"),
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


def restamp_weights(weights_path: str | Path, cfg: dict,
                    config_name: str) -> dict:
    """Add a train_meta record to an exported file that has none.

    Exports made before the trainer stamped provenance carry no framing
    envelope, and a file that cannot say what it is gets described as
    unknown by every script that loads it. This writes the record from the
    training config the user names, and marks it as stamped after the fact
    so it is never mistaken for the trainer's own record. It changes no
    weights; the state dict is written back unchanged.
    """
    weights_path = Path(weights_path)
    ck = torch.load(weights_path, map_location="cpu", weights_only=False)
    if "model" not in ck:
        raise ValueError(f"{weights_path} is not a weights file from this project")
    existing = ck.get("train_meta")
    if existing and existing.get("framing"):
        raise ValueError(
            f"{weights_path} already records framing "
            f"{existing['framing']}; refusing to overwrite a real record.")
    def optional(key, cast):
        """Record only what the named config actually contains. A run config
        that predates a setting should leave a gap rather than have one
        invented, which is the mistake this whole feature exists to undo."""
        try:
            return cast(require(cfg, key))
        except Exception:
            return None

    framing = require(cfg, "train.augment.framing")
    record = {
        "framing": [float(framing[0]), float(framing[1])],
        "reference_expand": optional("preprocess.reference_expand", float),
        "cache_expand": optional("preprocess.crop_expand", float),
        "loss": optional("train.loss", str),
        "seed": optional("seed", int),
        "stamped_after_the_fact": config_name,
    }
    ck["train_meta"] = {k: v for k, v in record.items() if v is not None}
    torch.save(ck, weights_path)
    return ck["train_meta"]

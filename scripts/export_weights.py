#!/usr/bin/env python3
"""Export a training checkpoint to a weights-only file for deployment.

Training checkpoints carry optimiser, scheduler and RNG state so a run can
resume; the shippable model is just the weights. This writes the small file,
reloads it, and confirms the forward pass matches the original exactly.

    python scripts/export_weights.py --config configs/layer1_base.yaml \
        --checkpoint /kaggle/working/checkpoints/best.pth \
        --out /kaggle/working/landmarks24_wing.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dms_layer1.config import load_config
from dms_layer1.model.io import export_weights, restamp_weights


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", help="training checkpoint to export")
    ap.add_argument("--out", default="/kaggle/working/landmarks24.pt")
    ap.add_argument("--restamp", metavar="WEIGHTS.pt",
                    help="add a train_meta record to an already exported file "
                         "that has none, taken from --config's training "
                         "section and marked as stamped after the fact. "
                         "Weights are not touched.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.restamp:
        stamped = restamp_weights(args.restamp, cfg, Path(args.config).name)
        print(f"restamped {args.restamp} from {args.config}:")
        for k, v in stamped.items():
            print(f"    {k}: {v}")
        print("This is a record of what you assert the file was trained with, "
              "not the trainer's own stamp, and it is labelled as such "
              "wherever the weights are described. Point --config at the "
              "RUN's config_used.yaml, not at the base config: the base "
              "config's loss and framing are whatever was last edited there, "
              "and stamping those onto an old file would replace one wrong "
              "provenance claim with another.")
        return 0
    if not args.checkpoint:
        ap.error("--checkpoint is required unless --restamp is given")
    summary = export_weights(args.checkpoint, cfg, args.out)
    print(f"checkpoint : {args.checkpoint}  ({summary['checkpoint_mb']} MB)")
    print(f"exported   : {summary['out_path']}  ({summary['exported_mb']} MB, "
          f"{summary['params']:,} params)")
    print(f"trained epoch {summary['trained_epoch']}, "
          f"val NME {summary['val_nme']:.3f}%" if summary["val_nme"] is not None
          else f"trained epoch {summary['trained_epoch']}")
    print("verification: reloaded export matches the checkpoint output exactly")
    return 0


if __name__ == "__main__":
    sys.exit(main())

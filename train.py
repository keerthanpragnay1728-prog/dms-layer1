#!/usr/bin/env python3
"""Train the landmark model (milestone 4).

    python train.py --config configs/layer1_base.yaml

Everything - data location, model, loss, schedule, checkpoints, resume - is
in the config. To continue an interrupted run, set train.resume: true (or
pass --resume) and run the same command; the checkpoint restores optimiser,
scheduler, epoch counter and RNG state, so the continuation is identical to
an uninterrupted run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dms_layer1.config import load_config
from dms_layer1.train.loop import Trainer


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true",
                    help="same as setting train.resume: true in the config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.resume:
        cfg["train"]["resume"] = True
    Trainer(cfg).train()
    return 0


if __name__ == "__main__":
    sys.exit(main())

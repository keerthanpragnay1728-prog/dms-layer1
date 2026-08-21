"""The training loop (milestone 4), built for sessions that disconnect.

Session-survival rules implemented here, per the project brief:
  * a checkpoint EVERY epoch (atomic write: tmp file + rename, so a kill
    mid-save cannot corrupt the last good checkpoint),
  * checkpoints carry model, optimiser, scheduler, epoch counter, best-NME
    state, early-stop counter, and the python/numpy/torch RNG states,
  * train.resume: true restores all of it; augmentation and shuffling are
    seeded per epoch (see train/data.py), so a resumed run reproduces the
    exact batches an uninterrupted run would have seen,
  * metrics append to a CSV on disk after every epoch (on resume, rows from
    epochs that will be re-run are trimmed so the file never double-counts),
  * loss/NME curves are re-rendered to a PNG every epoch (the "TensorBoard
    or equivalent": CSV + curves, zero extra dependencies).

Validation is a seeded split of the train cache (WFLW has no subject IDs,
so a random face split is the only option; documented in the README). The
early-stopping metric is NME normalised by the outer-corner inter-ocular
distance, in percent.
"""

from __future__ import annotations

import csv
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dms_layer1.config import require, resolve_path, save_config_snapshot
from dms_layer1.data.cache import load_cache_from_cfg
from dms_layer1.landmarks.schema import load_schema
from dms_layer1.model.net import LandmarkNet, model_size_mb
from dms_layer1.train.data import AugmentParams, CachedFaceDataset
from dms_layer1.train.loss import make_loss

CSV_FIELDS = ["epoch", "lr", "train_loss", "val_loss", "val_nme_pct",
              "seconds", "timestamp"]


class Trainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.seed = int(require(cfg, "seed"))
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.schema = load_schema(resolve_path(cfg, require(cfg, "landmark_schema")))
        cache = load_cache_from_cfg(cfg, "train")
        n = len(cache.crops)
        val_fraction = float(require(cfg, "train.val_fraction"))
        perm = np.random.default_rng(self.seed).permutation(n)
        n_val = max(1, int(round(n * val_fraction)))
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        input_size = int(require(cfg, "model.input_size"))
        mean = float(require(cfg, "train.pixel_mean"))
        std = float(require(cfg, "train.pixel_std"))
        self.train_ds = CachedFaceDataset(
            cache.crops, cache.landmarks, train_idx, input_size, mean, std,
            augment=AugmentParams.from_cfg(cfg),
            flip_perm=self.schema.flip_permutation, base_seed=self.seed)
        self.val_ds = CachedFaceDataset(
            cache.crops, cache.landmarks, val_idx, input_size, mean, std)

        self.batch_size = int(require(cfg, "train.batch_size"))
        self.num_workers = int(require(cfg, "train.num_workers"))
        self.epochs = int(require(cfg, "train.epochs"))
        self.patience = int(require(cfg, "train.early_stopping_patience"))
        # Optional clean session cap (Kaggle's 12h limit): stop after N
        # completed epochs THIS session; resume continues identically.
        stop_after = require(cfg, "train.stop_after_epochs")
        self.stop_after = int(stop_after) if stop_after else None

        self.model = LandmarkNet(
            num_points=int(require(cfg, "model.num_points")),
            width=int(require(cfg, "model.width")),
            input_size=input_size).to(self.device)
        self.loss_fn = make_loss(cfg)
        opt_name = str(require(cfg, "train.optimizer")).lower()
        opt_cls = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam}.get(opt_name)
        if opt_cls is None:
            raise ValueError(f"Unknown train.optimizer '{opt_name}'")
        self.optimizer = opt_cls(self.model.parameters(),
                                 lr=float(require(cfg, "train.lr")),
                                 weight_decay=float(require(cfg, "train.weight_decay")))
        if str(require(cfg, "train.schedule")) != "cosine":
            raise ValueError("Only 'cosine' schedule is implemented")
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs)

        self.ckpt_dir = Path(require(cfg, "train.checkpoint_dir"))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = Path(require(cfg, "train.metrics_csv"))
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.curves_png = Path(require(cfg, "train.curves_png"))
        save_config_snapshot(cfg, self.ckpt_dir)

        self.start_epoch = 0
        self.best_nme = float("inf")
        self.since_best = 0
        if bool(require(cfg, "train.resume")):
            self._load_checkpoint()
        self._trim_csv()

        print(f"device={self.device}  model={model_size_mb(self.model):.2f} MB "
              f"({sum(p.numel() for p in self.model.parameters()):,} params)  "
              f"train={len(train_idx)} val={len(val_idx)}  "
              f"start_epoch={self.start_epoch}")

    # ---- checkpointing ----------------------------------------------------

    def _checkpoint_payload(self, epoch: int) -> dict:
        return {
            "epoch": epoch,                      # last COMPLETED epoch
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_nme": self.best_nme,
            "since_best": self.since_best,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": (torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else None),
            },
        }

    def _atomic_save(self, payload: dict, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)                    # kill-safe: never a torn file

    def _load_checkpoint(self) -> None:
        path = self.ckpt_dir / "last.pth"
        if not path.is_file():
            raise FileNotFoundError(
                f"train.resume is true but no checkpoint at {path}. "
                "Set resume: false for a fresh run.")
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model"])
        self.optimizer.load_state_dict(ck["optimizer"])
        self.scheduler.load_state_dict(ck["scheduler"])
        self.best_nme = ck["best_nme"]
        self.since_best = ck["since_best"]
        self.start_epoch = ck["epoch"] + 1
        random.setstate(ck["rng"]["python"])
        np.random.set_state(ck["rng"]["numpy"])
        torch.set_rng_state(ck["rng"]["torch"])
        if ck["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
        print(f"resumed from {path} (completed epoch {ck['epoch']}, "
              f"best NME {self.best_nme:.3f}%)")

    def _trim_csv(self) -> None:
        """Drop CSV rows for epochs that are about to be (re-)run, so a
        resume after a mid-epoch death never leaves duplicate rows."""
        if not self.csv_path.is_file():
            return
        with open(self.csv_path) as f:
            rows = list(csv.DictReader(f))
        keep = [r for r in rows if int(r["epoch"]) < self.start_epoch]
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(keep)

    def _append_csv(self, row: dict) -> None:
        new = not self.csv_path.is_file() or self.csv_path.stat().st_size == 0
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if new:
                writer.writeheader()
            writer.writerow(row)

    # ---- metrics ----------------------------------------------------------

    def _nme_percent(self, pred01: torch.Tensor, target01: torch.Tensor) -> torch.Tensor:
        """Per-face NME: mean point error / outer-corner inter-ocular
        distance, in percent. Coordinates may stay in [0,1] space — the
        ratio is scale-free."""
        err = torch.linalg.norm(pred01 - target01, dim=2).mean(dim=1)
        iod = torch.linalg.norm(
            target01[:, self.schema.nme_left_index]
            - target01[:, self.schema.nme_right_index], dim=1).clamp_min(1e-6)
        return 100.0 * err / iod

    def _validate(self) -> tuple[float, float]:
        self.model.eval()
        losses, nmes = [], []
        loader = DataLoader(self.val_ds, batch_size=self.batch_size,
                            shuffle=False, num_workers=self.num_workers)
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                losses.append(float(self.loss_fn(pred, y)) * len(x))
                nmes.append(self._nme_percent(pred, y).cpu())
        nme_all = torch.cat(nmes)
        return sum(losses) / len(self.val_ds), float(nme_all.mean())

    # ---- the loop ---------------------------------------------------------

    def train(self) -> float:
        for epoch in range(self.start_epoch, self.epochs):
            t0 = time.time()
            self.train_ds.set_epoch(epoch)
            g = torch.Generator()
            g.manual_seed(self.seed * 100_000 + epoch)   # deterministic shuffle
            loader = DataLoader(self.train_ds, batch_size=self.batch_size,
                                shuffle=True, num_workers=self.num_workers,
                                generator=g,
                                pin_memory=self.device.type == "cuda")
            self.model.train()
            total = 0.0
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                loss = self.loss_fn(self.model(x), y)
                loss.backward()
                self.optimizer.step()
                total += float(loss.detach()) * len(x)
            train_loss = total / len(self.train_ds)

            val_loss, val_nme = self._validate()
            self.scheduler.step()

            improved = val_nme < self.best_nme - 1e-9
            if improved:
                self.best_nme = val_nme
                self.since_best = 0
                self._atomic_save(self._checkpoint_payload(epoch),
                                  self.ckpt_dir / "best.pth")
            else:
                self.since_best += 1

            self._append_csv({
                "epoch": epoch,
                "lr": f"{self.scheduler.get_last_lr()[0]:.6g}",
                "train_loss": f"{train_loss:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_nme_pct": f"{val_nme:.4f}",
                "seconds": f"{time.time() - t0:.1f}",
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
            self._atomic_save(self._checkpoint_payload(epoch),
                              self.ckpt_dir / "last.pth")
            self._render_curves()
            print(f"epoch {epoch:3d}  train {train_loss:8.4f}  val {val_loss:8.4f}  "
                  f"NME {val_nme:6.3f}%{'  *best*' if improved else ''}  "
                  f"({time.time() - t0:.1f}s)")

            if self.since_best > self.patience:
                print(f"early stop: no val NME improvement for "
                      f"{self.patience} epochs (best {self.best_nme:.3f}%)")
                break
            if self.stop_after and (epoch - self.start_epoch + 1) >= self.stop_after:
                print(f"clean session stop after {self.stop_after} epochs "
                      "(train.stop_after_epochs); continue with train.resume: true")
                break
        print(f"done. best val NME {self.best_nme:.3f}%  "
              f"checkpoints in {self.ckpt_dir}")
        return self.best_nme

    def _render_curves(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with open(self.csv_path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return
        ep = [int(r["epoch"]) for r in rows]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6), dpi=120)
        ax1.plot(ep, [float(r["train_loss"]) for r in rows], color="#2a78d6",
                 linewidth=2, label="train")
        ax1.plot(ep, [float(r["val_loss"]) for r in rows], color="#eb6834",
                 linewidth=2, label="val")
        ax1.set_xlabel("epoch"), ax1.set_ylabel("loss"), ax1.legend(frameon=False)
        ax2.plot(ep, [float(r["val_nme_pct"]) for r in rows], color="#1baf7a",
                 linewidth=2)
        ax2.set_xlabel("epoch"), ax2.set_ylabel("val NME (%)")
        for ax in (ax1, ax2):
            ax.grid(color="#e6e6e6", linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        fig.tight_layout()
        self.curves_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.curves_png)
        plt.close(fig)

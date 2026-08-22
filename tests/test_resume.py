"""The brief's mandated resume test: kill a run, resume it, and confirm the
metrics are indistinguishable from an uninterrupted run - BEFORE the first
long training, not after losing one.

Run A trains 5 epochs straight. Run B trains the same 5-epoch schedule but
stops cleanly after 3 (train.stop_after_epochs, the same code path a Kaggle
session cap uses), then resumes. Both runs share the seed, so every epoch's
init, shuffle and augmentation draws are identical by construction; the test
asserts the two metrics CSVs match row for row. A separate test covers the
mid-epoch-death case: a stale CSV row for an epoch the resume will re-run
must be trimmed, never duplicated."""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dms_layer1.data.cache import build_cache
from dms_layer1.data.synthetic import write_synthetic_dataset
from dms_layer1.train.loop import Trainer

ATTRS = ["pose", "expression", "illumination", "makeup", "occlusion", "blur"]


def _cfg(cache_dir: Path, run_dir: Path, epochs: int, resume: bool = False,
         stop_after: int | None = None) -> dict:
    return {
        "_config_path": str(REPO / "configs" / "layer1_base.yaml"),
        "seed": 11,
        "landmark_schema": "landmarks_24.yaml",
        "cache": {"dir": str(cache_dir)},
        "model": {"input_size": 64, "num_points": 24, "width": 8},
        "train": {
            "batch_size": 8, "num_workers": 0, "val_fraction": 0.25,
            "pixel_mean": 0.5, "pixel_std": 0.5,
            "optimizer": "adamw", "lr": 0.001, "weight_decay": 0.0001,
            "schedule": "cosine", "epochs": epochs,
            "early_stopping_patience": 1000,
            "loss": "wing", "wing": {"w": 10.0, "epsilon": 2.0},
            "augment": {"rotation_deg": 10, "scale": [0.9, 1.1],
                        "translate_frac": 0.04, "brightness": 20,
                        "contrast": 0.15, "blur_prob": 0.2, "flip_prob": 0.5},
            "resume": resume, "stop_after_epochs": stop_after,
            "checkpoint_dir": str(run_dir / "ckpt"),
            "metrics_csv": str(run_dir / "metrics.csv"),
            "curves_png": str(run_dir / "curves.png"),
        },
    }


def _build_test_cache(tmp: Path) -> Path:
    root = write_synthetic_dataset(tmp / "ds", ATTRS, num_per_split=12, seed=4)
    cache_cfg = {
        "_config_path": str(REPO / "configs" / "layer1_base.yaml"),
        "landmark_schema": "landmarks_24.yaml", "seed": 4,
        "dataset": {
            "root": str(root), "images_dir": "WFLW_images",
            "annotations_dir": "WFLW_annotations",
            "train_list": "list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt",
            "test_list": "list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt",
            "attribute_names": ATTRS,
        },
        "preprocess": {"cache_size": 96, "crop_expand": 1.3,
                       "num_preview": 2, "out_dir": str(tmp / "cache")},
    }
    build_cache(cache_cfg, "train")
    return tmp / "cache"


def _rows(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def test_resumed_run_matches_uninterrupted_run():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cache_dir = _build_test_cache(tmp)

        Trainer(_cfg(cache_dir, tmp / "A", epochs=5)).train()

        # same 5-epoch schedule, "killed" cleanly after 3, then resumed
        Trainer(_cfg(cache_dir, tmp / "B", epochs=5, stop_after=3)).train()
        Trainer(_cfg(cache_dir, tmp / "B", epochs=5, resume=True)).train()

        a, b = _rows(tmp / "A" / "metrics.csv"), _rows(tmp / "B" / "metrics.csv")
        assert [r["epoch"] for r in a] == [r["epoch"] for r in b] == [str(i) for i in range(5)]
        for ra, rb in zip(a, b):
            for col in ("train_loss", "val_loss", "val_nme_pct", "lr"):
                da = abs(float(ra[col]) - float(rb[col]))
                assert da < 2e-6, (f"epoch {ra['epoch']} {col}: "
                                   f"{ra[col]} vs {rb[col]} - resume diverged")
        assert (tmp / "B" / "ckpt" / "best.pth").is_file()
        assert (tmp / "B" / "curves.png").is_file()


def test_stale_csv_row_is_trimmed_on_resume():
    """Simulates dying between the CSV append and the checkpoint save: the
    orphan row for the re-run epoch must be removed, not duplicated."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cache_dir = _build_test_cache(tmp)
        Trainer(_cfg(cache_dir, tmp / "C", epochs=5, stop_after=2)).train()

        csv_path = tmp / "C" / "metrics.csv"
        rows = _rows(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writerow({**rows[-1], "epoch": "2", "train_loss": "999.0"})

        Trainer(_cfg(cache_dir, tmp / "C", epochs=4, resume=True)).train()
        final = _rows(csv_path)
        assert [r["epoch"] for r in final] == ["0", "1", "2", "3"]
        assert float(final[2]["train_loss"]) != 999.0


def test_resume_without_checkpoint_fails_clearly():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cache_dir = _build_test_cache(tmp)
        try:
            Trainer(_cfg(cache_dir, tmp / "D", epochs=2, resume=True))
        except FileNotFoundError as e:
            assert "resume" in str(e)
        else:
            raise AssertionError("expected FileNotFoundError for resume without checkpoint")

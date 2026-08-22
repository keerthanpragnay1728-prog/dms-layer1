"""Export tests: the weights-only file must be small, reproduce the
checkpoint's outputs exactly, and refuse to load into a mismatched
architecture."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dms_layer1.model.io import build_model_from_cfg, export_weights, load_weights
from dms_layer1.train.loop import Trainer
from tests.test_resume import _build_test_cache, _cfg


def test_export_shrinks_and_reproduces():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cache_dir = _build_test_cache(tmp)
        cfg = _cfg(cache_dir, tmp / "run", epochs=2)
        Trainer(cfg).train()
        ckpt = tmp / "run" / "ckpt" / "best.pth"

        out = tmp / "export" / "landmarks24.pt"
        summary = export_weights(ckpt, cfg, out)
        assert out.is_file()
        # weights only: close to params * 4 bytes, well under the checkpoint
        assert summary["exported_mb"] < summary["checkpoint_mb"] / 2
        assert summary["exported_mb"] * 1e6 < summary["params"] * 4 * 1.2
        assert summary["val_nme"] is not None

        # exported file loads and matches the checkpoint's forward pass
        m_ckpt = build_model_from_cfg(cfg)
        load_weights(m_ckpt, ckpt, cfg)
        m_exp = build_model_from_cfg(cfg)
        meta = load_weights(m_exp, out, cfg)
        assert meta["epoch"] is not None
        x = torch.randn(2, 1, cfg["model"]["input_size"], cfg["model"]["input_size"])
        m_ckpt.eval(), m_exp.eval()
        with torch.no_grad():
            assert float((m_ckpt(x) - m_exp(x)).abs().max()) == 0.0


def test_load_model_uses_embedded_arch():
    """A width-8 checkpoint must load correctly even when the config says
    width 16, because the file carries its own arch record. This is the
    several-runs-one-config evaluation case."""
    from dms_layer1.model.io import load_model

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cache_dir = _build_test_cache(tmp)
        cfg = _cfg(cache_dir, tmp / "run", epochs=1)
        Trainer(cfg).train()

        other_cfg = {**cfg, "model": {**cfg["model"], "width": 16}}
        model, meta = load_model(tmp / "run" / "ckpt" / "best.pth", other_cfg)
        assert model.arch["width"] == 8
        assert meta["epoch"] == 0 and meta["val_nme"] is not None
        x = torch.randn(1, 1, cfg["model"]["input_size"], cfg["model"]["input_size"])
        model.eval()
        with torch.no_grad():
            assert model(x).shape == (1, 24, 2)


def test_arch_mismatch_fails_clearly():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cache_dir = _build_test_cache(tmp)
        cfg = _cfg(cache_dir, tmp / "run", epochs=1)
        Trainer(cfg).train()
        out = tmp / "landmarks24.pt"
        export_weights(tmp / "run" / "ckpt" / "best.pth", cfg, out)

        wrong = {**cfg, "model": {**cfg["model"], "width": 16}}
        try:
            load_weights(build_model_from_cfg(wrong), out, wrong)
        except ValueError as e:
            assert "mismatch" in str(e)
        else:
            raise AssertionError("expected ValueError for architecture mismatch")

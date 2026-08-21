"""YAML config loading.

Configs are plain nested dicts read with yaml.safe_load. Helpers here exist
to (a) fail with a message naming the missing key instead of a bare KeyError,
(b) resolve paths relative to the config file so scripts work from any
working directory, and (c) snapshot the config next to experiment outputs so
every reported number can be traced back to the settings that produced it.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """Raised when a config file is missing, malformed, or lacks a key."""


def load_config(path: str | Path) -> dict:
    """Load a YAML config file into a dict. The path is remembered under the
    reserved key ``_config_path`` so relative references can be resolved."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path.resolve()}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ConfigError(f"Config file {path} did not parse to a mapping.")
    cfg["_config_path"] = str(path.resolve())
    return cfg


def require(cfg: dict, dotted_key: str) -> Any:
    """Fetch ``cfg["a"]["b"]`` via ``require(cfg, "a.b")`` with a clear error
    that names the config file and the exact missing key."""
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            src = cfg.get("_config_path", "<config>")
            raise ConfigError(f"Missing required config key '{dotted_key}' in {src}")
        node = node[part]
    return node


def resolve_path(cfg: dict, value: str | Path) -> Path:
    """Resolve a possibly-relative path against the config file's directory."""
    p = Path(value)
    if p.is_absolute():
        return p
    base = Path(cfg.get("_config_path", ".")).parent
    return (base / p).resolve()


def save_config_snapshot(cfg: dict, out_dir: str | Path, name: str = "config_used.yaml") -> Path:
    """Write a copy of the config into an output directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = copy.deepcopy(cfg)
    snapshot.pop("_config_path", None)
    out_path = out_dir / name
    with open(out_path, "w") as f:
        yaml.safe_dump(snapshot, f, sort_keys=False)
    return out_path

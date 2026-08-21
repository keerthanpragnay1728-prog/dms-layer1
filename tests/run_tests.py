#!/usr/bin/env python3
"""Minimal test runner so the suite needs no extra dependency.

    python tests/run_tests.py

The test files are also plain pytest files; `pytest tests/` works identically
if pytest happens to be installed.
"""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODULES = ["tests.test_schema", "tests.test_wflw_parser", "tests.test_frame",
           "tests.test_crops", "tests.test_cache"]


def main() -> int:
    passed, failed = 0, []
    for mod_name in MODULES:
        mod = importlib.import_module(mod_name)
        for attr in sorted(dir(mod)):
            if not attr.startswith("test_"):
                continue
            try:
                getattr(mod, attr)()
                print(f"PASS  {mod_name}.{attr}")
                passed += 1
            except Exception:
                print(f"FAIL  {mod_name}.{attr}")
                traceback.print_exc()
                failed.append(f"{mod_name}.{attr}")
    print(f"\n{passed} passed, {len(failed)} failed")
    if failed:
        for f in failed:
            print(f"  FAILED: {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

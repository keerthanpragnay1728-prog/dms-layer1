#!/usr/bin/env python3
"""Magnitude diagnostics for the four pose-sensitive layout checks that
FAILed on real WFLW data (chin-lowest, pair (4,28) and (8,24) height match,
nose-midline). Reports magnitudes instead of pass/fail so "test artefact"
can be separated from "mapping error" BEFORE anything about the mapping moves.

Sections:
  0. Reproduction of the four checks with the exact verify_layout thresholds
     (shared code) — these numbers must match your verify_layout run.
  1. Chin: on failing faces, WHICH contour index is lowest and by how many
     pixels (also normalised by inter-ocular distance); pass rates under
     IOD-relative tolerances. A sub-percent margin at index 15/17 is a tie.
  2. Failure rate of all four checks vs estimated head yaw (contour
     half-width asymmetry, cross-checked against eye-width asymmetry), as a
     table and a PNG figure. Failures concentrating at high yaw = the checks
     are pose-sensitive, not the mapping wrong.
  3. Nose-midline: exact definition of the midline and tolerance, plus the
     offset distribution.
  4. The four checks re-run on the subset with NONE of the six attribute
     flags set (as close to truly frontal as WFLW gets).

The mapping itself is not touched by this script.

Usage (Kaggle):
    python scripts/diagnose_layout_failures.py --config configs/layer1_base.yaml --split test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dms_layer1.config import load_config, require, save_config_snapshot
from dms_layer1.data import frame as ff
from dms_layer1.data import wflw

YAW_BIN_EDGES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 1.001]
REL_CHIN_TOLS = [0.005, 0.01, 0.02, 0.03, 0.05]   # fractions of corner IOD
# dataviz-validated categorical order for a light surface (slots 1-4)
FIG_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

_lines: list[str] = []


def say(text: str = "") -> None:
    print(text)
    _lines.append(text)


def pct(mask: np.ndarray) -> str:
    return f"{100.0 * np.mean(mask):6.2f}%" if len(mask) else "   n/a"


def quantiles(x: np.ndarray, unit: str, scale: float = 1.0) -> str:
    if len(x) == 0:
        return "n/a (no faces)"
    p = np.percentile(x * scale, [10, 25, 50, 75, 90, 100])
    return (f"p10={p[0]:.3f}  p25={p[1]:.3f}  median={p[2]:.3f}  "
            f"p75={p[3]:.3f}  p90={p[4]:.3f}  max={p[5]:.3f} {unit}")


def corr(x: np.ndarray, y: np.ndarray) -> str:
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return "n/a (degenerate)"
    return f"{np.corrcoef(x, y)[0, 1]:+.3f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", default="/kaggle/working/m1_diagnostics")
    ap.add_argument("--synthetic", action="store_true",
                    help="run on generated schematic faces (code smoke test only)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.synthetic:
        import tempfile
        from dms_layer1.data.synthetic import write_synthetic_dataset
        say("SYNTHETIC MODE: schematic faces — smoke test of the code path only.")
        root = write_synthetic_dataset(tempfile.mkdtemp(prefix="wflw_synth_"),
                                       require(cfg, "dataset.attribute_names"),
                                       seed=require(cfg, "seed"))
        cfg["dataset"]["root"] = str(root)

    records, _ = wflw.load_split(cfg, args.split)
    lm = np.stack([r.landmarks98 for r in records])
    frontal = np.array([r.attributes["pose"] == 0 for r in records])
    noflag = np.array([sum(r.attributes.values()) == 0 for r in records])

    fr = ff.FaceFrame(lm)                      # per-face, so masks apply after
    lowest_idx, chin_margin = ff.chin_lowest(fr)
    mism_428 = ff.pair_height_mismatch(fr, 4, 28)
    mism_824 = ff.pair_height_mismatch(fr, 8, 24)
    n_off = ff.nose_offset(fr)
    iod = fr.iod_corner                        # NME normaliser: |wflw60 - wflw72|
    yaw = ff.yaw_proxy_contour(fr)
    yaw2 = ff.yaw_proxy_eyewidth(fr)

    checks = {                                 # True = pass, thresholds shared
        "chin lowest": ff.check_chin(fr),
        "pair (4,28)": ff.check_pair(fr, 4, 28),
        "pair (8,24)": ff.check_pair(fr, 8, 24),
        "nose midline": ff.check_nose_midline(fr),
    }

    say(f"\nSplit '{args.split}': {len(records)} faces | frontal (pose==0): "
        f"{frontal.sum()} | no-flag (all six attributes 0): {noflag.sum()}")

    # ---- 0: reproduction --------------------------------------------------
    say("\n=== 0. Check reproduction (verify_layout thresholds, frontal faces) ===")
    say("These must match your Kaggle verify_layout numbers; if not, stop and report.")
    for name, ok in checks.items():
        say(f"  {name:<14} pass {pct(ok[frontal])}")

    # ---- 1: chin ----------------------------------------------------------
    say("\n=== 1. Chin check: what is actually lowest on failing faces ===")
    fail_c = frontal & ~checks["chin lowest"]
    say(f"failing faces: {fail_c.sum()} of {frontal.sum()} frontal")
    if fail_c.sum():
        idxs, counts = np.unique(lowest_idx[fail_c], return_counts=True)
        order = np.argsort(-counts)
        say("  lowest contour index on failing faces:")
        for i in order:
            say(f"    index {idxs[i]:>2}: {counts[i]:>5}  ({100*counts[i]/fail_c.sum():.1f}%)")
        m_px = chin_margin[fail_c]
        m_iod = chin_margin[fail_c] / iod[fail_c]
        say(f"  margin below point 16, pixels : {quantiles(m_px, 'px')}")
        say(f"  margin / IOD (outer corners)  : {quantiles(m_iod, '', 1.0)}")
        near = np.isin(lowest_idx[fail_c], [15, 17])
        say(f"  failures where lowest is 15 or 17: {pct(near)}")
        for t in (0.01, 0.02, 0.05):
            say(f"  failures with margin <= {t:.0%} of IOD: {pct(m_iod <= t)}")
    say("  Tolerance is IOD-relative (scale-fair) since the milestone-1 verdict;"
        "\n  the originally used absolute 2.0px is shown for comparison:")
    say(f"    tol 2.0px absolute (original): {pct((chin_margin <= 2.0)[frontal])}")
    for t in REL_CHIN_TOLS:
        rel_ok = chin_margin <= t * iod
        mark = "  <- current" if abs(t - ff.CHIN_TOL_IOD) < 1e-9 else ""
        say(f"    tol {t:.1%} of IOD  : {pct(rel_ok[frontal])}{mark}")

    # ---- 2: failure rate vs yaw ------------------------------------------
    say("\n=== 2. Failure rate vs estimated head yaw (frontal faces) ===")
    say("yaw proxy = contour half-width asymmetry (right-left)/(right+left),")
    say("0 = symmetric; ~0.15 is already a clearly turned head.")
    say(f"proxy cross-check, r(contour proxy, eye-width proxy): "
        f"{corr(yaw[frontal], yaw2[frontal])}")
    say("(large |r| = both proxies track the same head-yaw signal; the sign only"
        "\n reflects annotation convention, not correctness)")
    ay = np.abs(yaw)
    say(f"\n  {'|yaw| bin':<12} {'n':>6}  " +
        "  ".join(f"{n:>12}" for n in checks) + "   (fail %)")
    bin_rates = {n: [] for n in checks}
    bin_labels = []
    for lo, hi in zip(YAW_BIN_EDGES[:-1], YAW_BIN_EDGES[1:]):
        in_bin = frontal & (ay >= lo) & (ay < hi)
        label = f"{lo:.2f}-{hi:.2f}" if hi <= 1 else f">={lo:.2f}"
        bin_labels.append(f"{label}\n(n={in_bin.sum()})")
        row = f"  {label:<12} {in_bin.sum():>6}  "
        for name, ok in checks.items():
            rate = 100 * np.mean(~ok[in_bin]) if in_bin.sum() else np.nan
            bin_rates[name].append(rate)
            row += f"{rate:>11.1f}%  " if np.isfinite(rate) else f"{'n/a':>12}  "
        say(row)

    say("\n  median |yaw|: failing vs passing faces per check")
    for name, ok in checks.items():
        f_m = np.median(ay[frontal & ~ok]) if (frontal & ~ok).sum() else np.nan
        p_m = np.median(ay[frontal & ok]) if (frontal & ok).sum() else np.nan
        say(f"    {name:<14} failing {f_m:.3f}  vs passing {p_m:.3f}")

    say("\n  correlation of the raw magnitudes with yaw (frontal faces):")
    say(f"    chin margin/IOD      vs |yaw|: {corr(chin_margin[frontal]/iod[frontal], ay[frontal])}")
    say(f"    (4,28) mismatch/faceh vs |yaw|: {corr((mism_428/fr.face_height)[frontal], ay[frontal])}")
    say(f"    (8,24) mismatch/faceh vs |yaw|: {corr((mism_824/fr.face_height)[frontal], ay[frontal])}")
    say(f"    signed nose offset/IOD vs signed yaw: {corr((n_off/iod)[frontal], yaw[frontal])}")
    fail_n = frontal & ~checks["nose midline"]
    if fail_n.sum():
        agree = np.sign(n_off[fail_n]) == -np.sign(yaw[fail_n])
        say(f"    nose-midline failures with sign(offset) opposite to sign(yaw): "
            f"{pct(agree)}")
        say("    (near 100% OR near 0% = the offset direction is locked to the"
            "\n    yaw direction, i.e. pose-driven; ~50% = unrelated to yaw)")

    fig_path = _plot(bin_labels, bin_rates, Path(args.out_dir))
    say(f"  figure: {fig_path}")

    # ---- 3: nose-midline definition --------------------------------------
    say("\n=== 3. Nose-midline check: definition and tolerance ===")
    say("  midline reference point: eye_mid = midpoint of the two eye-contour")
    say("  centroids (mean of WFLW 60-67, mean of 68-75).")
    say("  offset = projection of (point54 - eye_mid) onto u, the unit vector")
    say("  between the eye centroids. PASS iff |offset| < 0.35 * centroid IOD.")
    ratio = np.median(fr.iod_centroid[frontal] / iod[frontal]) if frontal.sum() else np.nan
    say(f"  centroid IOD is narrower than corner IOD (median ratio {ratio:.3f}),")
    say(f"  so the tolerance is ~{0.35 * ratio:.3f} of the corner-IOD NME normaliser.")
    say(f"  |offset|/IOD, frontal : {quantiles(np.abs(n_off[frontal]) / iod[frontal], '')}")
    say(f"  |offset|/IOD, no-flag : {quantiles(np.abs(n_off[noflag]) / iod[noflag], '')}")

    # ---- 4: no-flag subset ------------------------------------------------
    say("\n=== 4. The four checks on the no-flag subset ===")
    low_yaw = noflag & (ay <= (np.median(ay[noflag]) if noflag.sum() else 0))
    say(f"  {'check':<14} {'frontal(pose==0)':>18} {'no-flag':>12} {'no-flag & low |yaw|':>22}")
    for name, ok in checks.items():
        say(f"  {name:<14} {pct(ok[frontal]):>18} {pct(ok[noflag]):>12} {pct(ok[low_yaw]):>22}")
    say(f"  (n = {frontal.sum()}, {noflag.sum()}, {low_yaw.sum()}; "
        f"low |yaw| = below the no-flag median)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "diagnostics.txt").write_text("\n".join(_lines) + "\n")
    save_config_snapshot(cfg, out_dir)
    say(f"\nreport + config snapshot written to {out_dir}")
    return 0


def _plot(bin_labels: list[str], bin_rates: dict[str, list[float]],
          out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    x = np.arange(len(bin_labels))
    for (name, rates), color in zip(bin_rates.items(), FIG_COLORS):
        ax.plot(x, rates, marker="o", markersize=6, linewidth=2,
                color=color, label=name)
    ax.set_xticks(x, bin_labels, fontsize=8)
    ax.set_xlabel("|yaw proxy| (contour half-width asymmetry), frontal faces")
    ax.set_ylabel("failure rate (%)")
    ax.set_title("Layout-check failure rate vs estimated head yaw")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    path = out_dir / "failure_rate_vs_yaw.png"
    fig.savefig(path)
    plt.close(fig)
    return path


if __name__ == "__main__":
    sys.exit(main())

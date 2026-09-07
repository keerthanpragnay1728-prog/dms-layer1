"""Locating model assets that live outside the repo.

Three kinds of large file are attached rather than committed: the crop
cache, our trained weights, and MediaPipe's model bundle. They follow the
same rule, learned the hard way in milestones 2 and 6: an explicit path is
used as given; otherwise search the likely mounts and accept exactly ONE
unambiguous hit with a printed note; zero or several is an error listing
what was found, never a silent pick between runs.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_INPUT = Path("/kaggle/input")
SEARCH_DEPTH = 5


def search_dirs(extra: list[Path] | None = None) -> list[Path]:
    dirs = [REPO_ROOT / "assets", REPO_ROOT, Path("/kaggle/working")]
    if KAGGLE_INPUT.is_dir():
        dirs.append(KAGGLE_INPUT)
    return [d for d in (dirs + list(extra or [])) if d.is_dir()]


def find_candidates(patterns: list[str], extra: list[Path] | None = None) -> list[Path]:
    hits: set[Path] = set()
    for d in search_dirs(extra):
        for pattern in patterns:
            hits.update(p for p in d.glob(pattern) if p.is_file())
            if d == KAGGLE_INPUT or d.name == "assets":
                for depth in range(1, SEARCH_DEPTH):
                    prefix = "/".join(["*"] * depth)
                    hits.update(p for p in d.glob(f"{prefix}/{pattern}") if p.is_file())
    return sorted(hits)


def download(url: str, dest: Path) -> Path:
    """Fetch an asset once. Kaggle notebooks have internet on by default; if
    a session does not, upload the file as a dataset and point the config at
    it instead."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}\n         -> {dest}")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    return dest


def resolve_asset(explicit: str | Path, patterns: list[str], label: str,
                  url: str | None = None, allow_download: bool = False,
                  download_to: Path | None = None) -> Path:
    """Resolve one attached asset. `explicit` may be a real path or 'auto'."""
    path = Path(explicit)
    if str(explicit) != "auto" and path.is_file():
        return path

    hits = find_candidates(patterns)
    if len(hits) == 1:
        if str(explicit) != "auto":
            print(f"NOTE: {label} '{explicit}' not found; using the single "
                  f"candidate {hits[0]}. Set the config path to silence this.")
        return hits[0]
    if len(hits) > 1:
        listing = "\n".join(f"    {h}" for h in hits)
        raise FileNotFoundError(
            f"Several candidates for {label}; refusing to guess between "
            f"them:\n{listing}\nSet the config path to the exact file.")

    if url and allow_download:
        dest = Path(download_to or (REPO_ROOT / "assets" / Path(url).name))
        return download(url, dest)

    searched = "\n".join(f"    {d}" for d in search_dirs())
    raise FileNotFoundError(
        f"{label} not found (looked for {patterns}).\nSearched:\n{searched}\n"
        + ("Set allow_download: true to fetch it, or " if url else "")
        + "upload it as a Kaggle dataset and set the config path to its mount."
    )

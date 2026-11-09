"""Record (or verify) SHA-256 checksums of the FIRE archive and every file extracted from it.

FIRE is not redistributed here. Put the downloaded archive and its extracted `FIRE/` folder in one directory
(default D:/Downloads/fire-data, or set FIRE_DIR) and run

    python fire/checksums.py            # writes fire/data_manifest.csv (first run) or checks against it
    python fire/checksums.py --write    # overwrite the manifest

Every later script calls `verify()` first and stops if any image, mask or control-point file differs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "data_manifest.csv"


def data_dir() -> Path:
    return Path(os.environ.get("FIRE_DIR", "D:/Downloads/fire-data"))


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _files(root: Path) -> list[Path]:
    fire = root / "FIRE"
    files = sorted(p for p in fire.rglob("*") if p.is_file())
    archives = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in (".zip", ".7z"))
    return archives + files


def current(root: Path) -> list[dict]:
    return [{"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in _files(root)]


def verify(root: Path | None = None) -> None:
    """Raise if any FIRE file listed in the manifest is missing or changed (archives are not required)."""
    root = root or data_dir()
    want = {r["path"]: r["sha256"] for r in csv.DictReader(open(MANIFEST, encoding="utf-8"))
            if r["path"].startswith("FIRE/")}
    bad = [p for p, s in want.items() if not (root / p).is_file() or sha256(root / p) != s]
    if bad:
        raise RuntimeError(f"{len(bad)} FIRE files missing or changed, e.g. {bad[:3]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    root = data_dir()
    rows = current(root)
    if a.write or not MANIFEST.exists():
        with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["path", "bytes", "sha256"])
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {MANIFEST} ({len(rows)} files)")
        return 0
    verify(root)
    print(f"all FIRE files match {MANIFEST.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

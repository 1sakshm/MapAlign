"""Draw the tuning split: 10 FIRE pairs, stratified by category, with a fixed seed.

Only pair IDs are used (the 134 control-point file names), never images or results. The split is written to
fire/tuning_split.csv and committed with the analysis plan; these 10 pairs are excluded from every headline number.

    python fire/split.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from checksums import data_dir

HERE = Path(__file__).resolve().parent
SEED = 42
PER_CATEGORY = {"S": 5, "P": 4, "A": 1}   # 10 pairs, roughly proportional to 71 / 49 / 14


def pair_ids(root: Path | None = None) -> list[str]:
    gt = (root or data_dir()) / "FIRE" / "Ground Truth"
    ids = sorted(p.name.removeprefix("control_points_").removesuffix("_1_2.txt") for p in gt.glob("control_points_*_1_2.txt"))
    if len(ids) != 134:
        raise RuntimeError(f"expected 134 FIRE pairs, found {len(ids)}")
    return ids


def draw(ids: list[str]) -> list[str]:
    rng = np.random.default_rng(SEED)
    out = []
    for cat, k in PER_CATEGORY.items():
        pool = sorted(i for i in ids if i[0] == cat)
        out += sorted(rng.choice(pool, size=k, replace=False).tolist())
    return out


def main() -> None:
    tuning = draw(pair_ids())
    with open(HERE / "tuning_split.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "category"])
        w.writerows([t, t[0]] for t in tuning)
    print("tuning pairs:", " ".join(tuning))


if __name__ == "__main__":
    main()

"""Compare a fresh run's results against a reference copy (e.g. the committed results).

Usage:
    python tools/compare_results.py <reference_results_dir> <new_results_dir> [--tol-m 5]

Reports whether every transform and per-point error is bit-identical, the largest per-point difference in
metres, and whether any method's status changed. Exits 1 only if a status changes or a difference exceeds
--tol-m (cross-platform floating-point differences of a few metres are expected and reported, not hidden).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("reference")
    p.add_argument("new")
    p.add_argument("--tol-m", type=float, default=5.0)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    ref, new = Path(a.reference), Path(a.new)
    report = {"tolerance_m": a.tol_m}
    ok = True

    for m in ("sift", "loftr", "manual"):
        r, n = json.loads((ref / f"{m}_run.json").read_text()), json.loads((new / f"{m}_run.json").read_text())
        same_status = r["status"] == n["status"]
        H_r, H_n = r.get("H_full"), n.get("H_full")
        identical = H_r == H_n
        report[f"{m}_status"] = {"reference": r["status"], "new": n["status"], "same": same_status}
        report[f"{m}_H_bit_identical"] = identical
        if not same_status:
            ok = False

    er = pd.read_csv(ref / "evaluation_errors.csv")
    en = pd.read_csv(new / "evaluation_errors.csv")
    key = ["model", "method", "point_id"]
    j = er.merge(en, on=key, suffixes=("_ref", "_new"))
    d = (j.error_m_new - j.error_m_ref).abs()
    report["per_point_errors_bit_identical"] = bool(((j.error_m_ref == j.error_m_new) |
                                                     (j.error_m_ref.isna() & j.error_m_new.isna())).all())
    report["max_abs_error_difference_m"] = None if d.dropna().empty else round(float(d.max()), 3)
    worst = j.loc[d.idxmax()] if not d.dropna().empty else None
    if worst is not None:
        report["largest_difference_at"] = {k: str(worst[k]) for k in key}
    if report["max_abs_error_difference_m"] is not None and report["max_abs_error_difference_m"] > a.tol_m:
        ok = False

    mr = json.loads((ref / "metrics.json").read_text())["metrics"]
    mn = json.loads((new / "metrics.json").read_text())["metrics"]
    report["metrics"] = {k: {s: (mr[k].get(s), mn[k].get(s)) for s in ("median_m", "rmse_m")} for k in mr}
    report["passed"] = ok
    txt = json.dumps(report, indent=2)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

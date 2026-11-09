"""Run the FIRE benchmark end to end (fire/ANALYSIS_PLAN.md).

    python fire/run.py            # checksums -> tune -> evaluate -> stats -> figures
    python fire/run.py tune       # one stage only (tune | evaluate | stats | figures)

Matching runs in a child process per (method, size, CLAHE) so that a crash (e.g. out of memory) is recorded
instead of ending the run. Matches are cached in .cache/fire/matches/, so every stage can be resumed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark as B  # noqa: E402
from checksums import verify  # noqa: E402

RESULTS = B.HERE / "results"
GRID = {"sift": [728, 1456, 2912], "loftr": [640, 768, 1024]}
THRESH = [1, 2, 3, 5]
DEFAULT = {"sift": 2912, "loftr": 768, "threshold": 3, "clahe": False}    # MapAlign's frozen values (tie-break 2)


def cache_dir(method: str, side: int, clahe: bool) -> Path:
    return B.CACHE / "matches" / f"{method}_{side}_clahe{int(clahe)}"


# ---------------------------------------------------------------- matching (child process)

def match_worker(method: str, side: int, clahe: bool, pids: list[str]) -> None:
    out = cache_dir(method, side, clahe)
    out.mkdir(parents=True, exist_ok=True)
    for pid in pids:
        if (out / f"{pid}.npz").exists() or (out / f"{pid}.failed.json").exists():
            continue
        (out / f"{pid}.started").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
        t0 = time.perf_counter()
        try:
            test, ref = B.images(pid, clahe)
            m = B.match(method, side, test, ref)
            stats = {k: v for k, v in m.stats.items() if isinstance(v, (int, float, str, list, type(None)))}
            stats["seconds"] = round(time.perf_counter() - t0, 2)
            np.savez(out / f"{pid}.npz", pts0=m.pts0, pts1=m.pts1, common_scale=m.stats["common_scale"],
                     stats=json.dumps(stats))
        except Exception as e:  # recorded as a failed pair (plan §7), never retried with other settings
            (out / f"{pid}.failed.json").write_text(json.dumps({"reason": f"matching raised {type(e).__name__}: {e}"}))
        (out / f"{pid}.started").unlink(missing_ok=True)
        print(method, side, int(clahe), pid, f"{time.perf_counter() - t0:.1f}s", flush=True)


def ensure_matches(method: str, side: int, clahe: bool, pids: list[str]) -> str | None:
    """Run the child until every pair has a result. Returns a note if the setting could not run at all."""
    out = cache_dir(method, side, clahe)
    for _ in range(len(pids) + 1):
        todo = [p for p in pids if not (out / f"{p}.npz").exists() and not (out / f"{p}.failed.json").exists()]
        if not todo:
            return None
        cmd = [sys.executable, __file__, "_match", method, str(side), str(int(clahe)), *todo]
        r = subprocess.run(cmd, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        if r.returncode != 0:
            crashed = [p for p in todo if (out / f"{p}.started").exists()]
            for p in crashed:  # the process died inside this pair (e.g. out of memory)
                (out / f"{p}.failed.json").write_text(json.dumps({"reason": f"matcher process crashed (exit {r.returncode})"}))
                (out / f"{p}.started").unlink()
            if not crashed:
                return f"child exited {r.returncode} before matching"
    return None


def load_matches(method: str, side: int, clahe: bool, pid: str) -> dict:
    d = cache_dir(method, side, clahe)
    if (d / f"{pid}.failed.json").exists():
        return {"failed": json.loads((d / f"{pid}.failed.json").read_text())["reason"]}
    z = np.load(d / f"{pid}.npz")
    return {"pts0": z["pts0"], "pts1": z["pts1"], "common_scale": float(z["common_scale"]),
            "stats": json.loads(str(z["stats"]))}


def score(method: str, side: int, clahe: bool, pid: str, model: str, threshold: float) -> dict:
    mm = load_matches(method, side, clahe, pid)
    base = {"pair_id": pid, "category": pid[0], "method": method, "side": side, "clahe": int(clahe),
            "model": model, "threshold_matcher_px": threshold}
    if "failed" in mm:
        return {**base, "raw_matches": 0, "matches": 0, "inliers": 0, "valid": False, "reason": mm["failed"],
                "error_px": np.nan, "match_seconds": None}
    keep = B.feature_mask_filter(mm["pts0"], mm["pts1"])
    ref_pts, test_pts = B.control_points(pid)
    r = B.fit_and_score(mm["pts0"][keep], mm["pts1"][keep], mm["common_scale"], model, threshold, ref_pts, test_pts,
                        (2912, 2912))
    r.pop("H")
    r.pop("ransac_method_effective", None)
    return {**base, "raw_matches": int(len(keep)), **r, "match_seconds": mm["stats"].get("seconds")}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) and np.isfinite(v) else v) for k, v in r.items()})


# ---------------------------------------------------------------- stages

def tune() -> dict:
    split = B.split_of()
    pids = sorted(p for p, s in split.items() if s == "tuning")
    rows, unrunnable = [], []
    for method, clahe in product(("sift", "loftr"), (False, True)):
        for side in GRID[method]:
            note = ensure_matches(method, side, clahe, pids)
            if note:
                unrunnable.append({"method": method, "side": side, "clahe": int(clahe), "note": note})
                continue
            for th in THRESH:
                rows += [score(method, side, clahe, p, "homography", th) for p in pids]
    write_csv(RESULTS / "tuning.csv", rows)

    # A setting where every pair crashed the matcher could not run on this machine (plan §6): dropped, recorded.
    def configs():
        seen = {}
        for r in rows:
            seen.setdefault((r["method"], r["side"], r["clahe"], r["threshold_matcher_px"]), []).append(r)
        return seen
    table = []
    for (method, side, clahe, th), rs in configs().items():
        crashed = all("crashed" in str(r["reason"]) for r in rs)
        e = [r["error_px"] for r in rs]
        finite = np.nan_to_num(np.array(e, float), nan=np.inf)
        table.append({"method": method, "side": side, "clahe": clahe, "threshold": th, "auc": round(B.fire_auc(e), 4),
                      "median_error_px": float(np.median(finite)), "failures": int(np.isnan(np.array(e, float)).sum()),
                      "runnable": not crashed})
        if crashed:
            unrunnable.append({"method": method, "side": side, "clahe": clahe, "threshold": th,
                               "note": "matcher crashed on every tuning pair"})
    ok = [t for t in table if t["runnable"]]

    def closeness(t):  # tie-break 2: distance from MapAlign's frozen values
        return (abs(np.log2(t["side"] / DEFAULT[t["method"]])), abs(t["threshold"] - DEFAULT["threshold"]))

    def best(method, clahe):
        c = [t for t in ok if t["method"] == method and t["clahe"] == clahe]
        top = max(t["auc"] for t in c)
        tied = [t for t in c if t["auc"] >= top - 0.001]
        return sorted(tied, key=lambda t: (t["median_error_px"], *closeness(t)))[0]

    per_clahe = {}
    for clahe in (0, 1):
        b = {m: best(m, clahe) for m in ("sift", "loftr")}
        per_clahe[clahe] = {"best": b, "mean_auc": (b["sift"]["auc"] + b["loftr"]["auc"]) / 2}
    a0, a1 = per_clahe[0]["mean_auc"], per_clahe[1]["mean_auc"]
    clahe = 1 if a1 > a0 + 0.001 else 0   # within 0.001 -> CLAHE off (MapAlign's value)
    choice = {"clahe": bool(clahe), "clahe_mean_tuning_auc": {"off": round(a0, 4), "on": round(a1, 4)},
              "sift": {k: per_clahe[clahe]["best"]["sift"][k] for k in ("side", "threshold", "auc", "median_error_px")},
              "loftr": {k: per_clahe[clahe]["best"]["loftr"][k] for k in ("side", "threshold", "auc", "median_error_px")},
              "tuning_pairs": pids, "unrunnable": unrunnable, "grid_table": table}
    (RESULTS / "tuning_choice.json").write_text(json.dumps(choice, indent=1, default=float))
    print(json.dumps({k: choice[k] for k in ("clahe", "clahe_mean_tuning_auc", "sift", "loftr", "unrunnable")},
                     indent=1, default=float))
    return choice


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "_match":
        _, _, method, side, clahe, *pids = sys.argv
        match_worker(method, int(side), bool(int(clahe)), pids)
        return 0
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="all", choices=["all", "tune", "evaluate", "stats", "figures"])
    a = ap.parse_args()
    verify()
    if a.stage in ("all", "tune"):
        tune()
    return 0


if __name__ == "__main__":
    sys.exit(main())

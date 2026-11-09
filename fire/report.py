"""Statistics (plan §8–§11) and figures (plan §12) from fire/results/per_pair.csv."""

from __future__ import annotations

import csv
import json
import platform
from importlib.metadata import version

import cv2
import numpy as np
from scipy import stats as st

import benchmark as B

RESULTS = B.HERE / "results"
FIGURES = B.HERE / "figures"
N_BOOT = 10_000
CATS = ("S", "P", "A", "all")
LABEL = {"sift": "SIFT", "loftr": "LoFTR", "roma": "RoMa"}

# REMPE (Hernandez-Matas et al., IEEE JBHI 2020), Table VIII, as printed: S, P, A, FIRE
PUBLISHED = {"REMPE": (0.958, 0.542, 0.660, 0.773), "SIFT+WGTM": (0.837, 0.544, 0.407, 0.685),
             "GDB-ICP": (0.814, 0.303, 0.303, 0.576), "Harris-PIIFD": (0.900, 0.090, 0.443, 0.553)}


def load() -> list[dict]:
    rows = list(csv.DictReader(open(RESULTS / "per_pair.csv", encoding="utf-8")))
    for r in rows:
        r["error_px"] = float(r["error_px"]) if r["error_px"] not in ("", "nan", "NaN") else np.nan
    return rows


def errors(rows, method, model, cat, split=("evaluation",)) -> dict[str, float]:
    return {r["pair_id"]: r["error_px"] for r in rows if r["method"] == method and r["model"] == model
            and r["split"] in split and (cat == "all" or r["category"] == cat)}


def _median_inf(e: np.ndarray) -> float:
    return float(np.median(np.nan_to_num(e, nan=np.inf)))


def _num(x: float):
    return None if not np.isfinite(x) else round(float(x), 4)


def describe(e: dict[str, float], rng: np.random.Generator) -> dict:
    v = np.array(list(e.values()), float)
    ok = np.isfinite(v)
    idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    boot_auc = np.array([B.fire_auc(v[i]) for i in idx])
    boot_med = np.array([_median_inf(v[i]) for i in idx])
    lo, hi = np.percentile(boot_med, [2.5, 97.5]) if np.isfinite(boot_med).all() else (np.nan, np.nan)
    return {"n_pairs": int(len(v)), "failures": int((~ok).sum()), "auc": round(B.fire_auc(v), 4),
            "auc_ci95": [round(float(x), 4) for x in np.percentile(boot_auc, [2.5, 97.5])],
            "median_error_px": _num(_median_inf(v)),
            "median_error_ci95": [_num(lo), _num(hi)],
            "median_ci_note": None if np.isfinite(boot_med).all() else
            "some bootstrap resamples have a failed median (+inf); CI bound reported as null",
            "mean_error_px_successful_only": _num(v[ok].mean()) if ok.any() else None}


def paired(a: dict, b: dict, rng: np.random.Generator) -> dict:
    """a - b on the same pairs (plan §9): Wilcoxon on per-pair error, failures ranked worst, both-failed -> 0."""
    pids = sorted(set(a) & set(b))
    x, y = np.array([a[p] for p in pids]), np.array([b[p] for p in pids])
    finite = np.concatenate([x[np.isfinite(x)], y[np.isfinite(y)]])
    cap = (finite.max() if len(finite) else 0.0) * 10 + 1e3     # larger than every finite error; ranks only
    xc, yc = np.where(np.isfinite(x), x, cap), np.where(np.isfinite(y), y, cap)
    d = xc - yc
    out = {"n_pairs": len(pids), "a_better": int((d < 0).sum()), "b_better": int((d > 0).sum()),
           "ties_or_both_failed": int((d == 0).sum()), "median_diff_px": round(float(np.median(d)), 4)
           if np.all(np.isfinite(x) == np.isfinite(y)) else None}
    if (d != 0).sum() >= 1:
        exact = (d != 0).sum() <= 25 and len(set(np.abs(d[d != 0]))) == (d != 0).sum()
        w = st.wilcoxon(xc, yc, zero_method="wilcox", alternative="two-sided", method="exact" if exact else "auto")
        out.update(wilcoxon_statistic=float(w.statistic), p_two_sided=float(f"{w.pvalue:.3g}"),
                   wilcoxon_method="exact" if exact else "normal approximation (scipy 'auto', n > 25)")
    idx = rng.integers(0, len(pids), size=(N_BOOT, len(pids)))
    diff = np.array([B.fire_auc(x[i]) - B.fire_auc(y[i]) for i in idx])
    out.update(auc_diff=round(B.fire_auc(x) - B.fire_auc(y), 4),
               auc_diff_ci95=[round(float(q), 4) for q in np.percentile(diff, [2.5, 97.5])])
    return out


def holm(ps: dict[str, float]) -> dict[str, float]:
    keys = sorted(ps, key=ps.get)
    m, out, run = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * ps[k]))
        out[k] = float(f"{run:.3g}")
    return out


def stats() -> dict:
    rows = load()
    methods = [m for m in ("sift", "loftr", "roma") if any(r["method"] == m for r in rows)]
    rng = np.random.default_rng(B.SEED)
    summary = {"by_method": {}, "literature_row_all_134": {}, "published_REMPE_TableVIII": PUBLISHED}
    for method in methods:
        for model in ("homography", "affine"):
            summary["by_method"].setdefault(method, {})[model] = {
                c: describe(errors(rows, method, model, c), rng) for c in CATS}
        summary["literature_row_all_134"][method] = {
            c: round(B.fire_auc(list(errors(rows, method, "homography", c, ("evaluation", "tuning")).values())), 4)
            for c in CATS}
    comp = {}
    for c in CATS:
        comp[c] = paired(errors(rows, "loftr", "homography", c), errors(rows, "sift", "homography", c), rng)
    summary["primary_loftr_minus_sift_A"] = comp["A"]
    sec = {c: comp[c] for c in ("S", "P", "all")}
    adj = holm({c: v["p_two_sided"] for c, v in sec.items() if "p_two_sided" in v})
    for c in sec:
        sec[c]["p_holm"] = adj.get(c)
    summary["secondary_loftr_minus_sift"] = sec
    choice = json.loads((RESULTS / "tuning_choice.json").read_text())
    summary["frozen_settings"] = {k: choice[k] for k in ("clahe", "sift", "loftr")}
    summary["environment"] = {"python": platform.python_version(), "platform": platform.platform(),
                              **{p: version(p) for p in ("numpy", "scipy", "opencv-python-headless", "torch",
                                                         "kornia")},
                              "seeds": {"split_and_bootstrap": B.SEED, "loftr_torch": 0,
                                        "opencv_usac": "fixed internal state"}, "n_bootstrap": N_BOOT}
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


# ---------------------------------------------------------------- figures

def curves(rows) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.linspace(0, 25, 251)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    methods = [m for m in ("sift", "loftr", "roma") if any(r["method"] == m for r in rows)]
    for ax, c in zip(axes, CATS):
        for m in methods:
            e = errors(rows, m, "homography", c)
            ax.plot(t, 100 * B.success_curve(list(e.values()), t), label=f"{LABEL[m]} (AUC {B.fire_auc(list(e.values())):.3f})")
        ax.set_title({"all": "All evaluation pairs"}.get(c, f"Category {c}") + f" (n={len(e)})")
        ax.set_xlabel("Error threshold (px)")
        ax.set_xlim(0, 25)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("% pairs registered")
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "success_curves.png", dpi=150)
    plt.close(fig)


def _refit(method: str, pid: str):
    from run import cache_dir, load_matches  # noqa: PLC0415
    choice = json.loads((RESULTS / "tuning_choice.json").read_text())
    side = choice[method]["side"] if method in choice else 864
    th = choice[method]["threshold"] if method in choice else 3
    mm = load_matches(method, side, choice["clahe"], pid)
    if "failed" in mm:
        return None, choice["clahe"]
    keep = B.feature_mask_filter(mm["pts0"], mm["pts1"])
    ref_pts, test_pts = B.control_points(pid)
    r = B.fit_and_score(mm["pts0"][keep], mm["pts1"][keep], mm["common_scale"], "homography", th, ref_pts,
                        test_pts, (2912, 2912))
    return r["H"], choice["clahe"]


def overlay(method: str, pid: str, row: dict, kind: str) -> None:
    H, clahe = _refit(method, pid)
    test, ref = B.images(pid, clahe)
    ref_pts, test_pts = B.control_points(pid)
    if H is not None:
        warped = cv2.warpPerspective(test, H, (ref.shape[1], ref.shape[0]))
        img = cv2.merge([ref, warped, ref])           # BGR: reference magenta, warped test green
        pred = B.apply_h(H, test_pts)
    else:
        img = np.hstack([cv2.merge([ref] * 3), cv2.merge([test] * 3)])
        pred = None
    for p in ref_pts:
        cv2.circle(img, tuple(int(round(v)) for v in p), 18, (255, 255, 255), 4)
    if pred is not None:
        for p, q in zip(ref_pts, pred):
            cv2.drawMarker(img, tuple(int(round(v)) for v in q), (0, 255, 255), cv2.MARKER_CROSS, 36, 5)
            cv2.line(img, tuple(int(round(v)) for v in p), tuple(int(round(v)) for v in q), (0, 255, 255), 3)
    s = 1024 / img.shape[1] if H is not None else 1600 / img.shape[1]
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    err = row["error_px"]
    text = (f"{LABEL[method]} {pid} ({kind}): mean error {err:.1f} px" if np.isfinite(err)
            else f"{LABEL[method]} {pid} ({kind}): failed - {row['reason']}")
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(FIGURES / f"overlay_{method}_{kind}_{pid}.png"), img)


def figures() -> list[str]:
    rows = load()
    FIGURES.mkdir(exist_ok=True)
    curves(rows)
    chosen = []
    for method in ("sift", "loftr"):
        ev = [r for r in rows if r["method"] == method and r["model"] == "homography" and r["split"] == "evaluation"]
        ok = [r for r in ev if np.isfinite(r["error_px"])]
        med = float(np.median([r["error_px"] for r in ok]))
        good = min(ok, key=lambda r: (abs(r["error_px"] - med), r["pair_id"]))
        failed = sorted((r for r in ev if not np.isfinite(r["error_px"])), key=lambda r: r["pair_id"])
        bad = failed[0] if failed else max(ok, key=lambda r: r["error_px"])
        overlay(method, good["pair_id"], good, "success")
        overlay(method, bad["pair_id"], bad, "failure" if failed else "worst")
        chosen += [f"{method}: success {good['pair_id']}, {'failure' if failed else 'worst'} {bad['pair_id']}"]
    return chosen

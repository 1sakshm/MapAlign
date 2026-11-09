"""The three steps every MapAlign experiment shares: load a matcher, fit a transform, score held-out points.

The 1926 map experiment (pipeline, evaluate, extensions) and the art-history experiment (styles/) both call
these functions, so a matcher or a metric is defined once and every experiment uses the same version.

    m = get_matcher("loftr", params, device)(gray0, gray1)       # -> MatchResult in the caller's pixel frame
    tr = fit_transform(m, size0, size1, transform_cfg, seed)       # -> TransformResult (MAGSAC++ by default)
    e = positional_errors(truth, apply_h(tr.matrix, held_out_px))  # -> per-point error, in truth's units

Matchers return points in the coordinates of the arrays they were given, plus stats["common_scale"] when they
ran on shrunken copies, so that the inlier threshold can be set at the resolution where the points were found.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from affine import Affine
from scipy import stats

from mapalign.estimate_transform import TransformResult, apply_h, estimate_transform, estimator_params


@dataclass
class MatchResult:
    method: str
    pts0: np.ndarray                     # (N, 2) float, image0 working px
    pts1: np.ndarray                     # (N, 2) float, image1 working px
    scores: np.ndarray                   # (N,) per-match score (SIFT: ratio d1/d2, lower = better)
    stats: dict = field(default_factory=dict)


# ---------------------------------------------------------------- 1. load a matcher

MATCHERS = ("sift", "loftr", "disk_lightglue", "aliked_lightglue")


def get_matcher(name: str, params: dict | None = None, device: str = "auto") -> Callable[[np.ndarray, np.ndarray], MatchResult]:
    """A function (gray0, gray1) -> MatchResult. Heavy libraries are imported only when that matcher is used."""
    params = dict(params or {})
    if name == "sift":
        from mapalign.match_sift import match_sift
        return lambda g0, g1: match_sift(g0, g1, params)
    if name == "loftr":
        from mapalign.match_loftr import match_loftr
        return lambda g0, g1: match_loftr(g0, g1, params, device)
    if name.endswith("_lightglue"):
        from mapalign.match_lightglue import match_lightglue
        p = {**params, "features": name.removesuffix("_lightglue")}
        return lambda g0, g1: match_lightglue(g0, g1, p, device)
    raise ValueError(f"unknown matcher {name!r}; expected one of {MATCHERS}")


# ---------------------------------------------------------------- 2. fit a transform

def fit_transform(m: MatchResult, size0: tuple[int, int], size1: tuple[int, int], transform_cfg: dict,
                  seed: int) -> TransformResult:
    """Robust fit of image0 -> image1 from the matches, threshold set at the matcher's own resolution."""
    params = estimator_params(transform_cfg, m.stats.get("common_scale", 1.0))
    return estimate_transform(m.pts0, m.pts1, size0, size1, params, seed)


# ---------------------------------------------------------------- 3. score held-out points

def positional_errors(true_en: np.ndarray, pred_en: np.ndarray) -> np.ndarray:
    """e_i = sqrt((E_i - Ehat_i)^2 + (N_i - Nhat_i)^2), in the units of the inputs (metres here)."""
    true_en, pred_en = np.asarray(true_en, float), np.asarray(pred_en, float)
    if true_en.shape != pred_en.shape or true_en.ndim != 2 or true_en.shape[1] != 2:
        raise ValueError(f"expected matching (N, 2) arrays, got {true_en.shape} and {pred_en.shape}")
    return np.hypot(true_en[:, 0] - pred_en[:, 0], true_en[:, 1] - pred_en[:, 1])


def predict(H: np.ndarray | None, gt: Affine | None, px: np.ndarray) -> np.ndarray:
    """Historical pixels -> map coordinates through H and the reference geotransform; NaN when there is no fit."""
    from mapalign.georeference import ref_px_to_map

    if H is None or gt is None:
        return np.full((len(px), 2), np.nan)
    return ref_px_to_map(apply_h(H, px), gt)


def summarize_errors(e: np.ndarray) -> dict:
    """Summary statistics over valid (finite) errors. Population std (ddof=0) is reported as `std_m`."""
    e = np.asarray(e, float)
    valid = e[np.isfinite(e)]
    if len(valid) == 0:
        return {"n_valid": 0, "mean_m": None, "median_m": None, "rmse_m": None, "max_m": None, "min_m": None,
                "std_m": None, "p90_m": None}
    r = lambda v: round(float(v), 2)  # noqa: E731
    return {"n_valid": int(len(valid)), "mean_m": r(valid.mean()), "median_m": r(np.median(valid)),
            "rmse_m": r(np.sqrt(np.mean(valid ** 2))), "max_m": r(valid.max()), "min_m": r(valid.min()),
            "std_m": r(valid.std(ddof=0)), "p90_m": r(np.percentile(valid, 90))}


def paired_comparison(a: np.ndarray, b: np.ndarray, seed: int, n_boot: int = 20000) -> dict:
    """Paired comparison of per-point errors a vs b (same points).

    Wilcoxon signed-rank test (exact, two-sided) on d = a - b, and a percentile bootstrap 95% CI
    for the median of d. With n ~ 12, both are low-power and descriptive only.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    out = {"n_pairs": int(ok.sum()), "median_diff_m": round(float(np.median(d)), 2) if len(d) else None,
           "a_better_count": int((d < 0).sum()), "b_better_count": int((d > 0).sum())}
    if len(d) >= 5:
        w = stats.wilcoxon(d, alternative="two-sided", method="exact" if len(d) <= 25 else "auto")
        rng = np.random.default_rng(seed)
        boots = np.median(rng.choice(d, size=(n_boot, len(d)), replace=True), axis=1)
        out.update(wilcoxon_statistic=float(w.statistic), wilcoxon_p_two_sided=round(float(w.pvalue), 4),
                   bootstrap_ci95_median_diff_m=[round(float(np.percentile(boots, 2.5)), 2),
                                                 round(float(np.percentile(boots, 97.5)), 2)])
    return out

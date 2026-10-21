"""Method extensions added after the held-out evaluation (post hoc, secondary). Nothing here changes a frozen result.

Outputs go to results/extensions/.

Usage:
    python -m mapalign.extensions all
    python -m mapalign.extensions lightglue       # a fourth method: DISK or ALIKED keypoints + LightGlue + the frozen MAGSAC++
    python -m mapalign.extensions nonprojective   # polynomial and thin-plate-spline models instead of a homography

Both follow the protocol used for the original methods. Settings are chosen with development data only (the
graticule prior, the fitting points or the correspondences themselves), with the selection rule written down
before the held-out points are read, and then scored once on the same 12 locked evaluation points.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np
from scipy.interpolate import RBFInterpolator

from mapalign.config import resolve_path
from mapalign.diagnostics import aligned_tile_test, prior_consistency, prior_homography_work
from mapalign.estimate_transform import apply_h, compose_full_resolution, estimate_transform, estimator_params
from mapalign.evaluate import paired_comparison, positional_errors, predict
from mapalign.georeference import fit_gcp_transform, map_to_ref_px, ref_px_to_map
from mapalign.io import write_json
from mapalign.robustness import Ctx, _summ

log = logging.getLogger("mapalign.extensions")
OUT = resolve_path("results/extensions")


def _prepared(c: Ctx):
    from mapalign.preprocess import prepare_historical, prepare_reference

    if not hasattr(c, "_prep"):
        c._prep = (prepare_historical(c.cfg), prepare_reference(c.cfg))
    return c._prep


# ---------------------------------------------------------------- 1. DISK + LightGlue

LG_SIZES = (768, 1024, 1536, 2048)  # larger inputs risk exhausting the 15 GB CPU machine (DISK runs a full-resolution U-Net)


def lightglue(c: Ctx):
    return {f: _lightglue(c, f) for f in ("disk", "aliked")}


def _lightglue(c: Ctx, feat: str):
    """Fourth method. Input size chosen by the rule used for LoFTR in Phase 4 (fixed in advance): the most matches
    within 1 km of the graticule prior. Everything after matching is the frozen pipeline: MAGSAC++ with the same
    3 px threshold at matcher resolution, 1.2 M iterations, the same validity checks."""
    from mapalign.match_lightglue import lightglue_tile_matcher, match_lightglue
    from mapalign.visualize import plot_matches, plot_overlay

    OUT.mkdir(parents=True, exist_ok=True)
    hist, ref = _prepared(c)
    Hp = prior_homography_work(c.cfg, hist, ref)
    res_m = c.cfg["preprocessing"]["working_resolution_m"]
    size0 = (hist.gray.shape[1], hist.gray.shape[0])
    size1 = (ref.gray.shape[1], ref.gray.shape[0])
    sweep, runs = [], {}
    for side in LG_SIZES:
        row = {"max_side_px": side}
        try:
            m = match_lightglue(hist.gray, ref.gray, {"features": feat, "max_side_px": side}, c.cfg["compute"]["device"])
            pc = prior_consistency(m.pts0, m.pts1, Hp, res_m)
            row.update(matches=len(m.pts0), within_1km_of_prior=pc["matches_within"],
                       effective_resolution_m=round(res_m / m.stats["common_scale"], 1),
                       inference_s=m.stats["inference_s"])
            runs[side] = m
        except RuntimeError as e:  # out of memory is recorded, not hidden
            row["error"] = str(e).splitlines()[0][:200]
        sweep.append(row)
        log.info("%s+lightglue sweep %s", feat, row)
    ok = [r for r in sweep if "error" not in r]
    best = max(ok, key=lambda r: (r["within_1km_of_prior"], -r["max_side_px"]))
    m = runs[best["max_side_px"]]
    params = estimator_params(c.cfg["transform"], m.stats["common_scale"])
    t0 = time.perf_counter()
    tr = estimate_transform(m.pts0, m.pts1, size0, size1, params, c.seed)
    t_est = time.perf_counter() - t0
    tile = aligned_tile_test(hist, ref, Hp, lightglue_tile_matcher({"features": feat}, c.cfg["compute"]["device"]))
    out = {"purpose": f"fourth method (post hoc, secondary): {feat.upper()} + LightGlue matches into the frozen MAGSAC++ pipeline",
           "selection_rule": "input size with the most matches within 1 km of the graticule prior (the Phase 4 LoFTR rule)",
           "sweep": sweep, "selected_max_side_px": best["max_side_px"], "matching": m.stats,
           "estimation": tr.stats, "estimation_s": round(t_est, 3),
           "prior_consistency": prior_consistency(m.pts0, m.pts1, Hp, res_m, tr.inlier_mask),
           "aligned_tile_upper_bound": tile, "status": "valid" if tr.ok else "failed"}
    np.savez_compressed(OUT / f"{feat}_lightglue_correspondences.npz", pts0_work=m.pts0, pts1_work=m.pts1,
                        scores=m.scores, inlier_mask=tr.inlier_mask)
    plot_matches(hist.color, ref.color, m.pts0, m.pts1, tr.inlier_mask,
                 f"{feat.upper()} + LightGlue + MAGSAC++ (max side {best['max_side_px']} px)", OUT / f"{feat}_lightglue_matches.png", seed=c.seed)
    if tr.ok:
        H = compose_full_resolution(tr.matrix, hist.full_to_work, ref.full_to_work)
        e = positional_errors(c.truth, predict(H, ref.geotransform, c.px))
        out["H_full"] = H.tolist()
        out["heldout"] = _summ(e)
        out["per_point_m"] = dict(zip(c.ids, np.round(e, 1).tolist()))
        out["paired_vs_loftr"] = paired_comparison(c.errors("homography", "loftr"), e, c.seed)
        out["paired_vs_manual"] = paired_comparison(c.errors("homography", "manual"), e, c.seed)
        plot_overlay(hist.color, ref.color, tr.matrix, f"{feat.upper()} + LightGlue overlay (post hoc)", OUT / f"overlay_{feat}_lightglue.png")
    else:
        out["heldout"] = {"n_valid": 0, "note": "no valid transform, scored as failed like SIFT"}
    log.info("%s+lightglue: %s", feat, {k: out[k] for k in ("selected_max_side_px", "status", "heldout")})
    write_json(out, OUT / f"{feat}_lightglue.json")
    return out


# ---------------------------------------------------------------- 2. non-projective models

def _poly_terms(p: np.ndarray, deg: int) -> np.ndarray:
    x, y = p[:, 0], p[:, 1]
    return np.column_stack([x ** (i - j) * y ** j for i in range(deg + 1) for j in range(i + 1)])


class _Poly:
    def __init__(self, src, dst, deg):
        # normalise for conditioning
        self.mu, self.sd, self.deg = src.mean(0), src.std(0), deg
        A = _poly_terms((src - self.mu) / self.sd, deg)
        if len(src) < A.shape[1]:
            raise ValueError(f"polynomial degree {deg} needs >= {A.shape[1]} points, got {len(src)}")
        self.coef, *_ = np.linalg.lstsq(A, dst, rcond=None)

    def __call__(self, p):
        return _poly_terms((p - self.mu) / self.sd, self.deg) @ self.coef


class _H:
    def __init__(self, src, dst, model="homography"):
        self.H = fit_gcp_transform(src, dst, model)

    def __call__(self, p):
        return apply_h(self.H, p)


class _TPS:
    def __init__(self, src, dst, smoothing):
        self.mu, self.sd = src.mean(0), src.std().item()
        self.f = RBFInterpolator((src - self.mu) / self.sd, dst, kernel="thin_plate_spline",
                                 smoothing=smoothing, degree=1)

    def __call__(self, p):
        return self.f((p - self.mu) / self.sd)


def _model(name, src, dst):
    kind, _, arg = name.partition(":")
    if kind in ("affine", "homography"):
        return _H(src, dst, kind)
    if kind == "poly":
        return _Poly(src, dst, int(arg))
    if kind == "tps":
        return _TPS(src, dst, float(arg))
    raise ValueError(name)


def _block_cv(src, dst, name, res_m, k=4):
    """Spatial-block cross-validation RMSE in metres: a k x k grid of blocks, each left out in turn."""
    bx = np.minimum((src[:, 0] - src[:, 0].min()) / np.ptp(src[:, 0]) * k, k - 1).astype(int)
    by = np.minimum((src[:, 1] - src[:, 1].min()) / np.ptp(src[:, 1]) * k, k - 1).astype(int)
    blk = by * k + bx
    sq = []
    for b in np.unique(blk):
        te = blk == b
        try:
            f = _model(name, src[~te], dst[~te])
            sq.append(np.sum((f(src[te]) - dst[te]) ** 2, axis=1))
        except (ValueError, np.linalg.LinAlgError):
            return np.nan
    return float(np.sqrt(np.mean(np.concatenate(sq))) * res_m)


def _loo(src, dst_m, gt, name):
    dst = map_to_ref_px(dst_m, gt)
    e = []
    for i in range(len(src)):
        k = np.arange(len(src)) != i
        try:
            f = _model(name, src[k], dst[k])
            e.append(float(np.hypot(*(ref_px_to_map(f(src[i:i + 1]), gt)[0] - dst_m[i]))))
        except (ValueError, np.linalg.LinAlgError):
            return np.nan
    return float(np.sqrt(np.mean(np.square(e))))


TPS_SMOOTHING = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def nonprojective(c: Ctx):
    """Replace the single homography with second- and third-order polynomials and thin-plate splines.

    (a) On LoFTR's 390 MAGSAC++ inlier correspondences. Model and TPS smoothing chosen by 4 x 4 spatial-block
        cross-validation on the correspondences (development data).
    (b) On Method C's 7 fitting points. Model chosen by leave-one-out on the fitting points.
    Held-out errors are reported for every candidate, and the selected one is compared with the homography."""
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"purpose": "non-projective transform models (post hoc, secondary)",
           "selection": {"loftr": "lowest 4x4 spatial-block CV RMSE on the inlier correspondences",
                         "manual": "lowest leave-one-out RMSE on the 7 fitting points"}}

    # (a) LoFTR correspondences, working px; held-out points mapped into the same frames
    r = c.runs["loftr"]
    F0, F1 = np.array(r["historical_full_to_work"]), np.array(r["reference_full_to_work"])
    res_m = float(r["working_resolution_m"]["reference"]) if isinstance(r["working_resolution_m"], dict) else 10.0
    d = c.corr("loftr")
    inl = d["inlier_mask"].astype(bool)
    src, dst = d["pts0_work"][inl], d["pts1_work"][inl]
    px_work = apply_h(F0, c.px)

    def heldout(f):
        return positional_errors(c.truth, ref_px_to_map(apply_h(np.linalg.inv(F1), f(px_work)), c.gt))

    names = ["affine", "homography", "poly:2", "poly:3"] + [f"tps:{s:g}" for s in TPS_SMOOTHING]
    rows = []
    for n in names:
        f = _model(n, src, dst)
        e = heldout(f)
        rows.append({"model": n, "cv_rmse_m": round(_block_cv(src, dst, n, res_m), 2),
                     "fit_rmse_m": round(float(np.sqrt(np.mean(np.sum((f(src) - dst) ** 2, axis=1))) * res_m), 2),
                     **_summ(e), "_e": e})
    best = min((x for x in rows if np.isfinite(x["cv_rmse_m"])), key=lambda x: x["cv_rmse_m"])
    hom = next(x for x in rows if x["model"] == "homography")
    out["loftr"] = {"n_correspondences": int(inl.sum()), "working_resolution_m": res_m,
                    "note": "inliers were selected by the homography MAGSAC++ run (3 px at ~45 m/px), so a non-projective "
                            "model can only redistribute residuals within that tolerance",
                    "rows": [{k: v for k, v in x.items() if k != "_e"} for x in rows],
                    "selected_model": best["model"],
                    "selected_vs_homography": paired_comparison(hom["_e"], best["_e"], c.seed),
                    "per_point_m": {x["model"]: dict(zip(c.ids, np.round(x["_e"], 1).tolist())) for x in rows}}

    # (b) Method C fitting points, full-resolution px
    used = c.fit[c.fit.point_id.isin(c.runs["manual"]["fitting_point_ids"])]
    s = used[["historical_x_px", "historical_y_px"]].to_numpy(float)
    dm = used[["true_easting_m", "true_northing_m"]].to_numpy(float)
    dp = map_to_ref_px(dm, c.gt)
    mrows = []
    for n in ["affine", "homography", "poly:2", "tps:0", "tps:0.01", "tps:0.1", "tps:1"]:
        try:
            f = _model(n, s, dp)
        except ValueError as ex:
            mrows.append({"model": n, "error": str(ex)})
            continue
        e = positional_errors(c.truth, ref_px_to_map(f(c.px), c.gt))
        loo = _loo(s, dm, c.gt, n)
        mrows.append({"model": n, "loo_rmse_m": None if not np.isfinite(loo) else round(loo, 2), **_summ(e), "_e": e})
    ok = [x for x in mrows if x.get("loo_rmse_m") is not None]
    mbest = min(ok, key=lambda x: x["loo_rmse_m"])
    mhom = next(x for x in mrows if x["model"] == "homography")
    out["manual"] = {"n_points": len(s),
                     "note": "poly:2 has 6 coefficients per axis, so with 7 points each leave-one-out fit is exactly "
                             "determined; it is reported for completeness, not as a credible model",
                     "rows": [{k: v for k, v in x.items() if k != "_e"} for x in mrows],
                     "selected_model": mbest["model"],
                     "selected_vs_homography": paired_comparison(mhom["_e"], mbest["_e"], c.seed)}
    out["selected_loftr_vs_manual_homography"] = paired_comparison(c.errors("homography", "manual"), best["_e"], c.seed)
    log.info("nonprojective: loftr %s -> %s | manual %s -> %s", best["model"], _summ(best["_e"]),
             mbest["model"], _summ(mbest["_e"]))
    write_json(out, OUT / "nonprojective.json")
    return out


TASKS = {"lightglue": lightglue, "nonprojective": nonprojective}


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["all", *TASKS])
    args = p.parse_args(argv)
    logging.getLogger("mapalign.estimate_transform").setLevel(logging.WARNING)
    c = Ctx()
    for name in (TASKS if args.what == "all" else [args.what]):
        t0 = time.perf_counter()
        TASKS[name](c)
        log.info("== %s done in %.1f s", name, time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())

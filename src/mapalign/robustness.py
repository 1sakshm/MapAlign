"""Post-hoc robustness and sensitivity analyses (none of them changes a frozen result).

All outputs go to results/robustness/. The primary results in results/metrics.json are not modified.
These analyses were added after the held-out evaluation, so they are reported as sensitivity checks;
no parameter was re-selected from them.

Usage:
    python -m mapalign.robustness all            # everything below, in order
    python -m mapalign.robustness seeds          # RANSAC seed variability (200 seeds)
    python -m mapalign.robustness confidence     # LoFTR confidence threshold sweep
    python -m mapalign.robustness intervals      # bootstrap CIs for Table-1 metrics
    python -m mapalign.robustness spatial        # Moran's I + spatial cluster bootstrap
    python -m mapalign.robustness landmarks      # exclude landmark-centroid points
    python -m mapalign.robustness shared_source  # leave-neighbourhood-out LoFTR (OSM shared-source check)
    python -m mapalign.robustness style          # OSM rendering-style variants (re-runs LoFTR)
    python -m mapalign.robustness projection     # polyconic vs UTM: error floor of a homography
    python -m mapalign.robustness scan           # neatline straightness, scale anisotropy, JPEG quality
    python -m mapalign.robustness datum          # bootstrap CI of the graticule offset, two point sets
    python -m mapalign.robustness timing         # repeated runtime measurements
    python -m mapalign.robustness picker --repick <csv> --fitting <csv>   # independent human re-pick
    python -m mapalign.robustness extra_points --points <csv>             # supplementary points in empty cells
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from affine import Affine
from pyproj import Transformer
from scipy import stats

from mapalign.config import load_config, resolve_path
from mapalign.estimate_transform import apply_h, compose_full_resolution, estimate_transform, estimator_params
from mapalign.evaluate import (load_points, paired_comparison, positional_errors, predict, summarize_errors,
                               verify_lock)
from mapalign.georeference import graticule_px_to_map, map_to_ref_px, ref_px_to_map
from mapalign.io import read_json, write_json

log = logging.getLogger("mapalign.robustness")
OUT = resolve_path("results/robustness")


# ---------------------------------------------------------------- shared context

class Ctx:
    def __init__(self):
        self.cfg = load_config()
        if not self.cfg["experiment"]["frozen"]:
            raise RuntimeError("configuration must be frozen")
        verify_lock(self.cfg)
        self.seed = self.cfg["experiment"]["random_seed"]
        self.ev, self.fit = load_points(self.cfg)
        self.ids = self.ev.point_id.tolist()
        self.px = self.ev[["historical_x_px", "historical_y_px"]].to_numpy(float)
        self.truth = self.ev[["true_easting_m", "true_northing_m"]].to_numpy(float)
        self.runs = {m: read_json(resolve_path(f"results/{m}_run.json")) for m in ("sift", "loftr", "manual")}
        self.gt = Affine.from_gdal(*self.runs["loftr"]["reference_geotransform_gdal"])
        self.err = pd.read_csv(resolve_path("results/evaluation_errors.csv"))
        OUT.mkdir(parents=True, exist_ok=True)

    def corr(self, m):
        return np.load(resolve_path(f"results/{m}_correspondences.npz"))

    def heldout_from_work(self, H_work, m):
        r = self.runs[m]
        if H_work is None:
            return np.full(len(self.px), np.nan)
        H = compose_full_resolution(H_work, np.array(r["historical_full_to_work"]), np.array(r["reference_full_to_work"]))
        return positional_errors(self.truth, predict(H, self.gt, self.px))

    def params(self, m, **over):
        scale = self.runs[m]["matching"].get("common_scale", 1.0)
        return estimator_params({**self.cfg["transform"], **over}, scale)

    def sizes(self, m):
        w = self.runs[m]["working_sizes"]
        return tuple(w["historical"]), tuple(w["reference"])

    def errors(self, model, method):
        e = self.err[(self.err.model == model) & (self.err.method == method)].set_index("point_id").reindex(self.ids)
        return e.error_m.to_numpy(float)


def _summ(e):
    s = summarize_errors(e)
    return {k: s[k] for k in ("n_valid", "median_m", "mean_m", "rmse_m", "max_m")}


# ---------------------------------------------------------------- 1. RANSAC seeds

def seeds(c: Ctx, n=200):
    out = {"purpose": "variability of the MAGSAC++ outcome over random seeds (explicit cv2.UsacParams.randomGeneratorState "
                      "0..n-1); frozen settings otherwise. Seed 0 finds the published run's inlier set exactly; its polished matrix differs by < 1 m.",
           "n_seeds": n, "published_run_effective_seed": 0,
           "note": "cv2.setRNGSeed(42) in the pipeline has no effect on OpenCV's USAC; the published run used state 0"}
    for m in ("loftr", "sift"):
        d = c.corr(m)
        s0, s1 = c.sizes(m)
        rows = []
        for s in range(n):
            tr = estimate_transform(d["pts0_work"], d["pts1_work"], s0, s1, c.params(m), c.seed, usac_seed=s)
            e = c.heldout_from_work(tr.matrix if tr.ok else None, m)
            rows.append({"seed": s, "valid": tr.ok, "inliers": tr.stats.get("inliers", 0), **_summ(e)})
        df = pd.DataFrame(rows)
        df.to_csv(OUT / f"seeds_{m}.csv", index=False)
        v = df[df.valid]
        out[m] = {"valid_fraction": round(float(df.valid.mean()), 3),
                  "inliers": {"min": int(df.inliers.min()), "median": float(df.inliers.median()), "max": int(df.inliers.max())}}
        if len(v):
            for k in ("median_m", "rmse_m"):
                out[m][k] = {"min": round(float(v[k].min()), 2), "p5": round(float(v[k].quantile(.05)), 2),
                             "median": round(float(v[k].median()), 2), "p95": round(float(v[k].quantile(.95)), 2),
                             "max": round(float(v[k].max()), 2)}
            out[m]["published_seed0_rmse_percentile"] = round(float((v.rmse_m <= v[v.seed == 0].rmse_m.iloc[0]).mean() * 100), 1) \
                if (v.seed == 0).any() else None
        log.info("seeds %s: %s", m, out[m])
    write_json(out, OUT / "seeds.json")
    return out


# ---------------------------------------------------------------- 2. LoFTR confidence threshold

def confidence(c: Ctx):
    d = c.corr("loftr")
    s0, s1 = c.sizes("loftr")
    rows = []
    for t in np.round(np.arange(0.0, 0.95, 0.1), 2):
        keep = d["scores"] >= t
        tr = estimate_transform(d["pts0_work"][keep], d["pts1_work"][keep], s0, s1, c.params("loftr"), c.seed)
        e = c.heldout_from_work(tr.matrix if tr.ok else None, "loftr")
        rows.append({"threshold": float(t), "matches": int(keep.sum()), "valid": tr.ok,
                     "inliers": tr.stats.get("inliers", 0), **_summ(e)})
        log.info("confidence %.1f: %s", t, rows[-1])
    pd.DataFrame(rows).to_csv(OUT / "loftr_confidence.csv", index=False)
    write_json({"purpose": "LoFTR match-confidence threshold (frozen value 0.0 keeps all matches)", "rows": rows},
               OUT / "loftr_confidence.json")
    return rows


# ---------------------------------------------------------------- 3. bootstrap CIs for Table 1

def intervals(c: Ctx, n_boot=20000):
    rng = np.random.default_rng(c.seed)
    out = {"purpose": "percentile bootstrap 95% CIs over evaluation points (resampled with replacement)", "n_boot": n_boot}
    for model in ("homography", "affine"):
        for m in ("loftr", "manual"):
            e = c.errors(model, m)
            idx = rng.integers(0, len(e), (n_boot, len(e)))
            b = e[idx]
            res = {}
            for name, f in (("median_m", lambda x: np.median(x, 1)), ("mean_m", lambda x: x.mean(1)),
                            ("rmse_m", lambda x: np.sqrt((x ** 2).mean(1)))):
                v = f(b)
                point = float(f(e[None])[0])
                res[name] = {"estimate": round(point, 2), "ci95": [round(float(np.percentile(v, 2.5)), 2),
                                                                   round(float(np.percentile(v, 97.5)), 2)]}
            out[f"{model}/{m}"] = res
    log.info("intervals: %s", out)
    write_json(out, OUT / "intervals.json")
    return out


# ---------------------------------------------------------------- 4. spatial dependence

def _morans_i(values, xy, w_kind="inverse", n_perm=9999, rng=None):
    d = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    W = np.where(d > 0, 1.0 / np.maximum(d, 1.0), 0.0)
    W = W / W.sum()
    z = values - values.mean()
    stat = lambda zz: len(zz) * (zz @ W @ zz) / (zz @ zz)  # noqa: E731  (W already row-sum normalised to total 1)
    I = float(stat(z))
    perms = np.array([stat(rng.permutation(z)) for _ in range(n_perm)])
    p = float((np.sum(np.abs(perms - perms.mean()) >= abs(I - perms.mean())) + 1) / (n_perm + 1))
    return {"I": round(I, 3), "expected_under_null": round(-1 / (len(values) - 1), 3), "p_permutation": round(p, 4)}


def spatial(c: Ctx, n_boot=20000, link_km=3.0):
    rng = np.random.default_rng(c.seed)
    xy = c.truth
    out = {"purpose": "are held-out errors spatially correlated? and a cluster bootstrap that respects it"}
    for m in ("loftr", "manual"):
        r = c.err[(c.err.model == "homography") & (c.err.method == m)].set_index("point_id").reindex(c.ids)
        out[f"morans_I/{m}"] = {k: _morans_i(r[k].to_numpy(float), xy, rng=rng) for k in ("error_m", "dE_m", "dN_m")}
    # single-linkage clusters: points within link_km of each other share a cluster
    d = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    lab = -np.ones(len(xy), int)
    k = 0
    for i in range(len(xy)):
        if lab[i] >= 0:
            continue
        stack, lab[i] = [i], k
        while stack:
            j = stack.pop()
            for n in np.flatnonzero((d[j] <= link_km * 1000) & (lab < 0)):
                lab[n] = k
                stack.append(n)
        k += 1
    clusters = {int(g): [c.ids[i] for i in np.flatnonzero(lab == g)] for g in range(k)}
    a, b = c.errors("homography", "loftr"), c.errors("homography", "manual")
    diff = a - b
    meds, means = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, k, k)
        sel = np.concatenate([np.flatnonzero(lab == g) for g in pick])
        meds.append(np.median(diff[sel]))
        means.append(np.mean(diff[sel]))
    pt = paired_comparison(a, b, c.seed)
    out["clusters"] = {"link_km": link_km, "n_clusters": k, "members": clusters}
    out["cluster_bootstrap_loftr_minus_manual"] = {
        "median_diff_m": round(float(np.median(diff)), 2),
        "ci95_median_diff_m": [round(float(np.percentile(meds, 2.5)), 2), round(float(np.percentile(meds, 97.5)), 2)],
        "mean_diff_m": round(float(np.mean(diff)), 2),
        "ci95_mean_diff_m": [round(float(np.percentile(means, 2.5)), 2), round(float(np.percentile(means, 97.5)), 2)],
        "fraction_of_resamples_favouring_loftr_median": round(float(np.mean(np.array(meds) < 0)), 3)}
    out["point_bootstrap_reference"] = pt
    # sign-flip test at cluster level: averages within clusters, then exact sign-flip over clusters
    cl_mean = np.array([diff[lab == g].mean() for g in range(k)])
    signs = np.array(np.meshgrid(*[[-1, 1]] * k)).reshape(k, -1).T if k <= 16 else None
    if signs is not None:
        obs = cl_mean.mean()
        null = (signs * np.abs(cl_mean)).mean(1)
        out["cluster_sign_flip_test"] = {"cluster_mean_diffs_m": np.round(cl_mean, 2).tolist(),
                                         "p_two_sided": round(float(np.mean(np.abs(null) >= abs(obs) - 1e-9)), 4)}
    log.info("spatial: %s", out)
    write_json(out, OUT / "spatial.json")
    return out


# ---------------------------------------------------------------- 5. landmark definition

def landmarks(c: Ctx):
    keep = ~c.ev.feature_type.str.startswith("landmark").to_numpy()
    out = {"purpose": "drop points whose truth is a landmark centroid (definition differs from the 1926 symbol)",
           "excluded": c.ev.point_id[~keep].tolist(), "n": int(keep.sum())}
    for model in ("homography", "affine"):
        a, b = c.errors(model, "loftr")[keep], c.errors(model, "manual")[keep]
        out[model] = {"loftr": _summ(a), "manual": _summ(b), "paired": paired_comparison(a, b, c.seed)}
    log.info("landmarks: %s", out)
    write_json(out, OUT / "landmarks.json")
    return out


# ---------------------------------------------------------------- 6. shared OSM source (leave-neighbourhood-out)

def shared_source(c: Ctx, radii_km=(1.0, 2.0, 3.0)):
    """Remove every LoFTR correspondence whose OSM-side location is within r km of a held-out point, refit,
    and predict that point. If LoFTR's accuracy came from matching the very OSM features used as truth,
    errors should rise sharply."""
    d = c.corr("loftr")
    r = c.runs["loftr"]
    s0, s1 = c.sizes("loftr")
    F1 = np.array(r["reference_full_to_work"])
    ref_map = ref_px_to_map(apply_h(np.linalg.inv(F1), d["pts1_work"]), c.gt)
    base = c.errors("homography", "loftr")
    out = {"purpose": "test whether LoFTR's held-out accuracy depends on OSM features near the truth points", "rows": []}
    for rk in radii_km:
        errs, removed, valid = [], [], []
        for i in range(len(c.ids)):
            keep = np.linalg.norm(ref_map - c.truth[i], axis=1) > rk * 1000
            tr = estimate_transform(d["pts0_work"][keep], d["pts1_work"][keep], s0, s1, c.params("loftr"), c.seed)
            e = c.heldout_from_work(tr.matrix if tr.ok else None, "loftr")[i]
            errs.append(e)
            removed.append(int((~keep).sum()))
            valid.append(tr.ok)
        errs = np.array(errs)
        out["rows"].append({"radius_km": rk, "matches_removed_median": float(np.median(removed)),
                            "all_valid": bool(all(valid)), **_summ(errs),
                            "per_point_change_m": dict(zip(c.ids, np.round(errs - base, 1).tolist()))})
        log.info("shared_source r=%.0f km: %s", rk, out["rows"][-1])
    out["baseline"] = _summ(base)
    write_json(out, OUT / "shared_source.json")
    return out


# ---------------------------------------------------------------- 7. rendering style

def style(c: Ctx):
    from mapalign.match_loftr import match_loftr
    from mapalign.preprocess import prepare_historical, prepare_reference
    from mapalign.reference import BACKGROUND, LINE_STYLES, WATER, render_osm

    grey = [(k, (40, 40, 40), w) for k, _, w in LINE_STYLES]
    no_minor = [s for s in LINE_STYLES if not (s[0][0] == "highway" and "residential" in s[0][1])]
    variants = {
        "frozen": {},
        "thin_lines_x0.5": {"width_scale": 0.5},
        "thick_lines_x2": {"width_scale": 2.0},
        "monochrome": {"line_styles": grey, "water_colour": (150, 150, 150)},
        "no_water_fill": {"water": False},
        "no_minor_roads": {"line_styles": no_minor},
        "white_background": {"background": (255, 255, 255)},
    }
    vdir = OUT / "style_renders"
    vdir.mkdir(parents=True, exist_ok=True)
    hist = prepare_historical(c.cfg)
    rows = []
    for name, kw in variants.items():
        path = vdir / f"osm_{name}.tif"
        render_osm(c.cfg, out_path=path, style_name=f"robustness:{name}", **kw)
        cfg = copy.deepcopy(c.cfg)
        cfg["modern_reference"]["rendered_path"] = str(path)
        ref = prepare_reference(cfg)
        m = match_loftr(hist.gray, ref.gray, cfg["loftr"], "cpu")
        params = estimator_params(cfg["transform"], m.stats["common_scale"])
        tr = estimate_transform(m.pts0, m.pts1, hist.gray.shape[::-1], ref.gray.shape[::-1], params, c.seed)
        e = np.full(len(c.px), np.nan)
        if tr.ok:
            H = compose_full_resolution(tr.matrix, hist.full_to_work, ref.full_to_work)
            e = positional_errors(c.truth, predict(H, ref.geotransform, c.px))
        rows.append({"variant": name, "matches": len(m.pts0), "valid": tr.ok, "inliers": tr.stats.get("inliers", 0),
                     **_summ(e)})
        log.info("style %s: %s", name, rows[-1])
        for f in vdir.glob(f"osm_{name}.*"):
            if f.suffix in (".tif",):
                f.unlink()  # large; the variant is fully described by its parameters
    pd.DataFrame(rows).to_csv(OUT / "style.csv", index=False)
    write_json({"purpose": "sensitivity of LoFTR to the OSM rendering style (frozen LoFTR/RANSAC settings, seed 42, CPU)",
                "variants": {k: {kk: (str(vv) if kk == "line_styles" else vv) for kk, vv in v.items()} for k, v in variants.items()},
                "rows": rows}, OUT / "style.json")
    return rows


# ---------------------------------------------------------------- 8. projection

def projection(c: Ctx):
    """Survey of India one-inch sheets were drawn on a polyconic projection (Everest spheroid). How well can a
    single homography map polyconic sheet coordinates onto UTM 43N over this sheet? That residual is an error
    floor no homography-based method can remove."""
    g = c.cfg["historical_map"]["neatline_graticule_deg"]
    lon0, lat0 = (g["west"] + g["east"]) / 2, (g["south"] + g["north"]) / 2
    lons, lats = np.meshgrid(np.linspace(g["west"], g["east"], 31), np.linspace(g["south"], g["north"], 31))
    ll = np.column_stack([lons.ravel(), lats.ravel()])
    out = {"purpose": "projection mismatch between a polyconic sheet and UTM 43N, as residual after best-fit transforms"}
    utm = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
    U = np.column_stack(utm.transform(ll[:, 0], ll[:, 1]))
    for name, proj in (("polyconic_everest_sheet_centre", f"+proj=poly +lat_0={lat0} +lon_0={lon0} +ellps=evrst30 +units=m"),
                       ("polyconic_everest_lon0_76", f"+proj=poly +lat_0=12 +lon_0=76 +ellps=evrst30 +units=m")):
        P = np.column_stack(Transformer.from_crs("EPSG:4326", proj, always_xy=True).transform(ll[:, 0], ll[:, 1]))
        res = {}
        H, _ = cv2.findHomography(P, U, 0)
        r = np.linalg.norm(apply_h(H, P) - U, axis=1)
        res["homography"] = {"max_m": round(float(r.max()), 3), "rms_m": round(float(np.sqrt((r ** 2).mean())), 3)}
        X = np.hstack([P, np.ones((len(P), 1))])
        A, *_ = np.linalg.lstsq(X, U, rcond=None)
        r = np.linalg.norm(X @ A - U, axis=1)
        res["affine"] = {"max_m": round(float(r.max()), 3), "rms_m": round(float(np.sqrt((r ** 2).mean())), 3)}
        out[name] = res
    log.info("projection: %s", out)
    write_json(out, OUT / "projection.json")
    return out


# ---------------------------------------------------------------- 9. scan geometry

def scan(c: Ctx):
    from PIL import Image

    from mapalign.io import load_image

    hm = c.cfg["historical_map"]
    img = load_image(resolve_path(hm["path"]), grayscale=True).astype(float)
    k = hm["neatline_corners_px"]
    sides = {"north": ("nw", "ne", "h"), "south": ("sw", "se", "h"), "west": ("nw", "sw", "v"), "east": ("ne", "se", "v")}
    out = {"purpose": "is the scan geometrically clean? straightness of the printed neatline and scale anisotropy"}
    lengths = {}
    for name, (a, b, ori) in sides.items():
        p0, p1 = np.array(k[a], float), np.array(k[b], float)
        L = np.linalg.norm(p1 - p0)
        lengths[name] = L
        ts = np.linspace(0.04, 0.96, 140)
        pts = []
        for t in ts:
            q = p0 + t * (p1 - p0)
            if ori == "h":
                x = int(round(q[0]))
                ys = np.arange(int(q[1]) - 20, int(q[1]) + 21)
                prof = img[ys, x - 2:x + 3].mean(1)
                pts.append((x, ys[np.argmin(prof)]))
            else:
                y = int(round(q[1]))
                xs = np.arange(int(q[0]) - 20, int(q[0]) + 21)
                prof = img[y - 2:y + 3, xs].mean(0)
                pts.append((xs[np.argmin(prof)], y))
        pts = np.array(pts, float)
        u, v = (pts[:, 0], pts[:, 1]) if ori == "h" else (pts[:, 1], pts[:, 0])
        # robust straight-line fit: samples that hit tick marks or lettering are dropped (> 3 robust sigma)
        ok = np.ones(len(u), bool)
        for _ in range(5):
            lin = np.polyfit(u[ok], v[ok], 1)
            r_all = v - np.polyval(lin, u)
            sig = 1.4826 * np.median(np.abs(r_all[ok] - np.median(r_all[ok]))) + 1e-6
            ok = np.abs(r_all) <= max(3 * sig, 1.5)
        quad = np.polyfit(u[ok], v[ok], 2)
        resid = (v - np.polyval(lin, u))[ok]
        mid_bow = np.polyval(quad, u.mean()) - np.polyval(lin, u.mean())
        out[f"neatline_{name}"] = {"rms_dev_from_straight_px": round(float(np.sqrt((resid ** 2).mean())), 2),
                                   "max_dev_px": round(float(np.abs(resid).max()), 2),
                                   "bow_at_midpoint_px": round(float(mid_bow), 2),
                                   "rms_dev_m": round(float(np.sqrt((resid ** 2).mean())) * 5.36, 1),
                                   "samples_used": f"{int(ok.sum())}/{len(ok)}"}
    # expected ground lengths on the Everest spheroid (same for any projection to first order)
    from pyproj import Geod
    geod = Geod(ellps="evrst30")
    g = hm["neatline_graticule_deg"]
    exp = {"north": geod.inv(g["west"], g["north"], g["east"], g["north"])[2],
           "south": geod.inv(g["west"], g["south"], g["east"], g["south"])[2],
           "west": geod.inv(g["west"], g["south"], g["west"], g["north"])[2],
           "east": geod.inv(g["east"], g["south"], g["east"], g["north"])[2]}
    scale = {s: exp[s] / lengths[s] for s in lengths}
    ew = (scale["north"] + scale["south"]) / 2
    ns = (scale["west"] + scale["east"]) / 2
    out["metres_per_scan_pixel"] = {s: round(v, 4) for s, v in scale.items()}
    out["nominal_metres_per_pixel_at_300dpi"] = round(0.0254 * 63360 / 300, 4)
    out["anisotropy_ns_over_ew"] = round(ns / ew, 5)
    out["anisotropy_note"] = "1.0 = no differential paper shrinkage or scanner stretch between the two directions"
    out["implied_paper_scale_vs_nominal"] = round(((ew + ns) / 2) / (0.0254 * 63360 / 300), 4)
    q = Image.open(resolve_path(hm["path"])).quantization
    if q:
        lum = np.array(q[0], float)
        # IJG standard luminance table; quality estimated from the mean scaling factor
        base = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                         14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                         49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], float)
        s = float(np.mean(np.sort(lum) * 100 / np.sort(base)))
        out["jpeg_quality_estimate"] = int(round(100 - s / 2)) if s <= 100 else int(round(5000 / s))
        out["jpeg_luma_table_mean"] = round(float(lum.mean()), 2)
    log.info("scan: %s", out)
    write_json(out, OUT / "scan.json")
    return out


# ---------------------------------------------------------------- 10. datum offset intervals

def datum(c: Ctx, n_boot=20000):
    rng = np.random.default_rng(c.seed)
    grat = graticule_px_to_map(c.cfg)
    mrun = c.runs["manual"]
    used = c.fit[c.fit.point_id.isin(mrun["fitting_point_ids"])]
    sets = {"fitting_points_qc_passed": used, "evaluation_points": c.ev}
    out = {"purpose": "offset of true WGS84 positions from the printed graticule read as WGS84, with bootstrap CIs; "
                      "the evaluation points give a second, larger, independent sample (no method involved)"}
    for name, df in sets.items():
        off = df[["true_easting_m", "true_northing_m"]].to_numpy(float) - grat(df[["historical_x_px", "historical_y_px"]].to_numpy(float))
        b = np.median(off[rng.integers(0, len(off), (n_boot, len(off)))], axis=1)
        out[name] = {"n": int(len(off)), "median_E_N_m": np.round(np.median(off, 0), 1).tolist(),
                     "ci95_E_m": np.round(np.percentile(b[:, 0], [2.5, 97.5]), 1).tolist(),
                     "ci95_N_m": np.round(np.percentile(b[:, 1], [2.5, 97.5]), 1).tolist(),
                     "magnitude_m": round(float(np.hypot(*np.median(off, 0))), 1)}
    log.info("datum: %s", out)
    write_json(out, OUT / "datum.json")
    return out


# ---------------------------------------------------------------- 11. repeated timing

def timing(c: Ctx, repeats=5):
    from mapalign.match_loftr import match_loftr
    from mapalign.match_sift import match_sift
    from mapalign.preprocess import prepare_historical, prepare_reference

    rows = []
    for i in range(repeats):
        t0 = time.perf_counter()
        hist, ref = prepare_historical(c.cfg), prepare_reference(c.cfg)
        t_prep = time.perf_counter() - t0
        for m in ("sift", "loftr"):
            t1 = time.perf_counter()
            res = match_sift(hist.gray, ref.gray, c.cfg["sift"]) if m == "sift" else \
                match_loftr(hist.gray, ref.gray, c.cfg["loftr"], "cpu")
            t2 = time.perf_counter()
            estimate_transform(res.pts0, res.pts1, hist.gray.shape[::-1], ref.gray.shape[::-1], c.params(m), c.seed)
            t3 = time.perf_counter()
            rows.append({"repeat": i, "method": m, "preprocess_s": t_prep, "match_s": t2 - t1, "estimate_s": t3 - t2,
                         "total_s": t_prep + t3 - t1})
        t4 = time.perf_counter()
        from mapalign.georeference import fit_gcp_transform
        used = c.fit[c.fit.point_id.isin(c.runs["manual"]["fitting_point_ids"])]
        fit_gcp_transform(used[["historical_x_px", "historical_y_px"]].to_numpy(float),
                          map_to_ref_px(used[["true_easting_m", "true_northing_m"]].to_numpy(float), c.gt))
        rows.append({"repeat": i, "method": "manual_fit_only", "preprocess_s": 0.0, "match_s": 0.0,
                     "estimate_s": time.perf_counter() - t4, "total_s": time.perf_counter() - t4})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "timing.csv", index=False)
    summ = {m: {k: {"median": round(float(g[k].median()), 3), "min": round(float(g[k].min()), 3), "max": round(float(g[k].max()), 3)}
                for k in ("match_s", "estimate_s", "total_s")} for m, g in df.groupby("method")}
    import platform
    out = {"purpose": f"{repeats} repeated CPU runs (first run includes model loading / cache warm-up)",
           "machine": platform.processor(), "summary": summ,
           "not_measurable": "the human (or assisted) time spent picking the manual control points was not recorded"}
    log.info("timing: %s", summ)
    write_json(out, OUT / "timing.json")
    return out


def _refit_manual(c: Ctx, fit_df: pd.DataFrame):
    """Method C refitted on a set of fitting picks with the frozen QC rule (|z| > 3 robust sigma in E or N)."""
    from mapalign.gcp import graticule_offset_z
    from mapalign.georeference import QC_Z, fit_gcp_transform

    z = graticule_offset_z(c.cfg, fit_df)
    keep = ~np.any(np.abs(z) > QC_Z, axis=1)
    used = fit_df[keep]
    src = used[["historical_x_px", "historical_y_px"]].to_numpy(float)
    dst = map_to_ref_px(used[["true_easting_m", "true_northing_m"]].to_numpy(float), c.gt)
    return fit_gcp_transform(src, dst, "homography"), used.point_id.tolist(), fit_df.point_id[~keep].tolist()


def _read_picks(path) -> pd.DataFrame:
    rp = pd.read_csv(path, dtype={"point_id": str})
    return rp.dropna(subset=["historical_x_px", "historical_y_px"])


def picker(c: Ctx, repick_path: str | None = None, fitting_path: str | None = None):
    """Agreement between the original (AI-proposed, human-reviewed) picks and an independent human re-pick, and the
    effect on the evaluation.

    repick_path  (evaluation points): truth is unchanged; only the historical pixel of each evaluation point is
                 replaced by the second picker's, and the frozen transforms are re-scored.
    fitting_path (fitting points): Method C is refitted on the human picks with the frozen QC rule; its human
                 interaction time is the sum of the recorded seconds. Scored on the original and, if given, the
                 re-picked evaluation pixels.
    """
    out = {"purpose": "inter-picker agreement and its effect on held-out errors"}
    ev2 = None
    loftr_H, manual_H = np.array(c.runs["loftr"]["H_full"]), np.array(c.runs["manual"]["H_full"])
    if repick_path:
        rp = _read_picks(repick_path)
        m = c.ev.set_index("point_id").join(rp.set_index("point_id")[["historical_x_px", "historical_y_px", "seconds_spent"]],
                                            rsuffix="_2", how="inner")
        d = np.hypot(m.historical_x_px - m.historical_x_px_2, m.historical_y_px - m.historical_y_px_2).to_numpy()
        ev2 = m
        res = {"n_points": int(len(m)),
               "picker_difference_px": {"median": round(float(np.median(d)), 1), "max": round(float(d.max()), 1)},
               "picker_difference_m_nominal": {"median": round(float(np.median(d)) * 5.36, 1),
                                               "max": round(float(d.max()) * 5.36, 1)},
               "per_point_px": dict(zip(m.index, np.round(d, 1).tolist())),
               "human_seconds_total": float(pd.to_numeric(m.seconds_spent, errors="coerce").sum())}
        px2 = m[["historical_x_px_2", "historical_y_px_2"]].to_numpy(float)
        tr = m[["true_easting_m", "true_northing_m"]].to_numpy(float)
        idx = [c.ids.index(i) for i in m.index]
        e2 = {}
        for meth, H in (("loftr", loftr_H), ("manual", manual_H)):
            e2[meth] = positional_errors(tr, predict(H, c.gt, px2))
            res[meth] = {"original_picks": _summ(c.errors("homography", meth)[idx]), "second_picker": _summ(e2[meth])}
        res["loftr_vs_manual_second_picker"] = paired_comparison(e2["loftr"], e2["manual"], c.seed)
        out["evaluation"] = res
    if fitting_path:
        fp = _read_picks(fitting_path)
        f2 = c.fit.set_index("point_id").drop(columns=["historical_x_px", "historical_y_px"]).join(
            fp.set_index("point_id")[["historical_x_px", "historical_y_px", "seconds_spent"]], how="inner").reset_index()
        H, used, excluded = _refit_manual(c, f2)
        secs = pd.to_numeric(f2.seconds_spent, errors="coerce")
        e = positional_errors(c.truth, predict(H, c.gt, c.px))
        orig = c.fit.set_index("point_id").loc[f2.point_id]
        res = {"n_picked": int(len(f2)), "qc_excluded": excluded, "used": used,
               "human_seconds_total": float(secs.sum()), "human_seconds_per_point_median": float(secs.median()),
               "picker_difference_px": dict(zip(f2.point_id, np.round(np.hypot(
                   orig.historical_x_px.to_numpy() - f2.historical_x_px.to_numpy(),
                   orig.historical_y_px.to_numpy() - f2.historical_y_px.to_numpy()), 1).tolist())),
               "heldout_original_eval_picks": _summ(e),
               "loftr_vs_human_manual": paired_comparison(c.errors("homography", "loftr"), e, c.seed),
               "H_full": H.tolist()}
        if ev2 is not None:
            px2 = ev2[["historical_x_px_2", "historical_y_px_2"]].to_numpy(float)
            tr = ev2[["true_easting_m", "true_northing_m"]].to_numpy(float)
            eh, el = positional_errors(tr, predict(H, c.gt, px2)), positional_errors(tr, predict(loftr_H, c.gt, px2))
            res["fully_human"] = {"manual": _summ(eh), "loftr": _summ(el), "loftr_vs_manual": paired_comparison(el, eh, c.seed)}
        out["fitting"] = res
    write_json(out, OUT / "picker_agreement.json")
    log.info("picker: %s", out)
    return out


def extra_points(c: Ctx, path: str):
    """Score the frozen transforms on supplementary evaluation points picked with `gcp blind extra` (cells that had
    no evaluation point). The file is hashed and the hash recorded, as for the main set. Reported alone and pooled
    with the original 12 points; the original analysis is unchanged."""
    import hashlib

    from mapalign.gcp import grid_cell

    raw = Path(path).read_bytes()
    pts = _read_picks(path)
    px = pts[["historical_x_px", "historical_y_px"]].to_numpy(float)
    tr = pts[["true_easting_m", "true_northing_m"]].to_numpy(float)
    out = {"purpose": "supplementary evaluation points in previously empty cells (post hoc, secondary)",
           "file_sha256": hashlib.sha256(raw).hexdigest(), "n_points": int(len(pts)),
           "n_skipped": int(pd.read_csv(path).notes.fillna("").astype(str).str.startswith("SKIPPED").sum()),
           "cells": sorted(set(grid_cell(c.cfg, px))),
           "human_seconds_total": float(pd.to_numeric(pts.seconds_spent, errors="coerce").sum())}
    errs = {}
    for meth in ("loftr", "manual"):
        e = positional_errors(tr, predict(np.array(c.runs[meth]["H_full"]), c.gt, px))
        errs[meth] = e
        out[meth] = {"supplementary": _summ(e),
                     "pooled_with_original_12": _summ(np.concatenate([c.errors("homography", meth), e]))}
    out["paired_supplementary"] = paired_comparison(errs["loftr"], errs["manual"], c.seed)
    out["paired_pooled"] = paired_comparison(np.concatenate([c.errors("homography", "loftr"), errs["loftr"]]),
                                             np.concatenate([c.errors("homography", "manual"), errs["manual"]]), c.seed)
    # the picker's own confidence flag, recorded before any result was seen; the only subset that does not depend on
    # either method's output
    conf = (pts.get("confident_yes_no", pd.Series("", index=pts.index)).astype(str).str.lower() == "yes").to_numpy()
    out["confident_only"] = {"n_points": int(conf.sum()),
                             **{m: _summ(e[conf]) for m, e in errs.items()},
                             "paired": paired_comparison(errs["loftr"][conf], errs["manual"][conf], c.seed)}
    out["both_methods_over_300_m"] = int(((errs["loftr"] > 300) & (errs["manual"] > 300)).sum())
    out["per_point_m"] = {m: dict(zip(pts.point_id, np.round(e, 1).tolist())) for m, e in errs.items()}
    write_json(out, OUT / "extra_points.json")
    log.info("extra_points: %s", out)
    return out


TASKS = {"seeds": seeds, "confidence": confidence, "intervals": intervals, "spatial": spatial, "landmarks": landmarks,
         "shared_source": shared_source, "projection": projection, "scan": scan, "datum": datum, "style": style,
         "timing": timing}


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["all", "picker", "extra_points", *TASKS])
    p.add_argument("--repick", help="filled evaluation re-pick template (for 'picker')")
    p.add_argument("--fitting", help="filled fitting re-pick template (for 'picker')")
    p.add_argument("--points", help="filled supplementary-point template (for 'extra_points')")
    args = p.parse_args(argv)
    c = Ctx()
    if args.what == "picker":
        if not (args.repick or args.fitting):
            p.error("picker needs --repick and/or --fitting <csv>")
        picker(c, args.repick, args.fitting)
        return 0
    if args.what == "extra_points":
        if not args.points:
            p.error("extra_points needs --points <csv>")
        extra_points(c, args.points)
        return 0
    for name in (TASKS if args.what == "all" else [args.what]):
        logging.getLogger("mapalign.estimate_transform").setLevel(logging.WARNING)
        t0 = time.perf_counter()
        TASKS[name](c)
        log.info("== %s done in %.1f s", name, time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())

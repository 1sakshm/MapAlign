"""Second set of post-hoc robustness analyses. Nothing here changes a frozen result.

Outputs go to results/robustness/. Every analysis is a sensitivity check added after the held-out
evaluation; no parameter of the main comparison is re-selected from it.

Usage:
    python -m mapalign.robustness_extra all
    python -m mapalign.robustness_extra synthetic        # known-transform benchmark: method error floor
    python -m mapalign.robustness_extra plates           # colour-plate misregistration on the scan
    python -m mapalign.robustness_extra symbols          # cartographic symbol widths and forced displacement
    python -m mapalign.robustness_extra thresholds       # sensitivity to the 600 m / 1 km development radii
    python -m mapalign.robustness_extra classical        # printed-corner georeferencing (+ ECC refinement)
    python -m mapalign.robustness_extra manual_models    # Method C model chosen by leave-one-out on fitting points
    python -m mapalign.robustness_extra multiple_testing # Holm / Benjamini-Hochberg over all reported p-values
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
import time

import cv2
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

from mapalign.config import resolve_path
from mapalign.diagnostics import colour_layers, prior_homography_work, sift_tile_matcher
from mapalign.estimate_transform import apply_h, compose_full_resolution, estimate_transform, estimator_params
from mapalign.evaluate import paired_comparison, positional_errors, predict
from mapalign.georeference import fit_gcp_transform, map_to_ref_px, ref_px_to_map
from mapalign.io import load_image, read_json, write_json
from mapalign.robustness import OUT, Ctx, _summ

log = logging.getLogger("mapalign.robustness_extra")


def _prepared(c: Ctx):
    from mapalign.preprocess import prepare_historical, prepare_reference

    if not hasattr(c, "_prep"):
        c._prep = (prepare_historical(c.cfg), prepare_reference(c.cfg))
    return c._prep


def _hsv_masks(bgr: np.ndarray) -> dict:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = (hsv[..., i].astype(np.int16) for i in range(3))
    return {"blue": ((h >= 85) & (h <= 130) & (s >= 25) & (v >= 90)).astype(np.uint8),
            "red": (((h >= 165) | (h <= 6)) & (s >= 70) & (v >= 90)).astype(np.uint8),
            "black": (v < 70).astype(np.uint8)}


# ---------------------------------------------------------------- 1. synthetic known-transform benchmark

def _H_syn(w, h, rot_deg=2.0, scale=0.97, persp=(2e-6, -1.5e-6), shift=(-300.0, -350.0)):
    cx, cy = w / 2, h / 2
    a = np.deg2rad(rot_deg)
    T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]])
    R = np.array([[scale * np.cos(a), -scale * np.sin(a), 0], [scale * np.sin(a), scale * np.cos(a), 0], [0, 0, 1.0]])
    P = np.array([[1, 0, 0], [0, 1, 0], [persp[0], persp[1], 1.0]])
    T2 = np.array([[1, 0, cx + shift[0]], [0, 1, cy + shift[1]], [0, 0, 1.0]])
    return T2 @ P @ R @ T1


def synthetic(c: Ctx):
    """Warp an image by a known homography and ask each matcher to recover it. Content is identical on both
    sides, so the result is the method's own precision floor, free of map error, OSM error and picking error."""
    from mapalign.match_loftr import match_loftr
    from mapalign.match_sift import match_sift

    hist, ref = _prepared(c)
    rng = np.random.default_rng(c.seed)
    cases = {"osm_render_vs_itself": ref.gray, "scan_1926_vs_itself": hist.gray}
    out = {"purpose": "matcher + MAGSAC precision on pairs with a known transform (no map/truth/picking error)",
           "metric": "distance between estimated and true mapping on a 21x21 grid inside the overlap, in metres "
                     "(working resolution 10 m/px)", "cases": {}}
    for name, base in cases.items():
        h, w = base.shape
        Hs = _H_syn(w, h)                     # base px -> synthetic px
        for degrade in ("clean", "blur_noise_jpeg"):
            syn = cv2.warpPerspective(base, Hs, (w, h), borderValue=245)
            if degrade != "clean":
                syn = cv2.GaussianBlur(syn, (0, 0), 1.2)
                syn = np.clip(syn + rng.normal(0, 6, syn.shape), 0, 255).astype(np.uint8)
                ok, buf = cv2.imencode(".jpg", syn, [cv2.IMWRITE_JPEG_QUALITY, 60])
                syn = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
            H_true = np.linalg.inv(Hs)           # synthetic px -> base px
            grid = np.stack(np.meshgrid(np.linspace(0.15 * w, 0.85 * w, 21), np.linspace(0.15 * h, 0.85 * h, 21)), -1).reshape(-1, 2)
            for meth in ("loftr", "sift"):
                t0 = time.perf_counter()
                m = match_loftr(syn, base, c.cfg["loftr"], "cpu") if meth == "loftr" else match_sift(syn, base, c.cfg["sift"])
                params = estimator_params(c.cfg["transform"], m.stats.get("common_scale", 1.0))
                tr = estimate_transform(m.pts0, m.pts1, (w, h), (w, h), params, c.seed)
                row = {"matches": len(m.pts0), "inliers": tr.stats.get("inliers", 0), "valid": tr.ok,
                       "seconds": round(time.perf_counter() - t0, 1)}
                if tr.matrix is not None:
                    e = np.linalg.norm(apply_h(tr.matrix, grid) - apply_h(H_true, grid), axis=1) * 10.0
                    row.update(rmse_m=round(float(np.sqrt((e ** 2).mean())), 2), max_m=round(float(e.max()), 2),
                               median_m=round(float(np.median(e)), 2))
                out["cases"][f"{name}/{degrade}/{meth}"] = row
                log.info("synthetic %s/%s/%s: %s", name, degrade, meth, row)
    write_json(out, OUT / "synthetic.json")
    return out


# ---------------------------------------------------------------- 2. colour-plate registration

def plates(c: Ctx, n_samples=160, min_contrast=12.0):
    """Registration of the background-tint plate(s) against the black plate, measured at the neatline.

    Inside the neatline the map is tinted (buff ground, grey hill shading, green vegetation); outside it the
    margin is untinted paper. The tint is printed from separate plates and should stop at the black neatline.
    For each side we measure, perpendicular to the neatline, the signed gap between the neatline centre (darkest
    pixel) and the tint edge (half-way crossing of colour saturation between margin and interior). A plate shift
    of (dx, dy) changes opposite sides in opposite directions:
        dx = (gap_west - gap_east) / 2,   dy = (gap_north - gap_south) / 2
    while (gap_west + gap_east) / 2 is the designed overlap/underlap. Line plates (red roads, blue water, brown
    contours) have no shared edge with the black plate on this sheet and cannot be measured this way.
    """
    img = load_image(resolve_path(c.cfg["historical_map"]["path"]))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    S, V = hsv[..., 1].astype(float), hsv[..., 2].astype(float)
    k = c.cfg["historical_map"]["neatline_corners_px"]
    sides = {"west": ("nw", "sw", (1, 0)), "east": ("ne", "se", (-1, 0)),
             "north": ("nw", "ne", (0, 1)), "south": ("sw", "se", (0, -1))}
    gaps, line_half = {}, {}
    for name, (a, b, inward) in sides.items():
        p0, p1 = np.array(k[a], float), np.array(k[b], float)
        vals, starts = [], []
        for t in np.linspace(0.05, 0.95, n_samples):
            q = p0 + t * (p1 - p0)
            ts = np.arange(-24, 25)
            pts = q[None] + ts[:, None] * np.array(inward, float)[None]
            xs, ys = np.round(pts[:, 0]).astype(int), np.round(pts[:, 1]).astype(int)
            if inward[0]:   # vertical side: average 5 rows
                sp = np.array([S[y - 2:y + 3, x].mean() for x, y in zip(xs, ys)])
                vp = np.array([V[y - 2:y + 3, x].mean() for x, y in zip(xs, ys)])
            else:
                sp = np.array([S[y, x - 2:x + 3].mean() for x, y in zip(xs, ys)])
                vp = np.array([V[y, x - 2:x + 3].mean() for x, y in zip(xs, ys)])
            iline = int(np.argmin(vp[14:35])) + 14            # neatline near the centre of the profile
            outside, inside = np.median(sp[:iline - 4][-12:]), np.median(sp[iline + 4:][:12])
            if abs(inside - outside) < min_contrast:
                continue
            half = (inside + outside) / 2
            # end of the dark neatline on the inner side: value recovers half-way to the interior brightness
            v_in = np.median(vp[iline + 4:][:12])
            after = np.flatnonzero(vp[iline:] >= (vp[iline] + v_in) / 2)
            start = iline + (int(after[0]) if len(after) else 2)
            seg = sp[start:]
            cross = np.flatnonzero((seg - half) * np.sign(inside - outside) >= 0)
            if not len(cross):
                continue
            j = cross[0]
            if j > 0:                                          # sub-pixel crossing
                y0, y1 = seg[j - 1] - half, seg[j] - half
                pos = start + j - 1 + (y0 / (y0 - y1) if y1 != y0 else 0.0)
            else:
                pos = float(start)
            vals.append(pos - iline)
            starts.append(start - iline)
        vals = np.array(vals)
        gaps[name] = vals
        line_half[name] = float(np.median(starts)) if starts else None
    rng = np.random.default_rng(c.seed)

    def med_ci(v):
        if len(v) < 5:
            return None, [None, None]
        b = np.median(v[rng.integers(0, len(v), (5000, len(v)))], axis=1)
        return float(np.median(v)), np.round(np.percentile(b, [2.5, 97.5]), 2).tolist()

    out = {"purpose": __doc__ if False else "tint-plate registration against the black neatline (see docstring)",
           "gap_px_inward_from_neatline_centre": {}, "n_profiles": {s: int(len(v)) for s, v in gaps.items()},
           "neatline_inner_half_width_px": line_half,
           "resolution_note": "a gap equal to the neatline half-width means the tint reaches the line itself"}
    m = {}
    for s, v in gaps.items():
        m[s], ci = med_ci(v)
        out["gap_px_inward_from_neatline_centre"][s] = {"median": None if m[s] is None else round(m[s], 2), "ci95": ci}
    if all(m[s] is not None for s in m):
        dx, dy = (m["west"] - m["east"]) / 2, (m["north"] - m["south"]) / 2
        boot = []
        for _ in range(3000):
            bm = {s: np.median(v[rng.integers(0, len(v), len(v))]) for s, v in gaps.items()}
            boot.append(((bm["west"] - bm["east"]) / 2, (bm["north"] - bm["south"]) / 2))
        boot = np.array(boot)
        out["tint_plate_shift_px"] = {"dx": round(dx, 2), "dy": round(dy, 2),
                                      "ci95_dx": np.round(np.percentile(boot[:, 0], [2.5, 97.5]), 2).tolist(),
                                      "ci95_dy": np.round(np.percentile(boot[:, 1], [2.5, 97.5]), 2).tolist(),
                                      "magnitude_m": round(float(np.hypot(dx, dy)) * 5.36, 1)}
        out["designed_overlap_px"] = {"east_west": round((m["west"] + m["east"]) / 2, 2),
                                      "north_south": round((m["north"] + m["south"]) / 2, 2)}
        out["not_measurable"] = "line plates (red roads, blue water, brown contours) share no edge with the black plate"
    log.info("plates: %s", out)
    write_json(out, OUT / "plates.json")
    return out


# ---------------------------------------------------------------- 3. symbol widths and forced displacement

def symbols(c: Ctx, step_m=150.0, half_m=150.0):
    """Measure printed symbol widths (red road bands, black railway) perpendicular to OSM centrelines on the
    LoFTR GeoTIFF, convert to ground metres, and estimate the displacement a cartographer must introduce where
    a road and a railway run closer together than their symbols allow."""
    path = resolve_path("results/georef_loftr.tif")
    data = read_json(resolve_path(c.cfg["modern_reference"]["raw_path"]))
    tf = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
    lines = {"road": [], "rail": []}
    for el in data["elements"]:
        t = el.get("tags", {})
        kind = "rail" if t.get("railway") == "rail" else "road" if t.get("highway") in ("trunk", "primary", "secondary") else None
        if kind and el["type"] == "way" and "geometry" in el:
            e, n = tf.transform([p["lon"] for p in el["geometry"]], [p["lat"] for p in el["geometry"]])
            lines[kind].append(np.column_stack([e, n]))
    with rasterio.open(path) as ds:
        img = np.moveaxis(ds.read([1, 2, 3]), 0, 2)[:, :, ::-1].copy()
        alpha = ds.read(4)
        T = ds.transform
    masks = _hsv_masks(img)
    inv = ~T
    widths = {"road": [], "rail": []}
    for kind, layer in (("road", "red"), ("rail", "black")):
        m = masks[layer]
        for xy in lines[kind]:
            seg = np.diff(xy, axis=0)
            L = np.hypot(seg[:, 0], seg[:, 1])
            for (p, d, l) in zip(xy[:-1], seg, L):
                if l < 1:
                    continue
                u = d / l
                nrm = np.array([-u[1], u[0]])
                for s in np.arange(0, l, step_m):
                    q = p + u * s
                    ts = np.arange(-half_m, half_m + 1, 1.0)
                    pts = q + ts[:, None] * nrm
                    cols = (inv.a * pts[:, 0] + inv.b * pts[:, 1] + inv.c).astype(int)
                    rows = (inv.d * pts[:, 0] + inv.e * pts[:, 1] + inv.f).astype(int)
                    ok = (cols >= 0) & (rows >= 0) & (cols < m.shape[1]) & (rows < m.shape[0])
                    if not ok.all() or alpha[rows, cols].min() == 0:
                        continue
                    prof = m[rows, cols].astype(bool)
                    runs, start = [], None
                    for i, v in enumerate(prof):
                        if v and start is None:
                            start = i
                        if (not v or i == len(prof) - 1) and start is not None:
                            end = i if not v else i + 1
                            runs.append((start, end))
                            start = None
                    if not runs:
                        continue
                    # the symbol is the run whose centre is nearest the (method-aligned) centreline
                    centre = len(prof) // 2
                    a, b = min(runs, key=lambda r: abs((r[0] + r[1]) / 2 - centre))
                    w = (b - a) * 1.0
                    if 3 <= w <= 120:
                        widths[kind].append(w)
    out = {"purpose": "printed symbol widths in ground metres (1 m profile spacing on the 5 m GeoTIFF; resolution-limited)"}
    for kind, v in widths.items():
        v = np.array(v)
        out[f"{kind}_symbol_width_m"] = {"n_profiles": int(len(v)), "median": round(float(np.median(v)), 1),
                                         "iqr": np.round(np.percentile(v, [25, 75]), 1).tolist(),
                                         "map_mm": round(float(np.median(v)) / 63.36, 2)} if len(v) else None
    # road-rail pairs that run close together in reality
    from shapely.geometry import LineString
    from shapely.strtree import STRtree
    rails = [LineString(x) for x in lines["rail"]]
    tree = STRtree(rails)
    seps = []
    for r in lines["road"]:
        ls = LineString(r)
        for t in np.arange(0, ls.length, 100.0):
            p = ls.interpolate(t)
            near = tree.query(p.buffer(80.0))
            if len(near):
                seps.append(min(rails[i].distance(p) for i in near))
    seps = np.array(seps)
    if out.get("road_symbol_width_m") and out.get("rail_symbol_width_m") and len(seps):
        need = (out["road_symbol_width_m"]["median"] + out["rail_symbol_width_m"]["median"]) / 2 + 5.0  # 5 m white gap
        close = seps[seps < need]
        out["road_rail_adjacency"] = {
            "samples_with_rail_within_80m": int(len(seps)), "required_centre_separation_m": round(need, 1),
            "samples_closer_than_required": int(len(close)),
            "forced_displacement_m": {"median": round(float(np.median(need - close)), 1) if len(close) else 0.0,
                                      "max": round(float(np.max(need - close)), 1) if len(close) else 0.0},
            "note": "where true separation is less than the symbols need, one feature must be displaced by at least this"}
    log.info("symbols: %s", out)
    write_json(out, OUT / "symbols.json")
    return out


# ---------------------------------------------------------------- 4. development-threshold sensitivity

def thresholds(c: Ctx):
    from mapalign.diagnostics import prior_consistency
    from mapalign.match_loftr import loftr_tile_matcher, match_loftr

    hist, ref = _prepared(c)
    Hp = prior_homography_work(c.cfg, hist, ref)
    out = {"purpose": "do the hand-chosen development radii (1 km prior radius, 600 m tile radius) drive any conclusion?"}
    # (a) LoFTR input-size selection under different prior radii
    sizes = [1280, 1024, 896, 768, 640, 512, 384]   # largest first, while memory is free
    radii = [500, 750, 1000, 1500, 2000]
    table = []
    for side in sizes:
        try:
            m = match_loftr(hist.gray, ref.gray, {**c.cfg["loftr"], "max_side_px": side}, "cpu")
        except RuntimeError as ex:
            log.warning("size %d failed: %s", side, str(ex).splitlines()[0][:120])
            continue
        row = {"side": side, "matches": len(m.pts0)}
        for r in radii:
            pc = prior_consistency(m.pts0, m.pts1, Hp, 10.0, radius_m=float(r))
            row[f"near_{r}m"] = pc["matches_within"]
        table.append(row)
        log.info("thresholds size %d: %s", side, row)
    df = pd.DataFrame(table)
    out["size_selection"] = {"table": table,
                             "selected_by_count_at_radius": {f"{r}m": int(df.loc[df[f"near_{r}m"].idxmax(), "side"]) for r in radii},
                             "selected_by_fraction_at_radius": {f"{r}m": int(df.loc[(df[f"near_{r}m"] / df.matches).idxmax(), "side"]) for r in radii},
                             "note": "the frozen rule is 'most matches near the prior' at 1000 m"}
    # (b) aligned-tile test under different correctness radii
    h, w = ref.gray.shape
    warped = cv2.warpPerspective(hist.gray, Hp, (w, h), borderValue=255)
    tile = 512
    matchers = {"sift": sift_tile_matcher(c.cfg["sift"]), "loftr": loftr_tile_matcher(c.cfg["loftr"], "cpu")}
    dists = {k: [] for k in matchers}
    for y in range(450, min(3000, h - tile), tile):
        for x in range(450, min(3000, w - tile), tile):
            for k, f in matchers.items():
                pa, pb = f(warped[y:y + tile, x:x + tile], ref.gray[y:y + tile, x:x + tile])
                if len(pa):
                    dists[k].extend(np.linalg.norm(pa - pb, axis=1).tolist())
    rows = []
    for rad_m in (200, 300, 400, 600, 800, 1000):
        rp = rad_m / 10.0
        chance = np.pi * rp ** 2 / tile ** 2
        row = {"radius_m": rad_m, "chance_pct": round(100 * chance, 2)}
        for k, d in dists.items():
            d = np.array(d)
            frac = float((d < rp).mean())
            # one-sided binomial test against the chance rate
            pval = float(__import__("scipy.stats", fromlist=["binomtest"]).binomtest(int((d < rp).sum()), len(d), chance,
                                                                                      alternative="greater").pvalue)
            row[f"{k}_pct"] = round(100 * frac, 2)
            row[f"{k}_ratio_to_chance"] = round(frac / chance, 2)
            row[f"{k}_p_above_chance"] = float(f"{pval:.3g}")
        rows.append(row)
        log.info("thresholds tile radius %d m: %s", rad_m, row)
    out["tile_test"] = {"rows": rows, "n_matches": {k: len(v) for k, v in dists.items()}}
    write_json(out, OUT / "thresholds.json")
    return out


# ---------------------------------------------------------------- 5. classical corner-based georeferencing

def classical(c: Ctx):
    """Two classical baselines evaluated on the same held-out points (secondary, added post hoc):
    (a) georeference from the four printed sheet corners and their graticule values (the standard manual practice
        when nothing else is available), and (b) (a) refined by ECC (enhanced correlation coefficient) image
        alignment of the colour-separated linework, a classical intensity-based registration."""
    hist, ref = _prepared(c)
    Hp = prior_homography_work(c.cfg, hist, ref)
    out = {"purpose": "classical baselines (secondary, post hoc): printed corners only; printed corners + ECC refinement"}
    r = c.runs["loftr"]

    def score(H_work):
        H = compose_full_resolution(H_work, hist.full_to_work, ref.full_to_work)
        return positional_errors(c.truth, predict(H, ref.geotransform, c.px))

    e_corner = score(Hp)
    out["printed_corners"] = _summ(e_corner)
    a = colour_layers(hist.color, 0).astype(np.float32)
    b = colour_layers(ref.color, 0).astype(np.float32)
    res = {}
    for scale in (0.25, 0.5):
        S = np.diag([scale, scale, 1.0])
        a_s = cv2.GaussianBlur(cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), (0, 0), 2.0)
        b_s = cv2.GaussianBlur(cv2.resize(b, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), (0, 0), 2.0)
        # ECC estimates W: template(b) coords -> input(a) coords, i.e. reference -> historical
        W0 = (S @ np.linalg.inv(res.get("H", Hp)) @ np.linalg.inv(S)).astype(np.float32)
        W0 /= W0[2, 2]
        try:
            cc, W = cv2.findTransformECC(b_s, a_s, W0, cv2.MOTION_HOMOGRAPHY,
                                         (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-6), None, 5)
            H = np.linalg.inv(np.linalg.inv(S) @ W.astype(np.float64) @ S)
            res = {"H": H / H[2, 2], "cc": float(cc), "converged": True}
        except cv2.error as ex:
            res = {"H": res.get("H", Hp), "cc": None, "converged": False, "error": str(ex)[:120]}
            break
    e_ecc = score(res["H"])
    out["printed_corners_plus_ecc"] = {**_summ(e_ecc), "ecc_correlation": res.get("cc"), "converged": res.get("converged")}
    loftr = c.errors("homography", "loftr")
    out["paired_vs_loftr"] = {"printed_corners": paired_comparison(loftr, e_corner, c.seed),
                              "printed_corners_plus_ecc": paired_comparison(loftr, e_ecc, c.seed)}
    out["per_point_m"] = {"printed_corners": dict(zip(c.ids, np.round(e_corner, 1).tolist())),
                          "printed_corners_plus_ecc": dict(zip(c.ids, np.round(e_ecc, 1).tolist()))}
    log.info("classical: corners %s | ecc %s", out["printed_corners"], out["printed_corners_plus_ecc"])
    write_json(out, OUT / "classical.json")
    return out


# ---------------------------------------------------------------- 6. manual model selection by LOO

def manual_models(c: Ctx):
    """Give Method C the same kind of development tuning the matchers got: choose the transform model by
    leave-one-out error on its own fitting points only, then report held-out error for every model."""
    used = c.fit[c.fit.point_id.isin(c.runs["manual"]["fitting_point_ids"])]
    src = used[["historical_x_px", "historical_y_px"]].to_numpy(float)
    dst_m = used[["true_easting_m", "true_northing_m"]].to_numpy(float)
    dst = map_to_ref_px(dst_m, c.gt)

    def fit(model, s, d):
        if model == "similarity":
            # linear least squares for x' = a x - b y + tx, y' = b x + a y + ty (rotation + uniform scale + shift)
            A = np.zeros((2 * len(s), 4))
            A[0::2] = np.column_stack([s[:, 0], -s[:, 1], np.ones(len(s)), np.zeros(len(s))])
            A[1::2] = np.column_stack([s[:, 1], s[:, 0], np.zeros(len(s)), np.ones(len(s))])
            (a, b, tx, ty), *_ = np.linalg.lstsq(A, d.reshape(-1), rcond=None)
            return np.array([[a, -b, tx], [b, a, ty], [0, 0, 1.0]])
        return fit_gcp_transform(s, d, model)

    rows = []
    for model in ("similarity", "affine", "homography"):
        loo = []
        for i in range(len(src)):
            k = np.arange(len(src)) != i
            try:
                H = fit(model, src[k], dst[k])
                loo.append(float(np.hypot(*(ref_px_to_map(apply_h(H, src[i:i + 1]), c.gt)[0] - dst_m[i]))))
            except ValueError:
                loo.append(np.nan)
        H = fit(model, src, dst)
        e = positional_errors(c.truth, ref_px_to_map(apply_h(H, c.px), c.gt))
        rows.append({"model": model, "loo_rmse_m": round(float(np.sqrt(np.nanmean(np.square(loo)))), 2), **_summ(e)})
    best = min(rows, key=lambda r: r["loo_rmse_m"])
    out = {"purpose": "Method C model chosen by leave-one-out RMSE on its 7 fitting points (no evaluation data used "
                      "for the choice); held-out errors reported for all models",
           "rows": rows, "selected_model": best["model"],
           "selected_vs_loftr": paired_comparison(c.errors("homography", "loftr"),
                                                  positional_errors(c.truth, ref_px_to_map(apply_h(fit(best["model"], src, dst), c.px), c.gt)),
                                                  c.seed)}
    log.info("manual_models: %s", out)
    write_json(out, OUT / "manual_models.json")
    return out


# ---------------------------------------------------------------- 7. multiple testing

def multiple_testing(c: Ctx):
    met = read_json(resolve_path("results/metrics.json"))
    R = OUT
    tests = [("primary: LoFTR vs manual, homography (Wilcoxon)", met["paired_comparisons"]["homography"]["loftr_vs_manual"]["wilcoxon_p_two_sided"], "primary")]
    tests.append(("LoFTR vs manual, affine (Wilcoxon)", met["paired_comparisons"]["affine"]["loftr_vs_manual"]["wilcoxon_p_two_sided"], "secondary"))
    lm = read_json(R / "landmarks.json")
    tests += [("landmarks excluded, homography", lm["homography"]["paired"]["wilcoxon_p_two_sided"], "secondary"),
              ("landmarks excluded, affine", lm["affine"]["paired"]["wilcoxon_p_two_sided"], "secondary")]
    sp = read_json(R / "spatial.json")
    tests.append(("cluster sign-flip, homography", sp["cluster_sign_flip_test"]["p_two_sided"], "secondary"))
    for m in ("loftr", "manual"):
        for k in ("error_m", "dE_m", "dN_m"):
            tests.append((f"Moran's I {m} {k}", sp[f"morans_I/{m}"][k]["p_permutation"], "secondary"))
    num = read_json(resolve_path("paper/numbers.json")) if resolve_path("paper/numbers.json").is_file() else {}
    for k, v in num.get("distance_correlation", {}).items():
        tests.append((f"Spearman {k}", v["p"], "secondary"))
    for f, key in (("classical.json", "printed_corners"), ("classical.json", "printed_corners_plus_ecc")):
        if (R / f).is_file():
            p = read_json(R / f)["paired_vs_loftr"][key].get("wilcoxon_p_two_sided")
            if p is not None:
                tests.append((f"LoFTR vs {key} (Wilcoxon)", p, "secondary"))
    X = resolve_path("results/extensions")
    if (X / "aliked_lightglue.json").is_file():
        lg = read_json(X / "aliked_lightglue.json")
        for key in ("paired_vs_loftr", "paired_vs_manual"):
            if key in lg:
                tests.append((f"ALIKED+LightGlue {key.replace('paired_', '')} (Wilcoxon)", lg[key]["wilcoxon_p_two_sided"], "secondary"))
    if (X / "nonprojective.json").is_file():
        npj = read_json(X / "nonprojective.json")
        for m in ("loftr", "manual"):
            tests.append((f"{m} selected non-projective model vs homography (Wilcoxon)",
                          npj[m]["selected_vs_homography"]["wilcoxon_p_two_sided"], "secondary"))
    if (R / "picker_agreement.json").is_file():
        pa = read_json(R / "picker_agreement.json")
        if "evaluation" in pa:
            tests.append(("human re-pick of evaluation pixels: LoFTR vs manual (Wilcoxon)",
                          pa["evaluation"]["loftr_vs_manual_second_picker"]["wilcoxon_p_two_sided"], "secondary"))
        if "fitting" in pa:
            tests.append(("LoFTR vs manual refitted on human picks (Wilcoxon)",
                          pa["fitting"]["loftr_vs_human_manual"]["wilcoxon_p_two_sided"], "secondary"))
            if "fully_human" in pa["fitting"]:
                tests.append(("fully human: LoFTR vs human manual on human evaluation pixels (Wilcoxon)",
                              pa["fitting"]["fully_human"]["loftr_vs_manual"]["wilcoxon_p_two_sided"], "secondary"))
    if (R / "extra_points.json").is_file():
        xp = read_json(R / "extra_points.json")
        tests.append(("supplementary points, all: LoFTR vs manual (Wilcoxon)", xp["paired_supplementary"]["wilcoxon_p_two_sided"], "secondary"))
        if "confident_only" in xp:
            tests.append(("supplementary points, confident: LoFTR vs manual (Wilcoxon)",
                          xp["confident_only"]["paired"]["wilcoxon_p_two_sided"], "secondary"))
    df = pd.DataFrame(tests, columns=["test", "p", "family"])
    sec = df[df.family == "secondary"].copy()
    m = len(sec)
    order = np.argsort(sec.p.to_numpy())
    holm = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * sec.p.iloc[idx]))
        holm[idx] = running
    bh = np.empty(m)
    prev = 1.0
    for rank, idx in list(enumerate(order))[::-1]:
        prev = min(prev, sec.p.iloc[idx] * m / (rank + 1))
        bh[idx] = prev
    sec["p_holm"], sec["q_bh"] = np.round(holm, 4), np.round(bh, 4)
    out = {"purpose": "family-wise (Holm) and false-discovery-rate (Benjamini-Hochberg) adjustment over all secondary "
                      "tests; the single pre-specified primary test is reported unadjusted",
           "primary": df[df.family == "primary"].to_dict("records"),
           "secondary": sec.to_dict("records"), "n_secondary": m}
    log.info("multiple_testing: %s", out)
    write_json(out, OUT / "multiple_testing.json")
    return out


TASKS = {"synthetic": synthetic, "plates": plates, "symbols": symbols, "thresholds": thresholds,
         "classical": classical, "manual_models": manual_models, "multiple_testing": multiple_testing}


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

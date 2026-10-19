"""Coordinate chains, manual-GCP fitting (Method C) and georeferenced GeoTIFF output.

Chain used throughout:
    historical full-res px --H_full--> reference full-res px --geotransform--> EPSG:32643 (m)

GeoTIFF output. A homography cannot be stored as a GeoTIFF affine geotransform. Instead, the scan is
resampled onto a north-up EPSG:32643 grid whose geotransform IS affine. Each output pixel centre is
mapped back to the historical scan through the exact inverse chain
    out px -> UTM -> reference px -> inv(H_full) -> historical px
(cv2.warpPerspective with WARP_INVERSE_MAP), so no projective-to-affine approximation is made. The only
approximation is the bilinear resampling.

Usage:
    python -m mapalign.georeference manual            # fit Method C, write results/manual_run.json + figures
    python -m mapalign.georeference raster loftr|manual
    python -m mapalign.georeference verify loftr|manual
    python -m mapalign.georeference datum             # graticule offset from fitting GCPs (no method involved)
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
import time

import cv2
import numpy as np
import rasterio
from affine import Affine
from pyproj import Transformer
from rasterio.transform import from_origin

from mapalign.config import load_config, resolve_path
from mapalign.estimate_transform import apply_h, check_transform
from mapalign.io import load_image, read_json, write_json

log = logging.getLogger("mapalign.georeference")

QC_Z = 3.0  # fitting-point QC threshold in robust standard deviations (fixed before evaluation)


# ---------------------------------------------------------------- pixel <-> map

def ref_px_to_map(pts: np.ndarray, gt: Affine) -> np.ndarray:
    """Reference pixel coordinates (OpenCV centre convention) -> map coordinates.

    rasterio geotransforms address pixel corners, hence the +0.5.
    """
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    c, r = pts[:, 0] + 0.5, pts[:, 1] + 0.5
    return np.stack([gt.a * c + gt.b * r + gt.c, gt.d * c + gt.e * r + gt.f], axis=1)


def map_to_ref_px(xy: np.ndarray, gt: Affine) -> np.ndarray:
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    inv = ~gt
    c = inv.a * xy[:, 0] + inv.b * xy[:, 1] + inv.c
    r = inv.d * xy[:, 0] + inv.e * xy[:, 1] + inv.f
    return np.stack([c - 0.5, r - 0.5], axis=1)


def affine_px_to_map_matrix(gt: Affine) -> np.ndarray:
    """3x3 matrix equivalent of ref_px_to_map (centre convention)."""
    A = np.array([[gt.a, gt.b, gt.c], [gt.d, gt.e, gt.f], [0.0, 0.0, 1.0]])
    return A @ np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])


def historical_px_to_map(pts: np.ndarray, H_full: np.ndarray, gt: Affine) -> np.ndarray:
    return ref_px_to_map(apply_h(H_full, pts), gt)


def graticule_corner_check(cfg: dict, H_full: np.ndarray, gt: Affine, map_crs: str) -> dict:
    """Coarse plausibility check: where does a transform put the sheet's printed corners?

    The printed corner values are on an unverified pre-WGS84 Indian datum and are treated as WGS84
    here, so a correct alignment can still be off by a few hundred metres. Development diagnostic only.
    """
    hm = cfg["historical_map"]
    g = hm["neatline_graticule_deg"]
    corners_ll = {"nw": (g["west"], g["north"]), "ne": (g["east"], g["north"]),
                  "se": (g["east"], g["south"]), "sw": (g["west"], g["south"])}
    tf = Transformer.from_crs(cfg["crs"]["geographic"], map_crs, always_xy=True)
    out = {}
    for k, (lon, lat) in corners_ll.items():
        px = np.array([hm["neatline_corners_px"][k]], np.float64)
        pred = historical_px_to_map(px, H_full, gt)[0]
        exp = np.array(tf.transform(lon, lat))
        d = pred - exp
        out[k] = {"predicted_m": pred.round(1).tolist(), "graticule_m": exp.round(1).tolist(),
                  "offset_m": d.round(1).tolist(), "distance_m": round(float(np.hypot(*d)), 1)}
    dists = [v["distance_m"] for v in out.values()]
    return {"note": "development diagnostic only; graticule datum unverified (treated as WGS84)",
            "corners": out, "max_distance_m": max(dists), "mean_distance_m": round(float(np.mean(dists)), 1)}


def graticule_px_to_map(cfg: dict):
    """Historical px -> map coords implied by the printed corners alone (graticule treated as WGS84)."""
    hm = cfg["historical_map"]
    g = hm["neatline_graticule_deg"]
    order = ("nw", "ne", "se", "sw")
    ll = {"nw": (g["west"], g["north"]), "ne": (g["east"], g["north"]),
          "se": (g["east"], g["south"]), "sw": (g["west"], g["south"])}
    tf = Transformer.from_crs(cfg["crs"]["geographic"], cfg["crs"]["evaluation"], always_xy=True)
    dst = np.array([tf.transform(*ll[k]) for k in order], np.float32)
    src = np.array([hm["neatline_corners_px"][k] for k in order], np.float32)
    H = cv2.getPerspectiveTransform(src, dst).astype(np.float64)
    return lambda px: apply_h(H, px)


# ---------------------------------------------------------------- Method C: manual GCP fit

def fit_gcp_transform(src: np.ndarray, dst: np.ndarray, model: str = "homography") -> np.ndarray:
    """Least-squares fit over ALL points (no RANSAC): src (historical px) -> dst (reference px)."""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    if model == "homography":
        if len(src) < 4:
            raise ValueError(f"homography needs >= 4 GCPs, got {len(src)}")
        H, _ = cv2.findHomography(src, dst, 0)
        if H is None:
            raise ValueError("homography least-squares fit failed (degenerate GCP layout?)")
        return H / H[2, 2]
    if model == "affine":
        if len(src) < 3:
            raise ValueError(f"affine needs >= 3 GCPs, got {len(src)}")
        X = np.hstack([src, np.ones((len(src), 1))])
        sol, *_ = np.linalg.lstsq(X, dst, rcond=None)
        return np.vstack([sol.T, [0.0, 0.0, 1.0]])
    raise ValueError(model)


def gcp_fit_diagnostics(src: np.ndarray, dst_m: np.ndarray, H: np.ndarray, gt: Affine, model: str) -> dict:
    """Fitting residuals and leave-one-out residuals, in metres. NOT the held-out evaluation error."""
    dst_px = map_to_ref_px(dst_m, gt)
    res = ref_px_to_map(apply_h(H, src), gt) - dst_m
    loo = []
    for i in range(len(src)):
        keep = np.arange(len(src)) != i
        try:
            Hi = fit_gcp_transform(src[keep], dst_px[keep], model)
            loo.append(float(np.hypot(*(ref_px_to_map(apply_h(Hi, src[i:i + 1]), gt)[0] - dst_m[i]))))
        except ValueError:
            loo.append(None)
    d = np.hypot(res[:, 0], res[:, 1])
    valid_loo = [v for v in loo if v is not None]
    return {"fit_residual_m": d.round(2).tolist(), "fit_rmse_m": round(float(np.sqrt(np.mean(d ** 2))), 2),
            "leave_one_out_error_m": [None if v is None else round(v, 2) for v in loo],
            "leave_one_out_rmse_m": round(float(np.sqrt(np.mean(np.square(valid_loo)))), 2) if valid_loo else None,
            "note": "fitting diagnostics only; the independent evaluation is Phase 6"}


def run_manual(cfg: dict) -> dict:
    from mapalign.gcp import read_gcps
    from mapalign.visualize import plot_gcps, plot_overlay
    from mapalign.preprocess import prepare_historical, prepare_reference

    t0 = time.perf_counter()
    all_fit = read_gcps(cfg["gcp"]["fitting_points"])
    ev = read_gcps(cfg["gcp"]["evaluation_points"])
    overlap = set(all_fit.point_id) & set(ev.point_id)
    if overlap:
        raise ValueError(f"fitting and evaluation sets share point ids: {overlap}")
    qc = datum_offset(cfg)
    excluded = sorted(qc["qc_flagged_z"])
    df = all_fit[~all_fit.point_id.isin(excluded)].reset_index(drop=True)
    log.info("QC excluded fitting points: %s", excluded or "none")
    model = cfg["transform"]["model"]
    with rasterio.open(resolve_path(cfg["modern_reference"]["rendered_path"])) as ds:
        gt, crs, size1 = ds.transform, ds.crs.to_string(), (ds.width, ds.height)
    src = df[["historical_x_px", "historical_y_px"]].to_numpy(float)
    dst_m = df[["true_easting_m", "true_northing_m"]].to_numpy(float)
    H = fit_gcp_transform(src, map_to_ref_px(dst_m, gt), model)
    t_fit = time.perf_counter() - t0
    w, h = cfg["historical_map"]["image_size_px"]
    checks = check_transform(H, (w, h), size1)
    diag = gcp_fit_diagnostics(src, dst_m, H, gt, model)
    grat = graticule_corner_check(cfg, H, gt, crs)
    log.info("Method C (%s, %d GCPs): fit RMSE %.1f m, LOO RMSE %s m, valid=%s", model, len(df),
             diag["fit_rmse_m"], diag["leave_one_out_rmse_m"], checks["valid"])

    out_dir = resolve_path(cfg["experiment"]["output_dir"])
    hist, ref = prepare_historical(cfg), prepare_reference(cfg)
    H_work = ref.full_to_work @ H @ np.linalg.inv(hist.full_to_work)
    plot_overlay(hist.color, ref.color, H_work / H_work[2, 2],
                 f"Manual GCP overlay ({len(df)} AI-picked, human-reviewed fitting points, {model}, least squares)",
                 out_dir / "overlay_manual.png")
    plot_gcps(hist.color, ref.color, apply_h(hist.full_to_work, src), apply_h(ref.full_to_work, map_to_ref_px(dst_m, gt)),
              df.point_id.tolist(), apply_h(H_work, apply_h(hist.full_to_work, src)),
              "Method C fitting GCPs (AI-picked, human-reviewed): residual vectors x20",
              out_dir / "manual_gcp_visualization.png")

    record = {"method": "manual", "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "status": "valid" if checks["valid"] else "failed", "model": model,
              "direction": cfg["transform"]["direction"], "fit": "least squares over all fitting points (no RANSAC)",
              "n_fitting_points": int(len(df)), "fitting_point_ids": df.point_id.tolist(),
              "qc_excluded_fitting_points": {p: qc["qc_flagged_z"][p] for p in excluded},
              "qc_rule": qc["qc_rule"],
              "picked_by": "Claude (AI assistant) from zoomed crops; reviewed unchanged by Saksham Sharma, 2026-10-19",
              "H_full": H.tolist(), "reference_geotransform_gdal": list(gt.to_gdal()), "reference_crs": crs,
              "estimation": checks, "fitting_diagnostics": diag, "graticule_corner_check": grat,
              "timing_s": {"fit_and_diagnostics": round(t_fit, 3)},
              "manual_interaction_time": "not reported: points were picked by an AI assistant, not a human"}
    write_json(record, out_dir / "manual_run.json")
    return record


# ---------------------------------------------------------------- datum / graticule offset

def datum_offset(cfg: dict) -> dict:
    """Offset between true (OSM, WGS84) positions and the printed graticule, from FITTING GCPs only.

    Independent of every automatic method. Compared with the available PROJ datum transformations.
    """
    from pyproj.transformer import TransformerGroup
    from mapalign.gcp import read_gcps

    df = read_gcps(cfg["gcp"]["fitting_points"])
    src = df[["historical_x_px", "historical_y_px"]].to_numpy(float)
    true = df[["true_easting_m", "true_northing_m"]].to_numpy(float)
    off = true - graticule_px_to_map(cfg)(src)
    med = np.median(off, axis=0)
    g = cfg["historical_map"]["neatline_graticule_deg"]
    lon, lat = (g["west"] + g["east"]) / 2, (g["south"] + g["north"]) / 2
    utm = Transformer.from_crs(cfg["crs"]["geographic"], cfg["crs"]["evaluation"], always_xy=True)
    e0, n0 = utm.transform(lon, lat)
    proj = {}
    for code in cfg["crs"]["historical_datum_candidates"]:
        ops = TransformerGroup(code, cfg["crs"]["geographic"], always_xy=True).transformers
        proj[code] = []
        for t in ops:
            lo, la = t.transform(lon, lat)
            e, n = utm.transform(lo, la)
            proj[code].append({"operation": t.description, "accuracy_m": t.accuracy,
                               "shift_E_m": round(e - e0, 1), "shift_N_m": round(n - n0, 1)})
    sigma = 1.4826 * np.median(np.abs(off - med), axis=0)
    z = (off - med) / np.maximum(sigma, 1e-6)
    flagged = {p: zz.round(2).tolist() for p, zz in zip(df.point_id, z) if np.any(np.abs(zz) > QC_Z)}
    return {"n_points": int(len(df)), "per_point_offset_m": {p: o.round(1).tolist() for p, o in zip(df.point_id, off)},
            "qc_rule": f"exclude a FITTING point if |offset - median| > {QC_Z} x 1.4826 x MAD in E or N "
                       "(graticule-based, independent of all methods and of evaluation points)",
            "qc_robust_sigma_E_N_m": sigma.round(1).tolist(), "qc_flagged_z": flagged,
            "median_offset_E_N_m": med.round(1).tolist(), "median_offset_distance_m": round(float(np.hypot(*med)), 1),
            "spread_mad_m": np.median(np.abs(off - med), axis=0).round(1).tolist(),
            "proj_datum_shifts_at_sheet_centre": proj,
            "interpretation": "offset = true WGS84 position minus position implied by the printed graticule "
                              "treated as WGS84; includes datum shift, graticule drafting error and paper distortion"}


# ---------------------------------------------------------------- GeoTIFF output

def output_grid(cfg: dict, H_full: np.ndarray, gt: Affine, res: float) -> tuple[Affine, int, int]:
    """North-up grid (EPSG of the reference) covering the transformed neatline, snapped to res."""
    c = cfg["historical_map"]["neatline_corners_px"]
    poly = np.array([c[k] for k in ("nw", "ne", "se", "sw")], np.float64)
    xy = historical_px_to_map(poly, H_full, gt)
    xmin, ymin = np.floor(xy.min(0) / res) * res
    xmax, ymax = np.ceil(xy.max(0) / res) * res
    return from_origin(xmin, ymax, res, res), int(round((xmax - xmin) / res)), int(round((ymax - ymin) / res))


def warp_to_grid(img: np.ndarray, H_full: np.ndarray, gt_ref: Affine, gt_out: Affine, size: tuple[int, int],
                 interp: int = cv2.INTER_LINEAR, border=0) -> np.ndarray:
    """Resample `img` (historical px) onto an output grid; exact inverse projective mapping per pixel."""
    out_px_to_map = affine_px_to_map_matrix(gt_out)
    map_to_ref = np.linalg.inv(affine_px_to_map_matrix(gt_ref))
    M = np.linalg.inv(H_full) @ map_to_ref @ out_px_to_map      # output px -> historical px
    return cv2.warpPerspective(img, M / M[2, 2], size, flags=interp | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def write_georeferenced(cfg: dict, method: str, H_full: np.ndarray, gt_ref: Affine, crs: str, res: float = 5.0) -> dict:
    img = load_image(resolve_path(cfg["historical_map"]["path"]))
    c = cfg["historical_map"]["neatline_corners_px"]
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.round([c[k] for k in ("nw", "ne", "se", "sw")]).astype(np.int32)], 255)
    gt_out, w, h = output_grid(cfg, H_full, gt_ref, res)
    rgb = warp_to_grid(img, H_full, gt_ref, gt_out, (w, h))[:, :, ::-1]
    alpha = warp_to_grid(mask, H_full, gt_ref, gt_out, (w, h), cv2.INTER_NEAREST)
    out = resolve_path(cfg["experiment"]["output_dir"]) / f"georef_{method}.tif"
    with rasterio.open(out, "w", driver="GTiff", width=w, height=h, count=4, dtype="uint8", crs=crs,
                       transform=gt_out, compress="deflate", predictor=2, tiled=True, photometric="RGB") as ds:
        ds.write(np.moveaxis(rgb, 2, 0), [1, 2, 3])
        ds.write(alpha, 4)
        ds.colorinterp = [rasterio.enums.ColorInterp.red, rasterio.enums.ColorInterp.green,
                          rasterio.enums.ColorInterp.blue, rasterio.enums.ColorInterp.alpha]
        ds.update_tags(method=method, source_sha256=cfg["historical_map"]["sha256"],
                       resampling="bilinear via exact inverse homography (no affine approximation)",
                       H_full=str(np.round(H_full, 10).tolist()))
    meta = {"path": str(out.relative_to(resolve_path("."))), "crs": crs, "resolution_m": res, "width": w, "height": h,
            "transform_gdal": list(gt_out.to_gdal()), "bands": "RGB + alpha (alpha=0 outside neatline)"}
    write_json(meta, out.with_suffix(".json"))
    log.info("Wrote %s (%dx%d at %.1f m)", out, w, h, res)
    return meta


def load_method_transform(cfg: dict, method: str) -> tuple[np.ndarray, Affine, str]:
    rec = read_json(resolve_path(cfg["experiment"]["output_dir"]) / f"{method}_run.json")
    if rec["status"] != "valid" or rec.get("H_full") is None:
        raise RuntimeError(f"{method} has no valid transform (status={rec['status']}); no raster produced")
    return np.array(rec["H_full"]), Affine.from_gdal(*rec["reference_geotransform_gdal"]), rec["reference_crs"]


def verify_raster(cfg: dict, method: str, n_random: int = 2000, seed: int = 0) -> dict:
    """Independent checks of the written file, reading only the file's own georeferencing."""
    from mapalign.gcp import read_gcps

    H, gt_ref, crs = load_method_transform(cfg, method)
    path = resolve_path(cfg["experiment"]["output_dir"]) / f"georef_{method}.tif"
    img = load_image(resolve_path(cfg["historical_map"]["path"]))
    rng = np.random.default_rng(seed)
    c = cfg["historical_map"]["neatline_corners_px"]
    poly = np.array([c[k] for k in ("nw", "ne", "se", "sw")], np.float32)
    pts = []
    while len(pts) < n_random:
        p = rng.uniform(poly.min(0) + 30, poly.max(0) - 30)
        if cv2.pointPolygonTest(poly.reshape(-1, 1, 2), (float(p[0]), float(p[1])), True) > 30:
            pts.append(p)
    pts = np.array(pts)
    xy = historical_px_to_map(pts, H, gt_ref)
    with rasterio.open(path) as ds:
        info = {"crs": ds.crs.to_string(), "transform_gdal": list(ds.transform.to_gdal()), "size": [ds.width, ds.height],
                "count": ds.count, "dtypes": list(ds.dtypes)}
        vals = np.array(list(ds.sample(xy, indexes=[1, 2, 3, 4])))
    src_rgb = img[np.round(pts[:, 1]).astype(int), np.round(pts[:, 0]).astype(int)][:, ::-1].astype(float)
    inside = vals[:, 3] > 0
    diff = np.abs(vals[inside, :3].astype(float) - src_rgb[inside]).mean(1)
    # Same comparison after a deliberate 2-pixel shift, to show the check is sensitive.
    shifted = np.array(list(rasterio.open(path).sample(xy + [2 * info["transform_gdal"][1], 0], indexes=[1, 2, 3])))
    diff_shift = np.abs(shifted[inside].astype(float) - src_rgb[inside]).mean(1)
    gcps = read_gcps(cfg["gcp"]["fitting_points"])
    fit_xy = historical_px_to_map(gcps[["historical_x_px", "historical_y_px"]].to_numpy(float), H, gt_ref)
    res = {"file": info, "crs_ok": info["crs"] == cfg["crs"]["evaluation"],
           "random_points": int(len(pts)), "inside_alpha": int(inside.sum()),
           "mean_abs_rgb_diff_at_mapped_points": round(float(diff.mean()), 2),
           "mean_abs_rgb_diff_with_2px_shift_control": round(float(diff_shift.mean()), 2),
           "fitting_gcps_mapped_m": {p: v.round(1).tolist() for p, v in zip(gcps.point_id, fit_xy)}}
    log.info("verify %s: %s", method, {k: res[k] for k in ("crs_ok", "mean_abs_rgb_diff_at_mapped_points",
                                                           "mean_abs_rgb_diff_with_2px_shift_control")})
    return res


def vector_check_figure(cfg: dict, method: str, out, half_m: float = 900.0) -> None:
    """QGIS-style check: OSM railways/major roads/rivers drawn on GeoTIFF windows using ONLY the file's
    own CRS + geotransform. Windows are centred on fixed landmarks chosen before evaluation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    windows = {"Srirangapatna": (683000, 1373300), "Gumbaz / Ganjam": (686300, 1372500),
               "Mysore station": (676600, 1361700), "Belagula rail": (672570, 1369980),
               "Elivala": (667740, 1366740), "Varuna": (689410, 1357010)}
    data = read_json(resolve_path(cfg["modern_reference"]["raw_path"]))
    tf = Transformer.from_crs(cfg["crs"]["geographic"], cfg["crs"]["evaluation"], always_xy=True)
    lines = []
    for el in data["elements"]:
        t = el.get("tags", {})
        style = ("k", 1.4) if t.get("railway") == "rail" else ("b", 1.2) if t.get("waterway") in ("river",) else \
            ("m", 1.0) if t.get("highway") in ("trunk", "primary", "secondary", "tertiary") else None
        if style and el["type"] == "way" and "geometry" in el:
            e, n = tf.transform([p["lon"] for p in el["geometry"]], [p["lat"] for p in el["geometry"]])
            lines.append((np.column_stack([e, n]), style))
    path = resolve_path(cfg["experiment"]["output_dir"]) / f"georef_{method}.tif"
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    with rasterio.open(path) as ds:
        for ax, (name, (cx, cy)) in zip(axes.ravel(), windows.items()):
            win = rasterio.windows.from_bounds(cx - half_m, cy - half_m, cx + half_m, cy + half_m, ds.transform)
            img = np.moveaxis(ds.read([1, 2, 3], window=win, boundless=True, fill_value=255), 0, 2)
            ax.imshow(img, extent=(cx - half_m, cx + half_m, cy - half_m, cy + half_m))
            for xy, (col, lw) in lines:
                m = (np.abs(xy[:, 0] - cx) < 2 * half_m) & (np.abs(xy[:, 1] - cy) < 2 * half_m)
                if m.any():
                    ax.plot(xy[:, 0], xy[:, 1], color=col, lw=lw, alpha=0.8)
            ax.set_xlim(cx - half_m, cx + half_m), ax.set_ylim(cy - half_m, cy + half_m)
            ax.set_title(name, fontsize=10)
            ax.tick_params(labelsize=7)
    fig.suptitle(f"georef_{method}.tif with OSM vectors (black rail, blue river, magenta roads), drawn via the file's own "
                 f"EPSG:32643 geotransform; axes in metres", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["manual", "raster", "verify", "datum"])
    p.add_argument("method", nargs="?", choices=["sift", "loftr", "manual"])
    args = p.parse_args(argv)
    cfg = load_config()
    out_dir = resolve_path(cfg["experiment"]["output_dir"])
    if args.cmd == "manual":
        run_manual(cfg)
    elif args.cmd == "datum":
        r = datum_offset(cfg)
        write_json(r, out_dir / "datum_offset.json")
        log.info("Graticule offset (median of %d fitting GCPs): %s m", r["n_points"], r["median_offset_E_N_m"])
    elif args.cmd in ("raster", "verify"):
        if args.method is None:
            p.error("method required")
        try:
            H, gt_ref, crs = load_method_transform(cfg, args.method)
        except RuntimeError as e:
            log.error("%s", e)
            return 2
        if args.cmd == "raster":
            write_georeferenced(cfg, args.method, H, gt_ref, crs)
        else:
            write_json(verify_raster(cfg, args.method), out_dir / f"georef_{args.method}_verification.json")
            vector_check_figure(cfg, args.method, out_dir / f"georef_{args.method}_vector_check.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())

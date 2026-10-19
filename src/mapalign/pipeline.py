"""Run one automatic baseline end to end: prepare -> match -> RANSAC -> diagnostics -> outputs.

Usage:
    python -m mapalign.pipeline sift
    python -m mapalign.pipeline loftr      # Phase 4

Outputs (results/):
    <method>_matches.png          correspondences, inliers vs outliers
    overlay_<method>.png          historical map warped onto the reference
    <method>_correspondences.npz  raw matches (working px), scores, inlier mask
    <method>_run.json             parameters, counts, timings, transforms, diagnostics
No held-out evaluation points are read here.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata as md
import logging
import platform
import sys
import time

import numpy as np
from pyproj import Transformer

from mapalign.config import DEFAULT_CONFIG, load_config, resolve_path
from mapalign.diagnostics import (aligned_tile_test, corner_deviation_from_prior, prior_consistency,
                                  prior_homography_work, sift_tile_matcher)
from mapalign.estimate_transform import apply_h, compose_full_resolution, estimate_transform, estimator_params
from mapalign.georeference import graticule_corner_check, map_to_ref_px
from mapalign.io import write_json
from mapalign.match_sift import MatchResult, match_sift
from mapalign.preprocess import PreparedImage, prepare_historical, prepare_reference
from mapalign.visualize import plot_matches, plot_overlay

log = logging.getLogger("mapalign.pipeline")


def _run_matcher(method: str, hist: PreparedImage, ref: PreparedImage, cfg: dict) -> MatchResult:
    if method == "sift":
        return match_sift(hist.gray, ref.gray, cfg["sift"])
    if method == "loftr":
        from mapalign.match_loftr import match_loftr  # Phase 4
        return match_loftr(hist.gray, ref.gray, cfg["loftr"], cfg["compute"]["device"])
    raise ValueError(method)


def _tile_matcher(method: str, cfg: dict):
    if method == "sift":
        return sift_tile_matcher(cfg["sift"])
    if method == "loftr":
        from mapalign.match_loftr import loftr_tile_matcher  # Phase 4
        return loftr_tile_matcher(cfg["loftr"], cfg["compute"]["device"])
    return None


def _corner_points_work(cfg: dict, hist: PreparedImage, ref: PreparedImage, H_work: np.ndarray | None):
    """Printed sheet corners in reference working px: from graticule, and via the transform."""
    g = cfg["historical_map"]["neatline_graticule_deg"]
    order = ("nw", "ne", "se", "sw")
    ll = {"nw": (g["west"], g["north"]), "ne": (g["east"], g["north"]),
          "se": (g["east"], g["south"]), "sw": (g["west"], g["south"])}
    tf = Transformer.from_crs(cfg["crs"]["geographic"], ref.crs, always_xy=True)
    utm = np.array([tf.transform(*ll[k]) for k in order])
    grat = apply_h(ref.full_to_work, map_to_ref_px(utm, ref.geotransform))
    pred = None
    if H_work is not None:
        px = np.array([cfg["historical_map"]["neatline_corners_px"][k] for k in order], np.float64)
        pred = apply_h(H_work @ hist.full_to_work, px)
    return grat, pred


def run(method: str, cfg: dict) -> dict:
    seed = cfg["experiment"]["random_seed"]
    np.random.seed(seed)
    out_dir = resolve_path(cfg["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    t0 = time.perf_counter()
    hist, ref = prepare_historical(cfg), prepare_reference(cfg)
    t_prep = time.perf_counter() - t0

    t0 = time.perf_counter()
    m = _run_matcher(method, hist, ref, cfg)
    t_match = time.perf_counter() - t0

    size0 = (hist.gray.shape[1], hist.gray.shape[0])
    size1 = (ref.gray.shape[1], ref.gray.shape[0])
    t0 = time.perf_counter()
    matcher_scale = m.stats.get("common_scale", 1.0)
    tr = estimate_transform(m.pts0, m.pts1, size0, size1, estimator_params(cfg["transform"], matcher_scale), seed)
    t_est = time.perf_counter() - t0
    t_total = time.perf_counter() - t_start

    H_full = None if tr.matrix is None else compose_full_resolution(tr.matrix, hist.full_to_work, ref.full_to_work)
    grat_check = None if H_full is None else graticule_corner_check(cfg, H_full, ref.geotransform, ref.crs)
    if grat_check:
        log.info("Graticule corner check (coarse): mean %.0f m, max %.0f m",
                 grat_check["mean_distance_m"], grat_check["max_distance_m"])

    # Development diagnostics against the graticule-corner prior (no evaluation points involved).
    res_m = cfg["preprocessing"]["working_resolution_m"]
    H_prior = prior_homography_work(cfg, hist, ref)
    diag = {"prior_consistency": prior_consistency(m.pts0, m.pts1, H_prior, res_m, tr.inlier_mask),
            "max_corner_deviation_from_prior_m": corner_deviation_from_prior(tr.matrix, H_prior, size0, res_m)}
    tile_matcher = _tile_matcher(method, cfg)
    if tile_matcher is not None:
        diag["aligned_tile_upper_bound"] = aligned_tile_test(hist, ref, H_prior, tile_matcher)
    log.info("Prior diagnostics: %s", diag)
    prior_png = out_dir / "overlay_graticule_prior.png"
    if not prior_png.exists():
        plot_overlay(hist.color, ref.color, H_prior,
                     "Diagnostic only: alignment from the 4 printed sheet corners (graticule treated as WGS84)",
                     prior_png)

    # Outputs
    np.savez_compressed(out_dir / f"{method}_correspondences.npz", pts0_work=m.pts0, pts1_work=m.pts1,
                        scores=m.scores, inlier_mask=tr.inlier_mask)
    plot_matches(hist.color, ref.color, m.pts0, m.pts1, tr.inlier_mask,
                 f"{method.upper()} + {cfg['transform']['ransac_method']} ({cfg['transform']['model']})",
                 out_dir / f"{method}_matches.png", seed=seed)
    grat_px, pred_px = _corner_points_work(cfg, hist, ref, tr.matrix)
    if tr.matrix is not None:
        status = "VALID" if tr.ok else "INVALID: " + "; ".join(tr.stats.get("problems", []))
        plot_overlay(hist.color, ref.color, tr.matrix,
                     f"{method.upper()} overlay ({status}). Blue + = printed sheet corner from graticule "
                     f"(datum unverified), red x = same corner via estimated transform",
                     out_dir / f"overlay_{method}.png", grat_px, pred_px)

    record = {
        "method": method,
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "status": "valid" if tr.ok else "failed",
        "direction": cfg["transform"]["direction"],
        "frames": {
            "H_work": "historical working px -> reference working px",
            "H_full": "historical full-res scan px -> reference GeoTIFF px (both OpenCV centre convention)",
            "reference_geotransform": "rasterio corner-based; map = T*(col+0.5, row+0.5)",
        },
        "H_work": None if tr.matrix is None else tr.matrix.tolist(),
        "H_full": None if H_full is None else H_full.tolist(),
        "historical_full_to_work": hist.full_to_work.tolist(),
        "reference_full_to_work": ref.full_to_work.tolist(),
        "reference_geotransform_gdal": list(ref.geotransform.to_gdal()),
        "reference_crs": ref.crs,
        "working_sizes": {"historical": list(size0), "reference": list(size1)},
        "working_resolution_m": {"historical_nominal": hist.nominal_resolution_m, "reference": ref.nominal_resolution_m},
        "matching": m.stats,
        "estimation": tr.stats,
        "graticule_corner_check": grat_check,
        "prior_diagnostics": diag,
        "timing_s": {"preprocess": round(t_prep, 2), "matching": round(t_match, 2),
                     "estimation": round(t_est, 3), "total": round(t_total, 2)},
        "device": "cpu" if method == "sift" else m.stats.get("device"),
        "parameters": {"preprocessing": cfg["preprocessing"], method: cfg[method], "transform": cfg["transform"]},
        "config_frozen": cfg["experiment"]["frozen"],
        "inputs": {"historical_sha256": cfg["historical_map"]["sha256"],
                   "reference_render": cfg["modern_reference"]["rendered_path"]},
        "software": {p: md.version(p) for p in ("opencv-python-headless", "numpy", "torch", "kornia")},
        "platform": platform.platform(),
    }
    write_json(record, out_dir / f"{method}_run.json")
    log.info("%s finished: status=%s, %d matches, %d inliers, total %.1f s",
             method, record["status"], len(m.pts0), tr.stats.get("inliers", 0), t_total)
    return record


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("method", choices=["sift", "loftr"])
    p.add_argument("--config", default=DEFAULT_CONFIG)
    args = p.parse_args(argv)
    rec = run(args.method, load_config(args.config))
    return 0 if rec["status"] == "valid" else 2


if __name__ == "__main__":
    sys.exit(main())

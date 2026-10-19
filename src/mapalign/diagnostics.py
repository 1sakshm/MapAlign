"""Development diagnostics shared by all automatic methods (no held-out evaluation data).

The coarse *graticule prior* is the homography defined by the four printed neatline corners and
their printed latitude/longitude, with the latter treated as WGS84. It ignores the unverified
datum shift and paper distortion, so it is only trusted to a few hundred metres. It is used to:
  * tell whether proposed matches are roughly correct (prior consistency), and
  * run an upper-bound test: can a matcher find correct matches on tiles already aligned by the prior?

Usage:
    python -m mapalign.diagnostics sift-variants   # exploratory preprocessing sweep for SIFT
    python -m mapalign.diagnostics loftr-sizes     # exploratory LoFTR input-size sweep
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
import time

import cv2
import numpy as np
from pyproj import Transformer

from mapalign.config import load_config, resolve_path
from mapalign.estimate_transform import apply_h, estimate_transform, estimator_params
from mapalign.georeference import map_to_ref_px
from mapalign.io import write_json
from mapalign.preprocess import PreparedImage, prepare_historical, prepare_reference

log = logging.getLogger("mapalign.diagnostics")

ORDER = ("nw", "ne", "se", "sw")


def prior_homography_work(cfg: dict, hist: PreparedImage, ref: PreparedImage) -> np.ndarray:
    """Historical working px -> reference working px, from the 4 printed corners only."""
    g = cfg["historical_map"]["neatline_graticule_deg"]
    ll = {"nw": (g["west"], g["north"]), "ne": (g["east"], g["north"]),
          "se": (g["east"], g["south"]), "sw": (g["west"], g["south"])}
    tf = Transformer.from_crs(cfg["crs"]["geographic"], ref.crs, always_xy=True)
    dst = apply_h(ref.full_to_work, map_to_ref_px(np.array([tf.transform(*ll[k]) for k in ORDER]), ref.geotransform))
    px = cfg["historical_map"]["neatline_corners_px"]
    src = apply_h(hist.full_to_work, np.array([px[k] for k in ORDER], np.float64))
    return cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32)).astype(np.float64)


def prior_consistency(pts0: np.ndarray, pts1: np.ndarray, H_prior: np.ndarray, res_m: float,
                      inlier_mask: np.ndarray | None = None, radius_m: float = 1000.0) -> dict:
    """How many matches land within radius_m of where the prior predicts."""
    if len(pts0) == 0:
        return {"radius_m": radius_m, "matches_within": 0, "matches": 0, "inliers_within": 0, "inliers": 0}
    d = np.linalg.norm(apply_h(H_prior, pts0) - pts1, axis=1) * res_m
    near = d < radius_m
    inl = np.zeros(len(pts0), bool) if inlier_mask is None else inlier_mask.astype(bool)
    return {"radius_m": radius_m, "matches": int(len(pts0)), "matches_within": int(near.sum()),
            "inliers": int(inl.sum()), "inliers_within": int((near & inl).sum()),
            "median_distance_m": round(float(np.median(d)), 1)}


def corner_deviation_from_prior(H: np.ndarray | None, H_prior: np.ndarray, size0: tuple[int, int],
                                res_m: float) -> float | None:
    if H is None:
        return None
    w, h = size0
    cs = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float64)
    return round(float(np.max(np.linalg.norm(apply_h(H, cs) - apply_h(H_prior, cs), axis=1)) * res_m), 1)


def aligned_tile_test(hist: PreparedImage, ref: PreparedImage, H_prior: np.ndarray, matcher, tile: int = 512,
                      radius_px: float = 60.0, start: int = 450, stop: int = 3000) -> dict:
    """Upper bound: match prior-aligned tiles and count matches within radius_px of zero displacement.

    `matcher(gray_a, gray_b) -> (pts_a, pts_b)`. The chance rate is the fraction of the tile area
    within radius_px, i.e. what randomly placed matches would score.
    """
    h, w = ref.gray.shape
    warped = cv2.warpPerspective(hist.gray, H_prior, (w, h), borderValue=255)
    total = correct = n_tiles = 0
    for y in range(start, min(stop, h - tile), tile):
        for x in range(start, min(stop, w - tile), tile):
            pa, pb = matcher(warped[y:y + tile, x:x + tile], ref.gray[y:y + tile, x:x + tile])
            n_tiles += 1
            if len(pa):
                total += len(pa)
                correct += int((np.linalg.norm(pa - pb, axis=1) < radius_px).sum())
    chance = np.pi * radius_px ** 2 / tile ** 2
    return {"tiles": n_tiles, "tile_px": tile, "radius_px": radius_px, "matches": total, "correct": correct,
            "correct_fraction": round(correct / total, 4) if total else None, "chance_fraction": round(chance, 4)}


def sift_tile_matcher(params: dict):
    sift = cv2.SIFT_create(nfeatures=params["n_features"], contrastThreshold=params["contrast_threshold"],
                           edgeThreshold=params["edge_threshold"], sigma=params["sigma"])
    bf = cv2.BFMatcher(cv2.NORM_L2)

    def match(a, b):
        k0, d0 = sift.detectAndCompute(a, None)
        k1, d1 = sift.detectAndCompute(b, None)
        if d0 is None or d1 is None or len(k1) < 2:
            return np.empty((0, 2)), np.empty((0, 2))
        good = [m for m, n in bf.knnMatch(d0, d1, k=2) if m.distance < params["lowe_ratio"] * n.distance]
        return (np.array([k0[m.queryIdx].pt for m in good]).reshape(-1, 2),
                np.array([k1[m.trainIdx].pt for m in good]).reshape(-1, 2))
    return match


def colour_layers(color: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Exploratory representation: red linework (roads/settlements), blue water, dark linework."""
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    h, s, v = (hsv[..., i].astype(int) for i in range(3))
    out = np.full(color.shape[:2], 255, np.uint8)
    out[(h >= 85) & (h <= 130) & (s >= 25) & (v >= 90)] = 160
    out[((h >= 165) | (h <= 6)) & (s >= 70) & (v >= 90)] = 60
    out[v < 70] = 0
    return cv2.GaussianBlur(out, (0, 0), sigma) if sigma > 0 else out


def sift_variants(cfg: dict) -> list[dict]:
    """Exploratory sweep (documented, not used to tune on evaluation data)."""
    from mapalign.match_sift import match_sift

    rows = []
    for name, res, rep in [("gray_10m", 10, "gray"), ("clahe_10m", 10, "clahe"), ("gray_20m", 20, "gray"),
                           ("layers_10m", 10, "layers"), ("layers_20m", 20, "layers"), ("layers_30m", 30, "layers"),
                           # ~LoFTR's selected effective resolution (768 px input), added for parity
                           ("gray_45m", 45, "gray"), ("layers_45m", 45, "layers")]:
        c = copy.deepcopy(cfg)
        c["preprocessing"].update(working_resolution_m=res, clahe=rep == "clahe")
        hist, ref = prepare_historical(c), prepare_reference(c)
        g0, g1 = (colour_layers(hist.color), colour_layers(ref.color)) if rep == "layers" else (hist.gray, ref.gray)
        Hp = prior_homography_work(c, hist, ref)
        t0 = time.perf_counter()
        m = match_sift(g0, g1, c["sift"])
        tr = estimate_transform(m.pts0, m.pts1, g0.shape[::-1], g1.shape[::-1],
                                estimator_params(c["transform"], 1.0), c["experiment"]["random_seed"])
        rows.append({"variant": name, "working_resolution_m": res, "representation": rep,
                     "keypoints": [m.stats["keypoints0"], m.stats["keypoints1"]],
                     "ratio_matches": len(m.pts0), "inliers": tr.stats.get("inliers", 0), "valid": tr.ok,
                     "prior_consistency_1km": prior_consistency(m.pts0, m.pts1, Hp, res, tr.inlier_mask),
                     "max_corner_deviation_from_prior_m": corner_deviation_from_prior(tr.matrix, Hp, g0.shape[::-1], res),
                     "seconds": round(time.perf_counter() - t0, 1)})
        log.info("%s", rows[-1])
    return rows


def loftr_sizes(cfg: dict, sizes=(384, 512, 640, 768, 896, 1024, 1280, 1536)) -> list[dict]:
    """Exploratory LoFTR input-size sweep. Selection criterion (fixed in advance): the most
    matches within 1 km of the graticule prior. Out-of-memory sizes are recorded, not hidden."""
    from mapalign.match_loftr import match_loftr

    hist, ref = prepare_historical(cfg), prepare_reference(cfg)
    Hp = prior_homography_work(cfg, hist, ref)
    res = cfg["preprocessing"]["working_resolution_m"]
    rows = []
    for side in sizes:
        row: dict = {"max_side_px": side}
        try:
            m = match_loftr(hist.gray, ref.gray, {**cfg["loftr"], "max_side_px": side}, cfg["compute"]["device"])
            pc = prior_consistency(m.pts0, m.pts1, Hp, res)
            row.update(matches=len(m.pts0), within_1km_of_prior=pc["matches_within"],
                       fraction_within=round(pc["matches_within"] / max(len(m.pts0), 1), 3),
                       effective_resolution_m=round(res / m.stats["common_scale"], 1),
                       inference_s=m.stats["inference_s"])
        except RuntimeError as e:  # e.g. CPU out of memory
            row.update(error=str(e).splitlines()[0][:200])
        rows.append(row)
        log.info("%s", row)
    return rows


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["sift-variants", "loftr-sizes"])
    args = p.parse_args(argv)
    cfg = load_config()
    if args.what == "loftr-sizes":
        rows = loftr_sizes(cfg)
        out = resolve_path(cfg["experiment"]["output_dir"]) / "exploration" / "loftr_input_size_sweep.json"
        write_json({"purpose": "exploratory; selection criterion fixed in advance: most matches within 1 km "
                               "of the graticule prior (development data only)", "sizes": rows}, out)
        log.info("Wrote %s", out)
    if args.what == "sift-variants":
        rows = sift_variants(cfg)
        out = resolve_path(cfg["experiment"]["output_dir"]) / "exploration" / "sift_preprocessing_variants.json"
        write_json({"purpose": "exploratory; scored only against the graticule-corner prior (development data)",
                    "variants": rows}, out)
        log.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

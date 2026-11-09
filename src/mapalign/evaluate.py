"""Shared positional-error evaluation on the held-out GCPs (metres, EPSG:32643).

Every method is scored by ONE function: predicted = H_full applied to the historical pixel, then
the reference geotransform; error = Euclidean distance to the OSM-derived truth in EPSG:32643.
A method without a valid transform gets a recorded failure for every point (no metrics, no
silent dropping).

Usage:
    python -m mapalign.evaluate            # primary (homography) + secondary (affine) evaluation

Outputs (results/): evaluation_errors.csv, metrics.csv, metrics.json, error_vectors_<method>.png,
per_point_errors.png, experiment_metadata.json
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata as md
import logging
import platform
import sys

import numpy as np
import pandas as pd
from affine import Affine

from mapalign.config import load_config, resolve_path
from mapalign.core import paired_comparison, positional_errors, predict, summarize_errors  # noqa: F401 (re-exported)
from mapalign.estimate_transform import apply_h, check_transform, estimate_transform, estimator_params
from mapalign.georeference import fit_gcp_transform, map_to_ref_px, ref_px_to_map
from mapalign.io import read_json, write_json

log = logging.getLogger("mapalign.evaluate")

METHODS = ("sift", "loftr", "manual")
LABELS = {"sift": "SIFT + MAGSAC++", "loftr": "LoFTR + MAGSAC++", "manual": "Manual GCPs (Method C)"}
# The affine re-fit uses cv2.estimateAffine2D, which has no MAGSAC++, so those rows say RANSAC.
AFFINE_LABELS = {**LABELS, "sift": "SIFT + RANSAC", "loftr": "LoFTR + RANSAC"}


class EvaluationLockError(RuntimeError):
    """The evaluation-point file differs from the locked version."""


# ---------------------------------------------------------------- inputs

def verify_lock(cfg: dict) -> str:
    path = resolve_path(cfg["gcp"]["evaluation_points"])
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    lock = read_json(resolve_path("data/raw/manifest.json")).get("evaluation_points_lock")
    if lock is None:
        raise EvaluationLockError("evaluation points were never locked (data/raw/manifest.json)")
    if lock["sha256"] != h:
        raise EvaluationLockError(f"{path} changed after locking (sha256 {h} != {lock['sha256']})")
    return h


def load_points(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    from mapalign.gcp import read_gcps

    ev = read_gcps(cfg["gcp"]["evaluation_points"])
    fit = read_gcps(cfg["gcp"]["fitting_points"])
    if set(ev.point_id) & set(fit.point_id):
        raise ValueError("evaluation and fitting point ids overlap")
    return ev, fit


def method_transforms(cfg: dict) -> dict:
    """{method: (H_full or None, geotransform or None, record)} from the saved run files."""
    out = {}
    res_dir = resolve_path(cfg["experiment"]["output_dir"])
    for m in METHODS:
        p = res_dir / f"{m}_run.json"
        if not p.is_file():
            out[m] = (None, None, {"status": "not run"})
            continue
        rec = read_json(p)
        H = np.array(rec["H_full"]) if rec.get("status") == "valid" and rec.get("H_full") else None
        gt = Affine.from_gdal(*rec["reference_geotransform_gdal"]) if rec.get("reference_geotransform_gdal") else None
        out[m] = (H, gt, rec)
    return out


# ---------------------------------------------------------------- secondary: affine vs homography

def affine_transforms(cfg: dict, fit: pd.DataFrame, tfs: dict) -> dict:
    """Re-estimate each method with an affine model, using the SAME inputs and RANSAC settings.

    Automatic methods: saved correspondences (working px) -> RANSAC affine -> full-res composition.
    Manual: least-squares affine on the same QC-passed fitting points. Evaluation points unused.
    """
    from mapalign.estimate_transform import compose_full_resolution

    res_dir = resolve_path(cfg["experiment"]["output_dir"])
    out = {}
    for m in ("sift", "loftr"):
        rec = tfs[m][2]
        npz = res_dir / f"{m}_correspondences.npz"
        if not npz.is_file() or "historical_full_to_work" not in rec:
            out[m] = (None, None, {"status": "not run"})
            continue
        d = np.load(npz)
        scale = rec["matching"].get("common_scale", 1.0)
        params = estimator_params({**cfg["transform"], "model": "affine"}, scale)
        size0, size1 = tuple(rec["working_sizes"]["historical"]), tuple(rec["working_sizes"]["reference"])
        tr = estimate_transform(d["pts0_work"], d["pts1_work"], size0, size1, params, cfg["experiment"]["random_seed"])
        gt = Affine.from_gdal(*rec["reference_geotransform_gdal"])
        H = None
        if tr.ok:
            H = compose_full_resolution(tr.matrix, np.array(rec["historical_full_to_work"]),
                                        np.array(rec["reference_full_to_work"]))
        out[m] = (H, gt, {"status": "valid" if tr.ok else "failed", "estimation": tr.stats})
    rec = tfs["manual"][2]
    if rec.get("status") == "valid":
        gt = Affine.from_gdal(*rec["reference_geotransform_gdal"])
        used = fit[fit.point_id.isin(rec["fitting_point_ids"])]
        src = used[["historical_x_px", "historical_y_px"]].to_numpy(float)
        dst = map_to_ref_px(used[["true_easting_m", "true_northing_m"]].to_numpy(float), gt)
        A = fit_gcp_transform(src, dst, "affine")
        w, h = cfg["historical_map"]["image_size_px"]
        out["manual"] = (A, gt, {"status": "valid", "estimation": check_transform(A, (w, h), (1, 1)) | {"valid": True}})
    else:
        out["manual"] = (None, None, {"status": "failed"})
    return out


# ---------------------------------------------------------------- run

def evaluate(cfg: dict) -> dict:
    lock_hash = verify_lock(cfg)
    ev, fit = load_points(cfg)
    valid = ev[ev.valid_for_evaluation.astype(bool)].reset_index(drop=True)
    excluded = ev[~ev.valid_for_evaluation.astype(bool)]
    px = valid[["historical_x_px", "historical_y_px"]].to_numpy(float)
    truth = valid[["true_easting_m", "true_northing_m"]].to_numpy(float)
    tfs = method_transforms(cfg)
    seed = cfg["experiment"]["random_seed"]

    rows, metrics = [], {}
    models = {"homography": tfs, "affine": affine_transforms(cfg, fit, tfs)}
    for model, table in models.items():
        for m in METHODS:
            H, gt, rec = table[m]
            pred = predict(H, gt, px)
            e = positional_errors(truth, pred)
            for i, pid in enumerate(valid.point_id):
                rows.append({"model": model, "method": m, "point_id": pid,
                             "true_easting_m": truth[i, 0], "true_northing_m": truth[i, 1],
                             "pred_easting_m": round(pred[i, 0], 2) if np.isfinite(pred[i, 0]) else None,
                             "pred_northing_m": round(pred[i, 1], 2) if np.isfinite(pred[i, 1]) else None,
                             "dE_m": round(pred[i, 0] - truth[i, 0], 2) if np.isfinite(pred[i, 0]) else None,
                             "dN_m": round(pred[i, 1] - truth[i, 1], 2) if np.isfinite(pred[i, 1]) else None,
                             "error_m": round(float(e[i]), 2) if np.isfinite(e[i]) else None,
                             "prediction_status": "ok" if np.isfinite(e[i]) else f"failed: method status {rec.get('status')}",
                             "confidence": valid.confidence[i], "feature_type": valid.feature_type[i]})
            entry = {"model": model, "method": m, "label": (LABELS if model == "homography" else AFFINE_LABELS)[m], "method_status": rec.get("status"),
                     "n_evaluation_points": int(len(valid)), "n_failed_predictions": int((~np.isfinite(e)).sum()),
                     **summarize_errors(e)}
            if model == "homography":
                entry |= _method_extras(m, rec)
            metrics[f"{model}/{m}"] = entry

    err = pd.DataFrame(rows)
    comparisons = {}
    for model in models:
        e = {m: err[(err.model == model) & (err.method == m)].error_m.to_numpy(float) for m in METHODS}
        comparisons[model] = {"loftr_vs_manual": paired_comparison(e["loftr"], e["manual"], seed)}
    # Primary-model sensitivity: drop low-confidence points (reported, NOT a replacement result).
    hi = valid.confidence != "low"
    sens = {}
    for m in METHODS:
        e = err[(err.model == "homography") & (err.method == m)].error_m.to_numpy(float)
        sens[m] = summarize_errors(e[hi.to_numpy()])
    return {"errors": err, "metrics": metrics, "comparisons": comparisons, "lock_sha256": lock_hash,
            "excluded_points": excluded[["point_id", "exclusion_reason"]].to_dict("records"),
            "sensitivity_without_low_confidence": {"n_points": int(hi.sum()), "metrics": sens},
            "valid_points": valid, "transforms": models}


def _method_extras(m: str, rec: dict) -> dict:
    est, mat = rec.get("estimation", {}), rec.get("matching", {})
    out = {"runtime_s": rec.get("timing_s", {}).get("total") or rec.get("timing_s", {}).get("fit_and_diagnostics"),
           "device": rec.get("device")}
    if m in ("sift", "loftr"):
        out |= {"proposed_matches": est.get("input_correspondences"), "ransac_inliers": est.get("inliers"),
                "inlier_ratio": est.get("inlier_ratio"),
                "inlier_reprojection_rmse_working_px": est.get("inlier_reproj_rmse_px")}
    else:
        d = rec.get("fitting_diagnostics", {})
        out |= {"n_fitting_points": rec.get("n_fitting_points"), "fit_rmse_m": d.get("fit_rmse_m"),
                "leave_one_out_rmse_m": d.get("leave_one_out_rmse_m"),
                "manual_interaction_time": rec.get("manual_interaction_time")}
    return out


def write_outputs(cfg: dict, r: dict) -> None:
    from mapalign.visualize import plot_error_vectors, plot_per_point_errors
    from mapalign.preprocess import prepare_historical

    out = resolve_path(cfg["experiment"]["output_dir"])
    r["errors"].to_csv(out / "evaluation_errors.csv", index=False)
    mdf = pd.DataFrame(r["metrics"].values())
    mdf.to_csv(out / "metrics.csv", index=False)
    write_json({"definitions": {
                    "error_m": "Euclidean distance in EPSG:32643 between predicted and OSM-derived true position",
                    "rmse_m": "sqrt(mean(e_i^2))", "std_m": "population standard deviation (ddof=0)",
                    "p90_m": "90th percentile (linear interpolation)",
                    "inlier_reprojection_rmse_working_px": "image-space fit of RANSAC inliers; NOT positional accuracy",
                    "fit_rmse_m / leave_one_out_rmse_m": "Method C fitting diagnostics on fitting points; NOT evaluation"},
                "primary_model": "homography", "metrics": r["metrics"], "paired_comparisons": r["comparisons"],
                "sensitivity_without_low_confidence": r["sensitivity_without_low_confidence"],
                "excluded_points": r["excluded_points"], "evaluation_points_sha256": r["lock_sha256"],
                "caveats": ["12 evaluation points (target was 20); none in the southern fifth or Mysore city",
                            "fitting and evaluation points were picked by an AI assistant and reviewed unchanged by a human (gcp/PROTOCOL.md)",
                            "truth is OSM vector geometry of unknown accuracy (plausibly 5-20 m)",
                            "single historical sheet; results do not generalise beyond it"]},
               out / "metrics.json")
    hist = prepare_historical(cfg)
    valid = r["valid_points"]
    px_full = valid[["historical_x_px", "historical_y_px"]].to_numpy(float)
    for m in METHODS:
        sub = r["errors"][(r["errors"].model == "homography") & (r["errors"].method == m)]
        plot_error_vectors(hist.color, apply_h(hist.full_to_work, px_full), sub, LABELS[m],
                           r["metrics"][f"homography/{m}"], out / f"error_vectors_{m}.png")
    plot_per_point_errors(r["errors"], out / "per_point_errors.png")

    write_json({"created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "config_frozen": cfg["experiment"]["frozen"], "frozen_on": cfg["experiment"]["frozen_on"],
                "random_seed": cfg["experiment"]["random_seed"],
                "inputs": {"historical_sha256": cfg["historical_map"]["sha256"],
                           "evaluation_points_sha256": r["lock_sha256"],
                           "manifest": read_json(resolve_path("data/raw/manifest.json"))},
                "crs": cfg["crs"], "transform": cfg["transform"], "preprocessing": cfg["preprocessing"],
                "sift": cfg["sift"], "loftr": cfg["loftr"],
                "software": {p: md.version(p) for p in ("torch", "kornia", "opencv-python-headless", "numpy",
                                                        "rasterio", "pyproj", "scipy", "pandas")},
                "platform": platform.platform(), "python": sys.version}, out / "experiment_metadata.json")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    if not cfg["experiment"]["frozen"]:
        log.error("Refusing to evaluate: configuration is not frozen")
        return 2
    r = evaluate(cfg)
    write_outputs(cfg, r)
    for k, v in r["metrics"].items():
        log.info("%-20s status=%-7s n=%s median=%s RMSE=%s max=%s", k, v["method_status"], v["n_valid"],
                 v["median_m"], v["rmse_m"], v["max_m"])
    log.info("Paired LoFTR vs manual (homography): %s", r["comparisons"]["homography"]["loftr_vs_manual"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

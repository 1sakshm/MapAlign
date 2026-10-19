"""Robustness check: rerun LoFTR on CUDA and compare it with the reported CPU run.

Usage (in an environment with CUDA torch):
    python -m mapalign.gpu_check

Writes results/gpu/ (the run record, figures, correspondences) and results/gpu/gpu_check.json. The reported
(CPU) results are not modified. All frozen parameters are reused; only compute.device changes.
"""

from __future__ import annotations

import copy
import logging
import sys

import numpy as np
import torch
from affine import Affine

from mapalign.config import load_config, resolve_path
from mapalign.estimate_transform import apply_h
from mapalign.evaluate import load_points, positional_errors, predict, summarize_errors, verify_lock
from mapalign.io import read_json, write_json

log = logging.getLogger("mapalign.gpu_check")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not torch.cuda.is_available():
        log.error("CUDA is not available in this environment (torch %s)", torch.__version__)
        return 2
    from mapalign.pipeline import run

    base = load_config()
    verify_lock(base)
    cfg = copy.deepcopy(base)
    cfg["compute"]["device"] = "cuda"
    cfg["experiment"]["output_dir"] = "results/gpu"
    torch.cuda.reset_peak_memory_stats()
    rec = run("loftr", cfg)
    peak_mb = torch.cuda.max_memory_allocated() / 2**20

    cpu = read_json(resolve_path("results/loftr_run.json"))
    ev, _ = load_points(base)
    px = ev[["historical_x_px", "historical_y_px"]].to_numpy(float)
    truth = ev[["true_easting_m", "true_northing_m"]].to_numpy(float)
    out = {"gpu_name": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
           "peak_gpu_memory_mb": round(peak_mb, 1)}
    runs = {}
    for name, r in (("cpu", cpu), ("gpu", rec)):
        H = np.array(r["H_full"]) if r.get("H_full") else None
        gt = Affine.from_gdal(*r["reference_geotransform_gdal"])
        e = positional_errors(truth, predict(H, gt, px))
        runs[name] = {"device": r["device"] if name == "gpu" else (r.get("device") or "cpu"), "status": r["status"],
                      "matches": r["estimation"]["input_correspondences"], "inliers": r["estimation"].get("inliers"),
                      "inference_s": r["matching"].get("inference_s"), "total_s": r["timing_s"]["total"],
                      "heldout": summarize_errors(e), "errors_m": np.round(e, 2).tolist(), "H": H}
    Hc, Hg = runs["cpu"].pop("H"), runs["gpu"].pop("H")
    if Hc is not None and Hg is not None:
        w, h = base["historical_map"]["image_size_px"]
        grid = np.stack(np.meshgrid(np.linspace(0, w - 1, 21), np.linspace(0, h - 1, 21)), -1).reshape(-1, 2)
        gt = Affine.from_gdal(*cpu["reference_geotransform_gdal"])
        d = np.linalg.norm(predict(Hc, gt, grid) - predict(Hg, gt, grid), axis=1)
        out["transform_difference_over_sheet_m"] = {"max": round(float(d.max()), 3), "mean": round(float(d.mean()), 3)}
        out["identical_matrices"] = bool(np.allclose(Hc, Hg, rtol=0, atol=1e-12))
    out |= runs
    write_json(out, resolve_path("results/gpu/gpu_check.json"))
    log.info("GPU check: %s", {k: out[k] for k in out if k not in ("cpu", "gpu")})
    log.info("CPU held-out: %s | GPU held-out: %s", runs["cpu"]["heldout"], runs["gpu"]["heldout"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

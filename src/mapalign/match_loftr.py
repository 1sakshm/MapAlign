"""Kornia LoFTR matching (Method A).

LoFTR's coarse stage matches every 1/8-resolution cell of image0 against every cell of image1, so
memory grows with the product of the two image areas. Full-sheet matching at the 10 m working
resolution would need a ~10^10-entry similarity matrix, so both images are downscaled by ONE
common factor so that the largest side is <= `loftr.max_side_px`. Equal ground resolution is
kept, and each image is cropped (right/bottom only) to a multiple of 8 px. Matches are mapped
back to working-resolution pixels, so RANSAC and all diagnostics are shared with SIFT.
"""

from __future__ import annotations

import functools
import logging
import time

import cv2
import numpy as np
import torch

from mapalign.config import ensure_torch_home
from mapalign.match_sift import MatchResult

log = logging.getLogger(__name__)


def resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("compute.device is 'cuda' but CUDA is not available")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


@functools.lru_cache(maxsize=2)
def _load_model(pretrained: str, device: str):
    import kornia.feature as KF

    ensure_torch_home()
    return KF.LoFTR(pretrained=pretrained).to(device).eval()


def _to_tensor(gray: np.ndarray, device: str) -> torch.Tensor:
    """uint8 HxW -> float32 1x1xHxW in [0, 1] (LoFTR's expected input)."""
    return torch.from_numpy(np.ascontiguousarray(gray)).float().div_(255.0)[None, None].to(device)


def _infer(model, g0: np.ndarray, g1: np.ndarray, device: str):
    with torch.inference_mode():
        out = model({"image0": _to_tensor(g0, device), "image1": _to_tensor(g1, device)})
    return (out["keypoints0"].cpu().numpy().astype(np.float64),
            out["keypoints1"].cpu().numpy().astype(np.float64),
            out["confidence"].cpu().numpy().astype(np.float64))


def _shrink(gray: np.ndarray, s: float) -> tuple[np.ndarray, float, float]:
    """Resize by s (pixel-centre aligned), then crop right/bottom to a multiple of 8."""
    h, w = gray.shape
    nw, nh = max(8, int(round(w * s))), max(8, int(round(h * s)))
    small = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA) if s < 1 else gray.copy()
    sx, sy = small.shape[1] / w, small.shape[0] / h
    return small[: small.shape[0] // 8 * 8, : small.shape[1] // 8 * 8], sx, sy


def match_loftr(gray0: np.ndarray, gray1: np.ndarray, params: dict, device: str = "auto") -> MatchResult:
    device = resolve_device(device)
    torch.manual_seed(0)
    max_side = int(params["max_side_px"])
    s = min(1.0, max_side / max(*gray0.shape, *gray1.shape))
    g0, sx0, sy0 = _shrink(gray0, s)
    g1, sx1, sy1 = _shrink(gray1, s)
    log.info("LoFTR input: %s and %s px (common scale %.4f), device=%s", g0.shape[::-1], g1.shape[::-1], s, device)

    t0 = time.perf_counter()
    model = _load_model(params["pretrained"], device)
    t_load = time.perf_counter() - t0
    t1 = time.perf_counter()
    k0, k1, conf = _infer(model, g0, g1, device)
    t_inf = time.perf_counter() - t1

    keep = conf >= params["confidence_threshold"]
    k0, k1, conf = k0[keep], k1[keep], conf[keep]
    # small px -> working px:  x_work = (x_small + 0.5) / s - 0.5
    pts0 = (k0 + 0.5) / [sx0, sy0] - 0.5
    pts1 = (k1 + 0.5) / [sx1, sy1] - 0.5
    stats = {"device": device, "pretrained": params["pretrained"], "common_scale": round(s, 6),
             "loftr_input_px": [list(g0.shape[::-1]), list(g1.shape[::-1])],
             "raw_matches": int(keep.size), "matches_after_confidence": int(keep.sum()),
             "confidence_threshold": params["confidence_threshold"],
             "confidence_mean": round(float(conf.mean()), 4) if len(conf) else None,
             "model_load_s": round(t_load, 2), "inference_s": round(t_inf, 2),
             "torch_threads": torch.get_num_threads()}
    log.info("LoFTR: %d matches (conf >= %.2f) in %.1f s", len(pts0), params["confidence_threshold"], t_inf)
    return MatchResult("loftr", pts0, pts1, conf, stats)


def loftr_tile_matcher(params: dict, device: str = "auto"):
    """Matcher for the aligned-tile upper-bound test: tiles are matched at working resolution."""
    device = resolve_device(device)
    model = _load_model(params["pretrained"], device)

    def match(a: np.ndarray, b: np.ndarray):
        a, b = a[: a.shape[0] // 8 * 8, : a.shape[1] // 8 * 8], b[: b.shape[0] // 8 * 8, : b.shape[1] // 8 * 8]
        k0, k1, conf = _infer(model, a, b, device)
        keep = conf >= params["confidence_threshold"]
        return k0[keep], k1[keep]
    return match

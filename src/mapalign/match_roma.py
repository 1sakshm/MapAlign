"""RoMa dense matching (romatch package), added for the art-history experiment and rerun on the 1926 map.

RoMa predicts a dense warp and a certainty map between the two images, then samples matches from it in
proportion to certainty. It resizes BOTH images to one fixed grid (upsample_res, default 864 px square)
whatever their aspect ratio, and returns coordinates in each original image, which are converted here to the
OpenCV pixel-centre convention used everywhere else.

Weights (RoMa outdoor + DINOv2 ViT-L/14) are cached under TORCH_HOME (the repo's .cache/torch by default).
Sampling is random, so torch is seeded before every call. On CPU the model runs in float32.
"""

from __future__ import annotations

import functools
import logging
import time

import numpy as np
import torch
from PIL import Image

from mapalign.config import ensure_torch_home
from mapalign.core import MatchResult
from mapalign.match_loftr import resolve_device

log = logging.getLogger(__name__)

DEFAULTS = {"coarse_res": 560, "upsample_res": 864, "num_samples": 5000, "seed": 0}


@functools.lru_cache(maxsize=2)
def _load(coarse_res: int, upsample_res: int, device: str):
    from romatch import roma_outdoor

    ensure_torch_home()
    dtype = torch.float16 if device == "cuda" else torch.float32
    return roma_outdoor(device=device, coarse_res=coarse_res, upsample_res=upsample_res, amp_dtype=dtype,
                        use_custom_corr=False)


def to_rgb_pil(img: np.ndarray) -> Image.Image:
    """uint8 gray (HxW) or OpenCV BGR (HxWx3) -> RGB PIL image."""
    if img.ndim == 2:
        return Image.fromarray(np.ascontiguousarray(img)).convert("RGB")
    return Image.fromarray(np.ascontiguousarray(img[..., ::-1]))


def match_roma(img0: np.ndarray, img1: np.ndarray, params: dict | None = None, device: str = "auto") -> MatchResult:
    params = {**DEFAULTS, **(params or {})}
    device = resolve_device(device)
    t0 = time.perf_counter()
    model = _load(int(params["coarse_res"]), int(params["upsample_res"]), device)
    t_load = time.perf_counter() - t0
    (h0, w0), (h1, w1) = img0.shape[:2], img1.shape[:2]
    torch.manual_seed(int(params["seed"]))
    t1 = time.perf_counter()
    with torch.inference_mode():
        warp, cert = model.match(to_rgb_pil(img0), to_rgb_pil(img1), device=device)
        m, c = model.sample(warp, cert, num=int(params["num_samples"]))
        k0, k1 = model.to_pixel_coordinates(m, h0, w0, h1, w1)
    t_inf = time.perf_counter() - t1
    # RoMa's pixel coordinates count from the image's top-left edge; OpenCV's count from the first pixel centre.
    pts0 = k0.cpu().numpy().astype(np.float64) - 0.5
    pts1 = k1.cpu().numpy().astype(np.float64) - 0.5
    # Localisation resolution for the inlier threshold: the coarser of the two resize factors.
    s = min(1.0, params["upsample_res"] / max(h0, w0, h1, w1))
    stats = {"device": device, "common_scale": round(s, 6), **{k: params[k] for k in DEFAULTS},
             "matches": int(len(pts0)), "model_load_s": round(t_load, 2), "inference_s": round(t_inf, 2)}
    log.info("RoMa: %d matches in %.1f s", len(pts0), t_inf)
    return MatchResult("roma", pts0, pts1, c.cpu().numpy().astype(np.float64), stats)

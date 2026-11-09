"""MASt3R matching (naver/mast3r), added for the art-history experiment and rerun on the 1926 map.

MASt3R predicts a 3D point map and a dense local descriptor for both images jointly. Matches are the
reciprocal nearest neighbours of those descriptors (the authors' fast_reciprocal_NNs, as in their README),
with a 3 px border dropped. Only the 2D matches are used here, so they go through the same estimator as every
other matcher.

Each image is resized so its long side is 512 px and centre-cropped to a multiple of 16, exactly as
dust3r.utils.image.load_images does. Matches are mapped back to the input pixels (OpenCV centre convention).

The code is not on PyPI: clone https://github.com/naver/mast3r (with --recursive) and point MAST3R_DIR at it
(default: ../vendor/mast3r next to the repo). The checkpoint is downloaded from Hugging Face into the repo's
.cache/huggingface/hub, whatever HF_HOME says. Licence: CC BY-NC-SA 4.0 (code and weights), research use only; neither is redistributed.
"""

from __future__ import annotations

import functools
import logging
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

from mapalign.config import REPO_ROOT
from mapalign.core import MatchResult
from mapalign.match_loftr import resolve_device
from mapalign.match_roma import to_rgb_pil

log = logging.getLogger(__name__)

CHECKPOINT = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
DEFAULTS = {"size": 512, "subsample": 8, "border_px": 3}


def _import_paths():
    root = os.environ.get("MAST3R_DIR", str(REPO_ROOT.parent / "vendor" / "mast3r"))
    for p in (root, os.path.join(root, "dust3r")):
        if p not in sys.path:
            sys.path.insert(0, p)


@functools.lru_cache(maxsize=1)
def _load(device: str):
    _import_paths()
    from mast3r.model import AsymmetricMASt3R

    from huggingface_hub import snapshot_download

    # An explicit cache_dir, so a user-wide HF_HOME does not trigger a second 2.7 GB download.
    local = snapshot_download(CHECKPOINT, cache_dir=str(REPO_ROOT / ".cache" / "huggingface" / "hub"))
    return AsymmetricMASt3R.from_pretrained(local).to(device).eval()


def _prepare(img: Image.Image, size: int, idx: int):
    """dust3r's load_images for one image; also returns (scale, crop_x0, crop_y0) to map points back."""
    import torchvision.transforms as tvf

    W1, H1 = img.size
    S = max(W1, H1)
    new = tuple(int(round(x * size / S)) for x in img.size)
    img = img.resize(new, Image.LANCZOS if S > size else Image.BICUBIC)
    W, H = img.size
    cx, cy = W // 2, H // 2
    halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
    if W == H:  # dust3r crops square images to 4:3
        halfh = 3 * halfw / 4
    x0, y0 = cx - halfw, cy - halfh
    img = img.crop((x0, y0, cx + halfw, cy + halfh))
    norm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5,) * 3, (0.5,) * 3)])
    view = dict(img=norm(img)[None], true_shape=np.int32([img.size[::-1]]), idx=idx, instance=str(idx))
    return view, (new[0] / W1, new[1] / H1, x0, y0)


def _back(pts: np.ndarray, sx: float, sy: float, x0: float, y0: float) -> np.ndarray:
    """Pixels of the resized, cropped image -> input image pixels (both pixel-centre convention)."""
    return np.stack([(pts[:, 0] + x0 + 0.5) / sx - 0.5, (pts[:, 1] + y0 + 0.5) / sy - 0.5], 1)


def match_mast3r(img0: np.ndarray, img1: np.ndarray, params: dict | None = None, device: str = "auto") -> MatchResult:
    params = {**DEFAULTS, **(params or {})}
    device = resolve_device(device)
    t0 = time.perf_counter()
    model = _load(device)
    t_load = time.perf_counter() - t0
    from dust3r.inference import inference
    from mast3r.fast_nn import fast_reciprocal_NNs

    v0, f0 = _prepare(to_rgb_pil(img0), int(params["size"]), 0)
    v1, f1 = _prepare(to_rgb_pil(img1), int(params["size"]), 1)
    torch.manual_seed(0)
    t1 = time.perf_counter()
    with torch.inference_mode():
        out = inference([(v0, v1)], model, device, batch_size=1, verbose=False)
        d0, d1 = out["pred1"]["desc"].squeeze(0), out["pred2"]["desc"].squeeze(0)
        m0, m1 = fast_reciprocal_NNs(d0, d1, subsample_or_initxy1=int(params["subsample"]), device=device,
                                     dist="dot", block_size=2 ** 13)
    t_inf = time.perf_counter() - t1
    b = int(params["border_px"])
    (H0, W0), (H1, W1) = v0["true_shape"][0], v1["true_shape"][0]
    keep = ((m0[:, 0] >= b) & (m0[:, 0] < W0 - b) & (m0[:, 1] >= b) & (m0[:, 1] < H0 - b)
            & (m1[:, 0] >= b) & (m1[:, 0] < W1 - b) & (m1[:, 1] >= b) & (m1[:, 1] < H1 - b))
    pts0 = _back(m0[keep].astype(np.float64), *f0)
    pts1 = _back(m1[keep].astype(np.float64), *f1)
    s = min(1.0, f0[0], f0[1], f1[0], f1[1])
    stats = {"device": device, "common_scale": round(float(s), 6), **params, "checkpoint": CHECKPOINT,
             "input_px": [[int(W0), int(H0)], [int(W1), int(H1)]], "matches": int(len(pts0)),
             "model_load_s": round(t_load, 2), "inference_s": round(t_inf, 2)}
    log.info("MASt3R: %d matches in %.1f s", len(pts0), t_inf)
    return MatchResult("mast3r", pts0, pts1, np.ones(len(pts0)), stats)

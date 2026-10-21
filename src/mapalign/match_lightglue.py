"""Sparse learned keypoints (DISK or ALIKED) + LightGlue matching (Kornia), a second family of learned matchers.

Added after the held-out evaluation (post hoc, secondary). It answers whether LoFTR is special or whether
another modern learned matcher does as well. It plugs into the same preprocessing, the same MAGSAC++
estimator and the same validity checks as Methods A and B.

DISK and ALIKED are learned keypoint detectors/descriptors; LightGlue is a learned matcher over sparse keypoints. Unlike
LoFTR it is detector-based, and its memory grows with the number of keypoints rather than with the product of
the image areas, so it can run at a larger input size. As for LoFTR, both images are scaled by ONE common
factor (largest side <= max_side_px) so that their ground resolution stays equal.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import time

import numpy as np
import torch

from mapalign.config import ensure_torch_home
from mapalign.match_loftr import _shrink, resolve_device
from mapalign.match_sift import MatchResult

log = logging.getLogger(__name__)

DEFAULTS = {"features": "disk", "max_side_px": 1536, "n_keypoints": 4096, "filter_threshold": 0.1}
# The SHA-256 of every checkpoint used is recorded in the output JSON (weight_hashes()).


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def weight_hashes() -> dict:
    """SHA-256 of every checkpoint this matcher loaded from the torch hub cache."""
    ck = ensure_torch_home() / "hub" / "checkpoints"
    keys = ("disk", "depth-save", "aliked")
    return {p.name: _sha256(p) for p in sorted(ck.glob("*")) if any(k in p.name.lower() for k in keys)}


@functools.lru_cache(maxsize=2)
def _load(features: str, n: int, device: str, filter_threshold: float):
    import kornia.feature as KF

    ensure_torch_home()
    if features == "disk":
        det = KF.DISK.from_pretrained("depth", device=torch.device(device))
    elif features == "aliked":
        det = KF.ALIKED.from_pretrained("aliked-n16", max_num_keypoints=n, device=torch.device(device)).eval()
    else:
        raise ValueError(features)
    lg = KF.LightGlueMatcher(features, {"filter_threshold": filter_threshold}).to(device).eval()
    return det, lg


def _features(det, features: str, gray: np.ndarray, n: int, device: str):
    t = torch.from_numpy(np.ascontiguousarray(gray)).float().div_(255.0)[None, None].to(device)
    with torch.inference_mode():
        if features == "disk":
            f = det(t.repeat(1, 3, 1, 1), n=n, pad_if_not_divisible=True)[0]
        else:
            f = det(t.repeat(1, 3, 1, 1))[0]
    return f.keypoints, f.descriptors


def lightglue_pair(gray0: np.ndarray, gray1: np.ndarray, params: dict, device: str):
    """Match two uint8 images at their given resolution. Returns kpts0, kpts1 (px) and match confidence."""
    import kornia.feature as KF

    feat, n = params["features"], int(params["n_keypoints"])
    det, lg = _load(feat, n, device, float(params["filter_threshold"]))
    k0, d0 = _features(det, feat, gray0, n, device)
    k1, d1 = _features(det, feat, gray1, n, device)
    if len(k0) < 4 or len(k1) < 4:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
    lafs0 = KF.laf_from_center_scale_ori(k0[None], torch.ones(1, len(k0), 1, 1, device=device))
    lafs1 = KF.laf_from_center_scale_ori(k1[None], torch.ones(1, len(k1), 1, 1, device=device))
    with torch.inference_mode():
        dists, idxs = lg(d0, d1, lafs0, lafs1, hw1=gray0.shape[:2], hw2=gray1.shape[:2])
    idxs = idxs.cpu().numpy()
    p0 = k0.cpu().numpy().astype(np.float64)[idxs[:, 0]]
    p1 = k1.cpu().numpy().astype(np.float64)[idxs[:, 1]]
    conf = 1.0 - dists.cpu().numpy().astype(np.float64).ravel()  # Kornia returns 1 - matching score
    return p0, p1, conf


def match_lightglue(gray0: np.ndarray, gray1: np.ndarray, params: dict | None = None,
                    device: str = "auto") -> MatchResult:
    params = {**DEFAULTS, **(params or {})}
    device = resolve_device(device)
    torch.manual_seed(0)
    s = min(1.0, int(params["max_side_px"]) / max(*gray0.shape, *gray1.shape))
    g0, sx0, sy0 = _shrink(gray0, s)
    g1, sx1, sy1 = _shrink(gray1, s)
    t0 = time.perf_counter()
    _load(params["features"], int(params["n_keypoints"]), device, float(params["filter_threshold"]))
    t_load = time.perf_counter() - t0
    t1 = time.perf_counter()
    k0, k1, conf = lightglue_pair(g0, g1, params, device)
    t_inf = time.perf_counter() - t1
    pts0 = (k0 + 0.5) / [sx0, sy0] - 0.5
    pts1 = (k1 + 0.5) / [sx1, sy1] - 0.5
    stats = {"device": device, "common_scale": round(s, 6), **{k: params[k] for k in DEFAULTS},
             "input_px": [list(g0.shape[::-1]), list(g1.shape[::-1])], "matches": int(len(pts0)),
             "model_load_s": round(t_load, 2), "inference_s": round(t_inf, 2), "weights_sha256": weight_hashes()}
    log.info("%s+LightGlue: %d matches at max side %d px in %.1f s", params["features"], len(pts0),
             params["max_side_px"], t_inf)
    return MatchResult(f"{params['features']}_lightglue", pts0, pts1, conf, stats)


def lightglue_tile_matcher(params: dict | None = None, device: str = "auto"):
    """Matcher for the aligned-tile upper-bound test (tiles matched at working resolution)."""
    params = {**DEFAULTS, **(params or {})}
    device = resolve_device(device)

    def match(a: np.ndarray, b: np.ndarray):
        k0, k1, _ = lightglue_pair(a, b, params, device)
        return k0, k1
    return match

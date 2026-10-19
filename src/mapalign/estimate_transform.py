"""Shared robust transform estimation and validity diagnostics.

Direction convention: every matrix estimated here maps points of image0 (historical) to
image1 (reference): dst ~ H @ [x, y, 1]. The RANSAC threshold is in image1 pixels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

log = logging.getLogger(__name__)

_METHODS = {"RANSAC": cv2.RANSAC, "USAC_MAGSAC": cv2.USAC_MAGSAC, "USAC_ACCURATE": cv2.USAC_ACCURATE,
            "LMEDS": cv2.LMEDS}


class EstimationError(RuntimeError):
    """Raised when no usable transform can be estimated."""


@dataclass
class TransformResult:
    model: str
    matrix: np.ndarray | None            # 3x3, image0 px -> image1 px
    inlier_mask: np.ndarray              # (N,) bool
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.matrix is not None and self.stats.get("valid", False)


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 projective matrix to (N, 2) points."""
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    hom = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return hom[:, :2] / hom[:, 2:3]


def reprojection_errors(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    return np.linalg.norm(apply_h(H, src) - dst, axis=1)


def check_transform(H: np.ndarray | None, size0: tuple[int, int], size1: tuple[int, int],
                    max_condition: float = 1e7, max_scale_ratio: float = 10.0) -> dict:
    """Sanity checks that catch degenerate or implausible transforms.

    size0/size1 are (width, height). Returns a dict of diagnostics plus `valid` and `problems`.
    """
    problems = []
    if H is None:
        return {"valid": False, "problems": ["no matrix"]}
    if not np.all(np.isfinite(H)):
        return {"valid": False, "problems": ["non-finite entries"]}
    H = H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H
    cond = float(np.linalg.cond(H))
    A = H[:2, :2]
    det = float(np.linalg.det(A))
    sv = np.linalg.svd(A, compute_uv=False)
    w0, h0 = size0
    corners = np.array([[0, 0], [w0 - 1, 0], [w0 - 1, h0 - 1], [0, h0 - 1]], np.float64)
    denom = np.hstack([corners, np.ones((4, 1))]) @ H[2]
    mapped = apply_h(H, corners) if np.all(denom > 0) else None

    if cond > max_condition:
        problems.append(f"ill-conditioned (cond={cond:.3g})")
    if det <= 0:
        problems.append("orientation flip (det<=0)")
    if sv.min() <= 0 or sv.max() / sv.min() > max_scale_ratio:
        problems.append(f"extreme anisotropy (singular values {sv.round(4).tolist()})")
    if mapped is None:
        problems.append("image0 corners cross the horizon line (w<=0)")
    else:
        if not cv2.isContourConvex(mapped.astype(np.float32).reshape(-1, 1, 2)):
            problems.append("mapped image0 outline is not convex")
        w1, h1 = size1
        margin = 0.5 * max(w1, h1)
        if (mapped[:, 0].min() < -margin or mapped[:, 0].max() > w1 + margin
                or mapped[:, 1].min() < -margin or mapped[:, 1].max() > h1 + margin):
            problems.append("mapped image0 corners far outside image1")
    return {"valid": not problems, "problems": problems, "condition_number": cond,
            "det_linear_part": det, "singular_values_linear_part": sv.tolist(),
            "mapped_corners_image1_px": None if mapped is None else mapped.tolist()}


def estimator_params(transform_cfg: dict, matcher_scale: float) -> dict:
    """Convert the config's matcher-pixel RANSAC threshold to image1 working pixels.

    matcher_scale = (matcher input size) / (working size); 1.0 for SIFT, LoFTR's common_scale for LoFTR.
    """
    if not 0 < matcher_scale <= 1:
        raise ValueError(f"matcher_scale must be in (0, 1], got {matcher_scale}")
    p = dict(transform_cfg)
    p["ransac_reproj_threshold_px"] = transform_cfg["ransac_reproj_threshold_matcher_px"] / matcher_scale
    p["matcher_scale"] = matcher_scale
    return p


def estimate_transform(src: np.ndarray, dst: np.ndarray, size0: tuple[int, int], size1: tuple[int, int],
                       params: dict, seed: int) -> TransformResult:
    """Estimate image0 -> image1 transform with RANSAC; never raises on poor data, reports instead."""
    model = params["model"]
    n = len(src)
    min_needed = 4 if model == "homography" else 3
    stats: dict = {"model": model, "input_correspondences": n,
                   "ransac_method": params["ransac_method"],
                   "ransac_reproj_threshold_px": params["ransac_reproj_threshold_px"],
                   "threshold_units": "image1 (reference) working pixels", "seed": seed,
                   "ransac_max_iters": params["ransac_max_iters"],
                   "ransac_reproj_threshold_matcher_px": params.get("ransac_reproj_threshold_matcher_px"),
                   "matcher_scale": params.get("matcher_scale")}
    empty = np.zeros(n, bool)
    if n < max(min_needed, params["min_inliers"]):
        stats.update(valid=False, problems=[f"only {n} correspondences"], inliers=0, inlier_ratio=0.0)
        log.warning("Not enough correspondences for %s: %d", model, n)
        return TransformResult(model, None, empty, stats)

    cv2.setRNGSeed(seed)
    src32, dst32 = src.astype(np.float32), dst.astype(np.float32)
    thr = float(params["ransac_reproj_threshold_px"])
    if model == "homography":
        H, mask = cv2.findHomography(src32, dst32, _METHODS[params["ransac_method"]], thr,
                                     maxIters=params["ransac_max_iters"], confidence=params["ransac_confidence"])
    elif model == "affine":
        # estimateAffine2D supports RANSAC/LMEDS only; MAGSAC requests fall back to RANSAC (recorded).
        method = cv2.LMEDS if params["ransac_method"] == "LMEDS" else cv2.RANSAC
        stats["ransac_method_effective"] = "LMEDS" if method == cv2.LMEDS else "RANSAC"
        A, mask = cv2.estimateAffine2D(src32, dst32, method=method, ransacReprojThreshold=thr,
                                       maxIters=params["ransac_max_iters"], confidence=params["ransac_confidence"])
        H = None if A is None else np.vstack([A, [0.0, 0.0, 1.0]])
    else:
        raise ValueError(f"Unknown model {model}")

    if H is None or mask is None:
        stats.update(valid=False, problems=["estimator returned no model"], inliers=0, inlier_ratio=0.0)
        log.warning("%s estimation failed", model)
        return TransformResult(model, None, empty, stats)

    H = H / H[2, 2]
    inl = mask.ravel().astype(bool)
    err = reprojection_errors(H, src, dst)
    stats.update(inliers=int(inl.sum()), inlier_ratio=round(float(inl.mean()), 4),
                 inlier_reproj_rmse_px=float(np.sqrt(np.mean(err[inl] ** 2))) if inl.any() else None,
                 inlier_reproj_median_px=float(np.median(err[inl])) if inl.any() else None)
    checks = check_transform(H, size0, size1)
    if inl.sum() < params["min_inliers"]:
        checks["problems"].append(f"only {int(inl.sum())} inliers (< {params['min_inliers']})")
        checks["valid"] = False
    stats.update(checks)
    log.info("%s: %d/%d inliers (%.1f%%), valid=%s %s", model, stats["inliers"], n,
             100 * stats["inlier_ratio"], stats["valid"], checks["problems"] or "")
    return TransformResult(model, H, inl, stats)


def compose_full_resolution(H_work: np.ndarray, full_to_work0: np.ndarray, full_to_work1: np.ndarray) -> np.ndarray:
    """Convert a working-resolution transform into full-resolution source pixel coordinates.

    H_full = inv(F1) @ H_work @ F0   maps full-res image0 px -> full-res image1 px.
    """
    H = np.linalg.inv(full_to_work1) @ H_work @ full_to_work0
    return H / H[2, 2]

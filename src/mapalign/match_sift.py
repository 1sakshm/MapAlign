"""OpenCV SIFT + Lowe ratio test matching (Method B).

Brute-force L2 k-nearest-neighbour matching (k=2), then Lowe's ratio test. RANSAC is applied
separately by mapalign.estimate_transform, so both automatic methods share one estimator.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class MatchResult:
    method: str
    pts0: np.ndarray                     # (N, 2) float, image0 working px
    pts1: np.ndarray                     # (N, 2) float, image1 working px
    scores: np.ndarray                   # (N,) per-match score (SIFT: ratio d1/d2, lower = better)
    stats: dict = field(default_factory=dict)


def match_sift(gray0: np.ndarray, gray1: np.ndarray, params: dict) -> MatchResult:
    sift = cv2.SIFT_create(nfeatures=params["n_features"], contrastThreshold=params["contrast_threshold"],
                           edgeThreshold=params["edge_threshold"], sigma=params["sigma"])
    t0 = time.perf_counter()
    k0, d0 = sift.detectAndCompute(gray0, None)
    k1, d1 = sift.detectAndCompute(gray1, None)
    t_detect = time.perf_counter() - t0
    log.info("SIFT keypoints: %d (image0), %d (image1) in %.1f s", len(k0), len(k1), t_detect)

    stats = {"keypoints0": len(k0), "keypoints1": len(k1), "detect_s": round(t_detect, 2)}
    if d0 is None or d1 is None or len(k0) < 2 or len(k1) < 2:
        log.warning("Too few SIFT keypoints to match")
        empty = np.empty((0, 2), np.float64)
        return MatchResult("sift", empty, empty, np.empty(0), {**stats, "candidate_matches": 0,
                                                                "ratio_matches": 0, "match_s": 0.0})

    if params["matcher"] != "bf_knn_l2":
        raise ValueError(f"Unsupported matcher {params['matcher']}")
    t1 = time.perf_counter()
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d0, d1, k=2)
    t_match = time.perf_counter() - t1

    ratio = params["lowe_ratio"]
    q, t, r = [], [], []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if n.distance > 0 and m.distance < ratio * n.distance:
            q.append(m.queryIdx)
            t.append(m.trainIdx)
            r.append(m.distance / n.distance)
    pts0 = np.array([k0[i].pt for i in q], np.float64).reshape(-1, 2)
    pts1 = np.array([k1[i].pt for i in t], np.float64).reshape(-1, 2)
    stats.update({"candidate_matches": len(knn), "ratio_matches": len(q), "match_s": round(t_match, 2)})
    log.info("SIFT matches: %d candidates, %d pass ratio %.2f (%.1f s)", len(knn), len(q), ratio, t_match)
    return MatchResult("sift", pts0, pts1, np.array(r), stats)

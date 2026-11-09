"""FIRE pipeline (fire/ANALYSIS_PLAN.md §4–§8): preprocess, match, mask, fit, score, AUC.

Matching, fitting and the transform checks come from mapalign.core / mapalign.estimate_transform with MapAlign's
frozen settings (configs/experiment.yaml). Nothing here changes how MapAlign computes its own results.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from mapalign.config import load_config
from mapalign.core import MatchResult, fit_transform, get_matcher
from mapalign.estimate_transform import apply_h

from checksums import data_dir

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / ".cache" / "fire"
CFG = load_config()
SEED = int(CFG["experiment"]["random_seed"])          # 42
FAR_OUTSIDE = "mapped image0 corners far outside image1"   # the one MapAlign check not applied to FIRE (plan §7)
CLAHE = dict(clipLimit=2.0, tileGridSize=(8, 8))
MIN_MATCHES = int(CFG["transform"]["min_inliers"])     # 12
THRESHOLDS = np.arange(1, 26)                           # FIRE AUC thresholds, px (plan §8)


# ---------------------------------------------------------------- data

def fire_root() -> Path:
    return data_dir() / "FIRE"


def pair_ids() -> list[str]:
    from split import pair_ids as ids
    return ids()


def split_of() -> dict[str, str]:
    tuning = {r["pair_id"] for r in csv.DictReader(open(HERE / "tuning_split.csv", encoding="utf-8"))}
    return {p: ("tuning" if p in tuning else "evaluation") for p in pair_ids()}


def control_points(pid: str) -> tuple[np.ndarray, np.ndarray]:
    """(reference_pts, test_pts), each (10, 2) full-resolution px."""
    a = np.loadtxt(fire_root() / "Ground Truth" / f"control_points_{pid}_1_2.txt", dtype=np.float64).reshape(-1, 4)
    return a[:, 0:2], a[:, 2:4]


@lru_cache(maxsize=2)
def masks() -> tuple[np.ndarray, np.ndarray]:
    """(field-of-view mask, feature mask), boolean 2912x2912."""
    m = cv2.imread(str(fire_root() / "Masks" / "mask.png"), cv2.IMREAD_GRAYSCALE) > 0
    f = cv2.imread(str(fire_root() / "Masks" / "feature_mask.png"), cv2.IMREAD_GRAYSCALE) > 0
    return m, f


def preprocess(path: Path, clahe: bool) -> np.ndarray:
    """Green channel, optional CLAHE, zero outside the field of view (plan §4 steps 1–3). uint8, full resolution."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    g = bgr[:, :, 1].copy()
    if clahe:
        g = cv2.createCLAHE(**CLAHE).apply(g)
    fov, _ = masks()
    if fov.shape != g.shape:
        raise ValueError(f"mask {fov.shape} does not match image {g.shape}")
    g[~fov] = 0
    return g


def images(pid: str, clahe: bool) -> tuple[np.ndarray, np.ndarray]:
    """(test, reference) preprocessed. The fitted transform maps test -> reference, as the FIRE protocol scores."""
    d = fire_root() / "Images"
    return preprocess(d / f"{pid}_2.jpg", clahe), preprocess(d / f"{pid}_1.jpg", clahe)


# ---------------------------------------------------------------- matching

def match(method: str, side: int, test: np.ndarray, ref: np.ndarray, device: str = "cpu") -> MatchResult:
    """Matches in full-resolution px, with stats['common_scale'] = matcher size / full size."""
    if method == "sift":
        s = min(1.0, side / max(*test.shape, *ref.shape))
        small = [cv2.resize(g, (round(g.shape[1] * s), round(g.shape[0] * s)), interpolation=cv2.INTER_AREA)
                 if s < 1 else g for g in (test, ref)]
        sx = [sm.shape[1] / g.shape[1] for sm, g in zip(small, (test, ref))]
        sy = [sm.shape[0] / g.shape[0] for sm, g in zip(small, (test, ref))]
        m = get_matcher("sift", CFG["sift"])(*small)
        m.pts0 = (m.pts0 + 0.5) / [sx[0], sy[0]] - 0.5
        m.pts1 = (m.pts1 + 0.5) / [sx[1], sy[1]] - 0.5
        m.stats["common_scale"] = round(s, 6)
        m.stats["sift_input_px"] = [list(sm.shape[::-1]) for sm in small]
        return m
    if method == "loftr":
        p = {**CFG["loftr"], "max_side_px": int(side)}
        return get_matcher("loftr", p, device)(test, ref)
    if method == "roma":
        p = {"coarse_res": 560, "upsample_res": 864, "num_samples": 5000, "seed": 0}
        return get_matcher("roma", p, device)(cv2.merge([test] * 3), cv2.merge([ref] * 3))
    raise ValueError(method)


def feature_mask_filter(pts0: np.ndarray, pts1: np.ndarray) -> np.ndarray:
    """True where both endpoints fall inside FIRE's feature_mask.png (plan §4 step 5)."""
    _, feat = masks()
    h, w = feat.shape

    def inside(p):
        x, y = np.rint(p[:, 0]).astype(int), np.rint(p[:, 1]).astype(int)
        ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        out = np.zeros(len(p), bool)
        out[ok] = feat[y[ok], x[ok]]
        return out
    return inside(pts0) & inside(pts1)


# ---------------------------------------------------------------- fit and score

def transform_cfg(model: str, threshold_matcher_px: float) -> dict:
    return {**CFG["transform"], "model": model, "ransac_reproj_threshold_matcher_px": float(threshold_matcher_px)}


def fit_and_score(pts0: np.ndarray, pts1: np.ndarray, common_scale: float, model: str, threshold: float,
                  ref_pts: np.ndarray, test_pts: np.ndarray, size: tuple[int, int]) -> dict:
    """Fit test -> reference and return the FIRE pair error (mean distance over the control points, px)."""
    row = {"matches": int(len(pts0)), "inliers": 0, "valid": False, "reason": "", "error_px": np.nan, "H": None}
    if len(pts0) < MIN_MATCHES:
        row["reason"] = f"only {len(pts0)} matches"
        return row
    m = MatchResult("x", pts0, pts1, np.zeros(len(pts0)), {"common_scale": common_scale})
    tr = fit_transform(m, size, size, transform_cfg(model, threshold), SEED)
    problems = [p for p in tr.stats.get("problems", []) if p != FAR_OUTSIDE]
    row["inliers"] = int(tr.stats.get("inliers", 0))
    if tr.matrix is None or problems:
        row["reason"] = "; ".join(problems) or "no model"
        return row
    row.update(valid=True, H=tr.matrix, error_px=pair_error(tr.matrix, ref_pts, test_pts),
               ransac_method_effective=tr.stats.get("ransac_method_effective", tr.stats["ransac_method"]))
    return row


def pair_error(H: np.ndarray, ref_pts: np.ndarray, test_pts: np.ndarray) -> float:
    """FIRE registration error: test points mapped through H, mean Euclidean distance to the reference points."""
    return float(np.mean(np.linalg.norm(apply_h(H, test_pts) - ref_pts, axis=1)))


def success_curve(errors, thresholds=THRESHOLDS) -> np.ndarray:
    """Fraction of pairs with error < t; failed pairs (NaN) never succeed."""
    e = np.nan_to_num(np.asarray(errors, float), nan=np.inf)
    return np.array([(e < t).mean() for t in thresholds])


def fire_auc(errors) -> float:
    """FIRE AUC: mean success over t = 1..25 px (reproduces the published FIRE AUCs, plan §8)."""
    return float(success_curve(errors).mean())

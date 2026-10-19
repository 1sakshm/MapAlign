import cv2
import numpy as np
import pytest

from mapalign.estimate_transform import (apply_h, check_transform, compose_full_resolution,
                                         estimate_transform, estimator_params, reprojection_errors)
from mapalign.preprocess import crop_resize_matrix

PARAMS = {"model": "homography", "ransac_method": "USAC_MAGSAC", "ransac_reproj_threshold_px": 3.0,
          "ransac_max_iters": 10000, "ransac_confidence": 0.999, "min_inliers": 12}
H_TRUE = np.array([[1.02, 0.03, 40.0], [-0.02, 0.98, -25.0], [2e-6, -1e-6, 1.0]])


def _synthetic(n_in=200, n_out=100, noise=0.5, seed=0):
    rng = np.random.default_rng(seed)
    src = rng.uniform(0, 1000, (n_in + n_out, 2))
    dst = apply_h(H_TRUE, src) + rng.normal(0, noise, (n_in + n_out, 2))
    dst[n_in:] = rng.uniform(0, 1000, (n_out, 2))  # gross outliers
    return src, dst, n_in


def test_apply_h_identity_and_translation():
    pts = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(apply_h(np.eye(3), pts), pts)
    T = np.array([[1, 0, 5], [0, 1, -2], [0, 0, 1]], float)
    assert np.allclose(apply_h(T, pts), pts + [5, -2])


def test_ransac_recovers_homography_with_outliers():
    src, dst, n_in = _synthetic()
    r = estimate_transform(src, dst, (1000, 1000), (1100, 1100), PARAMS, seed=42)
    assert r.ok, r.stats["problems"]
    assert r.inlier_mask[:n_in].mean() > 0.95 and r.inlier_mask[n_in:].mean() < 0.05
    grid = np.stack(np.meshgrid(np.linspace(0, 1000, 5), np.linspace(0, 1000, 5)), -1).reshape(-1, 2)
    assert np.max(np.linalg.norm(apply_h(r.matrix, grid) - apply_h(H_TRUE, grid), axis=1)) < 1.0
    assert r.stats["inlier_reproj_rmse_px"] < 1.5


def test_ransac_is_deterministic_with_seed():
    src, dst, _ = _synthetic(seed=3)
    a = estimate_transform(src, dst, (1000, 1000), (1100, 1100), PARAMS, seed=42)
    b = estimate_transform(src, dst, (1000, 1000), (1100, 1100), PARAMS, seed=42)
    assert np.array_equal(a.inlier_mask, b.inlier_mask) and np.allclose(a.matrix, b.matrix)


def test_affine_model():
    A = np.array([[0.9, 0.1, 10], [-0.05, 1.1, 5], [0, 0, 1]])
    rng = np.random.default_rng(1)
    src = rng.uniform(0, 500, (80, 2))
    r = estimate_transform(src, apply_h(A, src), (500, 500), (600, 600), {**PARAMS, "model": "affine"}, 42)
    assert r.ok and np.allclose(r.matrix, A, atol=1e-6)
    assert r.stats["ransac_method_effective"] == "RANSAC"


def test_estimator_params_scales_threshold_to_working_px():
    cfg = {**PARAMS, "ransac_reproj_threshold_matcher_px": 3.0}
    assert estimator_params(cfg, 1.0)["ransac_reproj_threshold_px"] == 3.0
    assert estimator_params(cfg, 0.25)["ransac_reproj_threshold_px"] == pytest.approx(12.0)
    with pytest.raises(ValueError):
        estimator_params(cfg, 0.0)


def test_insufficient_correspondences_reported_not_raised():
    src = np.zeros((5, 2))
    r = estimate_transform(src, src, (10, 10), (10, 10), PARAMS, 42)
    assert not r.ok and r.matrix is None and "only 5 correspondences" in r.stats["problems"][0]


def test_random_matches_do_not_yield_valid_model():
    rng = np.random.default_rng(7)
    src, dst = rng.uniform(0, 2000, (600, 2)), rng.uniform(0, 2000, (600, 2))
    r = estimate_transform(src, dst, (2000, 2000), (2000, 2000), PARAMS, 42)
    assert not r.ok


@pytest.mark.parametrize("H, problem", [
    (np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], float) * np.nan, "non-finite"),
    (np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]], float), "ill-conditioned"),
    (np.array([[-1, 0, 100], [0, 1, 0], [0, 0, 1]], float), "orientation flip"),
    (np.array([[1, 0, 0], [0, 1, 0], [-0.002, 0, 1]], float), "horizon"),
    (np.array([[1, 0, 1e5], [0, 1, 0], [0, 0, 1]], float), "far outside"),
])
def test_check_transform_flags_degenerate(H, problem):
    out = check_transform(H, (1000, 1000), (1000, 1000))
    assert not out["valid"] and any(problem in p for p in out["problems"])


def test_check_transform_accepts_identity():
    assert check_transform(np.eye(3), (100, 100), (100, 100))["valid"]


@pytest.mark.parametrize("factor, interp", [(0.5365, cv2.INTER_AREA), (0.5, cv2.INTER_AREA),
                                            (0.5, cv2.INTER_LINEAR)])
def test_crop_resize_matrix_matches_cv2_resize(factor, interp):
    # The centroid of a smooth blob must land where the matrix predicts after crop + cv2.resize.
    cx, cy = 333.3, 201.7
    ys, xs = np.mgrid[0:400, 0:600]
    img = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * 8.0 ** 2)).astype(np.float32)
    x0, y0 = 50, 30
    crop = img[y0:, x0:]
    size = (int(round(crop.shape[1] * factor)), int(round(crop.shape[0] * factor)))
    out = cv2.resize(crop, size, interpolation=interp)
    sx, sy = out.shape[1] / crop.shape[1], out.shape[0] / crop.shape[0]
    pred = apply_h(crop_resize_matrix(x0, y0, sx, sy), np.array([[cx, cy]]))[0]
    ys, xs = np.mgrid[0:out.shape[0], 0:out.shape[1]]
    centroid = np.array([(xs * out).sum(), (ys * out).sum()]) / out.sum()
    assert np.allclose(centroid, pred, atol=0.05)


def test_compose_full_resolution_roundtrip():
    F0 = crop_resize_matrix(412, 534, 0.5365, 0.5365)
    F1 = crop_resize_matrix(0, 0, 0.5, 0.5)
    H_work = np.array([[1.0026, 0.0043, 328.0], [-0.003, 0.9994, 341.6], [0, 0, 1]])
    H_full = compose_full_resolution(H_work, F0, F1)
    p_full = np.array([[1000.0, 2000.0], [5000.0, 300.0]])
    via_work = apply_h(np.linalg.inv(F1), apply_h(H_work, apply_h(F0, p_full)))
    assert np.allclose(apply_h(H_full, p_full), via_work)
    assert np.allclose(reprojection_errors(H_full, p_full, via_work), 0)

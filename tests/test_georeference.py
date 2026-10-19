import numpy as np
import pytest
from rasterio.transform import from_origin

from mapalign.estimate_transform import apply_h
from mapalign.georeference import (affine_px_to_map_matrix, fit_gcp_transform, gcp_fit_diagnostics,
                                   map_to_ref_px, ref_px_to_map, warp_to_grid)

GT_REF = from_origin(600000.0, 1400000.0, 5.0, 5.0)
H_TRUE = np.array([[0.93, 0.02, 15.0], [-0.03, 0.95, 22.0], [1e-6, -2e-6, 1.0]])


def test_affine_matrix_matches_function():
    px = np.random.default_rng(0).uniform(0, 1000, (20, 2))
    assert np.allclose(apply_h(affine_px_to_map_matrix(GT_REF), px), ref_px_to_map(px, GT_REF))


@pytest.mark.parametrize("model, H", [("homography", H_TRUE),
                                      ("affine", np.array([[0.9, 0.1, 5], [-0.1, 1.1, 7], [0, 0, 1]]))])
def test_fit_gcp_transform_exact(model, H):
    src = np.random.default_rng(1).uniform(0, 4000, (8, 2))
    assert np.allclose(fit_gcp_transform(src, apply_h(H, src), model), H, atol=1e-7)


def test_fit_gcp_transform_too_few_points():
    with pytest.raises(ValueError):
        fit_gcp_transform(np.zeros((3, 2)), np.zeros((3, 2)), "homography")
    with pytest.raises(ValueError):
        fit_gcp_transform(np.zeros((2, 2)), np.zeros((2, 2)), "affine")


def test_fit_diagnostics_perfect_data():
    src = np.random.default_rng(2).uniform(0, 4000, (7, 2))
    dst_m = ref_px_to_map(apply_h(H_TRUE, src), GT_REF)
    d = gcp_fit_diagnostics(src, dst_m, H_TRUE, GT_REF, "homography")
    assert d["fit_rmse_m"] < 1e-6 and d["leave_one_out_rmse_m"] < 1e-3


def test_warp_to_grid_is_exact_inverse_mapping():
    # Source image is a linear ramp, which bilinear interpolation reproduces exactly. Every output pixel
    # must hold the ramp value at the historical location given by the inverse chain.
    h, w = 300, 400
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    img = (0.5 * xs + 0.25 * ys).astype(np.float32)
    gt_out = from_origin(600050.0, 1399900.0, 7.0, 7.0)  # different origin and resolution
    out = warp_to_grid(img, H_TRUE, GT_REF, gt_out, (150, 120), border=-1)
    rng = np.random.default_rng(3)
    oc, orow = rng.integers(0, 150, 300), rng.integers(0, 120, 300)
    xy = np.stack([gt_out.c + (oc + 0.5) * gt_out.a, gt_out.f + (orow + 0.5) * gt_out.e], 1)
    hist = apply_h(np.linalg.inv(H_TRUE), map_to_ref_px(xy, GT_REF))
    inside = (hist[:, 0] > 1) & (hist[:, 0] < w - 2) & (hist[:, 1] > 1) & (hist[:, 1] < h - 2)
    assert inside.sum() > 50
    expected = 0.5 * hist[inside, 0] + 0.25 * hist[inside, 1]
    assert np.allclose(out[orow[inside], oc[inside]], expected, atol=0.02)

import numpy as np
import pytest

from mapalign.config import resolve_path
from mapalign.estimate_transform import estimate_transform

PARAMS = {"model": "homography", "ransac_method": "USAC_MAGSAC", "ransac_reproj_threshold_px": 3.0,
          "ransac_max_iters": 1200000, "ransac_confidence": 0.999, "min_inliers": 12}


def test_explicit_usac_seed0_matches_frozen_estimator():
    # The published runs used the flag-based call, whose internal RNG state is fixed at 0.
    from mapalign.io import read_json

    run = read_json(resolve_path("results/loftr_run.json"))
    p = {**PARAMS, "ransac_reproj_threshold_px": run["estimation"]["ransac_reproj_threshold_px"]}
    s0, s1 = tuple(run["working_sizes"]["historical"]), tuple(run["working_sizes"]["reference"])
    d = np.load(resolve_path("results/loftr_correspondences.npz"))
    a = estimate_transform(d["pts0_work"], d["pts1_work"], s0, s1, p, 42)
    b = estimate_transform(d["pts0_work"], d["pts1_work"], s0, s1, p, 42, usac_seed=0)
    assert np.array_equal(a.inlier_mask, d["inlier_mask"].astype(bool))  # the frozen estimator reproduces the run
    # Same inlier set; the polished matrices differ only by floating-point-level refinement differences.
    assert np.array_equal(a.inlier_mask, b.inlier_mask)
    grid = np.stack(np.meshgrid(np.linspace(0, s0[0] - 1, 9), np.linspace(0, s0[1] - 1, 9)), -1).reshape(-1, 2)
    from mapalign.estimate_transform import apply_h
    assert np.max(np.linalg.norm(apply_h(a.matrix, grid) - apply_h(b.matrix, grid), axis=1)) < 0.1  # < 1 m at 10 m/px
    assert "fixed" in a.stats["effective_rng"] and "randomGeneratorState=0" in b.stats["effective_rng"]


def test_setrngseed_does_not_change_flag_based_usac():
    d = np.load(resolve_path("results/sift_correspondences.npz"))
    r = [estimate_transform(d["pts0_work"], d["pts1_work"], (2708, 2773), (3392, 3450), PARAMS, s) for s in (1, 42)]
    assert np.array_equal(r[0].inlier_mask, r[1].inlier_mask)


def test_usac_seed_requires_magsac():
    src = np.random.default_rng(0).uniform(0, 100, (50, 2))
    with pytest.raises(ValueError):
        estimate_transform(src, src, (100, 100), (100, 100), {**PARAMS, "ransac_method": "RANSAC"}, 0, usac_seed=1)


def test_loftr_weights_checksum_and_mirror_failure(tmp_path, monkeypatch):
    import mapalign.match_loftr as ml

    monkeypatch.setenv("TORCH_HOME", str(tmp_path))
    ckpt = tmp_path / "hub" / "checkpoints" / "loftr_outdoor.ckpt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"corrupted")

    class Boom:
        def __init__(self, *a, **k):
            raise ConnectionError("offline")

    monkeypatch.setattr("requests.get", Boom)
    with pytest.raises(RuntimeError, match="Could not obtain verified LoFTR weights"):
        ml.ensure_loftr_weights("outdoor")


def test_morans_i_detects_spatial_pattern():
    from mapalign.robustness import _morans_i

    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 20000, (40, 2))
    trend = xy[:, 0] / 1000.0
    noise = rng.normal(0, 1, 40)
    assert _morans_i(trend, xy, n_perm=999, rng=rng)["p_permutation"] < 0.01
    assert _morans_i(noise, xy, n_perm=999, rng=rng)["p_permutation"] > 0.01


def test_projection_floor_is_submetre():
    import json

    from mapalign.robustness import Ctx, projection

    out = projection(Ctx())
    assert out["polyconic_everest_sheet_centre"]["homography"]["max_m"] < 1.0
    assert json.loads(json.dumps(out))  # serialisable

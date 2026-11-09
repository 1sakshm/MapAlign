"""FIRE scoring checks (fire/ANALYSIS_PLAN.md §8). No FIRE images are needed."""

import sys
from pathlib import Path

import numpy as np
import pytest

FIRE = Path(__file__).resolve().parents[1] / "fire"
sys.path.insert(0, str(FIRE))

import benchmark as B  # noqa: E402


def test_known_homography_scores_zero():
    """Exact matches from a known homography: the fit recovers it and the FIRE pair error is ~0 px."""
    rng = np.random.default_rng(0)
    H = np.array([[0.98, 0.05, 40.0], [-0.04, 1.01, -25.0], [2e-6, -1e-6, 1.0]])
    pts0 = rng.uniform(300, 2600, size=(400, 2))
    hom = np.c_[pts0, np.ones(len(pts0))] @ H.T
    pts1 = hom[:, :2] / hom[:, 2:]
    test_pts = rng.uniform(500, 2400, size=(10, 2))
    h = np.c_[test_pts, np.ones(10)] @ H.T
    ref_pts = h[:, :2] / h[:, 2:]
    for model in ("homography",):
        r = B.fit_and_score(pts0, pts1, 1.0, model, 3.0, ref_pts, test_pts, (2912, 2912))
        assert r["valid"], r["reason"]
        assert r["error_px"] < 1e-3


def test_pair_error_is_mean_distance():
    ref = np.array([[0.0, 0.0], [10.0, 0.0]])
    test = np.array([[3.0, 4.0], [10.0, 0.0]])
    assert B.pair_error(np.eye(3), ref, test) == pytest.approx(2.5)   # (5 + 0) / 2


def test_auc_counts_failures_and_uses_strict_threshold():
    assert B.fire_auc([np.nan, np.nan]) == 0.0
    assert B.fire_auc([0.0]) == 1.0
    assert B.fire_auc([1.0]) == pytest.approx(24 / 25)   # 1 < t only from t = 2


@pytest.mark.parametrize("name, published", [
    ("HernandezMatas_JBHI_2020", (0.958, 0.660, 0.773)),
    ("HernandezMatas_EMBC_2016", (0.945, 0.577, 0.721)),
    ("Chen_TBME_2010", (0.900, 0.443, 0.553)),
    ("Yang_TPAMI_2007", (0.814, 0.303, 0.576)),
])
def test_auc_reproduces_published_fire_scores(name, published):
    """S, A and overall AUC from FIRE's published per-pair errors (REMPE, IEEE JBHI 2020, Table VIII)."""
    rows = [line.split() for line in open(FIRE / "reference" / "published_scores" / f"{name}.txt")]
    err = {pid: float(v) for pid, v in rows}
    s = B.fire_auc([v for k, v in err.items() if k[0] == "S"])
    a = B.fire_auc([v for k, v in err.items() if k[0] == "A"])
    f = B.fire_auc(list(err.values()))
    assert (round(s, 3), round(a, 3), round(f, 3)) == published

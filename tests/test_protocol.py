"""Held-out integrity: graticule-offset QC and the evaluation-point lock."""

import numpy as np
import pandas as pd
import pytest

import mapalign.evaluate as ev
from mapalign.config import load_config
from mapalign.gcp import graticule_offset_z, prior_full_px


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _points(cfg, offsets_px):
    rng = np.random.default_rng(0)
    en = np.column_stack([rng.uniform(668000, 690000, len(offsets_px)), rng.uniform(1357000, 1381000, len(offsets_px))])
    px = prior_full_px(cfg)(en) + np.asarray(offsets_px, float)
    return pd.DataFrame({"historical_x_px": px[:, 0], "historical_y_px": px[:, 1],
                         "true_easting_m": en[:, 0], "true_northing_m": en[:, 1]})


def test_offset_qc_flags_only_the_outlier(cfg):
    offs = [[20 + d, 20 - d] for d in (-6, -3, -1, 0, 1, 2, 4, 5)] + [[-140, 20]]   # last: wrong feature
    z = graticule_offset_z(cfg, _points(cfg, offs))
    flagged = np.abs(z).max(axis=1) > 3
    assert flagged.tolist() == [False] * 8 + [True]


def test_constant_datum_offset_is_not_flagged(cfg):
    offs = [[19 + a, 19 + b] for a, b in [(-2, 1), (1, -1), (0, 2), (3, 0), (-1, -2), (2, 1)]]
    assert np.abs(graticule_offset_z(cfg, _points(cfg, offs))).max() <= 3


def test_verify_lock_rejects_modified_points(cfg, monkeypatch):
    monkeypatch.setattr(ev, "read_json", lambda p: {"evaluation_points_lock": {"sha256": "0" * 64}})
    with pytest.raises(ev.EvaluationLockError):
        ev.verify_lock(cfg)


def test_verify_lock_rejects_missing_lock(cfg, monkeypatch):
    monkeypatch.setattr(ev, "read_json", lambda p: {})
    with pytest.raises(ev.EvaluationLockError):
        ev.verify_lock(cfg)


def test_committed_points_match_lock(cfg):
    assert len(ev.verify_lock(cfg)) == 64

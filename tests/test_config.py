import copy

import pytest

from mapalign.config import ConfigError, load_config, validate_config


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_shipped_config_is_valid(cfg):
    assert cfg["transform"]["direction"] == "historical_px -> reference_px"
    assert cfg["crs"]["evaluation"] == "EPSG:32643"


@pytest.mark.parametrize(
    "section, key, value",
    [
        ("crs", "evaluation", "EPSG:4326"),          # geographic: degrees, not metres
        ("crs", "evaluation", "EPSG:2240"),          # projected, but US survey feet
        ("sift", "lowe_ratio", 1.2),
        ("transform", "model", "tps"),
        ("transform", "ransac_reproj_threshold_matcher_px", 0),
        ("transform", "ransac_max_iters", 0),
        ("transform", "direction", "reference_px -> historical_px"),
        ("compute", "device", "tpu"),
    ],
)
def test_invalid_values_rejected(cfg, section, key, value):
    bad = copy.deepcopy(cfg)
    bad[section][key] = value
    with pytest.raises(ConfigError):
        validate_config(bad)


def test_missing_section_rejected(cfg):
    bad = copy.deepcopy(cfg)
    del bad["gcp"]
    with pytest.raises(ConfigError):
        validate_config(bad)


def test_inverted_bbox_rejected(cfg):
    bad = copy.deepcopy(cfg)
    b = bad["modern_reference"]["bbox_wgs84_deg"]
    b["west"], b["east"] = b["east"], b["west"]
    with pytest.raises(ConfigError):
        validate_config(bad)

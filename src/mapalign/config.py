"""Load and validate the experiment configuration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pyproj import CRS

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiment.yaml"

_REQUIRED_SECTIONS = (
    "experiment", "historical_map", "modern_reference", "crs",
    "preprocessing", "sift", "loftr", "transform", "gcp", "compute",
)


class ConfigError(ValueError):
    """Raised when the configuration is missing keys or has invalid values."""


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    validate_config(cfg)
    log.info("Loaded config %s", path)
    return cfg


def resolve_path(p: str | Path) -> Path:
    """Resolve a config path relative to the repository root."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def ensure_torch_home() -> Path:
    """Keep downloaded model weights inside the repo (.cache/torch) unless TORCH_HOME is set."""
    import os

    home = Path(os.environ.setdefault("TORCH_HOME", str(REPO_ROOT / ".cache" / "torch")))
    home.mkdir(parents=True, exist_ok=True)
    return home


def validate_config(cfg: dict[str, Any]) -> None:
    if not isinstance(cfg, dict):
        raise ConfigError("Config must be a mapping")
    missing = [s for s in _REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ConfigError(f"Config missing sections: {missing}")

    for key in ("geographic", "evaluation"):
        try:
            CRS.from_user_input(cfg["crs"][key])
        except Exception as e:  # pyproj raises CRSError
            raise ConfigError(f"crs.{key} is not a valid CRS: {e}") from e

    eval_crs = CRS.from_user_input(cfg["crs"]["evaluation"])
    if not eval_crs.is_projected:
        raise ConfigError("crs.evaluation must be a projected CRS with metre units")
    units = {ax.unit_name for ax in eval_crs.axis_info}
    if units != {"metre"}:
        raise ConfigError(f"crs.evaluation must use metres, got {units}")

    ratio = cfg["sift"]["lowe_ratio"]
    if not 0 < ratio < 1:
        raise ConfigError(f"sift.lowe_ratio must be in (0, 1), got {ratio}")

    t = cfg["transform"]
    if t["model"] not in ("homography", "affine"):
        raise ConfigError(f"transform.model must be 'homography' or 'affine', got {t['model']}")
    if t["ransac_reproj_threshold_matcher_px"] <= 0:
        raise ConfigError("transform.ransac_reproj_threshold_matcher_px must be positive")
    if int(t["ransac_max_iters"]) < 1:
        raise ConfigError("transform.ransac_max_iters must be >= 1")
    if t["direction"] != "historical_px -> reference_px":
        raise ConfigError("transform.direction is fixed to 'historical_px -> reference_px'")

    b = cfg["modern_reference"]["bbox_wgs84_deg"]
    if not (b["west"] < b["east"] and b["south"] < b["north"]):
        raise ConfigError("modern_reference.bbox_wgs84_deg is not a valid box")

    if cfg["compute"]["device"] not in ("auto", "cpu", "cuda"):
        raise ConfigError("compute.device must be auto, cpu or cuda")

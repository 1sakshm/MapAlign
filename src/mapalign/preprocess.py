"""Prepare the historical scan and the reference raster for matching.

Both images are brought to the same nominal ground sampling distance
(`preprocessing.working_resolution_m`), so that local features appear at similar scales.

Every prepared image carries `full_to_work`, a 3x3 matrix that maps full-resolution pixel
coordinates of its source image to working-image pixel coordinates. Both use the OpenCV
pixel-centre convention. Matches found at working resolution are mapped back with
inv(full_to_work), so every stored transform refers to the original files.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
import rasterio
from affine import Affine

from mapalign.config import resolve_path
from mapalign.io import load_image

log = logging.getLogger(__name__)

INCH_M = 0.0254


@dataclass
class PreparedImage:
    name: str
    gray: np.ndarray                 # uint8, working resolution (matching input)
    color: np.ndarray                # uint8 BGR, working resolution (visualisation)
    full_to_work: np.ndarray         # 3x3, full-res source px -> working px
    full_size: tuple[int, int]       # (width, height) of the full-resolution source
    nominal_resolution_m: float      # ground size of one working pixel (nominal)
    geotransform: Affine | None = None  # full-res pixel corner -> map (reference only)
    crs: str | None = None


def crop_resize_matrix(x0: float, y0: float, sx: float, sy: float) -> np.ndarray:
    """Matrix for: crop at (x0, y0), then resize by (sx, sy) using pixel-centre alignment.

    x_work = sx * (x_full - x0 + 0.5) - 0.5   (the mapping cv2.resize uses)
    """
    return np.array([[sx, 0.0, sx * (0.5 - x0) - 0.5],
                     [0.0, sy, sy * (0.5 - y0) - 0.5],
                     [0.0, 0.0, 1.0]])


def _resize(img: np.ndarray, factor: float) -> tuple[np.ndarray, float, float]:
    h, w = img.shape[:2]
    nw, nh = max(1, int(round(w * factor))), max(1, int(round(h * factor)))
    interp = cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (nw, nh), interpolation=interp), nw / w, nh / h


def _to_gray(color: np.ndarray, clahe: bool) -> np.ndarray:
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    if clahe:
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return gray


def neatline_polygon(cfg: dict) -> np.ndarray:
    c = cfg["historical_map"]["neatline_corners_px"]
    return np.array([c["nw"], c["ne"], c["se"], c["sw"]], dtype=np.float64)


def prepare_historical(cfg: dict) -> PreparedImage:
    hm, pp = cfg["historical_map"], cfg["preprocessing"]
    img = load_image(resolve_path(hm["path"]))
    h, w = img.shape[:2]
    if [w, h] != list(hm["image_size_px"]):
        raise ValueError(f"Historical image is {w}x{h}, config says {hm['image_size_px']}")

    native_m = INCH_M * 63360 / hm["scan_dpi"]          # nominal ground metres per scan pixel
    factor = native_m / pp["working_resolution_m"]

    x0 = y0 = 0
    if pp["crop_to_neatline"]:
        poly = neatline_polygon(cfg)
        x0, y0 = int(np.floor(poly[:, 0].min())), int(np.floor(poly[:, 1].min()))
        x1, y1 = int(np.ceil(poly[:, 0].max())) + 1, int(np.ceil(poly[:, 1].max())) + 1
        img = img[y0:y1, x0:x1].copy()
        # Paint the slivers outside the (slightly rotated) neatline with paper colour, so
        # marginal text and the frame do not produce features.
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.round(poly - [x0, y0]).astype(np.int32)], 255)
        paper = np.median(img[mask > 0].reshape(-1, 3), axis=0).astype(np.uint8)
        img[mask == 0] = paper

    color, sx, sy = _resize(img, factor)
    m = crop_resize_matrix(x0, y0, sx, sy)
    log.info("Historical: crop origin (%d, %d), %dx%d -> %dx%d (scale %.4f), ~%.2f m/px",
             x0, y0, img.shape[1], img.shape[0], color.shape[1], color.shape[0], sx, native_m / sx)
    return PreparedImage("historical", _to_gray(color, pp["clahe"]), color, m, (w, h), native_m / sx)


def prepare_reference(cfg: dict) -> PreparedImage:
    mr, pp = cfg["modern_reference"], cfg["preprocessing"]
    with rasterio.open(resolve_path(mr["rendered_path"])) as ds:
        rgb = np.moveaxis(ds.read(), 0, 2)
        gt, crs = ds.transform, ds.crs.to_string()
        if abs(gt.b) > 0 or abs(gt.d) > 0:
            raise ValueError("Reference raster must be north-up (no rotation terms)")
        res = gt.a
    img = np.ascontiguousarray(rgb[:, :, ::-1])
    h, w = img.shape[:2]
    color, sx, sy = _resize(img, res / pp["working_resolution_m"])
    m = crop_resize_matrix(0, 0, sx, sy)
    log.info("Reference: %dx%d at %.1f m -> %dx%d", w, h, res, color.shape[1], color.shape[0])
    return PreparedImage("reference", _to_gray(color, pp["clahe"]), color, m, (w, h), res / sx, gt, crs)

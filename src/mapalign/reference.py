"""Render the OpenStreetMap extract into a georeferenced reference raster.

Usage:
    python -m mapalign.reference

The output is an RGB GeoTIFF in the evaluation CRS (EPSG:32643) with an exact north-up affine
geotransform. Pixel <-> map conversion is therefore known analytically. The style loosely follows
the Survey of India 1-inch palette (red roads, black railways, blue water), so the two images
are comparable in grayscale. The style is a provisional preprocessing choice and is recorded
in the output metadata.

Pixel convention: rasterio's transform maps pixel *corners*. The centre of pixel (col, row) is
at T * (col + 0.5, row + 0.5). OpenCV draws with integer coordinates at pixel centres, so map
coordinates are converted with  x_cv = col_corner - 0.5.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

from mapalign.config import load_config, resolve_path
from mapalign.io import read_json, write_json

log = logging.getLogger("mapalign.reference")

STYLE_VERSION = "v1"
BACKGROUND = (226, 236, 244)          # BGR, pale paper
WATER = (214, 190, 150)               # BGR, light blue
# (tag, values) -> (BGR colour, width in metres). Drawn in this order (later on top).
LINE_STYLES = [
    (("waterway", {"river"}), (190, 150, 90), 25.0),
    (("waterway", {"canal"}), (190, 150, 90), 8.0),
    (("waterway", {"stream", "drain", "ditch"}), (190, 150, 90), 3.0),
    (("highway", {"residential", "service", "living_street", "track"}), (150, 150, 215), 4.0),
    (("highway", {"tertiary", "tertiary_link", "unclassified", "road"}), (80, 80, 200), 7.0),
    (("highway", {"primary", "primary_link", "secondary", "secondary_link"}), (40, 40, 190), 11.0),
    (("highway", {"motorway", "motorway_link", "trunk", "trunk_link"}), (30, 30, 170), 14.0),
    (("railway", {"rail"}), (20, 20, 20), 7.0),
]
SUBPIXEL_SHIFT = 3  # OpenCV fixed-point bits


def _is_water_area(tags: dict) -> bool:
    return tags.get("natural") == "water" or tags.get("landuse") == "reservoir"


def _stitch_rings(segments: list[list[tuple[float, float]]]) -> list[list[tuple[float, float]]]:
    """Join multipolygon member ways that share endpoints into closed rings."""
    segs = [list(s) for s in segments if len(s) >= 2]
    rings = []
    while segs:
        ring = segs.pop(0)
        changed = True
        while ring[0] != ring[-1] and changed:
            changed = False
            for i, s in enumerate(segs):
                if s[0] == ring[-1]:
                    ring += s[1:]
                elif s[-1] == ring[-1]:
                    ring += s[::-1][1:]
                elif s[-1] == ring[0]:
                    ring = s[:-1] + ring
                elif s[0] == ring[0]:
                    ring = s[::-1][:-1] + ring
                else:
                    continue
                segs.pop(i)
                changed = True
                break
        if len(ring) >= 4:
            rings.append(ring)  # unclosed leftovers are closed implicitly by fillPoly
    return rings


def render_osm(cfg: dict, line_styles=None, width_scale: float = 1.0, water: bool = True,
               background=None, water_colour=None, out_path=None, style_name: str = STYLE_VERSION) -> dict:
    """Render the OSM extract. The defaults reproduce the frozen reference exactly; the keyword arguments
    exist only for the rendering-style robustness check (mapalign.robustness)."""
    line_styles = LINE_STYLES if line_styles is None else line_styles
    bg = BACKGROUND if background is None else background
    wc = WATER if water_colour is None else water_colour
    mr = cfg["modern_reference"]
    res = float(mr["render_resolution_m"])
    crs = mr["render_crs"]
    data = read_json(resolve_path(mr["raw_path"]))
    to_map = Transformer.from_crs(cfg["crs"]["geographic"], crs, always_xy=True)

    b = mr["bbox_wgs84_deg"]
    xmin, ymin, xmax, ymax = to_map.transform_bounds(b["west"], b["south"], b["east"], b["north"], densify_pts=50)
    xmin, ymin = np.floor(xmin / res) * res, np.floor(ymin / res) * res
    xmax, ymax = np.ceil(xmax / res) * res, np.ceil(ymax / res) * res
    width, height = int(round((xmax - xmin) / res)), int(round((ymax - ymin) / res))
    transform = from_origin(xmin, ymax, res, res)
    log.info("Canvas %d x %d px at %.1f m (%s), origin (%.0f, %.0f)", width, height, res, crs, xmin, ymax)

    def to_px(geom: list[dict]) -> np.ndarray:
        lon = np.array([p["lon"] for p in geom])
        lat = np.array([p["lat"] for p in geom])
        e, n = to_map.transform(lon, lat)
        col = (e - xmin) / res - 0.5
        row = (ymax - n) / res - 0.5
        return np.round(np.stack([col, row], 1) * (1 << SUBPIXEL_SHIFT)).astype(np.int32)

    canvas = np.empty((height, width, 3), np.uint8)
    canvas[:] = bg
    drawn: Counter = Counter()

    # 1. Water areas (ways and multipolygon relations), inner rings cut back out.
    inner_rings = []
    for el in data["elements"]:
        tags = el.get("tags", {})
        if not _is_water_area(tags):
            continue
        if el["type"] == "way" and "geometry" in el:
            if water:
                cv2.fillPoly(canvas, [to_px(el["geometry"])], wc, cv2.LINE_AA, SUBPIXEL_SHIFT)
            drawn["water_way"] += 1
        elif el["type"] == "relation":
            for role, store in (("outer", None), ("inner", inner_rings)):
                segs = [[(p["lon"], p["lat"]) for p in m["geometry"]]
                        for m in el.get("members", []) if m.get("role") == role and "geometry" in m]
                for ring in _stitch_rings(segs):
                    pts = to_px([{"lon": x, "lat": y} for x, y in ring])
                    if store is None:
                        if water:
                            cv2.fillPoly(canvas, [pts], wc, cv2.LINE_AA, SUBPIXEL_SHIFT)
                        drawn["water_relation_outer"] += 1
                    else:
                        store.append(pts)
    for pts in inner_rings:
        if water:
            cv2.fillPoly(canvas, [pts], bg, cv2.LINE_AA, SUBPIXEL_SHIFT)
        drawn["water_relation_inner"] += 1

    # 2. Linear features by style, in drawing order.
    ways = [el for el in data["elements"] if el["type"] == "way" and "geometry" in el]
    for (key, values), colour, width_m in line_styles:
        thickness = max(1, int(round(width_m * width_scale / res)))
        for el in ways:
            tags = el.get("tags", {})
            if tags.get(key) in values and not _is_water_area(tags):
                cv2.polylines(canvas, [to_px(el["geometry"])], False, colour, thickness,
                              cv2.LINE_AA, SUBPIXEL_SHIFT)
                drawn[f"{key}={tags[key]}"] += 1

    out = resolve_path(mr["rendered_path"]) if out_path is None else Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rgb = canvas[:, :, ::-1]
    with rasterio.open(out, "w", driver="GTiff", width=width, height=height, count=3, dtype="uint8",
                       crs=crs, transform=transform, compress="deflate", photometric="RGB") as ds:
        ds.write(np.moveaxis(rgb, 2, 0))
        ds.update_tags(source=mr["source"], attribution=mr["attribution"], style=STYLE_VERSION)

    preview = cv2.resize(canvas, None, fx=1000 / width, fy=1000 / width, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out.with_suffix(".preview.png")), preview)

    meta = {"path": mr["rendered_path"], "crs": crs, "resolution_m": res, "width": width, "height": height,
            "transform_gdal": list(transform.to_gdal()), "bounds_m": [xmin, ymin, xmax, ymax],
            "pixel_convention": "rasterio corner-based transform; centre of (col,row) = T*(col+0.5,row+0.5)",
            "style_version": style_name,
            "line_styles": [[k, sorted(v), list(c), w * width_scale] for (k, v), c, w in line_styles],
            "features_drawn": dict(sorted(drawn.items())),
            "osm_base_timestamp": data.get("osm3s", {}).get("timestamp_osm_base"),
            "attribution": mr["attribution"]}
    write_json(meta, out.with_suffix(".json"))
    log.info("Wrote %s  features=%s", out, dict(drawn))
    return meta


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    t0 = time.perf_counter()
    render_osm(load_config())
    log.info("Done in %.1f s", time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())

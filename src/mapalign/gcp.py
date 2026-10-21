"""Ground-control points: candidate generation, picking aids, CSV I/O, manual-GCP fitting (Method C).

Truth coordinates come from OSM *vector* geometry (e.g. the exact intersection of a railway and a
road), converted to EPSG:32643. Historical pixel positions are picked by visual inspection of
crops. The crops are centred with the graticule-corner prior, never with any method's
transform, so picking is not biased toward one method.

Usage:
    python -m mapalign.gcp candidates             # -> gcp/candidates.csv
    python -m mapalign.gcp sheet ID [ID ...]      # picking sheet(s) -> gcp/picking/
    python -m mapalign.gcp thumbs fitting|evaluation   # verification thumbnails for picked points
    python -m mapalign.gcp lock [--force]         # QC + lock evaluation_points.csv (SHA-256 in manifest)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from shapely import STRtree
from shapely.geometry import LineString, Point

from mapalign.config import load_config, resolve_path
from mapalign.estimate_transform import apply_h
from mapalign.georeference import map_to_ref_px
from mapalign.io import load_image, read_json

log = logging.getLogger("mapalign.gcp")

GCP_COLUMNS = ["point_id", "historical_x_px", "historical_y_px", "reference_x_px", "reference_y_px",
               "true_easting_m", "true_northing_m", "crs", "feature_type", "truth_source", "confidence",
               "valid_for_evaluation", "exclusion_reason", "notes"]
ROAD_CLASSES = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
                "trunk_link", "primary_link", "secondary_link", "tertiary_link"}


# ---------------------------------------------------------------- prior (crop centring only)

def prior_full_px(cfg: dict):
    """Return f(E, N) -> historical full-res px using only the 4 printed corners (graticule as WGS84)."""
    hm = cfg["historical_map"]
    g = hm["neatline_graticule_deg"]
    order = ("nw", "ne", "se", "sw")
    ll = {"nw": (g["west"], g["north"]), "ne": (g["east"], g["north"]),
          "se": (g["east"], g["south"]), "sw": (g["west"], g["south"])}
    tf = Transformer.from_crs(cfg["crs"]["geographic"], cfg["crs"]["evaluation"], always_xy=True)
    utm = np.array([tf.transform(*ll[k]) for k in order], np.float32)
    px = np.array([hm["neatline_corners_px"][k] for k in order], np.float32)
    H = cv2.getPerspectiveTransform(utm, px).astype(np.float64)
    return lambda en: apply_h(H, np.atleast_2d(en))


# ---------------------------------------------------------------- candidates from OSM vectors

def _lines(data: dict, tf: Transformer, pred) -> list[tuple[LineString, dict]]:
    out = []
    for el in data["elements"]:
        if el["type"] != "way" or "geometry" not in el or not pred(el.get("tags", {})):
            continue
        lon = [p["lon"] for p in el["geometry"]]
        lat = [p["lat"] for p in el["geometry"]]
        e, n = tf.transform(lon, lat)
        if len(e) >= 2:
            out.append((LineString(np.column_stack([e, n])), {"id": el["id"], **el.get("tags", {})}))
    return out


def _intersections(a, b, kind: str) -> list[dict]:
    tree = STRtree([g for g, _ in b])
    rows = []
    for ga, ta in a:
        for j in tree.query(ga, predicate="intersects"):
            gb, tb = b[j]
            inter = ga.intersection(gb)
            pts = [inter] if inter.geom_type == "Point" else [g for g in getattr(inter, "geoms", []) if g.geom_type == "Point"]
            for p in pts:
                rows.append({"feature_type": kind, "E": p.x, "N": p.y, "osm_a": ta["id"], "osm_b": tb["id"],
                             "label": f"{ta.get('name', ta.get('railway', ta.get('waterway', '')))} x "
                                      f"{tb.get('ref', tb.get('name', tb.get('highway', tb.get('waterway', ''))))}"})
    return rows


def _corner_vertices(ring: np.ndarray, n: int = 4) -> np.ndarray:
    """The n vertices of a closed ring with the sharpest turning angle (e.g. corners of an enclosure wall)."""
    pts = ring[:-1] if np.allclose(ring[0], ring[-1]) else ring
    a, b = np.roll(pts, 1, 0) - pts, np.roll(pts, -1, 0) - pts
    cosang = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    return pts[np.argsort(cosang)[::-1][:n]]  # cos near 0 (90 deg) ranks above straight (cos -1)


def landmark_candidates(cfg: dict, tf: Transformer) -> list[dict]:
    """Palace enclosure corners and temple/monument centroids from the landmark extract (if present)."""
    path = resolve_path("data/raw/osm_57D11_landmarks.json")
    if not path.is_file():
        return []
    rows = []
    for el in read_json(path)["elements"]:
        t = el.get("tags", {})
        name = t.get("name:en") or t.get("name", "")
        if el["type"] == "node" and t.get("natural") == "peak":
            x, y = tf.transform(el["lon"], el["lat"])
            rows.append({"feature_type": "landmark_peak", "E": x, "N": y, "osm_a": el["id"], "osm_b": None,
                         "label": f"{name} (ele {t.get('ele', '?')} m)"})
        if el["type"] == "way" and "geometry" in el:
            e, n = tf.transform([p["lon"] for p in el["geometry"]], [p["lat"] for p in el["geometry"]])
            ring = np.column_stack([e, n])
            if t.get("historic") == "castle":
                for i, (x, y) in enumerate(_corner_vertices(ring)):
                    rows.append({"feature_type": "landmark_corner", "E": x, "N": y, "osm_a": el["id"], "osm_b": i,
                                 "label": f"{name} enclosure corner"})
            elif t.get("amenity") == "place_of_worship" or t.get("historic") in ("monument", "memorial"):
                c = Point(ring.mean(0)) if len(ring) < 4 else LineString(ring).convex_hull.centroid
                rows.append({"feature_type": "landmark_centroid", "E": c.x, "N": c.y, "osm_a": el["id"],
                             "osm_b": None, "label": name})
    return rows


def generate_candidates(cfg: dict) -> pd.DataFrame:
    data = read_json(resolve_path(cfg["modern_reference"]["raw_path"]))
    tf = Transformer.from_crs(cfg["crs"]["geographic"], cfg["crs"]["evaluation"], always_xy=True)
    rail = _lines(data, tf, lambda t: t.get("railway") == "rail")
    road = _lines(data, tf, lambda t: t.get("highway") in ROAD_CLASSES)
    river = _lines(data, tf, lambda t: t.get("waterway") in {"river", "canal"})
    major = [(g, t) for g, t in road if t.get("highway", "").split("_")[0] in {"trunk", "primary", "secondary", "tertiary"}]
    junctions = [r for r in _intersections(major, major, "road_road_junction") if r["osm_a"] < r["osm_b"]]
    names = {t["id"]: (t.get("ref") or t.get("name") or f"way{t['id']}") for _, t in major}
    junctions = [r for r in junctions if names[r["osm_a"]] != names[r["osm_b"]]]
    rows = (_intersections(rail, road, "rail_road_crossing") + _intersections(road, river, "road_water_crossing")
            + _intersections(rail, river, "rail_water_crossing") + junctions)
    for el in data["elements"]:
        t = el.get("tags", {})
        if el["type"] == "node" and t.get("railway") == "station":
            e, n = tf.transform(el["lon"], el["lat"])
            rows.append({"feature_type": "railway_station", "E": e, "N": n, "osm_a": el["id"], "osm_b": None,
                         "label": t.get("name", "")})
    rows += landmark_candidates(cfg, tf)
    df = pd.DataFrame(rows)
    # De-duplicate: merge candidates closer than 60 m (keep first).
    keep, taken = [], np.zeros(len(df), bool)
    xy = df[["E", "N"]].to_numpy()
    for i in range(len(df)):
        if taken[i]:
            continue
        taken |= np.hypot(*(xy - xy[i]).T) < 60
        keep.append(i)
    df = df.iloc[keep].reset_index(drop=True)
    # Keep candidates well inside the neatline (prior estimate, 150 px margin).
    px = prior_full_px(cfg)(df[["E", "N"]].to_numpy())
    poly = np.array([cfg["historical_map"]["neatline_corners_px"][k] for k in ("nw", "ne", "se", "sw")], np.float32)
    inside = [cv2.pointPolygonTest(poly.reshape(-1, 1, 2), (float(x), float(y)), True) > 150 for x, y in px]
    df["prior_x_px"], df["prior_y_px"] = px[:, 0].round(1), px[:, 1].round(1)
    df = df[inside].reset_index(drop=True)
    df.insert(0, "cand_id", [f"C{i:03d}" for i in range(len(df))])
    return df.round({"E": 2, "N": 2})


# ---------------------------------------------------------------- picking aids

def _tick_grid(img: np.ndarray, x0: int, y0: int, step: int = 100) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    for gx in range((x0 // step + 1) * step, x0 + w, step):
        cv2.line(out, (gx - x0, 0), (gx - x0, h), (255, 0, 255), 1)
        cv2.putText(out, str(gx), (gx - x0 + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    for gy in range((y0 // step + 1) * step, y0 + h, step):
        cv2.line(out, (0, gy - y0), (w, gy - y0), (255, 0, 255), 1)
        cv2.putText(out, str(gy), (2, gy - y0 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    return out


def picking_sheet(cfg: dict, cand: pd.DataFrame, ids: list[str], out: Path, half: int = 160, scale: float = 1.6) -> None:
    """Per candidate: historical crop with full-res pixel grid (left) | OSM render with target (right)."""
    hist = load_image(resolve_path(cfg["historical_map"]["path"]))
    with rasterio.open(resolve_path(cfg["modern_reference"]["rendered_path"])) as ds:
        ref = np.ascontiguousarray(np.moveaxis(ds.read(), 0, 2)[:, :, ::-1])
        gt = ds.transform
    ref_half = int(round(half * 5.37 / gt.a))
    panels = []
    for cid in ids:
        r = cand.set_index("cand_id").loc[cid]
        cx, cy = int(r.prior_x_px), int(r.prior_y_px)
        x0, y0 = cx - half, cy - half
        hc = _tick_grid(hist[y0:y0 + 2 * half, x0:x0 + 2 * half], x0, y0)
        rx, ry = map_to_ref_px([[r.E, r.N]], gt)[0]
        rx, ry = int(round(rx)), int(round(ry))
        rc = ref[ry - ref_half:ry + ref_half, rx - ref_half:rx + ref_half].copy()
        cv2.drawMarker(rc, (ref_half, ref_half), (0, 0, 0), cv2.MARKER_CROSS, 30, 2)
        rc = cv2.resize(rc, (2 * half, 2 * half), interpolation=cv2.INTER_AREA)
        pair = np.hstack([hc, np.full((2 * half, 6, 3), 255, np.uint8), rc])
        pair = cv2.resize(pair, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        cv2.putText(pair, f"{cid} {r.feature_type}: {str(r.label)[:60]}", (5, pair.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
        panels.append(pair)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [np.hstack(panels[i:i + 2]) if i + 1 < len(panels) else
            np.hstack([panels[i], np.full_like(panels[i], 255)]) for i in range(0, len(panels), 2)]
    cv2.imwrite(str(out), np.vstack(rows))


def zoom_sheet(cfg: dict, centres: list[tuple[int, int]], out: Path, half: int = 100, scale: float = 3.0) -> None:
    """Fine picking aid: crops around given full-res pixels with 10 px gridlines (labels every 20 px)."""
    hist = load_image(resolve_path(cfg["historical_map"]["path"]))
    tiles = []
    for cx, cy in centres:
        x0, y0 = (cx - half) // 10 * 10, (cy - half) // 10 * 10  # snap so labelled gridlines exist
        c = cv2.resize(hist[y0:y0 + 2 * half, x0:x0 + 2 * half], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        for g in range(0, 2 * half + 1, 10):
            col = (255, 0, 255) if (x0 + g) % 20 == 0 else (200, 120, 255)
            # gridline at full-res pixel CENTRE coordinate x0+g  ->  zoomed position (g + 0.5)*scale - 0.5
            p = int(round((g + 0.5) * scale - 0.5))
            cv2.line(c, (p, 0), (p, c.shape[0]), col, 1)
            cv2.line(c, (0, p), (c.shape[1], p), col, 1)
            if (x0 + g) % 20 == 0:
                cv2.putText(c, str(x0 + g), (p + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 0, 180), 1)
            if (y0 + g) % 20 == 0:
                cv2.putText(c, str(y0 + g), (2, p - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 0, 180), 1)
        tiles.append(c)
    while len(tiles) % 2:
        tiles.append(np.full_like(tiles[0], 255))
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]))


def verification_thumbs(cfg: dict, df: pd.DataFrame, out: Path, half: int = 60, scale: float = 3.0) -> None:
    """Zoomed crops with each picked historical pixel marked, for human review."""
    hist = load_image(resolve_path(cfg["historical_map"]["path"]))
    tiles = []
    for _, r in df.iterrows():
        x, y = float(r.historical_x_px), float(r.historical_y_px)
        xi, yi = int(round(x)), int(round(y))
        c = hist[yi - half:yi + half, xi - half:xi + half].copy()
        c = cv2.resize(c, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        p = (int((x - xi + half + 0.5) * scale - 0.5), int((y - yi + half + 0.5) * scale - 0.5))
        cv2.circle(c, p, 9, (255, 0, 255), 2)
        cv2.putText(c, str(r.point_id), (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        tiles.append(c)
    while len(tiles) % 4:
        tiles.append(np.full_like(tiles[0], 255))
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))


# ---------------------------------------------------------------- CSV I/O

def read_gcps(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(resolve_path(path), dtype={"point_id": str})
    missing = set(GCP_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if df["point_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate point_id values")
    return df


def graticule_offset_z(cfg: dict, df: pd.DataFrame) -> np.ndarray:
    """Robust z-scores of each point's offset (picked px - graticule-prior px); method-independent QC."""
    d = df[["historical_x_px", "historical_y_px"]].to_numpy(float) - prior_full_px(cfg)(
        df[["true_easting_m", "true_northing_m"]].to_numpy(float))
    med = np.median(d, axis=0)
    sig = 1.4826 * np.median(np.abs(d - med), axis=0)
    return (d - med) / np.maximum(sig, 1e-6)


def lock_evaluation_points(cfg: dict, force: bool = False) -> str:
    """QC the evaluation points (|z| <= 3) and store their SHA-256 in data/raw/manifest.json."""
    import datetime as dt
    import hashlib
    from mapalign.io import write_json

    path = resolve_path(cfg["gcp"]["evaluation_points"])
    df = read_gcps(path)
    z = graticule_offset_z(cfg, df[df.valid_for_evaluation.astype(bool)])
    bad = np.abs(z).max(axis=1) > 3
    if bad.any():
        raise ValueError(f"graticule-offset QC failed for {df.point_id[bad.nonzero()[0]].tolist()}; re-examine before locking")
    manifest_path = resolve_path("data/raw/manifest.json")
    manifest = read_json(manifest_path)
    if "evaluation_points_lock" in manifest and not force:
        raise ValueError("evaluation points are already locked; pass --force to re-lock a documented revision")
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["evaluation_points_lock"] = {"file": cfg["gcp"]["evaluation_points"], "sha256": h,
                                          "locked_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                                          "n_points": int(len(df)),
                                          "note": "locked before any method prediction at these points was computed"}
    write_json(manifest, manifest_path)
    return h


def blind_kit(cfg: dict, which: str, out_dir: Path) -> Path:
    """Material for an independent person to re-pick points without seeing the existing picks.

    For each point: a crop of the 1926 scan centred with the printed-corner prior only (never with the
    existing pick), with a labelled 10 px grid, next to the OSM render with the target marked. Plus a CSV
    template with columns for the new pick and the time spent. Existing pixel coordinates are not included.
    """
    df = read_gcps(cfg["gcp"][f"{which}_points"])
    cand = df.rename(columns={"point_id": "cand_id", "true_easting_m": "E", "true_northing_m": "N"})
    cand["label"] = cand["feature_type"]
    px = prior_full_px(cfg)(cand[["E", "N"]].to_numpy(float))
    cand["prior_x_px"], cand["prior_y_px"] = px[:, 0], px[:, 1]
    out_dir.mkdir(parents=True, exist_ok=True)
    for pid in cand.cand_id:
        picking_sheet(cfg, cand, [pid], out_dir / f"{pid}.jpg", half=220, scale=1.5)
    tmpl = pd.DataFrame({"point_id": df.point_id, "feature_type": df.feature_type,
                         "true_easting_m": df.true_easting_m, "true_northing_m": df.true_northing_m,
                         "picker": "", "historical_x_px": "", "historical_y_px": "", "seconds_spent": "",
                         "confident_yes_no": "", "notes": ""})
    path = out_dir / f"repick_{which}_template.csv"
    tmpl.to_csv(path, index=False)
    (out_dir / "INSTRUCTIONS.md").write_text(
        "# Independent re-pick\n\n"
        "For each point, open `<point_id>.jpg`. The left panel is the 1926 scan with full-resolution pixel coordinates "
        "on its grid; the right panel is today's OSM with the target feature marked by a cross. Find the same feature "
        "on the 1926 map and write its pixel x/y (read from the grid labels, or measure in any image viewer on "
        "`data/raw/57_D11_Mysore_1926.jpg`) into the template. Record how many seconds each point took. Do not look at "
        "`gcp/*_points.csv` or any results before finishing.\n\n"
        "Then run: `python -m mapalign.robustness picker --repick <filled template>`\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("candidates")
    s = sub.add_parser("sheet")
    s.add_argument("ids", nargs="+")
    s.add_argument("--out", default=None)
    s.add_argument("--half", type=int, default=160)
    s.add_argument("--scale", type=float, default=1.6)
    z = sub.add_parser("zoom")
    z.add_argument("xy", nargs="+", help="full-res pixel centres as x,y")
    z.add_argument("--out", required=True)
    lk = sub.add_parser("lock")
    lk.add_argument("--force", action="store_true")
    bk = sub.add_parser("blind")
    bk.add_argument("which", choices=["fitting", "evaluation"])
    t = sub.add_parser("thumbs")
    t.add_argument("which", choices=["fitting", "evaluation"])
    args = p.parse_args(argv)
    cfg = load_config()
    if args.cmd == "candidates":
        df = generate_candidates(cfg)
        df.to_csv(resolve_path("gcp/candidates.csv"), index=False)
        log.info("Wrote %d candidates: %s", len(df), df.feature_type.value_counts().to_dict())
    elif args.cmd == "sheet":
        cand = pd.read_csv(resolve_path("gcp/candidates.csv"))
        out = Path(args.out) if args.out else resolve_path(f"gcp/picking/{'_'.join(args.ids)}.jpg")
        picking_sheet(cfg, cand, args.ids, out, half=args.half, scale=args.scale)
        log.info("Wrote %s", out)
    elif args.cmd == "zoom":
        centres = [tuple(int(v) for v in s.split(",")) for s in args.xy]
        zoom_sheet(cfg, centres, Path(args.out))
        log.info("Wrote %s", args.out)
    elif args.cmd == "lock":
        log.info("Locked evaluation points: sha256 %s", lock_evaluation_points(cfg, args.force))
    elif args.cmd == "blind":
        log.info("Wrote %s", blind_kit(cfg, args.which, resolve_path(f"gcp/repick_kit_{args.which}")))
    elif args.cmd == "thumbs":
        df = read_gcps(cfg["gcp"][f"{args.which}_points"])
        out = resolve_path(f"gcp/thumbnails_{args.which}.jpg")
        verification_thumbs(cfg, df, out)
        log.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

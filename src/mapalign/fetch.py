"""Acquire the raw input data and record provenance.

Usage:
    python -m mapalign.fetch historical   # download the Zenodo scan, verify md5, record sha256
    python -m mapalign.fetch osm          # download OSM features for the bbox via Overpass
    python -m mapalign.fetch all

Raw files are never modified after download. Provenance goes to data/raw/manifest.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import re
import sys
from pathlib import Path

import requests

from mapalign import __version__
from mapalign.config import DEFAULT_CONFIG, load_config, resolve_path
from mapalign.io import file_hash, read_json, write_json

log = logging.getLogger("mapalign.fetch")

USER_AGENT = f"MapAlign/{__version__} (research; historical map georeferencing)"
MANIFEST = resolve_path("data/raw/manifest.json")

# Features that can plausibly persist from the 1920s and be matched or used as GCPs.
OVERPASS_QUERY = """[out:json][timeout:300];
(
  way["highway"]({s},{w},{n},{e});
  way["railway"]({s},{w},{n},{e});
  way["waterway"]({s},{w},{n},{e});
  way["natural"="water"]({s},{w},{n},{e});
  relation["natural"="water"]({s},{w},{n},{e});
  way["landuse"="reservoir"]({s},{w},{n},{e});
  relation["landuse"="reservoir"]({s},{w},{n},{e});
  node["place"]({s},{w},{n},{e});
  node["railway"="station"]({s},{w},{n},{e});
);
out geom;
"""


# Stable landmarks for manual GCPs (palace walls, temples, peaks, tank bunds), added in Phase 5.
LANDMARK_QUERY = """[out:json][timeout:180];
(
  way["historic"]({s},{w},{n},{e});
  relation["historic"]({s},{w},{n},{e});
  way["building"="palace"]({s},{w},{n},{e});
  way["amenity"="place_of_worship"]["name"]({s},{w},{n},{e});
  node["natural"="peak"]({s},{w},{n},{e});
  way["man_made"="dyke"]({s},{w},{n},{e});
  way["waterway"="dam"]({s},{w},{n},{e});
  way["barrier"="wall"]["name"]({s},{w},{n},{e});
  way["barrier"="city_wall"]({s},{w},{n},{e});
);
out geom;
"""
LANDMARK_PATH = "data/raw/osm_57D11_landmarks.json"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _update_manifest(key: str, record: dict) -> None:
    manifest = read_json(MANIFEST) if MANIFEST.is_file() else {}
    manifest[key] = record
    write_json(manifest, MANIFEST)


def _download(url: str, dest: Path, **kwargs) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.request(kwargs.pop("method", "GET"), url, stream=True, timeout=600,
                          headers={"User-Agent": USER_AGENT}, **kwargs) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.replace(dest)


def fetch_historical(cfg: dict, config_path: Path, force: bool = False) -> dict:
    hm = cfg["historical_map"]
    dest = resolve_path(hm["path"])
    if dest.exists() and not force:
        log.info("Historical map already present: %s (not re-downloading)", dest)
    else:
        log.info("Downloading %s", hm["source"]["url"])
        _download(hm["source"]["url"], dest)

    md5 = file_hash(dest, "md5")
    if md5 != hm["md5"]:
        raise RuntimeError(f"MD5 mismatch for {dest}: got {md5}, expected {hm['md5']}. "
                           "The file is not the verified Zenodo scan.")
    sha256 = file_hash(dest, "sha256")
    if hm.get("sha256") and hm["sha256"] != sha256:
        raise RuntimeError(f"SHA-256 mismatch: got {sha256}, config has {hm['sha256']}")

    record = {"path": hm["path"], "url": hm["source"]["url"], "md5": md5, "sha256": sha256,
              "size_bytes": dest.stat().st_size, "verified_at": _now()}
    _update_manifest("historical_map", record)

    # Record the sha256 in the config without disturbing comments.
    if not hm.get("sha256"):
        text = config_path.read_text(encoding="utf-8")
        new = re.sub(r"(\n  sha256: )null", rf"\g<1>{sha256}", text, count=1)
        config_path.write_text(new, encoding="utf-8")
        log.info("Wrote sha256 into %s", config_path)
    log.info("Historical map OK  md5=%s  sha256=%s", md5, sha256)
    return record


def fetch_osm(cfg: dict, force: bool = False) -> dict:
    mr = cfg["modern_reference"]
    dest = resolve_path(mr["raw_path"])
    b = mr["bbox_wgs84_deg"]
    query = OVERPASS_QUERY.format(s=b["south"], w=b["west"], n=b["north"], e=b["east"])
    if dest.exists() and not force:
        log.info("OSM extract already present: %s (use --force to refresh)", dest)
    else:
        log.info("Querying Overpass (%s) for bbox %s", mr["overpass_url"], b)
        _download(mr["overpass_url"], dest, method="POST", data={"data": query})

    data = read_json(dest)
    if "elements" not in data:
        raise RuntimeError(f"Overpass response has no 'elements': {str(data)[:300]}")
    remark = data.get("remark")
    if remark:
        raise RuntimeError(f"Overpass returned a remark (likely timeout/partial result): {remark}")
    counts: dict[str, int] = {}
    for el in data["elements"]:
        counts[el["type"]] = counts.get(el["type"], 0) + 1

    record = {"path": mr["raw_path"], "source": mr["source"], "attribution": mr["attribution"],
              "overpass_url": mr["overpass_url"], "query": query, "bbox_wgs84_deg": b,
              "osm_base_timestamp": data.get("osm3s", {}).get("timestamp_osm_base"),
              "element_counts": counts, "sha256": file_hash(dest, "sha256"),
              "size_bytes": dest.stat().st_size, "acquired_at": _now()}
    _update_manifest("osm_extract", record)
    log.info("OSM extract OK  %s  base=%s", counts, record["osm_base_timestamp"])
    return record


def fetch_landmarks(cfg: dict, force: bool = False) -> dict:
    mr = cfg["modern_reference"]
    dest = resolve_path(LANDMARK_PATH)
    b = mr["bbox_wgs84_deg"]
    query = LANDMARK_QUERY.format(s=b["south"], w=b["west"], n=b["north"], e=b["east"])
    if dest.exists() and not force:
        log.info("Landmark extract already present: %s", dest)
    else:
        _download(mr["overpass_url"], dest, method="POST", data={"data": query})
    data = read_json(dest)
    if "elements" not in data or data.get("remark"):
        raise RuntimeError(f"Bad Overpass response: {str(data)[:300]}")
    record = {"path": LANDMARK_PATH, "source": mr["source"], "attribution": mr["attribution"],
              "query": query, "osm_base_timestamp": data.get("osm3s", {}).get("timestamp_osm_base"),
              "elements": len(data["elements"]), "sha256": file_hash(dest, "sha256"), "acquired_at": _now()}
    _update_manifest("osm_landmarks", record)
    log.info("Landmarks OK: %d elements", record["elements"])
    return record


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["historical", "osm", "landmarks", "all"])
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--force", action="store_true", help="re-download even if present")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if args.what in ("historical", "all"):
        fetch_historical(cfg, args.config, args.force)
    if args.what in ("osm", "all"):
        fetch_osm(cfg, args.force)
    if args.what in ("landmarks", "all"):
        fetch_landmarks(cfg, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())

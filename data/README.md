# Data

Nothing in `raw/` or `processed/` is committed, except `raw/manifest.json`.

| File | Created by | Notes |
|---|---|---|
| `raw/57_D11_Mysore_1926.jpg` | `python -m mapalign.fetch historical` | Original Zenodo scan, never modified. md5 `624bbfad28ccb57c3f062e70bd70be08`, sha256 `c308d1f440e9d12dd68fd007d32b277f4fbf1b8b0e0c50010ce3f07114167c13` |
| `raw/osm_57D11_overpass.json` | `python -m mapalign.fetch osm` | OSM snapshot. Its checksum and OSM base timestamp are in `manifest.json`. Re-running gives a newer snapshot, so keep the original for reproduction |
| `processed/osm_57D11_utm43n.tif` | Phase 3 | Rendered reference raster, EPSG:32643 |

See [`SOURCES.md`](SOURCES.md) for the Phase 1 verification log (candidate sheets, metadata, licensing).

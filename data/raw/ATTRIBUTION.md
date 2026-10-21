# Raw input data: sources and licences

These files are the exact inputs behind the published MapAlign results. Their SHA-256 checksums are in
`manifest.json`. Re-running `python -m mapalign.fetch` does not overwrite them. Re-downloading OSM data would
return a newer snapshot and change the evaluation truth, so use these files for exact reproduction.

| File | Source | Licence / attribution |
|---|---|---|
| `57_D11_Mysore_1926.jpg` | Survey of India sheet 57 D/11 *Mysore* (1926), scan from the Old Survey of India Maps collection, J. Brown, Zenodo record 8388121 (https://zenodo.org/records/8388121) | Zenodo deposit: CC BY 4.0. Underlying 1926 Government of India work believed to be out of copyright. The scan bears a University of Michigan Library stamp. |
| `osm_57D11_overpass.json` | OpenStreetMap features via the Overpass API, snapshot 2026-10-18T18:43:02Z | © OpenStreetMap contributors. Available under the Open Database License (ODbL) 1.0, https://opendatacommons.org/licenses/odbl/ |
| `osm_57D11_landmarks.json` | OpenStreetMap landmarks via the Overpass API, snapshot 2026-10-18T20:09:36Z | © OpenStreetMap contributors, ODbL 1.0 |

These OSM extracts are distributed as a database under the ODbL. If you publicly use an adapted version of
them, you must also offer that adapted database under the ODbL.

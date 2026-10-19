# MapAlign — Phase 1 source verification log

Verified 2026-10-18/19 by downloading the files and checking them by hand. Everything below was
checked against the actual files unless it is marked **UNVERIFIED**.

## Collection

- Old Survey of India Maps collection, compiled by John Brown. Zenodo, licence **CC BY 4.0** (as set by the depositor).
- Record named in the project brief: https://zenodo.org/records/14962132 (version of 2025-03-03)
  - `Map Selection and Download Spreadsheet 20250303.xlsx`: 9,967,676 bytes, md5 `e2175538a774cbc92723290ebc7eef38` (matches the md5 Zenodo lists).
  - The spreadsheet has 19 category tabs. The 1-inch sheets are on tab `63 50k South Asia`, and each filename cell links to the file.
- The latest version of the record (https://zenodo.org/records/18165194, 2026-05-26) **no longer contains the spreadsheet**.
  It now ships a PDF index instead. The 20250303 spreadsheet can still be downloaded from version 14962132.

## Candidate historical sheets (all Survey of India, 1 inch : 1 mile = 1:63,360, 15' x 15')

| Candidate | Zenodo file | Record | Size / md5 | Pixels (DPI tag) | Extent (read from the neatline) |
|---|---|---|---|---|---|
| **57 D/11 "Mysore" (1926)** | `57 D]11 Mysore District (1926).jpg` | [8388121](https://zenodo.org/records/8388121) | 4,002,790 B / `624bbfad28ccb57c3f062e70bd70be08` | 5898 x 6828 (300) | 76°30'–76°45'E, 12°15'–12°30'N |
| 57 H/9 "Madras & Mysore" (3rd revised ed., 1927) | `57 H]09 Salem District (1927).jpg` | [8388121](https://zenodo.org/records/8388121) | 4,423,655 B / `62eef722ee995b256f373f0c756b7164` | 6280 x 6930 (300) | 77°30'–77°45'E, 12°45'–13°00'N |
| 57 D/11 "Mysore" (2nd revised ed., 1947) | `57 D]11 Mandya District (1947).jpg` | [8388121](https://zenodo.org/records/8388121) | 10,723,149 B / `22a762afbd290a3bcd2528fea34aca17` | 5841 x 8026 (300) | 76°30'–76°45'E, 12°15'–12°30'N |

Download URL pattern: `https://zenodo.org/records/<record>/files/<url-encoded filename>?download=1`

### 57 D/11 (1926): details read from the sheet
- Title "MYSORE"; header "MYSORE DISTRICT"; sheet "No. 57 D/11"; "Season 1924-25".
- "Published under the direction of Colonel C. P. Gunter, O.B.E., R.E., Officiating Surveyor General of India, 1926."
- "Scale 1 Inch to a Mile, or 1:63,360". Contour interval 50 feet. Helio-zincographed at the Survey of India Offices, Calcutta.
- Graticule ticks every 5' with labelled corners. **No projected grid** is printed.
- Mean magnetic declination 2°10' W (in 1925).
- A University of Michigan General Library "Maps and Charts" stamp is visible, so the scan comes from a library copy.
- The Krishnarajasagara reservoir is marked "under construction" (NW corner), so that area has changed a lot since.
- Projection / datum: **UNVERIFIED.** Sheets of this era are expected to use the Everest 1830 spheroid on the Indian (Kalianpur) datum, possibly with a polyconic projection. This needs checking in Phase 5.

### 57 D/11 (1947): extra notes
- Prints "GRID IV A (Yards)", which is a projected India Zone IVA grid, plus "Mean Grid North ... 0°42' West of True North". This could be an independent cross-check.
- The scan shows heavy fold creases, fading and an uneven paper edge, so expect strong non-projective distortion.

## Modern reference options

- **OpenStreetMap data** (ODbL, attribution "© OpenStreetMap contributors"). Recommended. Download a small bbox extract
  and render it ourselves with an explicit CRS and geotransform. Do NOT bulk-download tile.openstreetmap.org tiles.
- Survey of India Open Series Map (OSM series) for the same tile, on Zenodo: `57 D]11 (2012) OSM.jpg`, record
  [14957672](https://zenodo.org/records/14957672), md5 `19e5b29e3f0f6916b23af0641ae9ec5b`. The neighbouring sheet D43X9 (= 57 H/9)
  was inspected. It shows "Projection - UTM, Datum - WGS 84", "© Government of India copyright 2011", and
  "Reproduction in whole or part by any means is prohibited without the written permission of Survey of India".
  So it must not be redistributed, whatever the Zenodo licence says.

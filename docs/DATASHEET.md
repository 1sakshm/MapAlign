# Datasheet: MapAlign control points and evaluation set

This datasheet follows the questions of *Datasheets for Datasets* (Gebru et al., 2021). It covers the data that
MapAlign releases: the control points, the human re-pick files and the evaluation outputs for Survey of India sheet
57 D/11 (1926). It does not change anything the paper or the results rely on.

Archive: Zenodo, [doi:10.5281/zenodo.23572419](https://doi.org/10.5281/zenodo.23572419) (all versions).

## Motivation

**Why was the dataset created?** To score georeferencing methods for a colonial Survey of India sheet in metres, on
points that no method sees during fitting. Most scans in this series are not georeferenced, and there was no public
held-out ground truth for any of them.

**Who created it?** Saksham Sharma. Candidate locations for the original 21 control points were proposed with an AI
assistant (Claude) and accepted unchanged by the author on 2026-10-19; the author later re-picked every point blind
and timed (see [Collection](#collection-process)).

**Funding.** None.

## Composition

| File | Rows | What one row is |
|---|---|---|
| `gcp/candidates.csv` | 1,053 | a candidate feature computed from OSM vector geometry (railway–road, railway–water and road–water crossings, road junctions, stations, landmark centroids), with its UTM 43N coordinates and the scan pixel predicted by the printed sheet corners |
| `gcp/fitting_points.csv` | 9 | a fitting point for the manual baseline (Method C); 2 are flagged excluded by quality control |
| `gcp/evaluation_points.csv` | 12 | a held-out evaluation point: historical pixel, OSM truth in EPSG:32643, feature type, confidence |
| `gcp/repick_kit_fitting/repick_fitting_template.csv` | 9 | the author's blind re-pick of a fitting point, with seconds spent and a confidence flag |
| `gcp/repick_kit_evaluation/repick_evaluation_template.csv` | 12 | the author's blind re-pick of an evaluation point's historical pixel |
| `gcp/repick_kit_extra/repick_extra_template.csv` | 57 | a supplementary point in one of the 15 grid cells without an evaluation point |
| `results/evaluation_errors.csv`, `results/metrics.json` | – | per-point errors and summary statistics for every method |

**Units and conventions.** Historical pixels are full-resolution scan pixels (OpenCV pixel-centre convention). Truth
is in metres, WGS 84 / UTM zone 43N (EPSG:32643). See the README section *Coordinate conventions*.

**Is it a sample?** Yes. Points were chosen by a written protocol (`gcp/PROTOCOL.md`): a 5 × 5 design grid, one fitting
point in each of 13 cells, evaluation points spread over the sheet and at least 1 km from every fitting point. Only
12 of the planned 20 evaluation points could be established. None lie in the southern fifth of the sheet or in
Mysore city.

**Known errors and noise.**
- Two pickers clicking the same feature differ by a median 7.3 px (about 39 m); two evaluation points differ by 58 and
  76 px.
- E11 is a likely mis-identified or realigned junction; it is kept.
- Of the 57 supplementary points, 21 have both methods more than 300 m off while the methods never disagree by more
  than 155 m. They are dominated by identification errors (a nearby look-alike, or a feature that postdates 1926) and
  must not be used to measure accuracy.
- OSM geometry has unknown absolute accuracy (plausibly 5–20 m).
- The `notes` column of the original fitting and evaluation files was written before the author's review and still
  says "NOT human-verified". The files are hash-locked, so the note was not edited; the review is recorded in the
  README and in `docs/DECISIONS.md`.

**Confidential or personal data.** None. The picker name in the re-pick files is the author's.

## Collection process

- **Sources.** Scan: Zenodo record 8388121, Old Survey of India Maps collection (John Brown), CC BY 4.0; MD5 and
  SHA-256 in `data/raw/manifest.json`. Modern reference: one OpenStreetMap Overpass snapshot (base timestamp
  2026-10-18 18:43 UTC), query and checksum in the same manifest.
- **Original picks (2026-10-19).** Zoomed crops of the scan, centred only by the printed sheet corners, were shown next
  to the OSM render with a 10 px grid. An AI assistant proposed each location; the author checked all 21 against the
  scan and the render and accepted them. A feature was kept only if it existed in 1926 with the same geometry and could
  be located to about ±5 px (±27 m).
- **Quality control.** A point is flagged if its offset from the printed-corner prediction is more than 3 robust
  standard deviations from the median. This removed F02 and F05 from fitting and exposed two evaluation picking errors
  (E06, E12), which were corrected rather than dropped. The rule uses no method output.
- **Lock.** `gcp/evaluation_points.csv` was hashed (SHA-256) before any prediction existed; the evaluation code refuses
  to run if it changes. `.gitattributes` keeps the CSV files byte-exact across platforms.
- **Human re-pick.** After every other analysis, the author re-picked all points blind in `mapalign.pickweb`: each
  window was centred on the printed-corner estimate, never on an earlier pick, and the time from display to acceptance
  was recorded (150 s for 9 fitting points, 121 s for 12 evaluation points). The 57 supplementary points were picked the
  same way, first attempt, and kept as recorded.

## Preprocessing

The scan is cropped to its neatline only inside the pipeline; the raw file is never modified. Candidates are exact
intersections of OSM geometries, computed by `src/mapalign/`. No point was moved after the evaluation lock.

## Uses

**Intended.** Scoring and comparing georeferencing methods on this sheet; studying picker disagreement; teaching
held-out evaluation for historical maps.

**Not suitable for.** Claims about historical maps in general (one sheet); accuracy in the south or in Mysore city
(no reliable points there); surveying-grade work.

**Has it been used?** In the MapAlign paper and its robustness analyses (`results/`).

## Distribution

Public on GitHub and archived on Zenodo. Code and derived data: MIT. The scan keeps its CC BY 4.0 licence;
OpenStreetMap data are under the ODbL and must be credited "© OpenStreetMap contributors".

## Maintenance

Maintained by the author through the GitHub repository. The locked files will not be edited; corrections or new
picks are added as new files with their own hashes and noted in `docs/DECISIONS.md`. Anyone can repeat the re-pick with
the kits in `gcp/repick_kit_*` and the commands in the README section *Re-picking the control points*.

## Reference

Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., & Crawford, K. (2021).
Datasheets for datasets. *Communications of the ACM, 64*(12), 86–92. https://doi.org/10.1145/3458723

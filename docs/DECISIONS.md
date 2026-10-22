# Design decisions

One line per choice: what was decided, when, and why. "Phase n" refers to the build order in the README;
everything up to Phase 5 was fixed before any held-out point existed. Post-hoc additions are marked as such.

## Data

| Decision | Why |
|---|---|
| Sheet 57 D/11 *Mysore* (1926), from Zenodo record 8388121 | Openly licensed (CC BY 4.0 deposit, 1926 work out of copyright); mixes a city, railways, the Cauvery and rural roads, so it has both easy and hard areas. |
| OpenStreetMap as the modern reference, fetched once from Overpass and rendered locally | Open (ODbL), vector, so truth coordinates are exact intersections, not pixels; no bulk tile scraping. Rendering it ourselves fixes its style. |
| Render in EPSG:32643 (WGS 84 / UTM 43N) at 5 m | A metric CRS, so every error is in metres; 5 m is finer than the scan's ≈ 5.4 m per pixel. |
| Raw inputs committed to the repo | A fresh Overpass download is a newer snapshot with different truth coordinates. Versioning the exact bytes is the only way to reproduce the numbers. |
| 2011 Survey of India Open Series sheets not used or committed | Still © Government of India. |

## Matching and estimation

| Decision | Why |
|---|---|
| Work at ≈ 10 m per pixel for both images, after cropping the scan to its neatline | Equal ground resolution helps every matcher; the margins hold no geography. |
| LoFTR with the "outdoor" weights, both images shrunk by one common factor to ≤ 768 px | LoFTR's coarse matching is quadratic in image area. 768 px was chosen in Phase 4 by a rule written down first: most matches within 1 km of the printed-corner prior. 1536 px ran out of memory. |
| SIFT with OpenCV defaults and Lowe ratio 0.75 | Textbook settings. Six preprocessing variants were tried in Phase 3; none gave a valid model, so the defaults stood rather than being tuned until something passed. |
| MAGSAC++ (OpenCV USAC) rather than plain RANSAC | Less sensitive to the inlier threshold, which matters when the inlier ratio may be 5 %. |
| 3 px threshold at the matcher's own resolution | A threshold in working pixels would be about 4.5× looser for LoFTR (≈ 45 m/px input) than for SIFT. Stating it where points are localised is fair to both. |
| 1.2 M iterations | `log(1−0.999)/log(1−0.05⁴)` ≈ 1.1 M: enough for a 5 % inlier ratio. The original 10 000 only covered ≥ 16 %. |
| Homography as the primary model | A 15′ sheet is ≈ 27 km across, so polyconic-vs-UTM and smooth paper distortion are close to projective. Measured later: the projection mismatch left by one homography is ≤ 0.46 m. Affine is reported as a secondary model. |
| Validity checks (determinant, conditioning, corner travel, minimum inliers) shared by all methods | A method only counts as "working" if its transform is plausible. SIFT's best model fails them and is scored as a failure, not as a huge error. |
| The seed 42 in the config | Intended for MAGSAC++, but OpenCV's flag-based call ignores `cv2.setRNGSeed`, so the effective state is OpenCV's fixed 0. Found in the robustness work; the 200-seed analysis gives the spread. |

## Control points and evaluation

| Decision | Why |
|---|---|
| Truth from OSM vector geometry (exact crossings, junctions, landmark centroids) | No pixel-picking error on the truth side; the error left is OSM's own accuracy and real change since 1926. |
| Picking crops centred with the printed-corner prior only | No method's transform can steer where points are picked. |
| Feature priority: rail–road and rail–water crossings first, stations last | Metre-gauge alignments of the 1920s mostly survive; a station's OSM node position is ambiguous. |
| 5 × 5 spatial grid; fitting and evaluation points ≥ 1 km apart | Errors are spatially correlated. A test point next to a fitting point would flatter Method C. |
| Pre-fit QC: drop a fitting point whose offset from the graticule prior is > 3 robust σ | Catches gross mis-picks without using any method's output. It removed F02 and F05. |
| Evaluation points locked by SHA-256 before any method was scored | The evaluation refuses to run if the file changes, so points cannot be quietly edited after seeing results. |
| 12 evaluation points, not the planned 20 | Many candidates are post-1926 (canals, ring roads) or realigned (the city railway). Points are never invented; the gap is reported. |
| Median as the headline, RMSE alongside | With 12 points one bad point (E11) moves the RMSE a lot. |
| Paired Wilcoxon test and a bootstrap CI of the median difference | Same points for every method, so the comparison is paired; n = 12 is too small for normal theory. |

## Post-hoc additions (after the held-out evaluation; none changed a frozen number)

| Decision | Why |
|---|---|
| Robustness analyses: seeds, styles, confidence, spatial statistics, shared-source check, datum, plates, symbols | Test how much the result depends on choices made; each answers an objection a reader would raise. |
| Known-transform synthetic benchmark | Separates the matcher's own error (2–3 m) from map, change and picking error. |
| Classical baselines: printed corners, corners + ECC | What a careful person does without control points or learned matching. |
| A fourth method: DISK or ALIKED keypoints + LightGlue | Asks whether LoFTR is special or any modern learned matcher works. Input size picked by the Phase 4 LoFTR rule. |
| Polynomial and thin-plate-spline models | Asks whether the homography limits accuracy. Chosen by spatial-block cross-validation on the correspondences, never by held-out error. |
| Holm and Benjamini–Hochberg corrections over all secondary tests | Many tests on 12 points; only corrected results are claimed. |
| Blind re-pick kits and a timed picking tool | The control points were AI-proposed and human-reviewed. An independent human re-pick, timed, is the fix; the tools make it a few hours' work. |
| Human re-pick done by the author, blind and timed, in a browser tool (`mapalign.pickweb`) | Same rules as the kits: window centred on the printed-corner estimate, no earlier pick shown, time from display to acceptance. A second picker, not an independent expert. |
| Supplementary-point picks kept as first recorded, not re-picked after seeing results | Many picks were clearly a different feature (21 points more than 300 m off for both methods, which never disagree by more than 155 m). Re-picking after seeing that, or filtering with LoFTR's prediction, would bias the comparison. The file is hashed and reported; only the picker's own confidence flag, recorded before any result, is used as a subset. |
| Kit instructions changed to "skip unless sure" after the first supplementary pick | No candidate was skipped the first time. The stricter rule is for future pickers and does not change the recorded file. |

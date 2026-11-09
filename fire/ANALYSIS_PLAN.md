# FIRE retinal registration: analysis plan

Written and committed before any method is run on any FIRE image. Every setting below is either fixed here or
chosen on the tuning split by the rule given here. Nothing is changed after evaluation results are seen; any
deviation will be listed in `fire/README.md` under "Deviations from the plan", with the reason.

## 1. Question

Does the MapAlign pipeline (a matcher, then MAGSAC++ homography, scored on held-out control points) transfer from
historical maps to colour fundus photographs? The main question is the same as MapAlign's: matching across time.
FIRE category A (two visits, with anatomical change) is the analogue of a 1926 map against today's OpenStreetMap.

## 2. Data

- **Source.** FIRE: Fundus Image Registration Dataset (Hernandez-Matas et al., Journal for Modeling in
  Ophthalmology 1(4):16–28, 2017, DOI 10.35119/maio.v1i4.42).
  - It has 134 pairs: S (71), P (49) and A (14).
  - All images are 2912×2912 px, taken with a Nidek AFC-210 camera at a 45° field of view.
  - Each pair has 10 control points.
- **Copy used.** The official server (projects.ics.forth.gr) refused connections on 2026-11-10, so the copy used
  is the Kaggle mirror `andrewmvd/fundus-image-registration`, downloaded by the author.
  - Archive: `archive (1).zip`, SHA-256 `1dbeac90…3a7a`.
  - It contains 268 JPEGs (134 pairs × 2), 134 control-point files, `mask.png` and `feature_mask.png`.
  - Only 124 of the 268 JPEG files are byte-distinct; images reused across pairs are stored once per pair.
  - The checksum of every file is in `fire/data_manifest.csv`, and every script verifies it first
    (`fire/checksums.py`).
  - There is no official checksum to compare against.
- **Terms.** The official page states no licence. It asks that the FIRE paper be cited if any part of the dataset is
  used. The data is not redistributed: it is never committed, the same rule as the SoI sheets.
- **Pairs.** Image `_1` is the reference and `_2` is the test image. Each line of
  `control_points_<id>_1_2.txt` reads `ref_x ref_y test_x test_y`.
  - Control points are taken as given, in full-resolution pixels.
  - They are assumed to use the same pixel-centre convention as OpenCV. This is not documented, and the effect is
    at most 0.5 px.

## 3. Tuning split and evaluation set

- **Tuning split.** `fire/split.py` draws 10 pairs, stratified by category, from the sorted pair IDs alone (no image
  or result is looked at), with `numpy.random.default_rng(42)`:
  - S: S06, S31, S46, S53, S71
  - P: P05, P25, P37, P47
  - A: A08
- **Evaluation set.** The remaining 124 pairs (66 S, 45 P, 13 A). Every headline number is on these pairs.
- **Literature row.** A secondary row reports the same frozen configuration on all 134 pairs. It is labelled as
  including the 10 tuning pairs, and exists only so the numbers can be set beside published FIRE results, which use
  all 134.

## 4. Pipeline (identical for every method except where stated)

1. **Read.** Load both JPEGs as BGR (OpenCV) and take the **green channel**. Green is the standard choice for fundus
   images: vessels have the most contrast there.
2. **CLAHE (on or off, tuned once for all methods; section 6).** If on, apply `cv2.createCLAHE(clipLimit=2.0,
   tileGridSize=(8, 8))` (OpenCV defaults) to the full-resolution green channel.
3. **Field-of-view mask.** Set pixels outside FIRE's `mask.png` to 0, so that no texture outside the fundus disc
   is left.
4. **Matching.** Matching runs on downscaled images. Matches are mapped back to full-resolution pixels with
   `x_full = (x_small + 0.5) / s − 0.5`, the convention MapAlign's LoFTR code uses.
   - **SIFT:** `mapalign.core.get_matcher("sift")` with MapAlign's frozen SIFT settings: OpenCV defaults
     (`nfeatures=0`, `contrastThreshold=0.04`, `edgeThreshold=10`, `sigma=1.6`), brute-force L2 2-NN, and Lowe
     ratio 0.75. The images are first resized (INTER_AREA) to the tuned working side.
   - **LoFTR:** `get_matcher("loftr")`, kornia 0.8.3, pretrained `outdoor`, the SHA-256-pinned weights MapAlign
     uses, `confidence_threshold=0`. It resizes internally to the tuned `max_side_px`.
5. **Match mask.** Drop any match whose full-resolution endpoint, in either image, falls outside FIRE's
   `feature_mask.png`. That mask is FIRE's own slightly eroded disc, meant for keypoints. This removes matches on
   the circular rim, which would favour the identity transform. The same mask is used for every method.
6. **Fit.** `mapalign.core.fit_transform` maps test → reference: cv2 USAC_MAGSAC, `maxIters=1,200,000`,
   `confidence=0.999`, `min_inliers=12`. The threshold is the tuned value in matcher pixels, converted to
   full-resolution pixels by `estimator_params`, exactly as in MapAlign.
7. **Score.** Map the 10 test control points through the fitted H. A pair's error is the **mean Euclidean distance
   to the 10 reference points, in full-resolution pixels**. This is the FIRE protocol: the paper maps t_j → r_j
   and uses the mean of |c_j − r_j|.

The shared modules (`mapalign.core`, `mapalign.estimate_transform`, `mapalign.match_sift`,
`mapalign.match_loftr`) are imported, not copied, and not modified. If a change ever becomes necessary, it will be
backward-compatible, and the MapAlign test suite will be re-run to show its numbers are unchanged.

## 5. Transform models

- **Primary: homography** (8 DoF), as in MapAlign.
- **Secondary: affine** (6 DoF), from the same matches with the same threshold. It is reported as a check, because
  the retina is curved and neither model is exact.
  - MapAlign's affine path uses `cv2.estimateAffine2D` with RANSAC, since OpenCV has no MAGSAC++ for affine. This
    is recorded in the outputs.
- No quadratic or eye-model method is fitted; that is out of scope.

## 6. Settings chosen on the tuning split

| Setting | Candidates | Applies to |
|---|---|---|
| SIFT working side (px) | 728, 1456, 2912 (scale 1/4, 1/2, 1) | SIFT |
| LoFTR `max_side_px` | 640, 768, 1024 | LoFTR |
| MAGSAC++ threshold (matcher px) | 1, 2, 3, 5 | each method separately |
| CLAHE | off, on | all methods together |

**Choosing CLAHE.** For each CLAHE setting, find each method's best (side, threshold) on the 10 tuning pairs. Keep
the CLAHE setting with the higher mean of SIFT's and LoFTR's best tuning AUCs.

**Choosing the rest.** With CLAHE fixed, each method keeps its (side, threshold) with the highest tuning AUC. Ties
(within 0.001 AUC) are broken in this order:
1. lower median per-pair error;
2. the candidate closest to MapAlign's frozen values (768 px for LoFTR, 3 px, CLAHE off).

**Unrunnable settings.** If a setting cannot run (for example LoFTR at 1024 px running out of memory on this
16 GB CPU machine), it is dropped from the grid and recorded.

The tuning results are written to `fire/results/tuning.csv` and reported separately. They are never part of a
headline number.

## 7. Failure

A pair **fails** for a method if any of these happens:
- matching raises an error;
- fewer than 12 matches survive;
- the estimator returns no model;
- the fit fails MapAlign's validity checks:
  - fewer than 12 inliers;
  - condition number above 1e7;
  - a flip (det ≤ 0);
  - singular-value ratio above 10;
  - corners crossing the horizon;
  - a non-convex mapped outline;

MapAlign's other check, "mapped corners more than half an image outside the reference", is **not** applied to
FIRE. P pairs can overlap well under 75%, so a correct fit can place the test image's far corners more than half an
image outside. `fire/` ignores that one problem string. The shared function is not changed.

A failed pair keeps its row, with `error_px = NaN` and the reason. It counts as **not registered at every
threshold**, the same way FIRE's published score files record failures (`NaN`). Failed pairs are never dropped or
re-run with other settings.

## 8. Measures

1. **AUC (headline), per category S, P, A and overall.** The FIRE success curve:
   - For a threshold t, success(t) is the fraction of pairs with error < t (failed pairs never succeed).
   - **AUC = mean of success(t) over t = 1, 2, …, 25 px.**
   - The papers do not spell this out. I fixed it by recomputing the published AUCs from the per-pair error files
     FIRE publishes for four methods (`fire/reference/published_scores/`):
     - this definition reproduces S, A and the overall FIRE AUC to three decimals for all four methods, matching
       the FIRE site table and REMPE (Hernandez-Matas et al., IEEE JBHI 2020) Table VIII;
     - none of the 11 other candidates tried does (≤ instead of <, thresholds 0–25, a trapezoid rule, a fine
       0.01 px grid).
   - Category P differs from the printed value by 0.006–0.009 for all four methods, in both directions. The site's own S, A and overall values imply
     P = 0.537 for REMPE, against the 0.542 printed. The published P numbers therefore appear internally
     inconsistent, and our P AUC is computed with the same definition.
   - `tests/test_fire.py` checks the definition against these files.
2. **Per-pair error.** Median (failures counted as +∞) and mean (over successful pairs only, reported with the
   failure count).
3. **Failure count** per method × category.
4. **Success curves** for t from 0 to 25 px, as figures.

**Uncertainty.** A percentile bootstrap over pairs, 10,000 resamples, `default_rng(42)`, gives the 95% CI for the
AUC and the median error of every method × category.

## 9. Primary comparison

**LoFTR vs SIFT on category A (13 evaluation pairs), homography.**
- **Test.** A two-sided Wilcoxon signed-rank test on per-pair error d = LoFTR − SIFT, exact (n ≤ 25), α = 0.05.
  - If one method fails a pair, its error is set larger than every finite error. The test uses ranks, so the
    exact value does not matter.
  - If both fail, d = 0 and the pair drops out (`zero_method="wilcox"`); the count is reported.
- **Effect.** The paired-bootstrap 95% CI of the AUC difference (LoFTR − SIFT, 10,000 resamples, seed 42) is
  reported with the number of pairs each method wins.
- **Power.** With 13 pairs, power is low. A non-significant result is reported as "no detectable difference", not
  as "equivalent".

**Secondary comparisons.** The same test on S, P and all evaluation pairs, with Holm correction across these three.
The affine model is reported descriptively only.

## 10. Optional third matcher (only after SIFT and LoFTR are complete)

- **Method.** RoMa (`get_matcher("roma")`), with the library defaults MapAlign used on the map:
  - `coarse_res` 560, `upsample_res` 864, 5,000 samples, seed 0.
  - The input is the preprocessed green channel repeated to 3 channels.
- **Settings.** Same CLAHE choice, same feature mask, threshold 3 matcher px (MapAlign's frozen value). It is not
  tuned.
- **Reporting.** Descriptive, with no test, and it does not change the primary comparison. If it is not run, the
  README says so.

## 11. Comparison with published results

- **What is shown.** Our literature row (all 134 pairs) beside REMPE Table VIII as printed:
  - REMPE: 0.958 / 0.542 / 0.660 / 0.773
  - SIFT+WGTM: 0.837 / 0.544 / 0.407 / 0.685
  - GDB-ICP: 0.814 / 0.303 / 0.303 / 0.576
  - Harris-PIIFD: 0.900 / 0.090 / 0.443 / 0.553
  - (all S / P / A / FIRE)
- **Labelling.** It is not presented as a controlled comparison: the methods, tuning and implementations differ.
- **Other papers.** Their results (e.g. SuperRetina) are quoted only if the exact figure and the scoring definition
  can be checked in the paper.

## 12. Figures and examples (rules fixed now)

- **Success curves.** One curve per method for each of S, P, A and all evaluation pairs, homography.
- **Example overlays, per method:**
  - one success: the evaluation pair whose error is closest to that method's median over successful pairs;
  - one failure: a failed pair (lowest ID) or, if none failed, the pair with the largest error.
  - Overlays use the reference in magenta and the warped test image in green.

## 13. Outputs and reproduction

- **One command:** `python fire/run.py` (after the data is in `FIRE_DIR`). It runs the checksum check, tuning,
  evaluation, statistics and figures.
- **Results.** `fire/results/`:
  - `tuning.csv`
  - `per_pair.csv` (pair, category, split, method, model, matches, inliers, valid, reason, error_px)
  - `summary.json` (method × model × category: AUC, median and mean error, failures and CIs; primary and secondary
    tests; chosen settings; package versions; seeds)
- **Figures.** `fire/figures/`.
- **Seeds.** 42 for the split and the bootstrap. LoFTR uses `torch.manual_seed(0)`, as in MapAlign. OpenCV USAC uses
  its fixed internal state, as documented in `mapalign.estimate_transform`.

## 14. Known limits (stated in advance)

- No training or fine-tuning on FIRE: both matchers are off-the-shelf.
- A homography on a curved retina is an approximation, which hurts P most. This is why affine is reported as well.
- Category A has only 13 evaluation pairs, so the CIs will be wide.
- There is no new method. This is MapAlign's benchmark extended to retinal images.
- The data comes from a mirror, with no official checksum.

## 15. Manual baseline (only if the author says go)

- **What.** On category A, the author clicks their own corresponding points in a tool that hides FIRE's control
  points and records the time spent per pair.
- **Lock and score.** The picks are locked by SHA-256 before scoring. The same homography is fitted and scored on
  FIRE's 10 points.
- **The A08 tuning pair.** It is reported separately, as it is for the matchers.

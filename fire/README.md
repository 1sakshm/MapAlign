# MapAlign on retinal images: the FIRE benchmark

This extends the MapAlign registration benchmark from a 1926 map sheet to colour fundus photographs. It uses the
same pipeline: a matcher, then a MAGSAC++ homography, scored on held-out control points. There is no new method and
no training on FIRE; both matchers are used off the shelf.

The analysis plan, `ANALYSIS_PLAN.md`, was committed in `e44e393`. That was before any method saw a FIRE image.
Settings were then chosen on a 10-pair tuning split (`a3a61d3`) and frozen before the 124 evaluation pairs were
run. Every number below is copied from `results/summary.json`, which `run.py` writes.

## Data

**FIRE** contains 134 pairs of 2912×2912 fundus images (Nidek AFC-210, 45° field of view) from 39 patients. Each
pair has 10 control points marked by hand. There are three categories:
- **S:** 71 pairs with more than 75% overlap.
- **P:** 49 pairs with less than 75% overlap.
- **A:** 14 pairs from different visits, with anatomical change.

Category A is the analogue of MapAlign's 1926 map against today's OpenStreetMap.

**Where the copy comes from.** The official FORTH server could not be reached on 2026-11-10, so the copy used here
is the Kaggle mirror (`andrewmvd/fundus-image-registration`). There is no official checksum, so it cannot be proved
identical to FORTH's archive. The SHA-256 of every file is in `data_manifest.csv`, and `run.py` checks them before
doing anything.

**Terms.** The official page states no licence and asks that the FIRE paper be cited. The images are not included in
this repository. For the same reason, the example overlays (which are derived patient images) are written locally to
`figures/` but not committed.

> C. Hernandez-Matas, X. Zabulis, A. Triantafyllou, P. Anyfanti, S. Douma, A. A. Argyros. "FIRE: Fundus Image
> Registration Dataset." *Journal for Modeling in Ophthalmology* 1(4):16–28, 2017. DOI 10.35119/maio.v1i4.42

## Reproduce

```bash
# FIRE extracted to D:/Downloads/fire-data/FIRE (or set FIRE_DIR)
python fire/run.py                 # checksums -> tuning -> evaluation -> statistics -> figures
python -m pytest tests/test_fire.py
```

On a 16-thread CPU, matching takes about 0.2 s per pair for SIFT and about 16 s for LoFTR.

## Method (details and reasons in `ANALYSIS_PLAN.md`)

- **Preprocessing:**
  - the green channel;
  - CLAHE (clip limit 2.0, 8×8 tiles), chosen on the tuning split;
  - zeros outside FIRE's field-of-view mask.

  The same preprocessing is used for both methods.
- **Matchers:** both come from `mapalign.core`, with MapAlign's frozen settings.
  - **SIFT:** OpenCV defaults and Lowe ratio 0.75, run at 728 px.
  - **LoFTR:** kornia, `outdoor` weights, run at 1024 px.
  - Matches are mapped back to full resolution. A match is dropped if either end falls outside FIRE's feature mask.
- **Fit:** a MAGSAC++ homography maps the test image onto the reference.
  - The threshold is 5 px at matcher resolution for both methods, chosen on the tuning split.
  - The fit needs at least 12 inliers, and MapAlign's validity checks apply.
  - The check for "corners more than half an image outside the reference" is skipped, because P pairs overlap
    little. The plan sets this out before any run.
- **Affine:** a secondary model, fitted with RANSAC.
- **Score:** a pair's error is the mean distance in full-resolution pixels between the 10 reference control points and
  the mapped test points. This is the FIRE protocol.
- **AUC:** the share of pairs with error below *t*, averaged over *t* = 1…25 px. Failed pairs count as never
  registered.
  - This definition reproduces the published S, A and overall FIRE AUCs of four methods from their published
    per-pair errors (`tests/test_fire.py`).
  - Category P does not reproduce. The published P values appear internally inconsistent; see plan §8.
- **Uncertainty:** 95% intervals come from 10,000 bootstrap resamples of pairs (seed 42).

### Tuning (10 pairs; not part of any result below)

**Pairs:** S06, S31, S46, S53, S71, P05, P25, P37, P47 and A08.

**Grid** (`results/tuning.csv`, choice in `results/tuning_choice.json`):
- SIFT size: 728, 1456 or 2912 px.
- LoFTR size: 640, 768 or 1024 px.
- Threshold: 1, 2, 3 or 5 px.
- CLAHE: off or on.

**What was chosen:**
- **CLAHE: on.** The mean of the two methods' best tuning AUCs was 0.588 with CLAHE and 0.380 without.
- **SIFT:** 728 px and 5 px (tuning AUC 0.544).
- **LoFTR:** 1024 px and 5 px (tuning AUC 0.632).

**Limits of this choice:**
- Three of the four choices sit at the edge of the grid, so better settings may lie outside it. The grid was fixed
  in advance and was not extended.
- Ten pairs separate settings only coarsely: neighbouring LoFTR settings differ by about 0.1 AUC with no clear
  pattern.

## Results: 124 evaluation pairs, homography

| | S (66) | P (45) | A (13) | All (124) |
|---|---|---|---|---|
| **SIFT AUC** [95% CI] | 0.9455 [0.9364, 0.9539] | 0.2631 [0.1956, 0.3351] | 0.5292 [0.3108, 0.7415] | 0.6542 [0.5874, 0.7194] |
| **LoFTR AUC** [95% CI] | 0.9582 [0.9515, 0.9648] | 0.3760 [0.3067, 0.4436] | 0.8246 [0.7754, 0.8708] | 0.7329 [0.6765, 0.7855] |
| SIFT failed pairs | 0 | 0 | 4 | 4 |
| LoFTR failed pairs | 0 | 0 | 0 | 0 |
| SIFT median error, px (failures = ∞) | 1.67 | 20.03 | 5.53 | 2.97 |
| LoFTR median error, px | 1.37 | 15.74 | 5.04 | 2.51 |

All values are from `results/summary.json` → `by_method.<method>.homography`. Per-pair rows are in
`results/per_pair.csv`, and the curves are in `figures/success_curves.png`.

**Primary comparison (declared in advance): LoFTR vs SIFT on category A.**
- LoFTR has the lower error on 12 of the 13 pairs.
- Exact two-sided Wilcoxon signed-rank test: p = 0.00464.
- AUC difference: +0.2954 (95% CI 0.1138 to 0.4955).
- Four of SIFT's 13 A pairs fail:
  - A05: only 7 matches.
  - A03 and A06: 10 inliers each, against the 12 required.
  - A02: a degenerate fit with 5 inliers.

**Secondary comparisons** (Holm-corrected; the normal approximation is used because n > 25):

| Category | Pairs where LoFTR has the lower error | Holm p | AUC difference [95% CI] |
|---|---|---|---|
| S | 54 of 66 | 1.56e-9 | +0.0127 [0.0067, 0.0194] |
| P | 33 of 45 | 0.00174 | +0.1129 [0.0551, 0.1724] |
| All | 99 of 124 | 5.34e-10 | +0.0787 [0.0461, 0.1139] |

**Affine** (secondary check), AUC for S / P / A / all:
- **SIFT:** 0.9376 / 0.1573 / 0.5446 / 0.6132
- **LoFTR:** 0.9503 / 0.1769 / 0.8062 / 0.6545

The homography does better than the affine model everywhere except SIFT on A. The gap is largest in P, where the
pairs overlap least.

## Next to published results (not a controlled comparison)

| | S | P | A | FIRE |
|---|---|---|---|---|
| SIFT, this pipeline (all 134 pairs) | 0.9397 | 0.2645 | 0.4914 | 0.6460 |
| LoFTR, this pipeline (all 134 pairs) | 0.9549 | 0.3739 | 0.7914 | 0.7254 |
| REMPE (Table VIII) | 0.958 | 0.542 | 0.660 | 0.773 |
| VOTUS (Table VIII) | 0.934 | 0.672 | 0.681 | 0.812 |
| SIFT+WGTM (Table VIII) | 0.837 | 0.544 | 0.407 | 0.685 |
| GDB-ICP (Table VIII) | 0.814 | 0.303 | 0.303 | 0.576 |
| Harris-PIIFD (Table VIII) | 0.900 | 0.090 | 0.443 | 0.553 |

The published rows are copied as printed from Hernandez-Matas, Zabulis and Argyros, "REMPE", *IEEE JBHI* 2020,
Table VIII. That table has more rows; only these five are shown.

To match the published protocol, our rows use all 134 pairs, which includes the 10 tuning pairs (`summary.json` →
`literature_row_all_134`).

**How to read this table:**
- The implementations, tuning and hardware all differ, so this does not show that one method is better than another.
- Category A has 14 pairs.
- Newer learned retinal methods, such as SuperRetina, are not listed. Their published figures and scoring were not
  checked, and they may well score higher.

What the table does support: a general-purpose matcher with a homography is close to REMPE on S. It is well below
REMPE and VOTUS on P, where REMPE's 3D eye model matters. On A, which has only 14 pairs, it scores above every row
listed.

## Figures

- **`figures/success_curves.png`:** success rate against threshold for S, P, A and all evaluation pairs.
- **Example overlays** (local only, not committed). They show the reference in magenta, the warped test image in
  green, FIRE's points as white circles and the mapped points as yellow crosses. The examples were picked by the
  rule in plan §12:
  - **SIFT:** a typical success (A12) and a failure (A02, with 5 inliers and a degenerate fit).
  - **LoFTR:** a typical success (A01). LoFTR failed no pairs, so the worst pair is shown instead: P09, mean error
    38.3 px. There the error grows toward the rim of the disc, which is where a planar model breaks on a curved
    retina.

## Limits

- **No training or fine-tuning on FIRE**, and no new method: this is MapAlign's benchmark applied to retinal images.
- **The homography is an approximation on a curved retina.** It costs most in P (P AUC 0.38 vs REMPE's 0.54).
- **Category A is small:** 13 evaluation pairs. The A intervals are wide (SIFT 0.31–0.74).
- **Three of SIFT's four A failures come from the 12-inlier minimum.** A05 failed on matches instead: only 7 survive.
  A lower minimum would change SIFT's A result. The minimum was MapAlign's frozen value and was not changed.
- **Tuning on 10 pairs is coarse**, and three of the four chosen settings sit at the edge of the grid.
- **The data is a Kaggle mirror** with no official checksum. Only 124 of its 268 image files are byte-distinct,
  while the paper reports 129 images. Shared images explain part of this gap, but this was not investigated further.
- **CPU only.** The timings come from one Windows machine.

## Deviations from the plan

None in the protocol. One change after the first statistics run: p-values are now stored to three significant
figures instead of five decimals, because very small values were being rounded to 0. The exact Wilcoxon method used
is now also recorded. Neither change affects any test result.

## Environment

Python 3.12.12; numpy 2.5.4, scipy 1.18.1, opencv-python-headless 5.0.0.93, torch 2.14.1+cpu, kornia 0.8.3.
Seeds: 42 for the split and the bootstrap; 0 for LoFTR's torch seed; OpenCV USAC uses its fixed internal state.

# Experiment 2: paintings, engravings and old photos

**Question.** Do modern image matchers still find the same points when one image is an engraving, a painting or an
old photograph and the other is a modern photograph of the same building? And does the answer depend on the style?

This extends the 1926 map experiment from one map sheet to depictions of buildings. It uses the same code
(`mapalign.core`) to load each matcher, fit the transform and score held-out points.

## Status

Protocol drafted. No pairs collected, no points marked, nothing run. Nothing in this folder is a result yet.

## Protocol (fixed before any matcher is run on these pairs)

**Pairs.** 30–50 pairs from Wikimedia Commons. Each pair is one historical image and one modern photograph of
the same building, taken from a similar side. The historical image is grouped by style: `photo`, `drawing`
(engravings, lithographs and line drawings) or `painting`. Every image is public domain or openly licensed. The
licence, author and file page are recorded in `pairs.csv`. Images are downloaded by script and are not committed.

**Ground truth.** For each pair, one annotator marks 8–10 corresponding points: well-defined building corners,
window corners or spire tips. The annotator has not seen any matcher output, and the marks are locked (SHA-256 in
`marks/LOCK.json`) before any matcher runs on the pairs. Marks go in `marks/<pair_id>.csv`, in full-resolution
pixel coordinates of both images (OpenCV pixel-centre convention).

**Methods.** SIFT, LoFTR, DISK and ALIKED + LightGlue, RoMa and MASt3R. Every matcher feeds the same robust
estimator with the same settings. A matcher's settings are fixed before the marks are read.

**Measures, reported by style.**

1. Success rate: the share of pairs where the fit is valid, meaning it has enough inliers and passes the
   validity checks.
2. Held-out point error on the marks, in pixels of the modern photo, divided by the photo's diagonal so that
   pairs of different sizes are comparable. Failed pairs are counted as failures, not dropped.

**Reported limits.** The sample is small, there is a single annotator, and it covers a narrow set of buildings
and styles.

## Files

| Path | Contents |
|---|---|
| `pairs.csv` | one row per pair: style, landmark, Wikimedia file pages and licences |
| `marks/<pair_id>.csv` | the annotator's points: `point_id, hist_x, hist_y, modern_x, modern_y` |
| `marks/LOCK.json` | SHA-256 of every mark file, written before the first matcher run |
| `results/` | per-matcher run outputs and the results table by style |

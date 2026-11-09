# Experiment 2: paintings, engravings and old photos

**Question.** Do modern image matchers still find the same points when one image is an engraving, a painting or an
old photograph and the other is a modern photograph of the same building? And does the answer depend on the style?

This extends the 1926 map experiment from one map sheet to depictions of buildings. It uses the same code
(`mapalign.core`) to load each matcher, fit the transform and score held-out points.

## Status

- **Pairs:** 53 pairs collected: 23 drawings, 17 paintings and 13 photos, from 14 landmarks in India and Europe.
- **Marks and results:** no points marked yet, and nothing run. Nothing in this folder is a result yet.

To mark the points:

```bash
python styles/fetch.py        # downloads the 106 images into .cache/styles/images (not committed)
python styles/mark.py         # opens http://127.0.0.1:8766; every point is saved as you go
python styles/mark.py --lock  # when all 53 pairs have 8-10 points: freezes the marks
```

## Protocol (fixed before any matcher is run on these pairs)

**Pairs.** 30–50 pairs from Wikimedia Commons. Each pair is one historical image and one modern photograph of
the same building, taken from a similar side. The historical image is grouped by style: `photo`, `drawing`
(engravings, lithographs and line drawings) or `painting`. Every image is public domain or openly licensed. The
licence, author and file page are recorded in `pairs.csv`. Images are downloaded by script and are not committed.

The pairs were chosen by looking at contact sheets of candidates before any matcher ran. The rules were:

- **Building:** the historical image and the modern photo show the same building from about the same side.
- **Licence:** each image is public domain or under an open licence (CC0, CC BY, CC BY-SA or the Free Art
  License).
- **Size:** each image's long side is at least 600 px.
- **Style:** the label comes from looking at the image, not from the Commons metadata. Three files catalogued as
  photographs or paintings are really engravings or etchings, and `notes` records each relabelling.
- **Exclusions:** a page of several small prints, a painted glass beaker, a 1906 etching still in copyright,
  three images under 600 px, and three pairs where the two images show different parts of the building.
- **Kept, with a note:** pairs where the modern photo shows only part of the building.

**Ground truth.** For each pair, one annotator marks 8–10 corresponding points: well-defined building corners,
window corners or spire tips. The annotator has not seen any matcher output, and the marks are locked (SHA-256 in
`marks/LOCK.json`) before any matcher runs on the pairs. Marks go in `marks/<pair_id>.csv`, in pixel coordinates of
the working images (long side at most 1600 px, written by `fetch.py`; OpenCV pixel-centre convention).

**Methods.** SIFT, LoFTR, DISK and ALIKED + LightGlue, RoMa and MASt3R. Every matcher feeds the same robust
estimator with the same settings. A matcher's settings are fixed before the marks are read.

**Measures, reported by style.**

1. Success rate: the share of pairs where the fit is valid, meaning it has enough inliers and passes the
   validity checks.
2. Held-out point error (primary). A homography is fitted from the matches with the same MAGSAC++ settings used
   for the map, and each marked historical point is sent through it. The error is the distance to the marked
   modern point, in pixels of the modern photo, divided by the photo's diagonal so that pairs of different sizes
   are comparable. Failed pairs are counted as failures, not dropped. A homography is exact only for a flat
   surface, which is why the pairs are mostly front-on facades.
3. Epipolar error (secondary). A fundamental matrix is fitted from the same matches, and each marked point is
   scored by its symmetric distance to the epipolar line, also divided by the diagonal. This does not assume a
   flat scene, so it shows how much of the primary error comes from the homography model rather than from the
   matcher.

These measures were chosen before any matcher ran on these pairs and are not changed after seeing results.

**Reported limits.** The sample is small, there is a single annotator, and it covers a narrow set of buildings
and styles.

## Files

| Path | Contents |
|---|---|
| `pairs.csv` | one row per pair: style, landmark, Wikimedia file pages and licences |
| `marks/<pair_id>.csv` | the annotator's points: `point_id, hist_x, hist_y, modern_x, modern_y` |
| `marks/LOCK.json` | SHA-256 of every mark file, written before the first matcher run |
| `results/` | per-matcher run outputs and the results table by style |

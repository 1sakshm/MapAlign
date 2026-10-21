"""Build every figure, results table and derived number used in paper/mapalign.tex.

Usage (from the repository root, after `python -m mapalign.run_all`):
    python paper/make_figures.py

Writes paper/figures/*.pdf, paper/tables/*.tex and paper/numbers.json. Nothing here fits a model or
changes a result; the "exploratory" analyses (bias, distance correlations, accuracy bands) are
computed after the fact from the frozen outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from affine import Affine  # noqa: E402
from matplotlib.patches import ConnectionPatch, Circle  # noqa: E402
from pyproj import Transformer  # noqa: E402
from scipy import stats  # noqa: E402

from mapalign.config import load_config, resolve_path  # noqa: E402
from mapalign.estimate_transform import apply_h  # noqa: E402
from mapalign.georeference import historical_px_to_map  # noqa: E402
from mapalign.io import read_json  # noqa: E402
from mapalign.preprocess import prepare_historical, prepare_reference  # noqa: E402

ROOT = resolve_path(".")
R = ROOT / "results"
OUT = ROOT / "paper"
FIG, TAB = OUT / "figures", OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

BLUE, RED, GREY, INK, BROWN = "#2a6fae", "#b5352b", "#8a8478", "#1f1a14", "#8b5a2b"
COL1, COL2 = 3.4, 7.1   # IEEE-like single / double column widths (inches)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})

cfg = load_config()
NUM: dict = {}


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------------ inputs
hist, ref = prepare_historical(cfg), prepare_reference(cfg)
ev = pd.read_csv(ROOT / "gcp/evaluation_points.csv")
fit = pd.read_csv(ROOT / "gcp/fitting_points.csv")
err = pd.read_csv(R / "evaluation_errors.csv")
met = read_json(R / "metrics.json")
lrun, srun, mrun = (read_json(R / f"{m}_run.json") for m in ("loftr", "sift", "manual"))
datum = read_json(R / "datum_offset.json")
sweep = read_json(R / "exploration/loftr_input_size_sweep.json")["sizes"]
sift_var = read_json(R / "exploration/sift_preprocessing_variants.json")["variants"]
gpu = read_json(R / "gpu/gpu_check.json") if (R / "gpu/gpu_check.json").is_file() else None
gt_ref = Affine.from_gdal(*lrun["reference_geotransform_gdal"])
H_l = np.array(lrun["H_full"])
H_m = np.array(mrun["H_full"])
ev_px = ev[["historical_x_px", "historical_y_px"]].to_numpy(float)
ev_true = ev[["true_easting_m", "true_northing_m"]].to_numpy(float)
used_fit = fit[fit.point_id.isin(mrun["fitting_point_ids"])]


def errs(model, method):
    return err[(err.model == model) & (err.method == method)].set_index("point_id").reindex(ev.point_id)


# ------------------------------------------------------------------ Fig 1: study sheet with GCP layout
LBL = {"E08": (4, 4), "E03": (-20, -10), "E01": (6, 5), "F08": (-6, -16), "F05": (-22, 4), "E09": (-18, -10)}
fig, ax = plt.subplots(figsize=(COL1, COL1 * 1.05))
ax.imshow(rgb(hist.color))
F = hist.full_to_work
poly = apply_h(F, np.array([cfg["historical_map"]["neatline_corners_px"][k] for k in ("nw", "ne", "se", "sw")], float))
x0, y0 = poly.min(0)
x1, y1 = poly.max(0)
for i in range(1, 5):
    ax.axvline(x0 + (x1 - x0) * i / 5, color="white", lw=0.5, alpha=0.8)
    ax.axhline(y0 + (y1 - y0) * i / 5, color="white", lw=0.5, alpha=0.8)
for _, f in fit.iterrows():
    p = apply_h(F, np.array([[f.historical_x_px, f.historical_y_px]]))[0]
    used = f.point_id in mrun["fitting_point_ids"]
    ax.scatter(*p, marker="^", s=34, facecolor=BROWN if used else "white", edgecolor=BROWN, lw=1, zorder=3)
    ax.annotate(f.point_id, p, xytext=LBL.get(f.point_id, (4, 3)), textcoords="offset points", fontsize=5.5, color="white",
                bbox=dict(boxstyle="round,pad=0.1", fc=BROWN, ec="none", alpha=0.85))
for _, e in ev.iterrows():
    p = apply_h(F, np.array([[e.historical_x_px, e.historical_y_px]]))[0]
    ax.scatter(*p, marker="o", s=26, facecolor=INK, edgecolor="white", lw=0.8, zorder=3)
    ax.annotate(e.point_id, p, xytext=LBL.get(e.point_id, (4, -8)), textcoords="offset points", fontsize=5.5, color="white",
                bbox=dict(boxstyle="round,pad=0.1", fc=INK, ec="none", alpha=0.85))
ax.set_xlim(x0 - 20, x1 + 20)
ax.set_ylim(y1 + 20, y0 - 20)
ax.set_xticks([]), ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(True)
save(fig, "fig1_sheet_gcps")

# ------------------------------------------------------------------ Fig 3: correspondences LoFTR vs SIFT
def corr_panel(ax_l, ax_r, npz, title):
    d = np.load(npz)
    inl = d["inlier_mask"].astype(bool)
    k = 0.35
    hs = cv2.resize(hist.color, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    rs = cv2.resize(ref.color, None, fx=k * hist.gray.shape[1] / ref.gray.shape[1] * 1.0, fy=k * hist.gray.shape[1] / ref.gray.shape[1], interpolation=cv2.INTER_AREA)
    kr = rs.shape[1] / ref.color.shape[1]
    ax_l.imshow(rgb(hs)), ax_r.imshow(rgb(rs))
    a = (d["pts0_work"] + .5) * k - .5
    b = (d["pts1_work"] + .5) * kr - .5
    rng = np.random.default_rng(0)
    out = np.flatnonzero(~inl)
    out = rng.choice(out, min(150, len(out)), replace=False) if len(out) else out
    for idx, col, al, lw in ((out, RED, .35, .35), (np.flatnonzero(inl), BLUE, .6, .4)):
        for i in idx:
            ax_r.add_artist(ConnectionPatch(xyA=a[i], coordsA=ax_l.transData, xyB=b[i], coordsB=ax_r.transData,
                                            color=col, lw=lw, alpha=al))
    for ax in (ax_l, ax_r):
        ax.set_xticks([]), ax.set_yticks([])
    ax_l.set_title(title, loc="left")


fig, axes = plt.subplots(2, 2, figsize=(COL2, COL2 * 0.98), gridspec_kw={"wspace": 0.04, "hspace": 0.08})
corr_panel(axes[0, 0], axes[0, 1], R / "loftr_correspondences.npz",
           f"(a) LoFTR: {lrun['estimation']['input_correspondences']} matches, {lrun['estimation']['inliers']} inliers")
corr_panel(axes[1, 0], axes[1, 1], R / "sift_correspondences.npz",
           f"(b) SIFT: {srun['estimation']['input_correspondences']} matches, no valid model")
save(fig, "fig3_correspondences")

# ------------------------------------------------------------------ Fig 4: diagnostics (tiles, size sweep)
tl, ts = lrun["prior_diagnostics"]["aligned_tile_upper_bound"], srun["prior_diagnostics"]["aligned_tile_upper_bound"]
fig, axes = plt.subplots(1, 3, figsize=(COL2, 1.75), gridspec_kw={"width_ratios": [1, 1.2, 1.2], "wspace": 0.45})
ax = axes[0]
vals = [ts["correct_fraction"] * 100, ts["chance_fraction"] * 100, tl["correct_fraction"] * 100]
ax.barh([2, 1, 0], vals, color=[GREY, "white", BLUE], edgecolor=[GREY, INK, BLUE], height=0.62, linestyle="-")
ax.barh([1], [vals[1]], color="none", edgecolor=INK, height=0.62, linestyle="--")
for y, v in zip([2, 1, 0], vals):
    ax.text(v + 0.3, y, f"{v:.2f}%", va="center", fontsize=7)
ax.set_yticks([2, 1, 0], ["SIFT", "chance", "LoFTR"])
ax.set_xlim(0, 16)
ax.set_xlabel("correct matches on aligned tiles (%)")
ax.set_title("(a)", loc="left")
sw = [r for r in sweep if r.get("fraction_within") is not None]
xs = [r["max_side_px"] for r in sw]
ax = axes[1]
ax.plot(xs, [r["fraction_within"] * 100 for r in sw], color=BLUE, marker="o", ms=3.5, lw=1.2)
ax.scatter([768], [next(r["fraction_within"] for r in sw if r["max_side_px"] == 768) * 100], s=40, color=RED, zorder=3)
ax.annotate("selected", (768, 71.4), xytext=(6, 6), textcoords="offset points", fontsize=6.5, color=RED)
ax.set_xlabel("LoFTR input, longest side (px)")
ax.set_ylabel("matches near prior (%)")
ax.set_ylim(0, 100)
ax.set_xticks(xs + [1536], [str(v) for v in xs] + ["1536"], rotation=45)
ax.axvline(1536, color=RED, ls=":", lw=0.8)
ax.text(1520, 92, "OOM", color=RED, fontsize=6.5, ha="right")
ax.set_title("(b)", loc="left")
ax = axes[2]
ax.plot(xs, [r["inference_s"] for r in sw], color=INK, marker="s", ms=3, lw=1)
ax.set_xlabel("LoFTR input, longest side (px)")
ax.set_ylabel("CPU inference (s)")
ax.set_xticks(xs, [str(v) for v in xs], rotation=45)
ax.set_title("(c)", loc="left")
save(fig, "fig4_diagnostics")

# ------------------------------------------------------------------ Fig 5: qualitative alignment windows
osm = read_json(ROOT / cfg["modern_reference"]["raw_path"])
tf = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
lines = []
for el in osm["elements"]:
    t = el.get("tags", {})
    style = (INK, 1.1) if t.get("railway") == "rail" else (BLUE, 1.0) if t.get("waterway") == "river" else \
        ("#c2185b", 0.8) if t.get("highway") in ("trunk", "primary", "secondary", "tertiary") else None
    if style and el["type"] == "way" and "geometry" in el:
        e, n = tf.transform([p["lon"] for p in el["geometry"]], [p["lat"] for p in el["geometry"]])
        lines.append((np.column_stack([e, n]), style))
wins = [("(a) Srirangapatna fort", 683000, 1373300), ("(b) Belagula level crossing", 672570, 1369980),
        ("(c) Mysore station (not fitted)", 676600, 1361700)]
fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.45), gridspec_kw={"wspace": 0.06})
half = 700
with rasterio.open(R / "georef_loftr.tif") as ds:
    for ax, (name, cx, cy) in zip(axes, wins):
        win = rasterio.windows.from_bounds(cx - half, cy - half, cx + half, cy + half, ds.transform)
        img = np.moveaxis(ds.read([1, 2, 3], window=win, boundless=True, fill_value=255), 0, 2)
        ax.imshow(img, extent=(cx - half, cx + half, cy - half, cy + half))
        for xy, (col, lw) in lines:
            m = (np.abs(xy[:, 0] - cx) < 2 * half) & (np.abs(xy[:, 1] - cy) < 2 * half)
            if m.any():
                ax.plot(xy[:, 0], xy[:, 1], color=col, lw=lw, alpha=0.85)
        ax.set_xlim(cx - half, cx + half), ax.set_ylim(cy - half, cy + half)
        ax.plot([cx + half - 560, cx + half - 60], [cy - half + 90] * 2, color="black", lw=2.2)
        ax.text(cx + half - 310, cy - half + 140, "500 m", ha="center", fontsize=6.5,
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.6))
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
        ax.set_title(name, loc="left")
save(fig, "fig5_alignment_windows")

# ------------------------------------------------------------------ Fig 6: error vectors on the sheet
fig, axes = plt.subplots(1, 2, figsize=(COL2, COL2 * 0.5), gridspec_kw={"wspace": 0.04})
pw = apply_h(F, ev_px)
EX = 20.0
for ax, m, col, lab in ((axes[0], "loftr", BLUE, "(a) LoFTR + MAGSAC"), (axes[1], "manual", RED, "(b) Manual GCP fit")):
    e = errs("homography", m)
    ax.imshow(rgb(hist.color), alpha=0.72)
    ax.quiver(pw[:, 0], pw[:, 1], e.dE_m.to_numpy() / 10 * EX, -e.dN_m.to_numpy() / 10 * EX, angles="xy",
              scale_units="xy", scale=1, color=col, width=0.005, zorder=3)
    ax.scatter(pw[:, 0], pw[:, 1], s=12, color=INK, zorder=4)
    for p, pid, v in zip(pw, ev.point_id, e.error_m):
        ax.annotate(f"{pid} {v:.0f}", p, xytext={"E08": (-4, 8), "E03": (6, -11), "E01": (7, -9)}.get(pid, (3, -8)),
                    textcoords="offset points", fontsize=5.5, bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.75))
    if m == "manual":
        for _, f in used_fit.iterrows():
            q = apply_h(F, np.array([[f.historical_x_px, f.historical_y_px]]))[0]
            ax.scatter(*q, marker="^", s=22, color=BROWN, zorder=2.5)
    ax.set_xlim(x0, x1), ax.set_ylim(y1, y0)
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(f"{lab}; arrows ×{EX:g}, north up", loc="left")
save(fig, "fig6_error_vectors")

# ------------------------------------------------------------------ Fig 7: per-point and ECDF
fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.5), gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.3})
ax = axes[0]
el_, em_ = errs("homography", "loftr").error_m.to_numpy(), errs("homography", "manual").error_m.to_numpy()
order = np.argsort(el_)
y = np.arange(len(order))
for yy, i in zip(y, order):
    ax.plot([el_[i], em_[i]], [yy, yy], color="#d8d2c4", lw=2.2, zorder=1)
ax.scatter(el_[order], y, color=BLUE, s=16, zorder=2, label="LoFTR")
ax.scatter(em_[order], y, color=RED, s=16, zorder=2, label="Manual")
ax.set_yticks(y, ev.point_id.to_numpy()[order])
ax.set_xlabel("held-out error (m)")
ax.set_xlim(0, 240)
ax.legend(frameon=False, loc="lower right")
ax.set_title("(a) paired per-point error (homography)", loc="left")
ax = axes[1]
for m, mod, col, ls in (("loftr", "homography", BLUE, "-"), ("loftr", "affine", BLUE, "--"),
                        ("manual", "homography", RED, "-"), ("manual", "affine", RED, "--")):
    v = np.sort(errs(mod, m).error_m.to_numpy())
    ax.step(np.r_[0, v], np.r_[0, np.arange(1, len(v) + 1) / len(v)], where="post", color=col, ls=ls, lw=1.1,
            label=f"{'LoFTR' if m == 'loftr' else 'Manual'}, {mod}")
nmas = 63360 * 0.0254 / 50
ax.axvline(nmas, color=GREY, lw=0.8, ls=":")
ax.text(nmas + 3, 0.97, f"NMAS {nmas:.0f} m", fontsize=6.5, color=GREY, va="top")
ax.set_xlabel("held-out error (m)")
ax.set_ylabel("fraction of points")
ax.set_xlim(0, 240)
ax.legend(frameon=False, loc="lower right")
ax.set_title("(b) empirical CDF", loc="left")
save(fig, "fig7_errors")

# ------------------------------------------------------------------ Fig 8: error components and bias
fig, axes = plt.subplots(1, 2, figsize=(COL1 * 1.35, COL1 * 0.72), sharex=True, sharey=True, gridspec_kw={"wspace": 0.08})
bias = {}
rng = np.random.default_rng(42)
for ax, m, col, lab in ((axes[0], "loftr", BLUE, "(a) LoFTR"), (axes[1], "manual", RED, "(b) Manual")):
    e = errs("homography", m)
    dE, dN = e.dE_m.to_numpy(), e.dN_m.to_numpy()
    for r in (50, 100, 200):
        ax.add_patch(Circle((0, 0), r, fill=False, lw=0.5, color=GREY, ls=":"))
    ax.axhline(0, color=GREY, lw=0.5), ax.axvline(0, color=GREY, lw=0.5)
    ax.scatter(dE, dN, s=14, color=col, alpha=0.85)
    mb = np.array([dE.mean(), dN.mean()])
    boots = np.array([[dE[i].mean(), dN[i].mean()] for i in rng.integers(0, len(dE), (5000, len(dE)))])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    bias[m] = {"mean_dE_m": round(mb[0], 1), "mean_dN_m": round(mb[1], 1), "magnitude_m": round(float(np.hypot(*mb)), 1),
               "ci95_dE_m": lo[0].round(1).tolist() and [round(lo[0], 1), round(hi[0], 1)],
               "ci95_dN_m": [round(lo[1], 1), round(hi[1], 1)]}
    ax.annotate("", xy=mb, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.2", color=INK, lw=1.4, shrinkA=0, shrinkB=0), zorder=5)
    ax.annotate(f"mean ({mb[0]:+.0f}, {mb[1]:+.0f}) m", mb, xytext=(6, 8 if mb[1] > 0 else -12), textcoords="offset points", fontsize=6.5)
    ax.set_aspect("equal")
    ax.set_xlim(-240, 240), ax.set_ylim(-240, 240)
    ax.set_xlabel(r"$\Delta E$ (m)")
    ax.set_title(lab, loc="left")
axes[0].set_ylabel(r"$\Delta N$ (m)")
save(fig, "fig8_error_components")
NUM["bias"] = bias

# ------------------------------------------------------------------ Fig 9: datum / graticule offsets
fig, ax = plt.subplots(figsize=(COL1, COL1 * 0.85))
off = np.array(list(datum["per_point_offset_m"].values()))
ids = list(datum["per_point_offset_m"].keys())
excl = set(mrun["qc_excluded_fitting_points"])
for r in (100, 200, 300, 500):
    ax.add_patch(Circle((0, 0), r, fill=False, lw=0.5, color=GREY, ls=":"))
    if r * 0.71 < 250:
        ax.text(r * 0.71, r * 0.71, f"{r} m", fontsize=5.5, color=GREY)
ax.axhline(0, color=GREY, lw=0.5), ax.axvline(0, color=GREY, lw=0.5)
for (x, yv), pid in zip(off, ids):
    ax.scatter(x, yv, marker="^", s=22, facecolor="white" if pid in excl else BROWN, edgecolor=BROWN, zorder=3)
    dx, dy = {"F04": (-14, 4), "F06": (4, 6)}.get(pid, (3, 3))
    ax.annotate(pid, (x, yv), xytext=(dx, dy), textcoords="offset points", fontsize=5.5)
proj = datum["proj_datum_shifts_at_sheet_centre"]
marks = [("median of fitting GCPs", datum["median_offset_E_N_m"], INK, "o"), ("LoFTR matches", [-128, 133], BLUE, "s")]
for code, lab in (("EPSG:4144", "PROJ Kalianpur 1937 (2)"), ("EPSG:4146", "PROJ Kalianpur 1975")):
    if proj.get(code):
        marks.append((lab, [proj[code][0]["shift_E_m"], proj[code][0]["shift_N_m"]], RED if "1937" in lab else GREY, "D"))
for lab, (x, yv), col, mk in marks:
    ax.scatter(x, yv, marker=mk, s=30, color=col, zorder=4, label=lab)
ax.set_aspect("equal")
ax.set_xlim(-560, 260), ax.set_ylim(-160, 300)
ax.set_xlabel("true − graticule, east (m)")
ax.set_ylabel("true − graticule, north (m)")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6)
save(fig, "fig9_datum")

# ------------------------------------------------------------------ Fig 10: distance analyses (exploratory)
d = np.load(R / "loftr_correspondences.npz")
inl_px = apply_h(np.linalg.inv(hist.full_to_work), d["pts0_work"][d["inlier_mask"].astype(bool)])
inl_map = historical_px_to_map(inl_px, H_l, gt_ref)
d_inl = np.min(np.linalg.norm(ev_true[:, None, :] - inl_map[None], axis=2), axis=1)
fit_map = used_fit[["true_easting_m", "true_northing_m"]].to_numpy(float)
d_fit = np.min(np.linalg.norm(ev_true[:, None, :] - fit_map[None], axis=2), axis=1)
sp_l = stats.spearmanr(d_inl, el_)
sp_m = stats.spearmanr(d_fit, em_)
NUM["distance_correlation"] = {
    "loftr_vs_nearest_inlier": {"rho": round(float(sp_l.statistic), 2), "p": round(float(sp_l.pvalue), 3),
                                "median_distance_m": round(float(np.median(d_inl)), 0)},
    "manual_vs_nearest_fitting_point": {"rho": round(float(sp_m.statistic), 2), "p": round(float(sp_m.pvalue), 3),
                                        "median_distance_m": round(float(np.median(d_fit)), 0)}}
fig, axes = plt.subplots(1, 2, figsize=(COL2 * 0.75, 2.0), sharey=True, gridspec_kw={"wspace": 0.12})
for ax, dd, ee, col, xl, sp in ((axes[0], d_inl / 1000, el_, BLUE, "distance to nearest LoFTR inlier (km)", sp_l),
                                (axes[1], d_fit / 1000, em_, RED, "distance to nearest fitting GCP (km)", sp_m)):
    ax.scatter(dd, ee, s=14, color=col)
    for x_, y_, pid in zip(dd, ee, ev.point_id):
        ax.annotate(pid, (x_, y_), xytext=(2, 2), textcoords="offset points", fontsize=5)
    ax.set_xlabel(xl)
    ax.text(0.97, 0.93, rf"Spearman $\rho$ = {sp.statistic:.2f} ($p$ = {sp.pvalue:.2f})", transform=ax.transAxes,
            ha="right", fontsize=6.5)
axes[0].set_ylabel("held-out error (m)")
save(fig, "fig10_distance")

# ------------------------------------------------------------------ Fig 11: Method C fit diagnostics
diag = mrun["fitting_diagnostics"]
fig, ax = plt.subplots(figsize=(COL1, 1.7))
xi = np.arange(len(mrun["fitting_point_ids"]))
ax.bar(xi - 0.19, diag["fit_residual_m"], 0.36, color=BROWN, label="fit residual")
ax.bar(xi + 0.19, diag["leave_one_out_error_m"], 0.36, color="white", edgecolor=BROWN, hatch="////", lw=0.6,
       label="leave-one-out error")
ax.set_xticks(xi, mrun["fitting_point_ids"])
ax.set_ylabel("m")
ax.legend(frameon=False, fontsize=6.5)
save(fig, "fig11_methodc_fit")

# ------------------------------------------------------------------ numbers
nmas_m = 63360 * 0.0254 / 50
for m in ("loftr", "manual"):
    for mod in ("homography", "affine"):
        v = errs(mod, m).error_m.to_numpy()
        NUM.setdefault("bands", {})[f"{mod}/{m}"] = {
            "within_50m": int((v <= 50).sum()), "within_100m": int((v <= 100).sum()),
            "within_nmas": int((v <= nmas_m).sum()), "n": int(len(v)),
            "rmse_in_map_mm": round(float(np.sqrt(np.mean(v ** 2))) / 63360 * 1000, 2),
            "median_in_map_mm": round(float(np.median(v)) / 63360 * 1000, 2)}
NUM["nmas_threshold_m"] = round(nmas_m, 1)
NUM["metrics"] = met["metrics"]
NUM["paired"] = met["paired_comparisons"]
NUM["sensitivity"] = met["sensitivity_without_low_confidence"]
NUM["gpu"] = gpu
NUM["matching"] = {"sift": srun["matching"], "loftr": lrun["matching"], "sift_est": srun["estimation"],
                   "loftr_est": lrun["estimation"], "tiles": {"sift": ts, "loftr": tl}}
NUM["datum"] = {k: datum[k] for k in ("median_offset_E_N_m", "qc_robust_sigma_E_N_m", "qc_flagged_z")}
NUM["methodc"] = {"fit": diag, "excluded": mrun["qc_excluded_fitting_points"], "graticule_corners": mrun["graticule_corner_check"]["mean_distance_m"]}
NUM["loftr_graticule_corner_mean_m"] = lrun["graticule_corner_check"]["mean_distance_m"]
(OUT / "numbers.json").write_text(json.dumps(NUM, indent=1, default=float), encoding="utf-8")

# ------------------------------------------------------------------ tables
def fm(v, nd=1):
    return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


M = met["metrics"]
rows = []
for key, name in (("homography/loftr", "LoFTR + MAGSAC"), ("homography/manual", "Manual GCPs (7)"),
                  ("homography/sift", "SIFT + MAGSAC")):
    v = M[key]
    rows.append(f"{name} & {v['n_valid']} & {fm(v['median_m'])} & {fm(v['mean_m'])} & {fm(v['rmse_m'])} & "
                f"{fm(v['std_m'])} & {fm(v['p90_m'])} & {fm(v['min_m'])} & {fm(v['max_m'])} \\\\")
(TAB / "results.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

rows = []
for _, e in ev.iterrows():
    a, b = errs("homography", "loftr").loc[e.point_id], errs("homography", "manual").loc[e.point_id]
    ft = {"rail_road_crossing": "rail $\\times$ road", "rail_water_crossing": "rail $\\times$ water",
          "road_water_crossing": "road $\\times$ water", "road_stream_crossing": "road $\\times$ stream",
          "road_road_junction": "road junction", "landmark_centroid": "landmark"}.get(e.feature_type, e.feature_type)
    rows.append(f"{e.point_id} & {ft} & {e.confidence} & {e.true_easting_m:.0f} & {e.true_northing_m:.0f} & "
                f"{a.dE_m:+.0f} & {a.dN_m:+.0f} & \\textbf{{{a.error_m:.1f}}} & {b.dE_m:+.0f} & {b.dN_m:+.0f} & "
                f"\\textbf{{{b.error_m:.1f}}} \\\\")
(TAB / "points.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

rows = []
for r in sweep:
    if r.get("error"):
        rows.append(f"{r['max_side_px']} & -- & -- & -- & -- & out of memory \\\\")
    else:
        rows.append(f"{r['max_side_px']} & {r['effective_resolution_m']:.1f} & {r['matches']} & {r['within_1km_of_prior']} & "
                    f"{100 * r['fraction_within']:.1f} & {r['inference_s']:.1f} \\\\")
(TAB / "sweep.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

nice = {"gray_10m": "grey, 10 m", "clahe_10m": "grey + CLAHE, 10 m", "gray_20m": "grey, 20 m", "layers_10m": "colour layers, 10 m",
        "layers_20m": "colour layers, 20 m", "layers_30m": "colour layers, 30 m", "gray_45m": "grey, 45 m", "layers_45m": "colour layers, 45 m"}
rows = [f"{nice.get(v['variant'], v['variant'])} & {v['keypoints'][0]:,} & {v['keypoints'][1]:,} & {v['ratio_matches']} & "
        f"{v['prior_consistency_1km']['matches_within']} & {v['inliers']} & {'yes' if v['valid'] else 'no'} \\\\"
        for v in sift_var]
(TAB / "sift_variants.tex").write_text("\n".join(rows).replace(",", "{,}") + "\n", encoding="utf-8")
print(json.dumps({k: NUM[k] for k in ("bias", "distance_correlation", "bands", "nmas_threshold_m")}, indent=1, default=float))
print("figures:", sorted(p.name for p in FIG.glob("*.pdf")))

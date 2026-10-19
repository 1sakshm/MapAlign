"""Match and overlay figures. All figures are generated from saved experiment outputs."""

from __future__ import annotations

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import ConnectionPatch  # noqa: E402

from mapalign.estimate_transform import apply_h  # noqa: E402


def _rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img


def plot_matches(img0: np.ndarray, img1: np.ndarray, pts0: np.ndarray, pts1: np.ndarray,
                 inliers: np.ndarray | None, title: str, out_path, max_lines: int = 250, seed: int = 0) -> None:
    """Side-by-side images with correspondence lines (green = RANSAC inlier, red = outlier)."""
    rng = np.random.default_rng(seed)
    inl = np.zeros(len(pts0), bool) if inliers is None else inliers.astype(bool)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(16, 8.6))
    for ax, img, name in ((a0, img0, "historical (working px)"), (a1, img1, "reference (working px)")):
        ax.imshow(_rgb(img), cmap="gray" if img.ndim == 2 else None)
        ax.set_title(name, fontsize=10)
        ax.set_xticks([]), ax.set_yticks([])
    for sel, colour, alpha in ((~inl, "red", 0.25), (inl, "lime", 0.7)):
        idx = np.flatnonzero(sel)
        if len(idx) > max_lines:
            idx = rng.choice(idx, max_lines, replace=False)
        for i in idx:
            fig.add_artist(ConnectionPatch(xyA=pts0[i], coordsA=a0.transData, xyB=pts1[i], coordsB=a1.transData,
                                           color=colour, lw=0.5, alpha=alpha))
        a0.scatter(pts0[idx, 0], pts0[idx, 1], s=3, c=colour)
        a1.scatter(pts1[idx, 0], pts1[idx, 1], s=3, c=colour)
    fig.suptitle(f"{title}\n{len(pts0)} matches, {int(inl.sum())} RANSAC inliers (green), "
                 f"{int((~inl).sum())} outliers (red); at most {max_lines} lines drawn per class", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_overlay(hist_color: np.ndarray, ref_color: np.ndarray, H_work: np.ndarray, title: str, out_path,
                 graticule_px: np.ndarray | None = None, predicted_px: np.ndarray | None = None) -> None:
    """Historical map warped into the reference frame. Three panels: blend, checkerboard, edges."""
    h, w = ref_color.shape[:2]
    warped = cv2.warpPerspective(hist_color, H_work, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    valid = cv2.warpPerspective(np.full(hist_color.shape[:2], 255, np.uint8), H_work, (w, h)) > 0
    blend = ref_color.copy()
    blend[valid] = (0.5 * warped[valid] + 0.5 * ref_color[valid]).astype(np.uint8)

    tile = max(32, w // 12)
    yy, xx = np.mgrid[0:h, 0:w]
    checker = ((yy // tile + xx // tile) % 2 == 0) & valid
    board = ref_color.copy()
    board[checker] = warped[checker]

    # Reference linework (dark/coloured pixels) drawn over the warped historical map.
    ref_gray = cv2.cvtColor(ref_color, cv2.COLOR_BGR2GRAY)
    lines = ref_gray < 200
    edges = warped.copy()
    edges[lines & valid] = (255, 0, 255)

    fig, axes = plt.subplots(1, 3, figsize=(21, 7.6))
    for ax, img, name in zip(axes, (blend, board, edges),
                             ("50/50 blend", "checkerboard (historical / reference)", "reference linework (magenta) on warped historical")):
        ax.imshow(_rgb(img))
        ax.set_title(name, fontsize=10)
        ax.set_xticks([]), ax.set_yticks([])
        if graticule_px is not None:
            ax.scatter(graticule_px[:, 0], graticule_px[:, 1], marker="+", s=120, c="blue", label="printed corner (graticule)")
        if predicted_px is not None:
            ax.scatter(predicted_px[:, 0], predicted_px[:, 1], marker="x", s=80, c="red", label="corner via transform")
    if graticule_px is not None or predicted_px is not None:
        axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_gcps(hist_color: np.ndarray, ref_color: np.ndarray, pts_hist: np.ndarray, pts_ref: np.ndarray,
              ids: list[str], pred_ref: np.ndarray | None, title: str, out_path, exaggeration: float = 20.0) -> None:
    """GCP locations on both images; on the reference, residual vectors (predicted - true) exaggerated."""
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(16, 8.4))
    a0.imshow(_rgb(hist_color))
    a1.imshow(_rgb(ref_color))
    a0.scatter(pts_hist[:, 0], pts_hist[:, 1], s=60, facecolors="none", edgecolors="magenta", linewidths=2)
    a1.scatter(pts_ref[:, 0], pts_ref[:, 1], s=60, facecolors="none", edgecolors="magenta", linewidths=2)
    for i, pid in enumerate(ids):
        a0.annotate(pid, pts_hist[i], xytext=(6, 6), textcoords="offset points", color="magenta", fontsize=9, weight="bold")
        a1.annotate(pid, pts_ref[i], xytext=(6, 6), textcoords="offset points", color="magenta", fontsize=9, weight="bold")
    if pred_ref is not None:
        d = (pred_ref - pts_ref) * exaggeration
        a1.quiver(pts_ref[:, 0], pts_ref[:, 1], d[:, 0], d[:, 1], angles="xy", scale_units="xy", scale=1,
                  color="blue", width=0.003)
    a0.set_title("historical (working px)", fontsize=10)
    a1.set_title(f"reference (working px); blue = residual x{exaggeration:g}", fontsize=10)
    for ax in (a0, a1):
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


SERIES = {"loftr": "#2a78d6", "manual": "#eb6834", "sift": "#1baf7a"}   # reference palette slots 1-3
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def plot_error_vectors(hist_color: np.ndarray, pts_work: np.ndarray, err_rows, label: str, metrics: dict,
                       out_path, exaggeration: float = 5.0) -> None:
    """Held-out points on the historical map with error vectors (predicted - true) exaggerated.

    Vectors are drawn in map orientation: +E to the right, +N up (image y is flipped).
    """
    fig, ax = plt.subplots(figsize=(9, 9.6))
    ax.imshow(_rgb(hist_color), alpha=0.55)
    ok = err_rows.error_m.notna().to_numpy()
    if ok.any():
        dE = err_rows.dE_m.to_numpy(float)[ok] / 10.0 * exaggeration   # metres -> ~10 m working px
        dN = err_rows.dN_m.to_numpy(float)[ok] / 10.0 * exaggeration
        ax.quiver(pts_work[ok, 0], pts_work[ok, 1], dE, -dN, angles="xy", scale_units="xy", scale=1,
                  color="#e34948", width=0.004, zorder=3)
    ax.scatter(pts_work[:, 0], pts_work[:, 1], s=50, c=np.where(ok, "#0b0b0b", "#9a9893"), marker="o",
               edgecolors="white", linewidths=1.5, zorder=4)
    for (x, y), pid, e in zip(pts_work, err_rows.point_id, err_rows.error_m):
        txt = f"{pid}  {e:.0f} m" if e == e and e is not None else f"{pid}  failed"
        ax.annotate(txt, (x, y), xytext=(7, -12), textcoords="offset points", fontsize=8.5, color=INK,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    if metrics.get("n_valid"):
        sub = (f"median {metrics['median_m']:.0f} m · RMSE {metrics['rmse_m']:.0f} m · max {metrics['max_m']:.0f} m "
               f"(n = {metrics['n_valid']}); red arrows = error ×{exaggeration:g}, map-oriented (N up)")
    else:
        sub = f"no valid transform (status: {metrics.get('method_status')}); every prediction failed"
    ax.set_title(f"{label}: held-out positional error\n{sub}", fontsize=10.5, color=INK, loc="left")
    ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_per_point_errors(err, out_path) -> None:
    """Dot plot of per-point errors for the methods with valid predictions (homography model)."""
    e = err[(err.model == "homography") & err.error_m.notna()]
    methods = [m for m in ("loftr", "manual", "sift") if (e.method == m).any()]
    pids = sorted(e.point_id.unique())
    labels = {"loftr": "LoFTR + RANSAC", "manual": "Manual GCPs", "sift": "SIFT + RANSAC"}
    fig, ax = plt.subplots(figsize=(10, 4.6))
    y = np.arange(len(pids))
    for m in methods:
        s = e[e.method == m].set_index("point_id").reindex(pids).error_m.to_numpy(float)
        ax.scatter(s, y, s=64, color=SERIES[m], edgecolors="white", linewidths=2, zorder=3, label=labels[m])
    for i, p in enumerate(pids):   # connector shows the paired difference
        vals = [e[(e.method == m) & (e.point_id == p)].error_m.iloc[0] for m in methods]
        ax.plot([min(vals), max(vals)], [i, i], color=GRID, lw=2, zorder=1)
    ax.set_yticks(y, pids)
    ax.invert_yaxis()
    ax.set_xlabel("positional error (m, EPSG:32643)", color=INK_2)
    ax.set_xlim(left=0)
    ax.grid(axis="x", color=GRID, lw=1)
    ax.tick_params(colors=INK_2)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=9)
    ax.set_title("Per-point held-out error, homography model (same 12 points for every method)", pad=26,
                 fontsize=10.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def mapped_outline(H: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    return apply_h(H, np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float64))

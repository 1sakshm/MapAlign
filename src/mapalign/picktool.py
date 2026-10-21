"""Timed, blind control-point picking in a window (matplotlib).

Works on any re-pick template written by `python -m mapalign.gcp blind fitting|evaluation|extra`. For each row
without a pick it shows the 1926 scan around the printed-corner prior (left; never centred on an existing pick)
and today's OSM render with the target feature marked (right). Click the same feature on the 1926 map; use the
toolbar's zoom/pan freely. The clock runs from when the point is shown until it is accepted, and the CSV is
saved after every point, so the session can be stopped and resumed.

Keys:  click = place/move the pick   Enter = accept   c = toggle "confident"   s = skip (not identifiable)
       q = save and quit

Usage:
    python -m mapalign.picktool gcp/repick_kit_evaluation/repick_evaluation_template.csv --picker "Your Name"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mapalign.config import load_config, resolve_path
from mapalign.gcp import prior_full_px
from mapalign.georeference import map_to_ref_px

PENDING_COLS = ("historical_x_px", "historical_y_px")


def pending_rows(df: pd.DataFrame) -> list[int]:
    """Rows with neither a pick nor a recorded skip."""
    picked = df[list(PENDING_COLS)].notna().all(axis=1)
    skipped = df["notes"].fillna("").astype(str).str.startswith("SKIPPED")
    return [i for i in df.index if not picked[i] and not skipped[i]]


def record(df: pd.DataFrame, i: int, xy, seconds: float, picker: str, confident: bool, skipped: bool = False) -> None:
    df.loc[i, "picker"] = picker
    df.loc[i, "seconds_spent"] = round(float(seconds), 1)
    if skipped:
        df.loc[i, "notes"] = "SKIPPED: not identifiable on the 1926 map"
        df.loc[i, "confident_yes_no"] = ""
    else:
        df.loc[i, "historical_x_px"], df.loc[i, "historical_y_px"] = round(float(xy[0]), 1), round(float(xy[1]), 1)
        df.loc[i, "confident_yes_no"] = "yes" if confident else "no"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("template")
    p.add_argument("--picker", required=True)
    p.add_argument("--half", type=int, default=320, help="half-width of the 1926 window in scan px (default 320 = ~1.7 km)")
    args = p.parse_args(argv)

    import cv2
    import matplotlib.pyplot as plt
    import rasterio

    cfg = load_config()
    path = Path(args.template)
    df = pd.read_csv(path, dtype={"point_id": str, "notes": str, "picker": str, "confident_yes_no": str})
    todo = pending_rows(df)
    if not todo:
        print("Nothing left to pick in", path)
        return 0
    hist = cv2.imread(str(resolve_path(cfg["historical_map"]["path"])))[:, :, ::-1]
    with rasterio.open(resolve_path(cfg["modern_reference"]["rendered_path"])) as ds:
        ref = np.moveaxis(ds.read(), 0, 2)
        gt = ds.transform
    prior = prior_full_px(cfg)
    ref_half = int(round(args.half * 5.37 / gt.a))

    for n, i in enumerate(todo, 1):
        r = df.loc[i]
        en = np.array([[r.true_easting_m, r.true_northing_m]], float)
        cx, cy = (int(v) for v in prior(en)[0])
        x0, y0 = max(cx - args.half, 0), max(cy - args.half, 0)
        x1, y1 = min(cx + args.half, hist.shape[1]), min(cy + args.half, hist.shape[0])
        rx, ry = (int(round(v)) for v in map_to_ref_px(en, gt)[0])
        fig, (a0, a1) = plt.subplots(1, 2, figsize=(15, 7.8))
        # extent in full-resolution scan px, so a click gives scan coordinates directly (pixel-centre convention)
        a0.imshow(hist[y0:y1, x0:x1], extent=(x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5), interpolation="nearest")
        a0.set_title("1926 scan: click the feature (toolbar zoom is fine)")
        a1.imshow(ref[max(ry - ref_half, 0):ry + ref_half, max(rx - ref_half, 0):rx + ref_half])
        a1.plot([min(rx, ref_half)], [min(ry, ref_half)], "+", color="crimson", ms=28, mew=2.5)
        a1.set_title(f"OSM today: {r.feature_type}")
        a1.set_xticks([]), a1.set_yticks([])
        fig.suptitle(f"{r.point_id}  ({n}/{len(todo)})   Enter = accept · c = confident on/off · s = skip · q = quit")
        st = {"xy": None, "conf": True, "done": None, "mark": None, "t0": time.perf_counter()}

        def redraw():
            if st["mark"] is not None:
                st["mark"].remove()
                st["mark"] = None
            if st["xy"] is not None:
                st["mark"], = a0.plot(*st["xy"], "x", color="crimson" if st["conf"] else "orange", ms=16, mew=2.5)
            a0.set_xlabel("pick: " + ("—" if st["xy"] is None else f"x={st['xy'][0]:.1f}, y={st['xy'][1]:.1f}")
                          + ("  (confident)" if st["conf"] else "  (NOT confident)"))
            fig.canvas.draw_idle()

        def on_click(ev):
            if ev.inaxes is a0 and ev.button == 1 and fig.canvas.toolbar.mode == "":
                st["xy"] = (ev.xdata, ev.ydata)
                redraw()

        def on_key(ev):
            if ev.key == "enter" and st["xy"] is not None:
                st["done"] = "accept"
            elif ev.key == "c":
                st["conf"] = not st["conf"]
                redraw()
                return
            elif ev.key == "s":
                st["done"] = "skip"
            elif ev.key == "q":
                st["done"] = "quit"
            else:
                return
            plt.close(fig)

        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("key_press_event", on_key)
        redraw()
        plt.show()
        secs = time.perf_counter() - st["t0"]
        if st["done"] in (None, "quit"):
            break
        record(df, i, st["xy"], secs, args.picker, st["conf"], skipped=st["done"] == "skip")
        df.to_csv(path, index=False)
        print(f"{r.point_id}: {st['done']} in {secs:.0f} s")
    left = len(pending_rows(df))
    print(f"Saved {path}. {left} point(s) left." + ("" if left else " All done."))
    return 0


if __name__ == "__main__":
    sys.exit(main())

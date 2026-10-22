"""Timed, blind control-point picking in the browser (localhost only).

Same rules and output as `mapalign.picktool`, served as a local web page instead of a matplotlib window. It
works on the three re-pick templates written by `python -m mapalign.gcp blind fitting|evaluation|extra`. For each
point it shows the 1926 scan around the printed-corner prior (never centred on an existing pick) and today's OSM
render with the target feature marked. The clock runs from when the point is shown until it is accepted, and the
CSV is saved after every point, so a session can be closed and resumed.

Usage:
    python -m mapalign.pickweb --picker "Your Name"            # then open http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from mapalign.config import load_config, resolve_path
from mapalign.gcp import prior_full_px
from mapalign.georeference import map_to_ref_px
from mapalign.picktool import pending_rows, record

KITS = {
    "fitting": "gcp/repick_kit_fitting/repick_fitting_template.csv",
    "evaluation": "gcp/repick_kit_evaluation/repick_evaluation_template.csv",
    "extra": "gcp/repick_kit_extra/repick_extra_template.csv",
}
DTYPES = {"point_id": str, "notes": str, "picker": str, "confident_yes_no": str}


def crop(img: np.ndarray, cx: int, cy: int, half: int, fill: int = 255):
    """Window of size 2*half centred on (cx, cy), padded where it leaves the image. Returns (patch, x0, y0)."""
    x0, y0 = cx - half, cy - half
    out = np.full((2 * half, 2 * half, img.shape[2]), fill, img.dtype)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + 2 * half, img.shape[1]), min(y0 + 2 * half, img.shape[0])
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out, x0, y0


class Session:
    def __init__(self, picker: str, half: int):
        import cv2
        import rasterio

        self.cv2, self.picker, self.half = cv2, picker, half
        cfg = load_config()
        self.hist = cv2.imread(str(resolve_path(cfg["historical_map"]["path"])))  # BGR, for imencode
        with rasterio.open(resolve_path(cfg["modern_reference"]["rendered_path"])) as ds:
            self.ref = np.ascontiguousarray(np.moveaxis(ds.read(), 0, 2)[:, :, 2::-1])
            self.gt = ds.transform
        self.prior = prior_full_px(cfg)
        self.ref_half = int(round(half * 5.37 / self.gt.a))
        self.lock = threading.Lock()

    def _path(self, kit: str) -> Path:
        if kit not in KITS:
            raise KeyError(kit)
        return resolve_path(KITS[kit])

    def _df(self, kit: str) -> pd.DataFrame:
        return pd.read_csv(self._path(kit), dtype=DTYPES)

    def state(self) -> dict:
        out = {"picker": self.picker, "kits": {}}
        for kit in KITS:
            df = self._df(kit)
            todo = set(pending_rows(df))
            rows = []
            for i, r in df.iterrows():
                status = "todo" if i in todo else ("skipped" if str(r.notes).startswith("SKIPPED") else "done")
                rows.append({"id": r.point_id, "feature": r.feature_type, "status": status})
            out["kits"][kit] = rows
        return out

    def _row(self, kit: str, pid: str):
        df = self._df(kit)
        hit = df.index[df.point_id == pid]
        if len(hit) != 1:
            raise KeyError(pid)
        return df, hit[0]

    def point(self, kit: str, pid: str) -> dict:
        df, i = self._row(kit, pid)
        r = df.loc[i]
        en = np.array([[r.true_easting_m, r.true_northing_m]], float)
        cx, cy = (int(v) for v in self.prior(en)[0])
        return {"id": pid, "feature": r.feature_type, "x0": cx - self.half, "y0": cy - self.half,
                "size": 2 * self.half, "hist": f"/img/hist?kit={kit}&id={pid}", "ref": f"/img/ref?kit={kit}&id={pid}"}

    def image(self, which: str, kit: str, pid: str) -> bytes:
        df, i = self._row(kit, pid)
        r = df.loc[i]
        en = np.array([[r.true_easting_m, r.true_northing_m]], float)
        if which == "hist":
            cx, cy = (int(v) for v in self.prior(en)[0])
            patch, _, _ = crop(self.hist, cx, cy, self.half)
        else:
            rx, ry = (int(round(v)) for v in map_to_ref_px(en, self.gt)[0])
            patch, _, _ = crop(self.ref, rx, ry, self.ref_half)
        return self.cv2.imencode(".png", patch)[1].tobytes()

    def save(self, kit: str, pid: str, body: dict) -> None:
        with self.lock:
            df, i = self._row(kit, pid)
            skipped = bool(body.get("skip"))
            xy = None if skipped else (float(body["x"]), float(body["y"]))
            secs = float(body["seconds"])
            if not pd.isna(df.loc[i, "seconds_spent"]):  # a redo adds to the time already spent on this point
                secs += float(df.loc[i, "seconds_spent"])
            if not skipped:
                df.loc[i, "notes"] = "" if str(df.loc[i, "notes"]).startswith("SKIPPED") else df.loc[i, "notes"]
            record(df, i, xy, secs, self.picker, bool(body.get("confident", True)), skipped=skipped)
            df.to_csv(self._path(kit), index=False)


def make_handler(s: Session):
    page = (Path(__file__).with_name("pickweb.html")).read_bytes()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path == "/":
                    self._send(200, page, "text/html; charset=utf-8")
                elif u.path == "/api/state":
                    self._json(s.state())
                elif u.path == "/api/point":
                    self._json(s.point(q["kit"], q["id"]))
                elif u.path in ("/img/hist", "/img/ref"):
                    self._send(200, s.image(u.path[5:], q["kit"], q["id"]), "image/png")
                else:
                    self._send(404, b"not found", "text/plain")
            except KeyError as e:
                self._json({"error": f"unknown {e}"}, 404)

        def do_POST(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path != "/api/save":
                return self._send(404, b"not found", "text/plain")
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                s.save(q["kit"], q["id"], body)
                self._json({"ok": True})
            except (KeyError, ValueError) as e:
                self._json({"error": str(e)}, 400)

    return H


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--picker", required=True)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--half", type=int, default=480, help="half-width of the 1926 window in scan px (default 480 = ~2.6 km)")
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args(argv)

    s = Session(args.picker, args.half)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(s))
    url = f"http://127.0.0.1:{args.port}"
    print(f"Picking as {args.picker!r} at {url}  (Ctrl+C to stop; every point is saved immediately)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

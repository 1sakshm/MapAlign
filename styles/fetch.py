"""Download the images for every pair in styles/pairs.csv from Wikimedia Commons.

Each image is fetched as Commons' rendition near 1600 px and resized so its long side is at most 1600 px, so the
same file comes back on every run. These working images are what the points are marked on and what the matchers see. Pairs whose
notes say `crop_left_half` (stereoview cards) keep only the left frame. Images go to .cache/styles/images, which
git ignores; they are not redistributed here, and every file's licence and author are in pairs.csv.

    python styles/fetch.py
"""

from __future__ import annotations

import csv
import io
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "styles" / "pairs.csv"
OUT = ROOT / ".cache" / "styles" / "images"
API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "MapAlign-research/1.0 (https://github.com/1sakshm/MapAlign; research use)"}
LONG_SIDE = 1600


def _get(url: str) -> bytes:
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                return r.read()
        except Exception:
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))


def rendition_url(file_page: str) -> tuple[str, list[int]]:
    """URL of Commons' rendition with long side <= LONG_SIDE, and the original size."""
    title = urllib.parse.unquote(file_page.split("/wiki/", 1)[1])

    def info(width: int) -> dict:
        q = urllib.parse.urlencode({"action": "query", "prop": "imageinfo", "titles": title, "iiprop": "url|size",
                                    "iiurlwidth": width, "format": "json"})
        return next(iter(json.loads(_get(f"{API}?{q}"))["query"]["pages"].values()))["imageinfo"][0]

    ii = info(LONG_SIDE)
    if ii["height"] > ii["width"] and ii["height"] > LONG_SIDE:  # iiurlwidth bounds the width only
        ii = info(max(1, int(ii["width"] * LONG_SIDE / ii["height"])))
    return ii.get("thumburl", ii["url"]), [ii["width"], ii["height"]]


def image_path(pair_id: str, side: str) -> Path:
    return OUT / f"{pair_id}_{side}.jpg"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(PAIRS, encoding="utf-8")))
    sizes = {}
    for r in rows:
        for side in ("historical", "modern"):
            p = image_path(r["pair_id"], side)
            if p.exists():
                continue
            url, original = rendition_url(r[f"{side}_file_page"])
            im = Image.open(io.BytesIO(_get(url))).convert("RGB")
            im.thumbnail((LONG_SIDE, LONG_SIDE), Image.LANCZOS)  # Commons may round up to a standard size
            if side == "historical" and "crop_left_half" in r["notes"]:
                im = im.crop((0, 0, im.width // 2, im.height))
            im.save(p, quality=95)
            sizes[p.name] = {"original_px": original, "working_px": list(im.size)}
            print(p.name, im.size)
            time.sleep(0.5)
    if sizes:
        manifest = OUT / "sizes.json"
        old = json.loads(manifest.read_text()) if manifest.exists() else {}
        manifest.write_text(json.dumps({**old, **sizes}, indent=1))
    print(f"{len(rows)} pairs ready in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

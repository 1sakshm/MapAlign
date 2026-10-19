"""Verify that the environment can run every stage of the pipeline.

Usage:
    python -m mapalign.envcheck            # full check, including LoFTR weight download (~45 MB once)
    python -m mapalign.envcheck --no-loftr

Writes results/environment.json (package versions, hardware, check outcomes).
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import logging
import platform
import sys
import tempfile
import time
import traceback
from pathlib import Path

from mapalign.config import REPO_ROOT, ensure_torch_home, load_config
from mapalign.io import write_json

log = logging.getLogger("mapalign.envcheck")

PACKAGES = ["torch", "kornia", "opencv-python-headless", "numpy", "pandas", "matplotlib",
            "pillow", "rasterio", "pyproj", "scipy", "pyyaml", "pytest"]


def check_versions() -> dict:
    out = {}
    for p in PACKAGES:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def check_torch() -> dict:
    import torch

    info = {"cuda_available": torch.cuda.is_available(), "torch_cuda_build": torch.version.cuda}
    if info["cuda_available"]:
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def check_sift() -> dict:
    import cv2
    import numpy as np

    rng = np.random.default_rng(0)
    img = (rng.random((400, 400)) * 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (5, 5), 1.5)
    shifted = np.roll(img, (7, 11), axis=(0, 1))
    sift = cv2.SIFT_create()
    k1, d1 = sift.detectAndCompute(img, None)
    k2, d2 = sift.detectAndCompute(shifted, None)
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    src = np.float32([k1[m.queryIdx].pt for m in good])
    dst = np.float32([k2[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 3.0)
    if H is None:
        raise RuntimeError("findHomography failed on synthetic pair")
    tx, ty = float(H[0, 2] / H[2, 2]), float(H[1, 2] / H[2, 2])
    if abs(tx - 11) > 0.5 or abs(ty - 7) > 0.5:
        raise RuntimeError(f"Recovered shift ({tx:.2f}, {ty:.2f}) != (11, 7)")
    return {"keypoints": [len(k1), len(k2)], "ratio_matches": len(good),
            "inliers": int(mask.sum()), "recovered_shift_px": [round(tx, 3), round(ty, 3)]}


def check_loftr(device: str) -> dict:
    import cv2
    import kornia.feature as KF
    import numpy as np
    import torch

    torch_home = ensure_torch_home()
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur((rng.random((480, 640)) * 255).astype(np.uint8), (0, 0), 2.0)
    shifted = np.roll(base, (8, 16), axis=(0, 1))
    to_t = lambda a: torch.from_numpy(a).float()[None, None].div(255.0).to(device)  # noqa: E731

    model = KF.LoFTR(pretrained="outdoor").to(device).eval()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model({"image0": to_t(base), "image1": to_t(shifted)})
    dt_s = time.perf_counter() - t0
    k0, k1 = out["keypoints0"].cpu().numpy(), out["keypoints1"].cpu().numpy()
    if len(k0) < 20:
        raise RuntimeError(f"LoFTR returned only {len(k0)} matches on synthetic pair")
    shift = np.median(k1 - k0, axis=0)
    return {"output_keys": sorted(out.keys()), "matches": int(len(k0)),
            "median_shift_px": [round(float(shift[0]), 2), round(float(shift[1]), 2)],
            "expected_shift_px": [16, 8], "inference_s_640x480": round(dt_s, 2),
            "device": device, "torch_home": str(torch_home)}


def check_pyproj(cfg: dict) -> dict:
    from pyproj import CRS, Geod, Transformer

    to_utm = Transformer.from_crs(cfg["crs"]["geographic"], cfg["crs"]["evaluation"], always_xy=True)
    back = Transformer.from_crs(cfg["crs"]["evaluation"], cfg["crs"]["geographic"], always_xy=True)
    lon, lat = 76.625, 12.375  # centre of sheet 57 D/11 (approx.)
    e, n = to_utm.transform(lon, lat)
    lon2, lat2 = back.transform(e, n)
    if abs(lon2 - lon) > 1e-9 or abs(lat2 - lat) > 1e-9:
        raise RuntimeError("UTM round trip failed")
    # Compare UTM distance with geodesic distance for a 10 km E-W segment.
    e2, n2 = to_utm.transform(lon + 0.0918, lat)
    d_utm = ((e2 - e) ** 2 + (n2 - n) ** 2) ** 0.5
    d_geo = Geod(ellps="WGS84").inv(lon, lat, lon + 0.0918, lat)[2]
    datums = {c: CRS.from_user_input(c).name for c in cfg["crs"]["historical_datum_candidates"]}
    return {"centre_utm43n_m": [round(e, 2), round(n, 2)], "utm_vs_geodesic_10km_m": round(d_utm - d_geo, 3),
            "proj_version": __import__("pyproj").proj_version_str, "historical_datums": datums}


def check_rasterio() -> dict:
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.tif"
        tf = from_origin(600000.0, 1370000.0, 5.0, 5.0)
        with rasterio.open(p, "w", driver="GTiff", width=10, height=8, count=1, dtype="uint8",
                           crs="EPSG:32643", transform=tf) as ds:
            ds.write(np.arange(80, dtype=np.uint8).reshape(1, 8, 10))
        with rasterio.open(p) as ds:
            ok = ds.crs.to_epsg() == 32643 and ds.transform == tf and ds.read(1)[7, 9] == 79
            centre_of_px00 = ds.xy(0, 0)
    if not ok:
        raise RuntimeError("GeoTIFF round trip failed")
    return {"gdal_version": rasterio.__gdal_version__, "pixel(0,0)_centre": centre_of_px00}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-loftr", action="store_true")
    args = p.parse_args(argv)

    cfg = load_config()
    report: dict = {"python": sys.version, "platform": platform.platform(),
                    "processor": platform.processor(), "packages": check_versions(), "checks": {}}
    torch_info = check_torch()
    report["torch"] = torch_info
    req = cfg["compute"]["device"]
    device = "cuda" if (req == "cuda" or (req == "auto" and torch_info["cuda_available"])) else "cpu"

    checks = [("sift_ransac", check_sift), ("pyproj", lambda: check_pyproj(cfg)), ("rasterio", check_rasterio)]
    if not args.no_loftr:
        checks.append(("loftr", lambda: check_loftr(device)))

    failed = 0
    for name, fn in checks:
        try:
            report["checks"][name] = {"ok": True, **fn()}
            log.info("PASS %-12s %s", name, report["checks"][name])
        except Exception as e:
            failed += 1
            report["checks"][name] = {"ok": False, "error": repr(e), "traceback": traceback.format_exc()}
            log.error("FAIL %-12s %r", name, e)

    out = REPO_ROOT / cfg["experiment"]["output_dir"] / "environment.json"
    write_json(report, out)
    log.info("Wrote %s  (%d failed)", out, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

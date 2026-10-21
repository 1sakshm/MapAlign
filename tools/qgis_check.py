"""Independent check of the georeferenced GeoTIFFs inside QGIS (PyQGIS, standalone).

Uses only QGIS and the Python standard library, no `mapalign` code, so the georeferencing is read and
reprojected by QGIS/GDAL/PROJ as a GIS user would see it.

Run with QGIS's Python, from the repository root, e.g.:
    D:\\QGIS\\bin\\python-qgis-ltr.bat tools\\qgis_check.py

Writes results/qgis/qgis_check.json and the rendered views results/qgis/*.png.
"""

import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))

from qgis.core import (QgsApplication, QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsFeature,  # noqa: E402
                       QgsField, QgsFillSymbol, QgsGeometry, QgsLineSymbol, QgsMapRendererParallelJob,
                       QgsMapSettings, QgsMarkerSymbol, QgsPointXY, QgsProject, QgsRasterLayer, QgsRectangle,
                       QgsVectorLayer, Qgis)
from qgis.PyQt.QtCore import QSize, Qt, QVariant  # noqa: E402
from qgis.PyQt.QtGui import QColor, QFont, QImage, QPainter, QPen  # noqa: E402

ROOT = Path.cwd()
OUT = ROOT / "results" / "qgis"
OUT.mkdir(parents=True, exist_ok=True)
UTM = QgsCoordinateReferenceSystem("EPSG:32643")
WGS = QgsCoordinateReferenceSystem("EPSG:4326")

qgs = QgsApplication([], False)
qgs.initQgis()
project = QgsProject.instance()
report = {"qgis_version": Qgis.version(), "gdal_check": {}, "layers": {}}


def raster(method):
    path = ROOT / "results" / f"georef_{method}.tif"
    lyr = QgsRasterLayer(str(path), f"georef_{method}")
    if not lyr.isValid():
        raise RuntimeError(f"QGIS could not open {path}")
    ext = lyr.extent()
    report["layers"][f"georef_{method}"] = {
        "valid": True, "crs": lyr.crs().authid(), "crs_description": lyr.crs().description(),
        "width": lyr.width(), "height": lyr.height(), "bands": lyr.bandCount(),
        "pixel_size_m": [round(lyr.rasterUnitsPerPixelX(), 6), round(lyr.rasterUnitsPerPixelY(), 6)],
        "extent_m": [round(ext.xMinimum(), 2), round(ext.yMinimum(), 2), round(ext.xMaximum(), 2), round(ext.yMaximum(), 2)],
    }
    return lyr


# ---------------------------------------------------------------- OSM vectors, reprojected by QGIS
osm = json.loads((ROOT / "data" / "raw" / "osm_57D11_overpass.json").read_text(encoding="utf-8"))
vec = QgsVectorLayer("LineString?crs=EPSG:4326&field=kind:string", "osm", "memory")
feats, counts = [], {}
for el in osm["elements"]:
    t = el.get("tags", {})
    kind = ("rail" if t.get("railway") == "rail" else "river" if t.get("waterway") == "river" else
            "road" if t.get("highway") in ("trunk", "primary", "secondary", "tertiary") else None)
    if kind and el["type"] == "way" and "geometry" in el:
        f = QgsFeature(vec.fields())
        f.setGeometry(QgsGeometry.fromPolylineXY([QgsPointXY(p["lon"], p["lat"]) for p in el["geometry"]]))
        f.setAttribute("kind", kind)
        feats.append(f)
        counts[kind] = counts.get(kind, 0) + 1
vec.dataProvider().addFeatures(feats)
report["layers"]["osm_vectors"] = {"crs": vec.crs().authid(), "features": counts, "reprojected_on_the_fly_to": "EPSG:32643"}
layers_by_kind = {}
for kind, colour, width in (("road", "#c2185b", 0.45), ("river", "#1565c0", 0.6), ("rail", "#000000", 0.7)):
    lyr = QgsVectorLayer("LineString?crs=EPSG:4326", kind, "memory")
    lyr.dataProvider().addFeatures([f for f in feats if f["kind"] == kind])
    lyr.renderer().setSymbol(QgsLineSymbol.createSimple({"color": colour, "width": str(width), "width_unit": "MM"}))
    layers_by_kind[kind] = lyr

# ---------------------------------------------------------------- evaluation points (truth + predictions)
def point_layer(name, rows, colour, shape, size):
    lyr = QgsVectorLayer("Point?crs=EPSG:32643&field=id:string", name, "memory")
    fs = []
    for pid, x, y in rows:
        f = QgsFeature(lyr.fields())
        f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(x, y)))
        f.setAttribute("id", pid)
        fs.append(f)
    lyr.dataProvider().addFeatures(fs)
    lyr.renderer().setSymbol(QgsMarkerSymbol.createSimple(
        {"name": shape, "color": colour, "outline_color": "#ffffff" if shape == "circle" else colour,
         "outline_width": "0.35", "size": str(size), "size_unit": "MM"}))
    return lyr


with open(ROOT / "gcp" / "evaluation_points.csv", newline="", encoding="utf-8") as fh:
    ev = list(csv.DictReader(fh))
truth = [(r["point_id"], float(r["true_easting_m"]), float(r["true_northing_m"])) for r in ev]
pred = {}
with open(ROOT / "results" / "evaluation_errors.csv", newline="", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        if r["model"] == "homography" and r["pred_easting_m"]:
            pred.setdefault(r["method"], []).append((r["point_id"], float(r["pred_easting_m"]), float(r["pred_northing_m"])))
truth_lyr = point_layer("truth", truth, "#000000", "circle", 2.4)


# ---------------------------------------------------------------- rendering
def render(layers, rect, size, path=None):
    ms = QgsMapSettings()
    ms.setLayers(layers)
    ms.setDestinationCrs(UTM)
    ms.setExtent(rect)
    ms.setOutputSize(QSize(*size))
    ms.setBackgroundColor(QColor("white"))
    job = QgsMapRendererParallelJob(ms)
    job.start()
    job.waitForFinished()
    img = job.renderedImage()
    if path:
        img.save(str(path))
    return img


def label(img, text, sub=None):
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    p.fillRect(0, 0, img.width(), 22 if not sub else 38, QColor(255, 255, 255, 215))
    p.setPen(QPen(QColor("black")))
    p.setFont(QFont("Arial", 10, QFont.Bold))
    p.drawText(6, 16, text)
    if sub:
        p.setFont(QFont("Arial", 8))
        p.drawText(6, 32, sub)
    # 200 m scale bar is drawn by the caller-provided pixel length
    p.end()
    return img


def scale_bar(img, metres, px):
    p = QPainter(img)
    p.setPen(QPen(QColor("black"), 4))
    x1, y = img.width() - 12, img.height() - 14
    p.drawLine(int(x1 - px), y, x1, y)
    p.setFont(QFont("Arial", 8, QFont.Bold))
    p.drawText(int(x1 - px), y - 6, f"{metres} m")
    p.end()


def grid(images, cols, path):
    w, h = images[0].width(), images[0].height()
    rows = (len(images) + cols - 1) // cols
    sheet = QImage(cols * w + (cols - 1) * 6, rows * h + (rows - 1) * 6, QImage.Format_RGB32)
    sheet.fill(QColor("white"))
    p = QPainter(sheet)
    for i, im in enumerate(images):
        p.drawImage((i % cols) * (w + 6), (i // cols) * (h + 6), im)
    p.end()
    sheet.save(str(path))


methods = [m for m in ("loftr", "manual") if (ROOT / "results" / f"georef_{m}.tif").exists()]
vectors = [layers_by_kind["rail"], layers_by_kind["river"], layers_by_kind["road"]]
for m in methods:
    r = raster(m)
    pl = point_layer(f"pred_{m}", pred.get(m, []), "#2a6fae" if m == "loftr" else "#b5352b", "cross2", 3.2)
    # 1) whole sheet
    ext = r.extent()
    img = render(vectors + [truth_lyr, r], ext, (1100, int(1100 * ext.height() / ext.width())))
    label(img, f"QGIS {Qgis.version()} | georef_{m}.tif ({r.crs().authid()}) + OSM vectors reprojected by QGIS",
          "black dots: held-out truth points")
    scale_bar(img, 5000, 5000 / ext.width() * 1100)
    img.save(str(OUT / f"qgis_{m}_sheet.png"))
    # 2) one window per evaluation point, +-350 m
    tiles = []
    half, S = 350.0, 330
    for pid, x, y in truth:
        rect = QgsRectangle(x - half, y - half, x + half, y + half)
        im = render([pl, truth_lyr] + vectors + [r], rect, (S, S))
        e = next((float(q["error_m"]) for q in csv.DictReader(open(ROOT / "results" / "evaluation_errors.csv", encoding="utf-8"))
                  if q["model"] == "homography" and q["method"] == m and q["point_id"] == pid), None)
        label(im, f"{pid}   error {e:.0f} m" if e is not None else pid)
        scale_bar(im, 200, 200 / (2 * half) * S)
        tiles.append(im)
    grid(tiles, 4, OUT / f"qgis_{m}_points.png")
    report["layers"][f"georef_{m}"]["rendered"] = [f"results/qgis/qgis_{m}_sheet.png", f"results/qgis/qgis_{m}_points.png"]

# ---------------------------------------------------------------- QGIS identify at the truth points (sanity: inside data)
for m in methods:
    lyr = QgsRasterLayer(str(ROOT / "results" / f"georef_{m}.tif"), "tmp")
    inside = 0
    for pid, x, y in truth:
        res = lyr.dataProvider().identify(QgsPointXY(x, y), Qgis.RasterIdentifyFormat.Value).results()
        if res and res.get(4, 0):  # alpha band > 0 -> inside the neatline
            inside += 1
    report["layers"][f"georef_{m}"]["truth_points_inside_alpha"] = f"{inside}/{len(truth)}"

(OUT / "qgis_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
sys.stdout.flush()
# QgsApplication.exitQgis() can crash on Windows after offscreen rendering; all outputs are already written.
os._exit(0)

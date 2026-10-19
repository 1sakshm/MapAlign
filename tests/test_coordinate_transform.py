import numpy as np
import pytest
import rasterio
from affine import Affine
from pyproj import Geod, Transformer
from rasterio.transform import from_origin

from mapalign.georeference import historical_px_to_map, map_to_ref_px, ref_px_to_map

GT = from_origin(659715.0, 1385815.0, 5.0, 5.0)


def test_pixel_centre_convention():
    # Centre of pixel (0, 0) is half a pixel inside the raster corner.
    assert np.allclose(ref_px_to_map([[0, 0]], GT), [[659717.5, 1385812.5]])
    assert np.allclose(ref_px_to_map([[-0.5, -0.5]], GT), [[659715.0, 1385815.0]])


def test_matches_rasterio_xy(tmp_path):
    p = tmp_path / "r.tif"
    with rasterio.open(p, "w", driver="GTiff", width=20, height=10, count=1, dtype="uint8",
                       crs="EPSG:32643", transform=GT) as ds:
        ds.write(np.zeros((1, 10, 20), np.uint8))
    with rasterio.open(p) as ds:
        for col, row in [(0, 0), (19, 9), (7, 3)]:
            assert np.allclose(ref_px_to_map([[col, row]], GT)[0], ds.xy(row, col))


def test_roundtrip_with_rotated_geotransform():
    gt = Affine(4.9, 0.3, 600000.0, 0.2, -5.1, 1400000.0)
    px = np.random.default_rng(0).uniform(-10, 7000, (50, 2))
    assert np.allclose(map_to_ref_px(ref_px_to_map(px, gt), gt), px, atol=1e-6)


def test_historical_chain_identity():
    px = np.array([[10.0, 20.0]])
    assert np.allclose(historical_px_to_map(px, np.eye(3), GT), ref_px_to_map(px, GT))


def test_utm_metres_vs_geodesic():
    # 1 km on the ground in the study area is ~1 km in EPSG:32643 (scale factor near 1 there).
    tf = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
    lon, lat = 76.625, 12.375
    lon2, lat2, _ = Geod(ellps="WGS84").fwd(lon, lat, 45.0, 1000.0)
    (e1, n1), (e2, n2) = tf.transform(lon, lat), tf.transform(lon2, lat2)
    assert np.hypot(e2 - e1, n2 - n1) == pytest.approx(1000.0, abs=1.0)


def test_degrees_are_not_metres():
    tf = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
    (e1, n1), (e2, n2) = tf.transform(76.5, 12.25), tf.transform(76.75, 12.25)
    assert 27000 < e2 - e1 < 27400  # 0.25 deg of longitude at 12.25 N is ~27.2 km

import numpy as np
import pytest
from PIL import Image

from mapalign.core import MATCHERS, get_matcher


def test_unknown_matcher_is_rejected():
    with pytest.raises(ValueError):
        get_matcher("nope")


def test_every_listed_matcher_resolves_without_loading_weights():
    pytest.importorskip("romatch")
    for name in MATCHERS:
        if name == "mast3r":
            continue  # needs the external clone; checked below when present
        assert callable(get_matcher(name, {}, "cpu"))


@pytest.mark.parametrize("w,h", [(1200, 800), (700, 1000), (900, 900)])
def test_mast3r_crop_maps_back_to_input_pixels(w, h):
    pytest.importorskip("torchvision")
    from mapalign.match_mast3r import _back, _prepare

    view, f = _prepare(Image.new("RGB", (w, h)), 512, 0)
    H, W = view["true_shape"][0]
    assert W % 16 == 0 and H % 16 == 0 and max(W, H) <= 512
    sx, sy, x0, y0 = f
    # input pixel -> resized -> cropped -> back again is the identity
    p = np.array([[w * 0.3, h * 0.6], [w / 2, h / 2]])
    cropped = np.stack([(p[:, 0] + 0.5) * sx - 0.5 - x0, (p[:, 1] + 0.5) * sy - 0.5 - y0], 1)
    assert np.allclose(_back(cropped, *f), p)
    # a corner of the crop maps inside the input, by the crop offset
    assert np.allclose(_back(np.array([[-0.5, -0.5]]), *f)[0], [x0 / sx - 0.5, y0 / sy - 0.5])


def test_rgb_conversion_swaps_bgr():
    from mapalign.match_roma import to_rgb_pil

    bgr = np.zeros((2, 2, 3), np.uint8)
    bgr[..., 0] = 255  # blue in OpenCV order
    assert to_rgb_pil(bgr).getpixel((0, 0)) == (0, 0, 255)
    assert to_rgb_pil(np.full((2, 2), 7, np.uint8)).getpixel((0, 0)) == (7, 7, 7)

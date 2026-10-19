import hashlib

import cv2
import numpy as np
import pytest

from mapalign.io import file_hash, load_image


def test_file_hash_matches_hashlib(tmp_path):
    p = tmp_path / "x.bin"
    payload = b"mapalign" * 100_000
    p.write_bytes(payload)
    assert file_hash(p) == hashlib.sha256(payload).hexdigest()
    assert file_hash(p, "md5") == hashlib.md5(payload).hexdigest()


def test_load_image_roundtrip_and_errors(tmp_path):
    img = np.zeros((5, 7, 3), np.uint8)
    img[2, 3] = (10, 20, 30)
    p = tmp_path / "ü.png"  # non-ASCII path
    ok, buf = cv2.imencode(".png", img)
    assert ok
    buf.tofile(p)
    out = load_image(p)
    assert out.shape == (5, 7, 3) and tuple(out[2, 3]) == (10, 20, 30)
    assert load_image(p, grayscale=True).shape == (5, 7)

    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "missing.png")
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        load_image(bad)

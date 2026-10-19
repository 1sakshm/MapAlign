"""File I/O helpers: checksums, image loading, JSON records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def file_hash(path: str | Path, algorithm: str = "sha256", chunk: int = 1 << 20) -> str:
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def load_image(path: str | Path, grayscale: bool = False) -> np.ndarray:
    """Load an image as uint8 (BGR, or single-channel if grayscale).

    Uses imdecode so that non-ASCII Windows paths work.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    data = np.fromfile(path, dtype=np.uint8)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imdecode(data, flag)
    if img is None:
        raise ValueError(f"Could not decode image: {path}")
    return img


def write_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)

"""Decode a photoshoot folder into a uniform grayscale tensor.

Every downstream feature (gradient descriptor, wavelet embedding, brightness,
masked correlation) is derived from a single 256x256 ``INTER_AREA`` grayscale
decode per image, so all decoding is centralized here.
"""
from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import numpy as np

RESOLUTION = 256
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _read_gray(path: Path) -> np.ndarray | None:
    """Robustly decode an image to a single-channel uint8 array.

    Falls back to Pillow (with truncated-image tolerance) when OpenCV cannot
    decode the file, which happens occasionally with corrupt JPEG tails.
    """
    try:
        im = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)
        if im is not None:
            return im
    except Exception:
        pass
    try:
        from PIL import Image, ImageFile

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        return np.array(Image.open(path).convert("L"))
    except Exception:
        return None


def _to_gray256(path: Path) -> np.ndarray:
    """Decode one image to a 256x256 grayscale tile (zeros if undecodable)."""
    im = _read_gray(path)
    if im is None:
        return np.zeros((RESOLUTION, RESOLUTION), np.uint8)
    return cv2.resize(im, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_AREA)


COLOR_RES = 128


def _to_color128(path: Path) -> np.ndarray:
    """Decode one image to a 128x128 BGR tile (zeros if undecodable). Used only
    by the exposure-invariant colour signature; channel order is irrelevant."""
    try:
        im = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        if im is not None:
            return cv2.resize(im, (COLOR_RES, COLOR_RES), interpolation=cv2.INTER_AREA)
    except Exception:
        pass
    return np.zeros((COLOR_RES, COLOR_RES, 3), np.uint8)


@dataclass
class Photoshoot:
    """The decoded photoshoot all features are computed from.

    Attributes:
        filenames: basenames, sorted, parallel to ``gray`` row order.
        gray: ``(N, 256, 256)`` uint8 grayscale tiles.
        brightness: ``(N,)`` per-image mean grayscale (exposure proxy).
    """

    filenames: list[str]
    gray: np.ndarray
    color: np.ndarray = None        # (N, 128, 128, 3) uint8 BGR, optional

    @property
    def count(self) -> int:
        return len(self.filenames)

    @property
    def brightness(self) -> np.ndarray:
        return self.gray.reshape(self.count, -1).mean(axis=1)


class ImageLoader:
    """Loads an image folder into a :class:`Photoshoot`."""

    def __init__(self, image_dir: str | Path):
        self.image_dir = Path(image_dir)

    def _image_paths(self) -> list[Path]:
        return sorted(
            p
            for p in self.image_dir.iterdir()
            if p.suffix.lower() in SUPPORTED_SUFFIXES
        )

    def load(self) -> Photoshoot:
        paths = self._image_paths()
        cv2.setNumThreads(1)  # we parallelize across images, not within a decode
        with Pool(max(1, cpu_count() - 1)) as pool:
            tiles = list(pool.imap(_to_gray256, paths, chunksize=16))
            ctiles = list(pool.imap(_to_color128, paths, chunksize=16))
        gray = np.asarray(tiles, np.uint8)
        color = np.asarray(ctiles, np.uint8)
        return Photoshoot([p.name for p in paths], gray, color)

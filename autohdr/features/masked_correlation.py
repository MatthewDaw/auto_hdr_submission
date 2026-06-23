"""Full-resolution masked edge correlation, the precision check for refinement.

The 64x64 gradient descriptor is fast but coarse: it can over-merge similar
layouts and under-link heavily-clipped brackets. The refinement passes settle
those borderline cases with a sharper 256x256 comparison that only correlates
pixels which are *well-exposed in both* images — clipped highlights/shadows
(value <8 or >247) carry no edge information and are excluded. Same-scene
brackets stay strongly correlated where they overlap; different rooms do not.
"""
from __future__ import annotations

import cv2
import numpy as np

_VALID_LOW, _VALID_HIGH = 8, 247   # grayscale range with usable edge detail
_MIN_OVERLAP = 200                 # co-valid pixels below this -> unreliable


class MaskedCorrelation:
    """Precomputes per-image edge magnitude + valid mask, then scores pairs."""

    def __init__(self, gray_stack: np.ndarray) -> None:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self._magnitude: list[np.ndarray] = []
        self._valid: list[np.ndarray] = []
        for gray256 in gray_stack:
            g = clahe.apply(gray256).astype(np.float32)
            gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
            self._magnitude.append(cv2.magnitude(gx, gy))
            self._valid.append((gray256 >= _VALID_LOW) & (gray256 <= _VALID_HIGH))

    def score(self, i: int, j: int) -> tuple[float, int]:
        """Masked ZNCC of edge maps i and j, plus the co-valid pixel count.

        Returns ``(-1.0, count)`` when the images overlap too little to judge.
        """
        valid = (self._valid[i] & self._valid[j]).ravel()
        count = int(valid.sum())
        if count < _MIN_OVERLAP:
            return -1.0, count
        a = self._magnitude[i].ravel()[valid]
        b = self._magnitude[j].ravel()[valid]
        a = a - a.mean()
        b = b - b.mean()
        zncc = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        return zncc, count

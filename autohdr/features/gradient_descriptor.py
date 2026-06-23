"""Gradient (edge-shape) descriptor.

A camera angle is defined by *where the edges are*, not by brightness. CLAHE
normalizes local contrast across exposures, the Sobel magnitude keeps edge
geometry while discarding absolute intensity, and the z-normalized vector makes
the dot product a zero-mean normalized cross-correlation (ZNCC) of edge maps —
high for two brackets of one angle, low for different angles.
"""
from __future__ import annotations

import cv2
import numpy as np

_SOBEL_RESOLUTION = 256  # CLAHE + Sobel are computed at this resolution
_GRID = 64               # then pooled to a 64x64 = 4096-d signature


class GradientDescriptor:
    def __init__(self) -> None:
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def _encode_one(self, gray256: np.ndarray) -> np.ndarray:
        g = self._clahe.apply(gray256).astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        z = cv2.resize(
            cv2.magnitude(gx, gy), (_GRID, _GRID), interpolation=cv2.INTER_AREA
        ).ravel()
        z = z - z.mean()
        norm = np.linalg.norm(z)
        return (z / norm if norm > 0 else z).astype(np.float32)

    def encode(self, gray_stack: np.ndarray) -> np.ndarray:
        """``(N, 256, 256)`` uint8 -> ``(N, 4096)`` L2-normalized descriptors."""
        return np.stack([self._encode_one(g) for g in gray_stack])

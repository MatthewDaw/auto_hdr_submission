"""Exposure-invariant colour signature (rg-chromaticity).

Grayscale edge structure cannot separate two *different* scenes that share a
layout (two beige rooms, a house vs. a field) — they correlate highly. Colour
can: different scenes differ in paint, wood tone, upholstery. The trick is making
colour exposure-invariant, since a bracket's frames span many exposures.

rg-chromaticity does that: ``r = R/(R+G+B)``, ``g = G/(R+G+B)`` divides out the
exposure scalar (a uniform brightness change cancels in the ratio). To stay in
the regime where the ratio is reliable we histogram only MID-tone pixels (away
from both clipping ends), and we only ever *compare* two frames at a similar
overall exposure level — comparing a frame's brightest regions against another's
darkest regions would sample different scene content, not different colour.

Used as a guarded SEPARATION signal only: a confident colour divergence between
two same-exposure-level frames means different scenes. Low divergence asserts
nothing (so it never forces a merge or breaks an HDR chain).
"""
from __future__ import annotations

import numpy as np

_BINS = 8
_MID_LOW, _MID_HIGH = 70, 185     # mid-tone pixel band (tightest exposure invariance)
_MIN_PIX = 200                    # need this many mid-tone pixels for a trustworthy hist


class ChromaSignature:
    def __init__(self, color: np.ndarray):
        # color: (N, H, W, 3) uint8. Channel order is irrelevant — only consistency
        # across frames matters for the divergence comparison.
        self._hist: list[np.ndarray | None] = [None] * (len(color) if color is not None else 0)
        if color is None:
            return
        for i, img in enumerate(color):
            self._hist[i] = self._compute(img)

    @staticmethod
    def _compute(img: np.ndarray) -> np.ndarray | None:
        x = img.astype(np.float32)
        s = x.sum(2) + 1e-6
        gray = x.mean(2)
        m = (gray > _MID_LOW) & (gray < _MID_HIGH)
        if int(m.sum()) < _MIN_PIX:
            return None
        a = (x[..., 0] / s)[m]
        b = (x[..., 1] / s)[m]
        h, _, _ = np.histogram2d(a, b, bins=_BINS, range=[[0, 1], [0, 1]])
        h = h.ravel()
        h /= (h.sum() + 1e-9)
        return h.astype(np.float32)

    def has(self, i: int) -> bool:
        return 0 <= i < len(self._hist) and self._hist[i] is not None

    def diverge(self, i: int, j: int) -> float | None:
        """Chi-square distance between two mid-tone chromaticity histograms, or
        None if either frame lacks a trustworthy mid-tone signature (or no colour
        was provided at all)."""
        if not (self.has(i) and self.has(j)):
            return None
        a, b = self._hist[i], self._hist[j]
        return float(0.5 * np.sum((a - b) ** 2 / (a + b + 1e-9)))

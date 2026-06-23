"""Shared inputs every refinement pass reads (none of them mutate it)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..features import ChromaSignature, MaskedCorrelation


@dataclass(frozen=True)
class RefinementContext:
    brightness: np.ndarray        # (N,) per-image mean grayscale
    embedding: np.ndarray         # (N, d) L2-normalized wavelet embedding
    masked: MaskedCorrelation     # full-resolution masked edge comparator
    filenames: list = None        # basenames in capture order, parallel to rows
    gray: np.ndarray = None       # (N, 256, 256) uint8 grayscale tiles
    chroma: ChromaSignature = None  # exposure-invariant colour signatures (or None)

    @property
    def count(self) -> int:
        return len(self.brightness)

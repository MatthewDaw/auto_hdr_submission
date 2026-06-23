"""Wavelet eigen-descriptor (training-free replacement for a learned CNN embed).

Per image: CLAHE -> multi-scale wavelet detail bands (db2, levels 1-3, the
approximation band is dropped because it just encodes brightness). The stack of
these descriptors is then projected onto its own principal components ("eigen-
wavelets"), fit per photoshoot in ~0.3s — so it adapts to each shoot and needs
no bundled model, no PyTorch/ONNX, and is portable to a browser.

The PCA uses ``svd_solver="full"`` for determinism: scikit-learn's size-triggered
'randomized' default is unseeded, which would make the embedding — and downstream
borderline merges — vary run to run. 'full' is exact and reproducible.
"""
from __future__ import annotations

import cv2
import numpy as np
import pywt
from sklearn.decomposition import PCA


class WaveletEmbedding:
    def __init__(self, dim: int = 128, drop_first: int = 1) -> None:
        # drop_first removes the leading principal component, which tends to
        # capture a global nuisance axis (residual exposure/contrast) rather
        # than scene identity.
        self.dim = dim
        self.drop_first = drop_first
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def _wavelet_features(self, gray256: np.ndarray) -> np.ndarray:
        g = self._clahe.apply(cv2.resize(gray256, (128, 128))).astype(np.float32)
        coeffs = pywt.wavedec2(g, "db2", level=3)
        bands = [
            cv2.resize(np.abs(band), (16, 16)).ravel()
            for level in (1, 2, 3)
            for band in coeffs[level]  # (cH, cV, cD) per level; skip cA approximation
        ]
        return np.concatenate(bands).astype(np.float32)

    def encode(self, gray_stack: np.ndarray) -> np.ndarray:
        """``(N, 256, 256)`` uint8 -> ``(N, dim - drop_first)`` L2-normalized embeddings.

        PCA is fit on exactly these N images (per-run, training-free).
        """
        feats = np.stack([self._wavelet_features(g) for g in gray_stack])
        k = min(self.dim, feats.shape[0], feats.shape[1])
        projected = PCA(
            n_components=k, whiten=True, svd_solver="full"
        ).fit_transform(feats)
        if self.drop_first and projected.shape[1] > self.drop_first:
            projected = projected[:, self.drop_first:]
        norms = np.linalg.norm(projected, axis=1, keepdims=True) + 1e-9
        return (projected / norms).astype(np.float32)

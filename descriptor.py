"""
Training-free per-image descriptor: CLAHE -> multi-scale wavelet detail bands
-> PCA ("eigen-wavelets"). Replaces the learned CNN embedding. The PCA basis is
fit per-run on the run's own images (~0.3s), so it adapts to each photoshoot and
needs no training, no bundled model, no PyTorch/ONNX. Browser-portable (CLAHE +
Haar/db2 wavelet + a single matrix multiply).

Benchmarks vs the old CNN embedding (see bench_large.py / bench_transforms.py):
  extract ~0.1-0.8 ms/img CPU (CNN ~3.2 ms batched, far slower in-browser WASM);
  large-set base fusion 0.987 (CNN 0.977); full pipeline 0.991 (CNN 0.992).
"""
import numpy as np, cv2, pywt
from sklearn.decomposition import PCA

_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

def wavelet_features(gray256):
    """One grayscale image (any size) -> multi-scale wavelet detail descriptor."""
    g = _clahe.apply(cv2.resize(gray256, (128, 128))).astype(np.float32)
    co = pywt.wavedec2(g, "db2", level=3)
    parts = [cv2.resize(np.abs(b), (16, 16)).ravel()        # |detail| bands, levels 1-3
             for lvl in (1, 2, 3) for b in co[lvl]]          # skip approximation cA (= brightness)
    return np.concatenate(parts).astype(np.float32)

def embed(gray_imgs, dim=128, drop_first=1):
    """
    gray_imgs: (N, H, W) uint8 grayscale array (e.g. the raw256 cache).
    Returns (N, dim-drop_first) L2-normalized eigen-wavelet embeddings.
    PCA is fit on these N images (per-run, training-free).
    """
    W = np.stack([wavelet_features(im) for im in gray_imgs])
    k = min(dim, W.shape[0], W.shape[1])
    P = PCA(n_components=k, whiten=True).fit_transform(W)
    if drop_first and P.shape[1] > drop_first:
        P = P[:, drop_first:]                                # drop leading (residual-lighting) component
    return (P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)).astype(np.float32)

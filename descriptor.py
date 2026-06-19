"""
Training-free per-image descriptor: CLAHE -> multi-scale wavelet detail bands
-> PCA ("eigen-wavelets"). Replaces the learned CNN embedding. The PCA basis is
fit per-run on the run's own images (~0.3s), so it adapts to each photoshoot and
needs no training, no bundled model, no PyTorch/ONNX. Browser-portable (CLAHE +
Haar/db2 wavelet + a single matrix multiply).

Benchmarks vs the old CNN embedding (see bench_large.py / bench_transforms.py):
  extract ~0.1-0.8 ms/img CPU (CNN ~3.2 ms batched, far slower in-browser WASM);
  large-set base fusion 0.987 (CNN 0.977); full pipeline 0.9931 (CNN 0.992).
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

def color_features(bgr):
    """Exposure-stable chroma signature: Lab a,b channels on a coarse grid."""
    lab = cv2.cvtColor(cv2.resize(bgr, (128, 128)), cv2.COLOR_BGR2Lab).astype(np.float32)
    a = cv2.resize(lab[:, :, 1], (10, 10)).ravel() - 128
    b = cv2.resize(lab[:, :, 2], (10, 10)).ravel() - 128
    return np.concatenate([a, b]).astype(np.float32)

def _pca(W, dim, drop_first):
    k = min(dim, W.shape[0], W.shape[1])
    # svd_solver='full' for DETERMINISM: the size-triggered 'randomized' default is
    # unseeded, making the embedding — and downstream borderline merges — vary run to
    # run. 'full' is exact and reproducible (a few extra seconds on the one-time batch
    # fit; per-image online extract is unaffected).
    P = PCA(n_components=k, whiten=True, svd_solver="full").fit_transform(W)
    if drop_first and P.shape[1] > drop_first:
        P = P[:, drop_first:]
    return (P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)).astype(np.float32)

def embed(gray_imgs, dim=128, drop_first=1):
    """
    gray_imgs: (N, H, W) uint8 grayscale array (e.g. the raw256 cache).
    Returns (N, dim-drop_first) L2-normalized eigen-wavelet embeddings.
    PCA is fit on these N images (per-run, training-free).
    """
    return _pca(np.stack([wavelet_features(im) for im in gray_imgs]), dim, drop_first)

def embed_cw(gray_imgs, bgr_imgs, dim=128, drop_first=1, color_w=4.0):
    """Wavelet (structure) + chroma (color identity) -> PCA. Both per-image blocks
    are z-normalized so PCA isn't dominated by one; chroma is upweighted by color_w."""
    Wv = np.stack([wavelet_features(im) for im in gray_imgs])
    Cl = np.stack([color_features(im) for im in bgr_imgs])
    Wv = (Wv - Wv.mean(0)) / (Wv.std(0) + 1e-6)
    Cl = (Cl - Cl.mean(0)) / (Cl.std(0) + 1e-6) * color_w
    return _pca(np.concatenate([Wv, Cl], axis=1), dim, drop_first)

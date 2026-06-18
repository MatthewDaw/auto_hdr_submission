"""
Spike 4: CLAHE preprocessing to fix over-splitting of high-dynamic-range groups.

The binding error is extreme-exposure frames (blown/crushed) losing gradient
signal so connected-components can't chain them. CLAHE (local adaptive contrast)
restores edges in shadows/highlights before Sobel. Tests global gradient-ZNCC
with and without CLAHE.
"""
import csv, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import cv2
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

SAMPLE = Path("sample"); IMG_DIR = SAMPLE / "images"
SIZE = 256; ZNCC = 64
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

def load_manifest():
    groups = defaultdict(set); f2g = {}
    for r in csv.DictReader(open(SAMPLE / "public_manifest.csv")):
        groups[r["group_id"]].add(r["filename"]); f2g[r["filename"]] = r["group_id"]
    return groups, f2g

def gradmap(path, use_clahe):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    g = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    if use_clahe:
        g = clahe.apply(g)
    g = g.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    z = cv2.resize(mag, (ZNCC, ZNCC), interpolation=cv2.INTER_AREA).ravel()
    z = z - z.mean(); z /= (np.linalg.norm(z) + 1e-9)
    return z

def score(pred_groups, ref_groups):
    ref = set(frozenset(v) for v in ref_groups.values())
    pred = set(frozenset(v) for v in pred_groups)
    return len(ref & pred)/len(ref), len(ref & pred), len(pred)

def run(files, groups, f2g, use_clahe, label):
    Z = np.array([gradmap(IMG_DIR / f, use_clahe) for f in files])
    sim = Z @ Z.T
    grp = np.array([f2g[f] for f in files]); same = grp[:, None] == grp[None, :]
    iu = np.triu_indices(len(files), k=1)
    pos, neg = sim[iu][same[iu]], sim[iu][~same[iu]]
    best = (0, 0, 0)
    for thr in np.arange(0.30, 0.90, 0.01):
        A = sim >= thr; np.fill_diagonal(A, False)
        _, lab = connected_components(csr_matrix(A), directed=False)
        pred = defaultdict(set)
        for i, f in enumerate(files): pred[lab[i]].add(f)
        sc, ex, npred = score(pred.values(), groups)
        if sc > best[1]: best = (round(thr, 3), round(sc, 4), npred)
    print(f"=== {label} ===")
    print(f"  SAME p10={np.percentile(pos,10):.3f} p25={np.percentile(pos,25):.3f} p50={np.percentile(pos,50):.3f}")
    print(f"  DIFF p99={np.percentile(neg,99):.3f} max={neg.max():.3f}")
    print(f"  best {best[1]} at thr={best[0]} (predicted {best[2]} vs 69 ref)")

def main():
    t0 = time.time()
    groups, f2g = load_manifest(); files = sorted(f2g.keys())
    run(files, groups, f2g, False, "no CLAHE (baseline)")
    run(files, groups, f2g, True,  "with CLAHE")
    print(f"total {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()

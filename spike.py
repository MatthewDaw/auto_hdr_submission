"""
Spike: measure exposure-invariant same-angle separation on the 500-sample.

Tests two candidate signals on all image pairs:
  - HOG-lite cosine: global gradient-orientation descriptor (blocking candidate)
  - gradient-ZNCC: zero-mean NCC on gradient-magnitude maps (verification candidate)

Then sweeps a merge threshold, builds groups via connected components,
and scores with the REAL exact-set scorer to find achievable score.
"""
import csv, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import cv2
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

SAMPLE = Path("sample")
IMG_DIR = SAMPLE / "images"
SIZE = 256          # working grayscale size
ZNCC_SIZE = 64      # gradient-map size for ZNCC

def load_manifest():
    groups = defaultdict(set)
    fname2grp = {}
    for r in csv.DictReader(open(SAMPLE / "public_manifest.csv")):
        groups[r["group_id"]].add(r["filename"])
        fname2grp[r["filename"]] = r["group_id"]
    return groups, fname2grp

def descriptors(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    g = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    ori = (np.rad2deg(np.arctan2(gy, gx)) % 180.0)  # unsigned orientation

    # HOG-lite: 8x8 cells, 9 orientation bins, magnitude-weighted, L2-normalized per cell
    cells, bins = 8, 9
    cs = SIZE // cells
    hog = np.zeros((cells, cells, bins), np.float32)
    binidx = np.minimum((ori / (180.0 / bins)).astype(int), bins - 1)
    for i in range(cells):
        for j in range(cells):
            b = binidx[i*cs:(i+1)*cs, j*cs:(j+1)*cs].ravel()
            w = mag[i*cs:(i+1)*cs, j*cs:(j+1)*cs].ravel()
            h = np.bincount(b, weights=w, minlength=bins)
            n = np.linalg.norm(h)
            hog[i, j] = h / n if n > 0 else h
    hog = hog.ravel()
    hog /= (np.linalg.norm(hog) + 1e-9)

    # ZNCC map: gradient magnitude downsized, zero-mean unit-norm
    zm = cv2.resize(mag, (ZNCC_SIZE, ZNCC_SIZE), interpolation=cv2.INTER_AREA).ravel()
    zm = zm - zm.mean()
    zm /= (np.linalg.norm(zm) + 1e-9)
    return hog, zm

def score_groups(pred_groups, ref_groups):
    ref = set(frozenset(v) for v in ref_groups.values())
    pred = set(frozenset(v) for v in pred_groups)
    return len(ref & pred) / len(ref)

def main():
    t0 = time.time()
    groups, fname2grp = load_manifest()
    files = sorted(fname2grp.keys())
    n = len(files)
    print(f"{n} images, {len(groups)} groups")

    HOG, ZM = [], []
    for k, f in enumerate(files):
        h, z = descriptors(IMG_DIR / f)
        HOG.append(h); ZM.append(z)
        if (k+1) % 100 == 0:
            print(f"  embedded {k+1}/{n}  ({time.time()-t0:.1f}s)")
    HOG = np.array(HOG); ZM = np.array(ZM)
    print(f"embedding done in {time.time()-t0:.1f}s  ({(time.time()-t0)/n*1000:.1f} ms/img)")

    # all-pairs similarity
    hog_sim = HOG @ HOG.T
    zncc = ZM @ ZM.T

    grp = np.array([fname2grp[f] for f in files])
    same = grp[:, None] == grp[None, :]
    iu = np.triu_indices(n, k=1)
    same_pairs = same[iu]

    for name, sim in [("HOG-cosine", hog_sim), ("gradient-ZNCC", zncc)]:
        s = sim[iu]
        pos, neg = s[same_pairs], s[~same_pairs]
        print(f"\n=== {name} ===")
        print(f"  SAME-group pairs ({pos.size}):  mean={pos.mean():.3f}  p10={np.percentile(pos,10):.3f}  p50={np.percentile(pos,50):.3f}")
        print(f"  DIFF-group pairs ({neg.size}): mean={neg.mean():.3f}  p90={np.percentile(neg,90):.3f}  p99={np.percentile(neg,99):.3f}  max={neg.max():.3f}")

        # threshold sweep -> connected components -> real score
        best = (0, 0)
        for thr in np.arange(0.50, 0.996, 0.01):
            A = (sim >= thr)
            np.fill_diagonal(A, False)
            ncomp, lab = connected_components(csr_matrix(A), directed=False)
            pred = defaultdict(set)
            for i, f in enumerate(files):
                pred[lab[i]].add(f)
            sc = score_groups(pred.values(), groups)
            if sc > best[1]:
                best = (round(thr, 3), round(sc, 4))
        print(f"  best exact-set score (connected-components sweep): {best[1]} at thr={best[0]}")

    print(f"\ntotal {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()

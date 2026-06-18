"""
Label-free per-run threshold selector.

Hypothesis: the best exact-set threshold sits at the PLATEAU of the
predicted-group-count vs threshold curve (the region where the clustering is
most stable). Detect it as the flattest local slope, no labels used.

Validates against the oracle (label-based best) on whatever dataset dir is given.
Run on both data/large and sample to check it picks ~0.62 and ~0.44 respectively.
"""
import csv, sys, time
from collections import defaultdict
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
import cv2
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "data/large")
IMG = DATA / "images"; CACHE = DATA / "sel_cache.npz"
SIZE = 256; ZNCC = 64

def read_gray(p):
    try:
        im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_GRAYSCALE)
        if im is not None: return im
    except Exception: pass
    try:
        from PIL import Image, ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        return np.array(Image.open(p).convert("L"))
    except Exception: return None

def desc(fname):
    im = read_gray(IMG / fname)
    if im is None: return np.zeros(ZNCC*ZNCC, np.float32)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g = clahe.apply(cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_AREA)).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    z = cv2.resize(cv2.magnitude(gx, gy), (ZNCC, ZNCC), interpolation=cv2.INTER_AREA).ravel()
    z = z - z.mean(); n = np.linalg.norm(z)
    return (z/n if n > 0 else z).astype(np.float32)

def build(files):
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        if list(d["files"]) == files: return d["M"]
    cv2.setNumThreads(1)
    with Pool(max(1, cpu_count()-1)) as pool:
        M = np.array(list(pool.imap(desc, files, chunksize=16)), np.float32)
    np.savez(CACHE, M=M, files=np.array(files)); return M

def main():
    groups = defaultdict(set)
    for r in csv.DictReader(open(DATA / "public_manifest.csv")):
        groups[r["group_id"]].add(r["filename"])
    files = sorted(f for v in groups.values() for f in v)
    M = build(files); n = len(files)
    sim = M @ M.T
    refsets = set(frozenset(v) for v in groups.values())

    grid = np.arange(0.35, 0.75, 0.01)
    counts, scores = [], []
    for t in grid:
        A = sim >= t; np.fill_diagonal(A, False)
        _, lab = connected_components(csr_matrix(A), directed=False)
        pred = defaultdict(set)
        for i, f in enumerate(files): pred[lab[i]].add(f)
        counts.append(len(pred))
        predlk = set(frozenset(v) for v in pred.values())
        scores.append(len(refsets & predlk)/len(refsets))
    counts = np.array(counts, float); scores = np.array(scores)

    # selector: flattest local slope of the count curve (window of +/- W steps)
    W = 3
    slope = np.full(len(grid), np.inf)
    for i in range(W, len(grid)-W):
        slope[i] = (counts[i+W] - counts[i-W]) / (2*W)
    # restrict to region where count is rising (avoid the merged low-thr floor)
    valid = counts > 0.5 * counts.max()
    slope[~valid] = np.inf
    # leading edge of the low-slope plateau: lowest valid thr whose slope is
    # within 1.3x (+0.5 abs) of the minimum slope -- the knee, not the flattest point
    fin = slope[np.isfinite(slope)]
    smin = fin.min()
    cut = 1.3 * smin + 0.5
    cand = np.where(np.isfinite(slope) & (slope <= cut))[0]
    sel_i = int(cand[0])
    t_sel = grid[sel_i]

    t_oracle = grid[int(np.argmax(scores))]
    print(f"{DATA}: n={n}, true_groups={len(refsets)}")
    print(f"  ORACLE best : score={scores.max():.4f} @ thr={t_oracle:.2f}")
    print(f"  SELECTOR    : picks thr={t_sel:.2f}  -> score={scores[int(np.argmin(slope))]:.4f}")
    print(f"  gap to oracle: {scores.max()-scores[int(np.argmin(slope))]:.4f}")
    # show curve around the selection
    print("  thr   count  slope  score")
    for i in range(0, len(grid), 2):
        s = f"{slope[i]:.1f}" if np.isfinite(slope[i]) else "  -"
        mark = " <sel" if abs(grid[i]-t_sel)<1e-6 else (" <oracle" if abs(grid[i]-t_oracle)<1e-6 else "")
        print(f"  {grid[i]:.2f}  {int(counts[i]):5d}  {s:>5}  {scores[i]:.3f}{mark}")

if __name__ == "__main__":
    main()

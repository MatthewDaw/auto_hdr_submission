"""
Spike 6: better exposure-invariance tooling to reduce over-splitting.

Compares descriptors on their ability to RAISE adjacent-exposure correlation
(so the connected-components chain stays unbroken through HDR ladders) WITHOUT
raising DIFF-group correlation (margin is thin). Reports for each variant:
  - best exact-set score + predicted group count
  - DIFF-group max  (merge-margin headroom)
  - intra-group min sim for the 3 known failing groups (chain weak links)

Variants:
  A baseline      : CLAHE -> Sobel|magnitude| -> ZNCC
  B log-grad      : Sobel|magnitude| on log(1+I) -> ZNCC   (exposure factor cancels)
  C CLAHE+log-grad: CLAHE -> log -> Sobel mag -> ZNCC
  D census        : 8-neighbor census transform -> Hamming similarity (ordinal-invariant)
"""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import cv2
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

SAMPLE = Path("sample"); IMG_DIR = SAMPLE / "images"
SIZE = 256; ZNCC = 64; CEN = 96
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
FAIL = ['22994', '40615', '40599']

def load_manifest():
    groups = defaultdict(set); f2g = {}
    for r in csv.DictReader(open(SAMPLE / "public_manifest.csv")):
        groups[r["group_id"]].add(r["filename"]); f2g[r["filename"]] = r["group_id"]
    return groups, f2g

def gray(path, size):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

def znorm(v):
    v = v.astype(np.float32).ravel(); v = v - v.mean(); n = np.linalg.norm(v)
    return v / n if n > 0 else v

def desc(path, mode):
    if mode == "census":
        g = gray(path, CEN).astype(np.int16)
        code = np.zeros((CEN, CEN), np.uint8); bit = 0
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0: continue
                sh = np.roll(np.roll(g, di, 0), dj, 1)
                code |= ((g > sh).astype(np.uint8) << bit); bit += 1
        return code.ravel()  # uint8 codes, compared by Hamming
    g = gray(path, SIZE).astype(np.float32)
    if mode == "baseline":
        g = clahe.apply(g.astype(np.uint8)).astype(np.float32)
    elif mode == "log":
        g = np.log1p(g)
    elif mode == "clahe_log":
        g = np.log1p(clahe.apply(g.astype(np.uint8)).astype(np.float32))
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return znorm(cv2.resize(mag, (ZNCC, ZNCC), interpolation=cv2.INTER_AREA))

def census_sim(C):
    # C: (n, P) uint8 census codes ; similarity = fraction of matching bits (8 bits/pixel)
    n, P = C.shape; S = np.zeros((n, n), np.float32)
    for i in range(n):
        x = np.bitwise_xor(C[i][None, :], C)               # (n, P)
        pc = np.unpackbits(x, axis=1).sum(1)               # set bits per row
        S[i] = 1.0 - pc / (P * 8.0)
    return S

def score(pred_groups, ref_groups):
    ref = set(frozenset(v) for v in ref_groups.values())
    pred = set(frozenset(v) for v in pred_groups)
    return len(ref & pred)/len(ref), len(pred)

def evaluate(name, sim, files, groups, f2g, lo, hi, step):
    grp = np.array([f2g[f] for f in files]); same = grp[:, None] == grp[None, :]
    iu = np.triu_indices(len(files), k=1)
    diffmax = sim[iu][~same[iu]].max()
    idxof = {f: i for i, f in enumerate(files)}
    fmin = {}
    for gid in FAIL:
        ix = [idxof[f] for f in groups[gid]]
        sub = sim[np.ix_(ix, ix)]; u = np.triu_indices(len(ix), 1)
        fmin[gid] = sub[u].min()
    best = (0, 0, 0)
    for thr in np.arange(lo, hi, step):
        A = sim >= thr; np.fill_diagonal(A, False)
        _, lab = connected_components(csr_matrix(A), directed=False)
        pred = defaultdict(set)
        for i, f in enumerate(files): pred[lab[i]].add(f)
        sc, npred = score(pred.values(), groups)
        if sc > best[0]: best = (round(sc, 4), round(thr, 3), npred)
    print(f"=== {name} ===")
    print(f"  best {best[0]} @thr={best[1]} (pred {best[2]} vs 69)  DIFFmax={diffmax:.3f}")
    print(f"  failing-group intra-min: " + "  ".join(f"{g}={fmin[g]:.3f}" for g in FAIL))

def main():
    groups, f2g = load_manifest(); files = sorted(f2g.keys())
    for mode, name, lo, hi, step in [
        ("baseline", "A CLAHE+mag", 0.30, 0.80, 0.01),
        ("log", "B log-grad", 0.30, 0.90, 0.01),
        ("clahe_log", "C CLAHE+log-grad", 0.30, 0.85, 0.01),
    ]:
        D = np.array([desc(IMG_DIR / f, mode) for f in files])
        evaluate(name, D @ D.T, files, groups, f2g, lo, hi, step)
    # census (separate similarity)
    C = np.array([desc(IMG_DIR / f, "census") for f in files], dtype=np.uint8)
    evaluate("D census", census_sim(C), files, groups, f2g, 0.55, 0.95, 0.01)

if __name__ == "__main__":
    main()

"""
Spike 7: combine high-precision gradient-ZNCC with gated census recall.

Edge(i,j) exists if:
   gradient-ZNCC(i,j) >= TG                      (precise: trusts edge structure)
   OR ( census(i,j) >= TC AND |bright_i-bright_j| >= DB )   (census ONLY bridges
        frames with a large exposure gap = the clipped-tail case where gradient
        fails; the brightness gate blocks census's false merges between
        similarly-exposed different rooms )

One decode per image computes gradient map, census code, and mean brightness.
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

def load_manifest():
    groups = defaultdict(set); f2g = {}
    for r in csv.DictReader(open(SAMPLE / "public_manifest.csv")):
        groups[r["group_id"]].add(r["filename"]); f2g[r["filename"]] = r["group_id"]
    return groups, f2g

def features(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    bright = float(img.mean())
    g = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    gc = clahe.apply(g).astype(np.float32)
    gx = cv2.Sobel(gc, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(gc, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.resize(cv2.magnitude(gx, gy), (ZNCC, ZNCC), interpolation=cv2.INTER_AREA).ravel()
    mag = mag - mag.mean(); n = np.linalg.norm(mag); mag = mag / n if n > 0 else mag
    # census on CLAHE'd image (helps clipped frames have ordering)
    c = cv2.resize(g, (CEN, CEN), interpolation=cv2.INTER_AREA).astype(np.int16)
    code = np.zeros((CEN, CEN), np.uint8); bit = 0
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0: continue
            sh = np.roll(np.roll(c, di, 0), dj, 1)
            code |= ((c > sh).astype(np.uint8) << bit); bit += 1
    return mag.astype(np.float32), code.ravel(), bright

def census_sim(C):
    n, P = C.shape; S = np.zeros((n, n), np.float32)
    for i in range(n):
        x = np.bitwise_xor(C[i][None, :], C)
        S[i] = 1.0 - np.unpackbits(x, axis=1).sum(1) / (P * 8.0)
    return S

def score(pred_groups, ref_groups):
    ref = set(frozenset(v) for v in ref_groups.values())
    pred = set(frozenset(v) for v in pred_groups)
    return len(ref & pred)/len(ref), len(pred)

def cluster_score(A, files, groups):
    np.fill_diagonal(A, False)
    _, lab = connected_components(csr_matrix(A), directed=False)
    pred = defaultdict(set)
    for i, f in enumerate(files): pred[lab[i]].add(f)
    return score(pred.values(), groups)

def main():
    groups, f2g = load_manifest(); files = sorted(f2g.keys())
    M, C, B = [], [], []
    for f in files:
        m, c, b = features(IMG_DIR / f); M.append(m); C.append(c); B.append(b)
    M = np.array(M); C = np.array(C, np.uint8); B = np.array(B)
    gradE = M @ M.T
    cenS = census_sim(C)
    dB = np.abs(B[:, None] - B[None, :])
    n = len(files)

    # baseline gradient-only
    best = (0, None)
    for tg in np.arange(0.38, 0.55, 0.01):
        sc, npred = cluster_score(gradE >= tg, files, groups)
        if sc > best[0]: best = (round(sc,4), (round(tg,3), npred))
    print(f"gradient-only baseline: {best[0]} @tg={best[1][0]} (pred {best[1][1]})")

    # gated combo sweep
    top = (0, None)
    for tg in [0.42, 0.44, 0.46]:
        for tc in np.arange(0.78, 0.94, 0.02):
            for db in [20, 30, 40, 50, 60]:
                A = (gradE >= tg) | ((cenS >= tc) & (dB >= db))
                sc, npred = cluster_score(A.copy(), files, groups)
                if sc > top[0]:
                    top = (round(sc,4), dict(tg=tg, tc=round(tc,2), db=db, pred=npred))
    print(f"gated combo BEST: {top[0]}  params={top[1]}")

if __name__ == "__main__":
    main()

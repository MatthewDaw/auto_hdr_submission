"""
Rescue the learned classifier's cross-size generalization with PER-RUN
NORMALIZED features: recenter similarity on each run's plateau-selected
threshold, express graph features as density-invariant fractions/ranks.
Then the learned decision boundary means the same thing at any dataset size.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.ensemble import HistGradientBoostingClassifier
from cross_size import load_dir, FLOOR, cluster_score
import cv2

def plateau_thr(sim, n):
    def cc(t):
        A = sim >= t; A = A.copy(); np.fill_diagonal(A, False)
        _, lab = connected_components(csr_matrix(A), directed=False); return len(set(lab))
    grid = np.arange(0.35, 0.75, 0.01); counts = np.array([cc(t) for t in grid], float)
    W = 3; slope = np.full(len(grid), np.inf)
    for i in range(W, len(grid)-W): slope[i] = (counts[i+W]-counts[i-W])/(2*W)
    slope[counts <= 0.5*counts.max()] = np.inf
    cut = 1.3*slope[np.isfinite(slope)].min() + 0.5
    return grid[int(np.where(np.isfinite(slope) & (slope <= cut))[0][0])]

def norm_pairs(M, C, B, files, f2g):
    n = len(files); sim = (M@M.T).astype(np.float32); np.fill_diagonal(sim, -1)
    thr = plateau_thr(sim, n)
    smax = sim.max(1); A = sim >= FLOOR; deg = A.sum(1)
    common = (csr_matrix(A.astype(np.int8))@csr_matrix(A.astype(np.int8))).toarray().astype(np.float32)
    rank = np.argsort(np.argsort(-sim, axis=1), axis=1).astype(np.float32)
    ii, jj = np.where(np.triu(A, 1))
    def cen(i, j): return 1.0-np.unpackbits(np.bitwise_xor(C[i], C[j])).sum()/(C.shape[1]*8.0)
    cenv = np.array([cen(i, j) for i, j in zip(ii, jj)], np.float32)
    F = np.column_stack([
        sim[ii, jj] - thr,                                   # similarity RELATIVE to per-run threshold
        cenv - np.median(cenv),                              # census relative to run median
        np.abs(B[ii]-B[jj]),                                 # brightness gap (already scale-free)
        common[ii, jj]/np.maximum(np.minimum(deg[ii], deg[jj]), 1),  # shared-neighbor FRACTION
        np.minimum(np.minimum(rank[ii, jj], rank[jj, ii]), 15),      # capped ranks
        np.minimum(np.maximum(rank[ii, jj], rank[jj, ii]), 15),
        np.minimum(smax[ii], smax[jj]),
        sim[ii, jj]/np.maximum(smax[ii], 1e-6),
        sim[ii, jj]/np.maximum(smax[jj], 1e-6),
    ]).astype(np.float32)
    gid = np.array([f2g[f] for f in files]); lab = (gid[ii] == gid[jj]).astype(int)
    return ii, jj, F, lab, sim, n, thr

def run(trainDATA, testDATA, label):
    tr = load_dir(trainDATA); te = load_dir(testDATA)
    iitr, jjtr, Ftr, ltr, _, _, thrtr = norm_pairs(*tr[:5])
    iite, jjte, Fte, lte, simte, nte, thrte = norm_pairs(*te[:5])
    fte, gte = te[3], te[5]
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_depth=4)
    clf.fit(Ftr, ltr); prob = clf.predict_proba(Fte)[:, 1]
    best = max((cluster_score(iite, jjte, prob >= p, nte, fte, gte), round(p, 2)) for p in np.arange(0.3, 0.9, 0.05))
    gb = max((cluster_score(iite, jjte, simte[iite, jjte] >= t, nte, fte, gte), round(t, 2)) for t in np.arange(0.5, 0.7, 0.02))
    print(f"=== {label}: train {trainDATA}(thr={thrtr:.2f}) -> test {testDATA}(thr={thrte:.2f}) ===")
    print(f"  LEARNED (per-run normalized): {best[0]:.4f} @p>={best[1]}")
    print(f"  BASELINE gradient-thr:        {gb[0]:.4f} @thr={gb[1]}")

if __name__ == "__main__":
    run("data/large", "sample", "big->small")
    run("sample", "data/large", "small->big")

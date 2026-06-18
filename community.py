"""
Replace connected-components with Leiden community detection to resist weak
between-group bridges while keeping HDR chains intact. Tests whether a single
resolution parameter generalizes across dataset sizes (the thing the learned
classifier failed at).
"""
import sys
from collections import defaultdict
import numpy as np
import igraph as ig
import leidenalg as la
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from cross_size import load_dir, FLOOR

def score(lab, files, groups):
    pred = defaultdict(set)
    for k, f in enumerate(files): pred[lab[k]].add(f)
    refsets = set(frozenset(v) for v in groups.values())
    predlk = set(frozenset(v) for v in pred.values())
    return len(refsets & predlk)/len(refsets), len(pred)

def cc(sim, t, n):
    A = sim >= t; np.fill_diagonal(A, False)
    _, lab = connected_components(csr_matrix(A), directed=False); return lab

def leiden(sim, t, n, res):
    A = sim >= t; np.fill_diagonal(A, False)
    ii, jj = np.where(np.triu(A, 1))
    g = ig.Graph(n=n, edges=list(zip(ii.tolist(), jj.tolist())))
    w = (sim[ii, jj]).astype(float).tolist()
    part = la.find_partition(g, la.RBConfigurationVertexPartition, weights=w,
                             resolution_parameter=res, n_iterations=-1, seed=0)
    return np.array(part.membership)

def run(DATA):
    M, C, B, files, f2g, groups = load_dir(DATA)
    n = len(files); sim = (M @ M.T).astype(np.float32); np.fill_diagonal(sim, -1)
    # CC baseline (best over threshold)
    ccbest = max((score(cc(sim, t, n), files, groups)[0], round(t, 2)) for t in np.arange(0.50, 0.68, 0.02))
    print(f"\n{DATA}: n={n}, groups={len(groups)}")
    print(f"  CC baseline best: {ccbest[0]:.4f} @thr={ccbest[1]}")
    # Leiden: sweep edge-floor threshold and resolution
    best = (0, None)
    for t in [0.45, 0.50, 0.55]:
        for res in [0.05, 0.1, 0.2, 0.4, 0.7, 1.0]:
            lab = leiden(sim, t, n, res); sc, npred = score(lab, files, groups)
            if sc > best[0]: best = (sc, dict(floor=t, res=res, pred=npred))
    print(f"  Leiden best: {best[0]:.4f}  {best[1]}")
    return best

if __name__ == "__main__":
    run("data/large")
    run("sample")

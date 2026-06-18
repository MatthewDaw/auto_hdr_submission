"""
Does the learned embedding ADD value on top of gradient-ZNCC?
Test on the fully-held-out 500-set: gradient sim, embedding sim, and fusions.
"""
import sys
from collections import defaultdict
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from train_embed import Model, embed_all, load, DEV

def plateau_thr(sim, lo=0.30, hi=0.97):
    grid = np.arange(lo, hi, 0.01); counts = []
    for t in grid:
        A = sim >= t; np.fill_diagonal(A, False)
        _, lab = connected_components(csr_matrix(A), directed=False); counts.append(len(set(lab)))
    counts = np.array(counts, float); W = 3; slope = np.full(len(grid), np.inf)
    for i in range(W, len(grid)-W): slope[i] = (counts[i+W]-counts[i-W])/(2*W)
    slope[counts <= 0.5*counts.max()] = np.inf
    cut = 1.3*slope[np.isfinite(slope)].min() + 0.5
    return grid[int(np.where(np.isfinite(slope) & (slope <= cut))[0][0])]

def score_sim(sim, gid, files):
    # ORACLE threshold (best over sweep) -- isolates signal quality from selector
    sim = sim.copy(); np.fill_diagonal(sim, -1)
    groups = defaultdict(set)
    for f, g in zip(files, gid): groups[g].add(f)
    refsets = set(frozenset(v) for v in groups.values())
    best = (0, 0)
    for thr in np.arange(0.20, 0.97, 0.01):
        A = sim >= thr; np.fill_diagonal(A, False)
        _, lab = connected_components(csr_matrix(A), directed=False)
        pred = defaultdict(set)
        for i, f in enumerate(files): pred[lab[i]].add(f)
        sc = len(refsets & set(frozenset(v) for v in pred.values()))/len(refsets)
        if sc > best[0]: best = (sc, round(thr, 2))
    return best

def main():
    imgs, gid, files = load("sample/img128.npz")
    d = np.load("sample/feat_cache.npz", allow_pickle=True)
    assert list(d["files"]) == files, "order mismatch"
    G = (d["M"] @ d["M"].T).astype(np.float32)          # gradient-ZNCC
    model = Model().to(DEV); model.load_state_dict(torch.load("embed_best.pt"))
    E = (embed_all(model, imgs) @ embed_all(model, imgs).T).astype(np.float32)  # embedding cosine

    sg, tg = score_sim(G, gid, files); se, te = score_sim(E, gid, files)
    print(f"gradient-only : {sg:.4f} @thr={tg:.2f}")
    print(f"embedding-only: {se:.4f} @thr={te:.2f}")
    # fusions
    best = (0, None)
    for w in np.arange(0.0, 1.01, 0.1):
        sc, t = score_sim(w*G + (1-w)*E, gid, files)
        if sc > best[0]: best = (sc, f"weighted w_grad={w:.1f} @thr={t:.2f}")
    # max / min / product
    for name, F in [("max", np.maximum(G, E)), ("min", np.minimum(G, E)),
                    ("product", np.clip(G,0,1)*np.clip(E,0,1))]:
        sc, t = score_sim(F, gid, files)
        if sc > best[0]: best = (sc, f"{name} @thr={t:.2f}")
    print(f"FUSION best   : {best[0]:.4f}  ({best[1]})")

if __name__ == "__main__":
    main()

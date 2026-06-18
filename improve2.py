"""
Data-driven adaptive threshold (replaces global threshold AND the drone binary).

edge(i,j) iff sim(i,j) >= max(FLOOR, RATIO * min(smax_i, smax_j))
where smax_i = i's strongest similarity to any other image. The bar auto-rises
in dense/homogeneous regions (drone angles) and auto-lowers in sparse regions
(clipped interior ladders) -- one mechanism, no labels, no classifier.

Compares against the global-threshold baseline, then layers orphan re-attach.
"""
import csv, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "data/large")
d = np.load(DATA / "feat_cache.npz", allow_pickle=True)
M, C, B = d["M"], d["C"], d["B"]; files = list(d["files"])
n = len(files)
groups = defaultdict(set)
for r in csv.DictReader(open(DATA / "public_manifest.csv")):
    groups[r["group_id"]].add(r["filename"])

sim = M @ M.T
np.fill_diagonal(sim, -1)
smax = sim.max(1)                       # strongest neighbor per image
mins = np.minimum.outer(smax, smax)     # min(smax_i, smax_j)

def score(A):
    Aw = A.copy(); np.fill_diagonal(Aw, False)
    _, lab = connected_components(csr_matrix(Aw), directed=False)
    pred = defaultdict(set)
    for i, f in enumerate(files): pred[lab[i]].add(f)
    refsets = set(frozenset(v) for v in groups.values())
    predlk = set(frozenset(v) for v in pred.values())
    return len(refsets & predlk)/len(refsets), len(pred)

# global baseline for reference
gbest = max((score(sim >= t)[0], round(t,2)) for t in np.arange(0.50, 0.64, 0.02))
print(f"global-threshold best: {gbest[0]:.4f} @ {gbest[1]}")

print("\nadaptive: edge iff sim >= max(FLOOR, RATIO*min(smax_i,smax_j))")
best = (0, None)
for floor in [0.40, 0.44, 0.48, 0.52]:
    for ratio in [0.80, 0.85, 0.90, 0.93, 0.96]:
        thr = np.maximum(floor, ratio * mins)
        sc, npred = score(sim >= thr)
        if sc > best[0]: best = (sc, dict(floor=floor, ratio=ratio, pred=npred))
print(f"adaptive BEST: {best[0]:.4f}  {best[1]}")

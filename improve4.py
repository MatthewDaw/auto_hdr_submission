"""
Direction-1 training-free fixes on top of auto-threshold + orphan re-attach:
  (1) De-bridge: keep edge only if SUPPORTED (shares a common neighbor => part
      of a triangle) OR it's a mutual nearest-neighbor pair (protects true
      2-image groups). Kills weak lone bridges that chain groups via connected-
      components.
  (2) Relaxed orphan re-attach: also re-attach larger split pieces (not just
      size<=2) when they are exposure-extreme and census-match a single cluster.
"""
import csv, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "data/large")
d = np.load(DATA / "feat_cache.npz", allow_pickle=True)
M, C, B = d["M"], d["C"], d["B"]; files = list(d["files"]); n = len(files)
groups = defaultdict(set)
for r in csv.DictReader(open(DATA / "public_manifest.csv")):
    groups[r["group_id"]].add(r["filename"])
sim = M @ M.T
refsets = set(frozenset(v) for v in groups.values())
idxof = {f: i for i, f in enumerate(files)}

def score_of(lab):
    pred = defaultdict(set)
    for i, f in enumerate(files): pred[lab[i]].add(f)
    predlk = set(frozenset(v) for v in pred.values())
    return len(refsets & predlk)/len(refsets), len(pred)

def cc(A):
    A = A.copy(); np.fill_diagonal(A, False)
    _, lab = connected_components(csr_matrix(A), directed=False); return lab

def auto_thr():
    grid = np.arange(0.35, 0.75, 0.01); counts = np.array([len(set(cc(sim >= t))) for t in grid], float)
    W = 3; slope = np.full(len(grid), np.inf)
    for i in range(W, len(grid)-W): slope[i] = (counts[i+W]-counts[i-W])/(2*W)
    slope[counts <= 0.5*counts.max()] = np.inf
    cut = 1.3*slope[np.isfinite(slope)].min() + 0.5
    return grid[int(np.where(np.isfinite(slope) & (slope <= cut))[0][0])]

thr = auto_thr()
A = sim >= thr
print(f"auto-threshold={thr:.2f}  CC baseline={score_of(cc(A))[0]:.4f}")

# --- (1) de-bridge ---
nn = np.argsort(sim - 2*np.eye(n), axis=1)[:, -1]   # nearest neighbor (excl self)
mutual = np.zeros((n, n), bool)
for i in range(n):
    if nn[nn[i]] == i: mutual[i, nn[i]] = mutual[nn[i], i] = True
np.fill_diagonal(A, False)                  # remove self-loops BEFORE path counting
Asp = csr_matrix(A.astype(np.int8))
common = (Asp @ Asp).toarray() > 0          # triangle: real length-2 path i->k->j
keep = A & (common | mutual)
labD = cc(keep)
scD, npD = score_of(labD)
print(f"+ de-bridge          = {scD:.4f} (pred {npD})")

# --- (2) relaxed orphan re-attach (on de-bridged labels) ---
def census_rows(i, allidx):
    x = np.bitwise_xor(C[i][None, :], C[allidx])
    return 1.0 - np.unpackbits(x, axis=1).sum(1) / (C.shape[1]*8.0)
lab = labD.copy()
cl = defaultdict(list)
for i, l in enumerate(lab): cl[l].append(i)
EXTREME = (B < 45) | (B > 210); moved = 0
for l, members in sorted(cl.items(), key=lambda kv: len(kv[1])):
    if len(members) > 2 or not any(EXTREME[i] for i in members): continue
    for i in list(members):
        if not EXTREME[i]: continue
        others = [j for j in range(n) if lab[j] != lab[i]]
        cs = census_rows(i, others); order = np.argsort(cs)[::-1]
        bestj = others[order[0]]
        secj = next((others[k] for k in order[1:] if lab[others[k]] != lab[bestj]), None)
        sec = census_rows(i, [secj])[0] if secj is not None else 0
        if cs[order[0]] >= 0.62 and cs[order[0]]-sec >= 0.03 and abs(B[i]-B[bestj]) >= 25:
            lab[i] = lab[bestj]; moved += 1
scF, npF = score_of(lab)
print(f"+ relaxed orphan     = {scF:.4f} (moved {moved}, pred {npF})")

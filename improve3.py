"""Find true global-threshold peak (full sweep) + re-apply orphan re-attach on it."""
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

def cluster(A):
    Aw = A.copy(); np.fill_diagonal(Aw, False)
    _, lab = connected_components(csr_matrix(Aw), directed=False)
    return lab

def score(lab):
    pred = defaultdict(set)
    for i, f in enumerate(files): pred[lab[i]].add(f)
    predlk = set(frozenset(v) for v in pred.values())
    return len(refsets & predlk)/len(refsets), len(pred)

print("full global sweep:")
best = (0, None, None)
for t in np.arange(0.54, 0.78, 0.01):
    lab = cluster(sim >= t); sc, npq = score(lab)
    mark = ""
    if sc > best[0]: best = (sc, round(t,2), lab); mark = " *"
    print(f"  thr={t:.2f}: {sc:.4f} (pred {npq}){mark}")
scG, thrG, labG = best
print(f"GLOBAL BEST {scG:.4f} @ {thrG}")

# orphan re-attach on the true-best baseline
def census_rows(idx_i, allidx):
    P8 = C.shape[1]*8.0; out=np.zeros(len(allidx),np.float32)
    x = np.bitwise_xor(C[idx_i][None,:], C[allidx])
    return 1.0 - np.unpackbits(x,axis=1).sum(1)/P8
cl = defaultdict(list)
for i,l in enumerate(labG): cl[l].append(i)
EXTREME = (B<45)|(B>210); labC=labG.copy(); moved=0
for l,members in list(cl.items()):
    if len(members)>2: continue
    if not any(EXTREME[i] for i in members): continue
    for i in members:
        others=[j for j in range(n) if labC[j]!=labC[i]]
        cs=census_rows(i,others); order=np.argsort(cs)[::-1]
        bestj=others[order[0]]
        secj=next((others[k] for k in order[1:] if labC[others[k]]!=labC[bestj]),None)
        sec=census_rows(i,[secj])[0] if secj is not None else 0
        if cs[order[0]]>=0.62 and (cs[order[0]]-sec)>=0.03 and abs(B[i]-B[bestj])>=25:
            labC[i]=labC[bestj]; moved+=1
scC,npC=score(labC)
print(f"+ orphan re-attach: {scC:.4f} (moved {moved})")

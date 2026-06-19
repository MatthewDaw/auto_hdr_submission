"""
Re-tune the gradient/descriptor fusion weight for the wavelet+eigenface descriptor
(was W_GRAD=0.65, tuned for the CNN). Sweep weight x threshold on the large set,
report best base-fusion fixable score + the still-missed groups for diagnosis.
"""
import csv, json, sys
from collections import defaultdict
import numpy as np
import descriptor
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

DATA=sys.argv[1] if len(sys.argv)>1 else "data/large"
fc=np.load(f"{DATA}/feat_cache.npz",allow_pickle=True); M=fc["M"]; files=list(fc["files"]); n=len(files)
raw=np.load(f"{DATA}/raw256.npz",allow_pickle=True)["imgs"]
gid={}
for r in csv.DictReader(open(f"{DATA}/public_manifest.csv",encoding="utf-8")): gid[r["filename"]]=r["group_id"]
groups=defaultdict(set)
for f in files: groups[gid[f]].add(f)
refsets=set(frozenset(v) for v in groups.values())
unfix=set(json.load(open(f"{DATA}/unfixable.json",encoding="utf-8"))["groups"].keys())
fixable={g for g in groups if g not in unfix}
G=(M@M.T).astype(np.float32); E=descriptor.embed(raw); Es=(E@E.T).astype(np.float32)

def evalfuse(w):
    sim=(w*G+(1-w)*Es).astype(np.float32); best=(0,0,None)
    for t in np.arange(0.15,0.97,0.01):
        A=sim>=t; np.fill_diagonal(A,False)
        _,lab=connected_components(csr_matrix(A),directed=False)
        pred=defaultdict(set)
        for i,f in enumerate(files): pred[lab[i]].add(f)
        pl=set(frozenset(v) for v in pred.values()); ok={g for g,v in groups.items() if frozenset(v) in pl}
        s=len(ok&fixable)/len(fixable)
        if s>best[0]: best=(s,round(t,2),ok)
    return best

print(f"{DATA}: {len(fixable)} fixable groups. Sweeping fusion weight (base fusion, oracle threshold):")
res=[]
for w in np.arange(0.0,1.01,0.1):
    s,t,ok=evalfuse(w); res.append((s,w,t,ok)); print(f"  w_grad={w:.1f}: base fixable={s:.4f} @thr={t}")
best=max(res,key=lambda r:r[0])
print(f"\nBEST base weight w_grad={best[1]:.1f} -> {best[0]:.4f}")
missed=sorted(fixable-best[3])
print(f"still-missed FIXABLE at base ({len(missed)}): {missed}")

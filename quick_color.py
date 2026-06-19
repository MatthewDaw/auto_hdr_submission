import csv, json
from collections import defaultdict
import numpy as np, descriptor
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
DATA="data/large"
fc=np.load(f"{DATA}/feat_cache.npz",allow_pickle=True); M=fc["M"]; files=list(fc["files"])
raw=np.load(f"{DATA}/raw256.npz",allow_pickle=True)["imgs"]
col=np.load(f"{DATA}/img128c.npz",allow_pickle=True)["imgs"]
gid={r["filename"]:r["group_id"] for r in csv.DictReader(open(f"{DATA}/public_manifest.csv",encoding="utf-8"))}
groups=defaultdict(set)
for f in files: groups[gid[f]].add(f)
unfix=set(json.load(open(f"{DATA}/unfixable.json",encoding="utf-8"))["groups"].keys())
fixable={g for g in groups if g not in unfix}
G=(M@M.T).astype(np.float32)
def best(E):
    Es=(E@E.T).astype(np.float32); bestv=(0,None)
    for w in [0.4,0.5,0.6,0.65,0.7]:
        sim=(w*G+(1-w)*Es).astype(np.float32)
        for t in np.arange(0.2,0.95,0.01):
            A=sim>=t; np.fill_diagonal(A,False); _,lab=connected_components(csr_matrix(A),directed=False)
            pred=defaultdict(set)
            for i,f in enumerate(files): pred[lab[i]].add(f)
            ok={g for g,v in groups.items() if frozenset(v) in set(frozenset(x) for x in pred.values())}
            s=len(ok&fixable)/len(fixable)
            if s>bestv[0]: bestv=(s,ok,w)
    return bestv
sg,okg,wg=best(descriptor.embed(raw)); print(f"gray  base fixable={sg:.4f} @w={wg}")
sc,okc,wc=best(descriptor.embed_cw(raw,col)); print(f"gray+color base fixable={sc:.4f} @w={wc}")
print("color recovers (vs gray):", sorted(okc-okg))
print("color loses:", sorted(okg-okc))

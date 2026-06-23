"""Probe current full-grouper behavior on each named case (isolation proxy).

LINK groups: run the group's frames alone -> want 1 cluster.
SPLIT pairs: run the union of the two groups -> want them separated (each GT
group lands entirely in its own cluster).
"""
import csv, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from autohdr import ImageGrouper
from autohdr.image_loader import Photoshoot

def load(ddir):
    raw = np.load(Path(ddir)/"raw256.npz", allow_pickle=True)
    files=list(raw["files"]); imgs=raw["imgs"]; idx={f:i for i,f in enumerate(files)}
    col=np.load(Path(ddir)/"img128c.npz", allow_pickle=True)
    cidx={f:i for i,f in enumerate(col["files"])}; cimgs=col["imgs"]
    gt=defaultdict(list)
    with open(Path(ddir)/"public_manifest.csv",encoding="utf-8") as f:
        for row in csv.DictReader(f): gt[row["group_id"]].append(row["filename"])
    return files,imgs,idx,cidx,cimgs,gt

def run(ddir, groups):
    files,imgs,idx,cidx,cimgs,gt=load(ddir)
    fr=[]; owner={}
    for g in groups:
        for f in gt.get(g,[]):
            if f in idx: fr.append(f); owner[f]=g
    if not fr: return None
    gray=np.stack([imgs[idx[f]] for f in fr])
    color=np.stack([cimgs[cidx[f]] for f in fr]) if all(f in cidx for f in fr) else None
    out=ImageGrouper().group(Photoshoot(fr,gray,color))
    # map each cluster to GT composition
    comp=[]
    for cl in out:
        c=defaultdict(int)
        for f in cl: c[owner[f]]+=1
        comp.append(dict(c))
    return comp

LINK=[["73234"],["10125"],["10129"],["10463"],["11992"],["19300"]]
SPLIT=[["10280","1038"],["10464","10613"],["11533","11604"],["14037","14288"],
       ["14279","14983"],["12226","13169"],["10411","10412"],["16527","16528"]]

for grp in LINK:
    ddir="data/large" if grp==["73234"] else "data/full_subset"
    comp=run(ddir,grp)
    if comp is None: print(f"LINK {grp}: MISSING"); continue
    ok = len(comp)==1
    print(f"LINK {grp}: {len(comp)} cluster(s) {'OK' if ok else 'FAIL'}  {comp}")
for grp in SPLIT:
    comp=run("data/full_subset",grp)
    if comp is None: print(f"SPLIT {grp}: MISSING"); continue
    # OK iff each GT group is the sole owner of distinct clusters (no cluster mixes both)
    mixed=any(len(c)>1 for c in comp)
    print(f"SPLIT {grp}: {len(comp)} cluster(s) {'FAIL-mixed' if mixed else 'OK-sep'}  {comp}")

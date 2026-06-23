"""FAST spike for the two WELL-EXPOSED cases (12226/13169 split, 10411/10412 move).

Loads from npz caches, runs the FULL grouper on the union, prints cluster
composition + cross masked-ZNCC for the cases of interest.
"""
import csv, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from autohdr import ImageGrouper
from autohdr.image_loader import Photoshoot
from autohdr.features import MaskedCorrelation

DDIR = "data/full_subset"

def load(ddir):
    raw = np.load(Path(ddir)/"raw256.npz", allow_pickle=True)
    files=list(raw["files"]); imgs=raw["imgs"]; idx={f:i for i,f in enumerate(files)}
    col=np.load(Path(ddir)/"img128c.npz", allow_pickle=True)
    cidx={f:i for i,f in enumerate(col["files"])}; cimgs=col["imgs"]
    gt=defaultdict(list)
    with open(Path(ddir)/"public_manifest.csv",encoding="utf-8") as f:
        for row in csv.DictReader(f): gt[row["group_id"]].append(row["filename"])
    return files,imgs,idx,cidx,cimgs,gt

def run(ddir, groups, focus=None):
    files,imgs,idx,cidx,cimgs,gt=load(ddir)
    fr=[]; owner={}
    for g in groups:
        for f in gt.get(g,[]):
            if f in idx: fr.append(f); owner[f]=g
    if not fr:
        print("MISSING", groups); return
    gray=np.stack([imgs[idx[f]] for f in fr])
    color=np.stack([cimgs[cidx[f]] for f in fr]) if all(f in cidx for f in fr) else None

    # brightness + masked ZNCC over the union frames
    B=gray.reshape(len(fr),-1).mean(1)
    mc=MaskedCorrelation(gray)
    print(f"\n=== {groups}  ({len(fr)} frames) ===")
    for f in fr:
        print(f"  {f:30s} GT={owner[f]:6s} B={B[fr.index(f)]:6.1f}")

    out=ImageGrouper().group(Photoshoot(fr,gray,color))
    comp=[]
    for cl in out:
        c=defaultdict(int)
        for f in cl: c[owner[f]]+=1
        comp.append((dict(c), sorted(cl)))
    print(" CLUSTERS:")
    for c,members in comp:
        print(f"   {c}")
    mixed=any(len(c)>1 for c,_ in comp)
    print(f" RESULT split-clean={not mixed}")

    if focus:
        print(" FOCUS cross masked-ZNCC:")
        for fa, fb in focus:
            if fa in fr and fb in fr:
                i,j=fr.index(fa),fr.index(fb)
                z,ov=mc.score(i,j)
                print(f"   {fa} vs {fb}: zncc={z:.3f} overlap={ov} |dB|={abs(B[i]-B[j]):.1f}")
    return comp, fr, owner

# Case 1: 12226 / 13169 should SPLIT
run(DDIR, ["12226","13169"])

# Case 2: 10411 / 10412 — B58 (GT 10411) should be with 10411
c2 = run(DDIR, ["10411","10412"])
# find frame whose stem ends 58
if c2:
    _,fr,owner=c2
    for f in fr:
        if f.split(".")[0].endswith("58"):
            print("  B58 candidate:", f, "GT", owner[f])

# explicit cross check for the misplaced 10411 frame vs both clusters
print("\n--- 10411/10412 cross masked-ZNCC matrix ---")
files,imgs,idx,cidx,cimgs,gt=load(DDIR)
groups=["10411","10412"]
fr=[]; owner={}
for g in groups:
    for f in gt.get(g,[]):
        if f in idx: fr.append(f); owner[f]=g
gray=np.stack([imgs[idx[f]] for f in fr])
B=gray.reshape(len(fr),-1).mean(1)
mc=MaskedCorrelation(gray)
target="g10411_A7406941.jpg"
ti=fr.index(target)
for f in fr:
    if f==target: continue
    j=fr.index(f)
    z,ov=mc.score(ti,j)
    print(f"  {target} vs {f:22s} GT={owner[f]} z={z:.3f} ov={ov} |dB|={abs(B[ti]-B[j]):.1f}")

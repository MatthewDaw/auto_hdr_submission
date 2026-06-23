"""Identify the stranded singleton orphan in each LINK case and why it doesn't merge.
Print orphan brightness, its masked-ZNCC overlap+zncc vs big-cluster well rep, anchor,
and coverage of any clipped big-cluster frame vs an orphan-built template."""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
from autohdr.features import MaskedCorrelation, GradientDescriptor, WaveletEmbedding, ChromaSignature
from autohdr.clustering import FusionClusterer
from autohdr.image_loader import Photoshoot
from autohdr.refinement import (OrphanReattachment, ClusterMerging, HighResSplitter,
    ClippedReattachment, CaptureRunSplitter, SceneChangeSplitter, MotionSplitter,
    AnchorSplitter, RefinementContext)
from autohdr.refinement.anchor_reattachment import _anchor_masks, _iou
from autohdr.features import extreme_anchor as ea

def load(ddir):
    raw=np.load(Path(ddir)/"raw256.npz",allow_pickle=True)
    files=list(raw["files"]); imgs=raw["imgs"]; idx={f:i for i,f in enumerate(files)}
    col=np.load(Path(ddir)/"img128c.npz",allow_pickle=True)
    cidx={f:i for i,f in enumerate(col["files"])}; cimgs=col["imgs"]
    gt=defaultdict(list)
    with open(Path(ddir)/"public_manifest.csv",encoding="utf-8") as f:
        for row in csv.DictReader(f): gt[row["group_id"]].append(row["filename"])
    return files,imgs,idx,cidx,cimgs,gt

def probe(ddir, groups):
    files,imgs,idx,cidx,cimgs,gt=load(ddir)
    fr=[]
    for g in groups:
        for f in gt.get(g,[]):
            if f in idx: fr.append(f)
    gray=np.stack([imgs[idx[f]] for f in fr])
    color=np.stack([cimgs[cidx[f]] for f in fr]) if all(f in cidx for f in fr) else None
    ps=Photoshoot(fr,gray,color)
    grad=GradientDescriptor().encode(gray); emb=WaveletEmbedding().encode(gray)
    graph=FusionClusterer(grad,emb).initial_graph()
    ctx=RefinementContext(brightness=ps.brightness,embedding=emb,
        masked=MaskedCorrelation(gray),filenames=fr,gray=gray,chroma=ChromaSignature(color))
    OrphanReattachment().apply(graph,ctx); ClusterMerging().apply(graph,ctx)
    labels=graph.labels()
    for p in [HighResSplitter(),ClippedReattachment(),CaptureRunSplitter(),
              SceneChangeSplitter(),MotionSplitter(),AnchorSplitter()]:
        labels=p.apply(labels,ctx)
    B=ctx.brightness
    clusters=defaultdict(list)
    for i,l in enumerate(labels): clusters[int(l)].append(i)
    print(f"CASE {groups}: {[(c,len(m),[round(B[k],0) for k in m]) for c,m in clusters.items()]}")
    sizes=sorted(clusters.items(),key=lambda kv:len(kv[1]))
    small=sizes[0][1]; big=[m for c,m in clusters.items() if c!=sizes[0][0]][0]
    masks=[_anchor_masks(gray[i]) for i in range(len(fr))]
    for oi in small:
        print(f" orphan idx{oi} mean={B[oi]:.1f} clipped={B[oi]<30 or B[oi]>225}")
        for j in big:
            z,ov=ctx.masked.score(oi,j)
            bi,di=masks[oi]; bj,dj=masks[j]
            an=0.5*(_iou(bi,bj)+_iou(di,dj))
            print(f"   vs big idx{j} mean={B[j]:.1f}: zncc={z:.3f} overlap={ov} anchor={an:.3f}")

for ddir,gs in [("data/large",["73234"]),("data/full_subset",["10125"]),
    ("data/full_subset",["10129"]),("data/full_subset",["19300"])]:
    probe(ddir,gs)

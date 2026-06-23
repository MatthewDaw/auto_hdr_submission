"""Trace labels after each refinement pass for in-scope cases."""
import csv, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from autohdr.clustering import FusionClusterer
from autohdr.features import ChromaSignature, GradientDescriptor, MaskedCorrelation, WaveletEmbedding
from autohdr.image_loader import Photoshoot
from autohdr.refinement import (AnchorReattachment, AnchorSplitter, CaptureRunSplitter,
    ClippedReattachment, ClusterMerging, HighResSplitter, MotionSplitter,
    OrphanReattachment, RefinementContext, SceneChangeSplitter)

def load(ddir):
    raw=np.load(Path(ddir)/"raw256.npz",allow_pickle=True)
    files=list(raw["files"]); imgs=raw["imgs"]; idx={f:i for i,f in enumerate(files)}
    col=np.load(Path(ddir)/"img128c.npz",allow_pickle=True)
    cidx={f:i for i,f in enumerate(col["files"])}; cimgs=col["imgs"]
    gt=defaultdict(list)
    with open(Path(ddir)/"public_manifest.csv",encoding="utf-8") as f:
        for row in csv.DictReader(f): gt[row["group_id"]].append(row["filename"])
    return files,imgs,idx,cidx,cimgs,gt

def show(tag, labels, owner, fr):
    cl=defaultdict(list)
    for f,l in zip(fr,labels): cl[int(l)].append(owner[f])
    comp=[dict((g,c.count(g)) for g in set(c)) for c in cl.values()]
    print(f"  {tag:22s} {len(cl)} cl  {comp}")

def trace(ddir, groups):
    files,imgs,idx,cidx,cimgs,gt=load(ddir)
    fr=[]; owner={}
    for g in groups:
        for f in gt.get(g,[]):
            if f in idx: fr.append(f); owner[f]=g
    gray=np.stack([imgs[idx[f]] for f in fr])
    color=np.stack([cimgs[cidx[f]] for f in fr]) if all(f in cidx for f in fr) else None
    ps=Photoshoot(fr,gray,color)
    print(f"CASE {groups} brightness={[round(float(b),1) for b in ps.brightness]}")
    gradient=GradientDescriptor().encode(gray); embedding=WaveletEmbedding().encode(gray)
    graph=FusionClusterer(gradient,embedding).initial_graph()
    ctx=RefinementContext(brightness=ps.brightness,embedding=embedding,
        masked=MaskedCorrelation(gray),filenames=fr,gray=gray,chroma=ChromaSignature(color))
    OrphanReattachment().apply(graph,ctx); ClusterMerging().apply(graph,ctx)
    labels=graph.labels(); show("init",labels,owner,fr)
    for name,p in [("HighRes",HighResSplitter()),("Clipped",ClippedReattachment()),
        ("CaptureRun",CaptureRunSplitter()),("SceneChange",SceneChangeSplitter()),
        ("Motion",MotionSplitter()),("AnchorSplit",AnchorSplitter()),
        ("AnchorReattach",AnchorReattachment())]:
        labels=p.apply(labels,ctx); show(name,labels,owner,fr)
    print()

for g in [["73234"]]:
    trace("data/large",g)
for g in [["10125"],["10129"],["19300"],["11533","11604"],["14037","14288"]]:
    trace("data/full_subset",g)

"""Probe AnchorReattachment internals for LINK cases: overlap of clipped orphan vs
well rep, anchor, coverage."""
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
    print(f"CASE {groups}: clusters {[(c,len(m)) for c,m in clusters.items()]}")
    for i in range(len(fr)):
        if B[i]<30 or B[i]>225:
            own=int(labels[i])
            print(f" clipped frame {i} mean={B[i]:.1f} own_cluster_size={len(clusters[own])}")
            for c,m in clusters.items():
                if c==own: continue
                well=[k for k in m if 30<=B[k]<=225]
                rep=min(well,key=lambda k:abs(B[k]-128.0)) if well else None
                ov=ctx.masked.score(i,rep)[1] if rep is not None else -1
                wt=[gray[k] for k in m if ea.is_well_exposed(gray[k])]
                tmpl=ea.build_template(wt) if wt else None
                pol=ea.clip_polarity(gray[i])
                cov=ea.coverage_score(gray[i],tmpl,pol) if tmpl else None
                print(f"   -> cluster {c} size={len(m)} rep_overlap={ov} cov={cov}")

for ddir,gs in [("data/large",["73234"]),("data/full_subset",["10125"]),
    ("data/full_subset",["10129"]),("data/full_subset",["19300"])]:
    probe(ddir,gs)

"""Probe AnchorSplitter._partition for the two in-scope SPLIT cases at the point it
runs (after the masked passes have merged them). Print: well-scene count, clip
clusters, multi_well, per-clip coverage vs each well-scene template."""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
from autohdr.features import MaskedCorrelation, GradientDescriptor, WaveletEmbedding, ChromaSignature
from autohdr.clustering import FusionClusterer
from autohdr.image_loader import Photoshoot
from autohdr.refinement import (OrphanReattachment, ClusterMerging, HighResSplitter,
    ClippedReattachment, CaptureRunSplitter, SceneChangeSplitter, MotionSplitter,
    RefinementContext)
from autohdr.refinement import anchor_splitter as AS
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

def probe(groups, ddir="data/full_subset"):
    files,imgs,idx,cidx,cimgs,gt=load(ddir)
    fr=[]; owner={}
    for g in groups:
        for f in gt.get(g,[]):
            if f in idx: fr.append(f); owner[f]=g
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
              SceneChangeSplitter(),MotionSplitter()]:
        labels=p.apply(labels,ctx)
    B=ctx.brightness
    clusters=defaultdict(list)
    for i,l in enumerate(labels): clusters[int(l)].append(i)
    print(f"CASE {groups}: pre-AnchorSplit clusters {[(c,[owner[fr[k]] for k in m]) for c,m in clusters.items()]}")
    for c,members in clusters.items():
        if len(members)<AS._MIN_MEMBERS: continue
        n=len(members)
        def AM(i,j): return AS.anchor_match(gray[members[i]],gray[members[j]])
        well=[i for i in range(n) if AS._WELL_LOW<=B[members[i]]<=AS._WELL_HIGH]
        # seed scenes
        w=len(well); link=np.zeros((w,w),bool)
        for a in range(w):
            for b in range(a+1,w):
                z,ov=ctx.masked.score(members[well[a]],members[well[b]])
                if ov>=AS._MIN_OVERLAP and z>=AS._SAME_SCENE: link[a,b]=link[b,a]=True
        comp=AS.AnchorSplitter._components(link)
        n_scene=(int(max(comp))+1) if w else 0
        multi=n_scene>=2
        print(f" cluster size {n}: well={[ (members[i],round(B[members[i]],0)) for i in well]} n_scene={n_scene} multi_well={multi}")
        reps=defaultdict(list)
        for li,wi in enumerate(well): reps[comp[li]].append(wi)
        clp=[i for i in range(n) if i not in set(well)]
        cc=AS.AnchorSplitter()._anchor_clusters(clp,B,members,AM)
        print(f"   clip_clusters: {[[ (members[i],owner[fr[members[i]]],round(B[members[i]],0)) for i in g] for g in cc]}")
        clipped=[i for i in range(n) if i not in set(well_idx for li,well_idx in enumerate(well))]
        clipped=[i for i in range(n) if i not in set(well)]
        print(f"   clipped frame idxs (mean): {[(members[i],round(B[members[i]],0)) for i in clipped]}")
        # build per-scene template, cov of each clipped
        for s,seeds in reps.items():
            wt=[gray[members[wi]] for wi in seeds if ea.is_well_exposed(gray[members[wi]])]
            t=ea.build_template(wt) if wt else None
            for ci in clipped:
                pol=ea.clip_polarity(gray[members[ci]])
                hard=B[members[ci]]<AS._CLIP_HARD_DARK or B[members[ci]]>AS._CLIP_HARD_WHITE
                cov=ea.coverage_score(gray[members[ci]],t,pol) if t else None
                print(f"     clip idx{members[ci]} owner={owner[fr[members[ci]]]} hard={hard} std={gray[members[ci]].std():.1f} cov_vs_scene{s}={cov}")

probe(["11533","11604"])
probe(["14037","14288"])

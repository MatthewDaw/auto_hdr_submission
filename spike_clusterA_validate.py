"""Validate Cluster A: anchor reattachment across large exposure jumps.

For each target case we build the RefinementContext, run every existing pass in
the production order, then OUR AnchorReattachment, and confirm the case collapses
to ONE correct cluster. We then confirm the safety set is unchanged (still one
cluster each, i.e. nothing wrongly merged or split).

Cases live in data/full_subset (checked against data/large too). Run:
    python spike_clusterA_validate.py
"""
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from autohdr.features import (
    ChromaSignature,
    GradientDescriptor,
    MaskedCorrelation,
    WaveletEmbedding,
)
from autohdr.clustering import FusionClusterer
from autohdr.image_loader import Photoshoot
from autohdr.refinement import (
    CaptureRunSplitter,
    ClippedReattachment,
    ClusterMerging,
    HighResSplitter,
    OrphanReattachment,
    RefinementContext,
    SceneChangeSplitter,
)
from autohdr.refinement.anchor_reattachment import AnchorReattachment, anchor_match

CASES = ["10370", "10463", "14055", "19300", "13675"]
SAFETY = ["14258", "14259", "14260", "14261", "14262",
          "10147", "10160", "13601", "13609", "12583",
          "13093", "13586", "13590", "14011"]


def load(dname):
    ddir = Path("data") / dname
    raw = np.load(ddir / "raw256.npz", allow_pickle=True)
    files = [str(f) for f in raw["files"]]
    imgs = raw["imgs"]
    idx = {f: i for i, f in enumerate(files)}
    col = np.load(ddir / "img128c.npz", allow_pickle=True)
    cidx = {str(f): i for i, f in enumerate(col["files"])}
    cimgs = col["imgs"]
    gt = defaultdict(list)
    for r in csv.DictReader(open(ddir / "public_manifest.csv", encoding="utf-8")):
        gt[r["group_id"]].append(r["filename"])
    return imgs, idx, cimgs, cidx, gt


SOURCES = {d: load(d) for d in ("full_subset", "large") if (Path("data") / d / "raw256.npz").exists()}


def find_case(g):
    """Return (dname, frames) for the first cache that has this group's frames."""
    for dname, (imgs, idx, cimgs, cidx, gt) in SOURCES.items():
        fr = [f for f in gt.get(g, []) if f in idx]
        if fr:
            return dname, fr
    return None, []


def run_pipeline(dname, frames, with_anchor):
    imgs, idx, cimgs, cidx, gt = SOURCES[dname]
    gray = np.stack([imgs[idx[f]] for f in frames])
    color = (np.stack([cimgs[cidx[f]] for f in frames])
             if all(f in cidx for f in frames) else None)
    ps = Photoshoot(frames, gray, color)

    grad = GradientDescriptor().encode(gray)
    emb = WaveletEmbedding().encode(gray)
    graph = FusionClusterer(grad, emb).initial_graph()
    ctx = RefinementContext(
        brightness=ps.brightness, embedding=emb,
        masked=MaskedCorrelation(gray), filenames=ps.filenames,
        gray=gray, chroma=ChromaSignature(color),
    )
    OrphanReattachment().apply(graph, ctx)
    ClusterMerging().apply(graph, ctx)
    labels = graph.labels()
    labels = HighResSplitter().apply(labels, ctx)
    labels = ClippedReattachment().apply(labels, ctx)
    labels = CaptureRunSplitter().apply(labels, ctx)
    labels = SceneChangeSplitter().apply(labels, ctx)
    if with_anchor:
        labels = AnchorReattachment().apply(labels, ctx)
    return ps, ctx, labels


def n_clusters(labels):
    return len(set(int(x) for x in labels))


print("=" * 72)
print("CASES — must collapse to 1 cluster after AnchorReattachment")
print("=" * 72)
case_ok = {}
for g in CASES:
    dname, frames = find_case(g)
    if not frames:
        print(f"  {g}: NOT FOUND in any cache"); case_ok[g] = False; continue
    ps, ctx, before = run_pipeline(dname, frames, with_anchor=False)
    _, _, after = run_pipeline(dname, frames, with_anchor=True)
    B = ps.brightness

    # report the clipped frames' best anchor scores to a different cluster
    clipped = [i for i in range(len(frames)) if B[i] < 30 or B[i] > 225]
    detail = []
    for i in clipped:
        best = -1.0
        for j in range(len(frames)):
            if before[j] != before[i]:
                s = anchor_match(ps.gray[i], ps.gray[j], B[i], B[j])
                best = max(best, s)
        detail.append(f"B{B[i]:.0f}->{best:.2f}" if best >= 0 else f"B{B[i]:.0f}->(none)")

    nb, na = n_clusters(before), n_clusters(after)
    ok = na == 1
    # classify a residual split: is the split-off frame clipped (in scope) or a
    # well-exposed frame (out of scope for an anchor-on-clipped-frames pass)?
    note = ""
    if not ok:
        sizes = Counter(int(x) for x in after)
        biggest = max(sizes.values())
        orphan_lbls = {lbl for lbl, n in sizes.items() if n < biggest}
        orphan_idx = [i for i in range(len(frames)) if int(after[i]) in orphan_lbls]
        if orphan_idx and all(30 <= B[i] <= 225 for i in orphan_idx):
            note = " (split frame is WELL-EXPOSED -> out of anchor scope)"
    case_ok[g] = ok
    flag = "OK" if ok else "OUT-OF-SCOPE" if note else "FAIL"
    print(f"  {g} [{dname}] {len(frames)}fr  before={nb} after={na}  "
          f"anchors[{' '.join(detail) if detail else '-'}]  {flag}{note}")

print()
print("=" * 72)
print("SAFETY — must stay 1 cluster (nothing wrongly merged/split)")
print("=" * 72)
safe_ok = True
for g in SAFETY:
    dname, frames = find_case(g)
    if not frames:
        print(f"  {g}: NOT FOUND"); continue
    _, _, before = run_pipeline(dname, frames, with_anchor=False)
    _, _, after = run_pipeline(dname, frames, with_anchor=True)
    nb, na = n_clusters(before), n_clusters(after)
    changed = nb != na
    if changed:
        safe_ok = False
    print(f"  {g} [{dname}] {len(frames)}fr  before={nb} after={na}  "
          f"{'CHANGED!' if changed else 'unchanged'}")

print()
print("=" * 72)
print("ANCHOR SCORE DISTRIBUTION (same-scene vs different-scene)")
print("=" * 72)
imgs, idx, cimgs, cidx, gt = SOURCES["full_subset"]
B = {f: float(imgs[idx[f]].mean()) for g in gt for f in gt[g] if f in idx}
groupfr = {g: [f for f in fs if f in idx] for g, fs in gt.items()}


def cl_best(fa, g):
    return max((anchor_match(imgs[idx[fa]], imgs[idx[fb]], B[fa], B[fb])
                for fb in groupfr[g] if fb != fa), default=0.0)


same, diff = [], []
others = [g for g in groupfr if len(groupfr[g]) >= 2 and g not in CASES]
import random
random.seed(0)
others = random.sample(others, min(400, len(others)))
for g in CASES:
    for f in groupfr.get(g, []):
        if B[f] < 30 or B[f] > 225:
            same.append(cl_best(f, g))
            diff.append(max(cl_best(f, gg) for gg in others if gg != g))
if same:
    print(f"  same-scene (clipped->own cluster): "
          f"min={min(same):.2f} med={np.median(same):.2f} max={max(same):.2f}")
    print(f"  diff-scene (clipped->best other) : "
          f"min={min(diff):.2f} med={np.median(diff):.2f} max={max(diff):.2f}")
    print(f"  chosen FLOOR=0.34  MARGIN=0.10  "
          f"(min same-margin over diff = {min(s - d for s, d in zip(same, diff)):+.2f})")

print()
print("=" * 72)
allcases = all(case_ok.values())
print(f"CASES all resolve to 1 cluster : {allcases}")
print(f"SAFETY set unchanged           : {safe_ok}")
print("=" * 72)

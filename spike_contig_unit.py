"""Deterministic unit test of ContiguityReattachment — no clustering run.

Construct the exact failure state (the clipped frame as its own singleton, the rest
of the bracket as one cluster) and run ONLY the pass. If it reattaches, the fix
works regardless of how the singleton arose in the full pipeline.
"""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
from autohdr.features import ChromaSignature, MaskedCorrelation, WaveletEmbedding
from autohdr.refinement import ContiguityReattachment, RefinementContext

d = Path("data/full_subset")
raw = np.load(d/"raw256.npz", allow_pickle=True)
files = list(raw["files"]); imgs = raw["imgs"]; idx = {f: i for i, f in enumerate(files)}
col = np.load(d/"img128c.npz", allow_pickle=True)
cidx = {f: i for i, f in enumerate(col["files"])}; cimgs = col["imgs"]
gt = defaultdict(list)
for r in csv.DictReader(open(d/"public_manifest.csv", encoding="utf-8")):
    gt[r["group_id"]].append(r["filename"])


def test(group, clip_pred):
    """clip_pred(B) -> True if this frame should start stranded as a singleton."""
    fr = sorted(gt[group], key=lambda f: imgs[idx[f]].mean())
    gray = np.stack([imgs[idx[f]] for f in fr])
    color = np.stack([cimgs[cidx[f]] for f in fr])
    B = gray.reshape(len(fr), -1).mean(1)
    ctx = RefinementContext(
        brightness=B, embedding=WaveletEmbedding().encode(gray),
        masked=MaskedCorrelation(gray), filenames=fr, gray=gray,
        chroma=ChromaSignature(color))
    # construct the failure state: stranded clipped frame(s) each own singleton,
    # everything else in cluster 0
    labels = np.zeros(len(fr), int); nxt = 1
    for i in range(len(fr)):
        if clip_pred(B[i]):
            labels[i] = nxt; nxt += 1
    before = len(set(labels))
    out = ContiguityReattachment().apply(labels, ctx)
    after = len(set(out.tolist()))
    print(f"{group}: B={[round(b) for b in B]}  before={before} clusters -> after={after}  "
          f"{'REATTACHED-OK' if after == 1 else 'STILL SPLIT'}")


print("11992 (near-white B254 stranded):")
test("11992", lambda b: b > 225)
print("12133 (near-black B9 stranded):")
test("12133", lambda b: b < 30)
